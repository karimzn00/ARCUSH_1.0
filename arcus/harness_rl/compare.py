# arcus/harness_rl/compare.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── Patch: CoinRun and Breakout removed from experiment ──────────────────────
DISCRETE_ENVS    = {"CartPole-v1", "Acrobot-v1", "FrozenLake-v1", "MountainCar-v0",
                    "ALE/Pong-v5"}
ON_POLICY_ALGOS  = {"a2c", "ppo", "trpo"}
OFF_POLICY_ALGOS = {"dqn", "ddpg", "sac", "td3"}
IDENTITY_COMPONENTS = ["competence", "coherence", "continuity", "integrity", "meaning"]

# ── Patch: publication-quality rcParams ──────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.dpi":        150,    # screen preview
    "savefig.dpi":       300,    # publication quality
    "pdf.fonttype":      42,     # embed fonts (required by most journals)
    "ps.fonttype":       42,
})


def _savefig(fig, path: Path):
    """Save as PNG (300 dpi) AND PDF (vector) for LaTeX inclusion."""
    path = Path(path)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}  +  {pdf_path.name}")


SCHEDULE_ORDER  = ["baseline", "concept_drift", "resource_constraint",
                   "trust_violation", "valence_inversion"]
SCHEDULE_SHORT  = {"baseline": "BL", "concept_drift": "CD",
                   "resource_constraint": "RC", "trust_violation": "TV",
                   "valence_inversion": "VI"}
SCHEDULE_COLORS = {
    "baseline":            "#9E9E9E",
    "concept_drift":       "#42A5F5",
    "resource_constraint": "#FFA726",
    "trust_violation":     "#EF5350",
    "valence_inversion":   "#AB47BC",
}
ALGO_MARKERS = {"a2c": "o", "ddpg": "s", "dqn": "D", "ppo": "^",
                "sac": "P", "td3": "X", "trpo": "v"}
ALGO_COLORS  = {"a2c": "#E53935", "ddpg": "#8E24AA", "dqn": "#1E88E5",
                "ppo": "#43A047", "sac": "#FB8C00", "td3": "#00ACC1",
                "trpo": "#6D4C41"}


def _short_env(e: str) -> str:
    return (e.replace("MountainCarContinuous", "MCC")
             .replace("MountainCar", "MC")
             .replace("FrozenLake", "FL")
             .replace("ALE/", "")
             .replace("-v1", "").replace("-v0", "")
             .replace("-v4", "").replace("-v5", ""))


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_eval_csv(run_root: Path) -> pd.DataFrame:
    p = run_root / "eval" / "eval_results.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run run_eval first.")
    return pd.read_csv(p)


def _load_per_episode_csv(run_root: Path):
    p = run_root / "eval" / "per_episode.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        "__".join([c for c in col if c]) if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]
    return df


def _is_single_run_root(root: Path) -> bool:
    return (root / "eval" / "eval_results.csv").exists()


def _discover_runs(root: Path) -> List[Path]:
    runs: List[Path] = []
    for p in root.rglob("eval/eval_results.csv"):
        runs.append(p.parent.parent)
    uniq, seen = [], set()
    for r in sorted(runs):
        if str(r) not in seen:
            uniq.append(r)
            seen.add(str(r))
    return uniq


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate_eval(df: pd.DataFrame) -> pd.DataFrame:
    group_cols  = ["env", "algo", "schedule", "eval_mode"]
    metric_cols = [c for c in df.columns
                   if c not in group_cols and c not in ("seed", "run_dir")]
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    return _flatten_cols(agg)


def _add_reward_norm(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.copy()
    if "reward_mean__mean" not in agg.columns:
        agg["reward_norm"] = np.nan
        return agg

    def _norm_group(g: pd.DataFrame) -> pd.Series:
        x = g["reward_mean__mean"].to_numpy(dtype=float)
        if not np.any(np.isfinite(x)):
            return pd.Series(np.full(len(g), np.nan), index=g.index)
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) < 1e-12:
            return pd.Series(np.full(len(g), 0.5), index=g.index)
        return pd.Series((x - xmin) / (xmax - xmin), index=g.index)

    agg["reward_norm"] = (
        agg.groupby(["env", "schedule", "eval_mode"], dropna=False, sort=False)
        .apply(_norm_group)
        .reset_index(level=[0, 1, 2], drop=True)
    )
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Leaderboard score
# ──────────────────────────────────────────────────────────────────────────────

def _default_weights() -> Tuple[float, float, float]:
    return 0.55, 0.30, 0.15


def _calib_by_env(calib: dict | None) -> dict:
    if not calib:
        return {}
    try:
        return calib.get("robust_score", {}).get("by_env", {}) or {}
    except Exception:
        return {}


def _pick_first_env_schedule(calib: dict | None) -> Tuple[str | None, str | None]:
    by_env = _calib_by_env(calib)
    if not by_env:
        return None, None
    env0   = sorted(by_env.keys())[0]
    scheds = by_env.get(env0, {}) or {}
    if not scheds:
        return env0, None
    return env0, sorted(scheds.keys())[0]


def _get_calib_weights(calib, env, schedule, calib_env_fallback=None):
    by_env = _calib_by_env(calib)
    if not by_env:
        return _default_weights()

    def _try(e, s):
        try:
            w = by_env[e][s]["weights"]
            return float(w["w_identity"]), float(w["w_cr_shock"]), float(w["w_reward"])
        except Exception:
            return None

    for e, s in [(env, schedule),
                 (calib_env_fallback, schedule) if calib_env_fallback else (None, None),
                 (calib_env_fallback, "baseline") if calib_env_fallback else (None, None),
                 (env, "baseline")]:
        if e and s:
            hit = _try(e, s)
            if hit:
                return hit

    e0, s0 = _pick_first_env_schedule(calib)
    if e0 and s0:
        hit = _try(e0, s0)
        if hit:
            return hit
    return _default_weights()


def _add_leaderboard_score(agg, calib, calib_env_fallback):
    agg   = agg.copy()
    ident = agg.get("identity_mean__mean",
                    pd.Series(np.nan, index=agg.index)).to_numpy(dtype=float)
    crs   = agg.get("collapse_rate_shock__mean",
                    pd.Series(np.nan, index=agg.index)).to_numpy(dtype=float)
    rnorm = agg.get("reward_norm",
                    pd.Series(np.nan, index=agg.index)).to_numpy(dtype=float)

    ident = np.nan_to_num(ident, nan=0.0)
    crs   = np.nan_to_num(crs,   nan=0.0)
    rnorm = np.nan_to_num(rnorm, nan=0.5)

    scores = []
    for i in range(len(agg)):
        w_id, w_cr, w_rw = _get_calib_weights(
            calib, str(agg.iloc[i]["env"]), str(agg.iloc[i]["schedule"]),
            calib_env_fallback=calib_env_fallback,
        )
        scores.append(
            w_id * ident[i]
            + w_cr * (1.0 - np.clip(crs[i], 0, 1))
            + w_rw * np.clip(rnorm[i], 0, 1)
        )

    agg["leaderboard_score"] = np.asarray(scores, dtype=float)
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# Print tables
# ──────────────────────────────────────────────────────────────────────────────

def _print_tables(ev: pd.DataFrame, agg: pd.DataFrame):
    cols_order = [c for c in SCHEDULE_ORDER if c in ev["schedule"].unique()]

    print("\n" + "=" * 80)
    print("COLLAPSE RATE (shock phase) — mean across seeds & eval modes")
    print("=" * 80)
    t = (ev.groupby(["env", "algo", "schedule"])["collapse_rate_shock"]
           .mean().round(3).unstack("schedule"))
    print(t[[c for c in cols_order if c in t.columns]].to_string())

    print("\n" + "=" * 80)
    print("COLLAPSE SCORE SHOCK MEAN — mean across seeds & eval modes")
    print("=" * 80)
    t2 = (ev.groupby(["env", "algo", "schedule"])["collapse_score_shock_mean"]
            .mean().round(4).unstack("schedule"))
    print(t2[[c for c in cols_order if c in t2.columns]].to_string())

    print("\n" + "=" * 80)
    print("REWARD MEAN (baseline schedule only) — mean across seeds & eval modes")
    print("=" * 80)
    bl = ev[ev["schedule"] == "baseline"]
    print(bl.groupby(["env", "algo"])["reward_mean"].mean().round(1)
            .unstack("algo").to_string())

    print("\n" + "=" * 80)
    print("ANCHOR CHECK — baseline lowest & vi highest (per env/algo)")
    print("=" * 80)
    fails = []
    for (env, algo), g in ev.groupby(["env", "algo"]):
        s = g.groupby("schedule")["collapse_score_shock_mean"].mean()
        stressors = [x for x in
                     ["concept_drift", "resource_constraint",
                      "trust_violation", "valence_inversion"]
                     if x in s.index]
        if not all(s["baseline"] < s[x] for x in stressors):
            fails.append(f"  FAIL  {env}/{algo}  baseline not lowest")
        if "valence_inversion" in s.index and s["valence_inversion"] != s.max():
            fails.append(f"  FAIL  {env}/{algo}  vi not highest")
    if fails:
        for f in fails:
            print(f)
    else:
        print("  All PASS")

    print("\n" + "=" * 80)
    print("ALGO VULNERABILITY PROFILE — worst middle stressor (cd/rc/tv) per env/algo")
    print("=" * 80)
    mid   = ev[ev["schedule"].isin(["concept_drift", "resource_constraint",
                                     "trust_violation"])]
    worst = (mid.groupby(["env", "algo", "schedule"])["collapse_score_shock_mean"]
               .mean().reset_index())
    worst = worst.loc[
        worst.groupby(["env", "algo"])["collapse_score_shock_mean"].idxmax()
    ]
    pivot = worst.pivot(index="algo", columns="env", values="schedule").fillna("-")
    print(pivot.to_string())

    if "leaderboard_score" in agg.columns:
        print("\n" + "=" * 80)
        print("LEADERBOARD SCORE — top algo per env/schedule/eval_mode")
        print("=" * 80)
        lb_cols = [c for c in ["env", "schedule", "eval_mode", "algo",
                                "leaderboard_score", "identity_mean__mean",
                                "collapse_rate_shock__mean",
                                "reward_mean__mean"] if c in agg.columns]
        agg_s = agg.sort_values(
            ["env", "schedule", "eval_mode", "leaderboard_score"],
            ascending=[True, True, True, False],
        )
        shown = agg_s.groupby(
            ["env", "schedule", "eval_mode"], dropna=False
        ).head(3)[lb_cols]
        print(shown.to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX table generation
# ──────────────────────────────────────────────────────────────────────────────

def _write_latex_tables(ev: pd.DataFrame, plots_dir: Path):
    """
    Write publication-ready LaTeX tables to plots_dir/tables/.

    Tables produced:
      collapse_rate_main.tex  — main results (env × stressor collapse rate)
      algo_vulnerability.tex  — worst middle stressor per algo × env
      baseline_fpr.tex        — per-env baseline FPR (scoring validation)
    """
    tables_dir = plots_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    scheds = [s for s in SCHEDULE_ORDER if s in ev["schedule"].unique()]
    envs   = sorted(ev["env"].unique())
    S      = SCHEDULE_SHORT

    # ── Table 1: Main collapse rate ───────────────────────────────────────
    data = (ev.groupby(["env", "schedule"])["collapse_rate_shock"]
              .mean().unstack("schedule")
              .reindex(index=envs, columns=scheds).fillna(0))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Mean collapse rate (shock phase) per environment and stressor, "
         r"averaged across algorithms, seeds, and evaluation modes. "
         r"Higher = greater identity collapse under stress. "
         r"\textbf{Bold} = worst stressor per row (excluding BL).}"),
        r"\label{tab:collapse_rate_main}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{l" + "r" * len(scheds) + "}",
        r"\toprule",
        "Environment & " + " & ".join(S.get(s, s) for s in scheds) + r" \\",
        r"\midrule",
    ]
    for env in envs:
        row_vals = [(s, float(data.loc[env, s])) for s in scheds]
        non_bl   = [(s, v) for s, v in row_vals if s != "baseline"]
        max_v    = max(v for _, v in non_bl) if non_bl else 0.0
        cells    = []
        for s, v in row_vals:
            fmt = f"{v:.2f}"
            if s != "baseline" and abs(v - max_v) < 0.001 and max_v > 0.1:
                fmt = r"\textbf{" + fmt + "}"
            cells.append(fmt)
        lines.append(
            r"\texttt{" + _short_env(env) + "} & "
            + " & ".join(cells) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out1 = tables_dir / "collapse_rate_main.tex"
    out1.write_text("\n".join(lines), encoding="utf-8")
    print(f"[table] {out1}")

    # ── Table 2: Algo vulnerability ────────────────────────────────────────
    mid   = ev[ev["schedule"].isin(["concept_drift", "resource_constraint",
                                     "trust_violation"])]
    worst = (mid.groupby(["env", "algo", "schedule"])["collapse_score_shock_mean"]
               .mean().reset_index())
    worst = worst.loc[
        worst.groupby(["env", "algo"])["collapse_score_shock_mean"].idxmax()
    ]
    pivot = (worst.pivot(index="algo", columns="env", values="schedule")
                  .fillna("--"))
    pivot.columns = [_short_env(c) for c in pivot.columns]
    abbrev = {"concept_drift": "CD", "resource_constraint": "RC",
              "trust_violation": "TV", "--": "--"}
    cols   = list(pivot.columns)

    lines2 = [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Most damaging stressor (excluding VI) per algorithm and environment. "
         r"CD = concept drift, RC = resource constraint, TV = trust violation. "
         r"Blank cells indicate the algorithm was not evaluated on that environment.}"),
        r"\label{tab:algo_vulnerability}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        "Algo & "
        + " & ".join(r"\texttt{" + c + "}" for c in cols) + r" \\",
        r"\midrule",
    ]
    for algo in sorted(pivot.index):
        cells = [abbrev.get(str(pivot.loc[algo, c]), str(pivot.loc[algo, c]))
                 for c in cols]
        lines2.append(
            r"\texttt{" + algo + "} & " + " & ".join(cells) + r" \\"
        )
    lines2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out2 = tables_dir / "algo_vulnerability.tex"
    out2.write_text("\n".join(lines2), encoding="utf-8")
    print(f"[table] {out2}")

    # ── Table 3: Baseline FPR ─────────────────────────────────────────────
    bl  = ev[ev["schedule"] == "baseline"]
    fpr = (bl.groupby("env")["collapse_rate_pre"]
             .agg(["mean", "std"])
             .reset_index()
             .rename(columns={"mean": "fpr_mean", "std": "fpr_std"})
             .sort_values("env"))

    lines3 = [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Baseline false positive rate (FPR) per environment, "
         r"averaged across algorithms, seeds, and evaluation modes. "
         r"Target $\alpha = 0.05$. "
         r"$\dagger$ denotes FPR $> 0.10$ (see \S\ref{sec:limitations}).}"),
        r"\label{tab:baseline_fpr}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Environment & FPR mean & FPR std \\",
        r"\midrule",
    ]
    for _, row in fpr.iterrows():
        flag = r" $\dagger$" if row["fpr_mean"] > 0.10 else ""
        lines3.append(
            r"\texttt{" + _short_env(row["env"]) + r"} & "
            + f"{row['fpr_mean']:.3f}{flag} & {row['fpr_std']:.3f} \\\\"
        )
    lines3 += [
        r"\midrule",
        f"Overall & {fpr['fpr_mean'].mean():.3f} & {fpr['fpr_std'].mean():.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out3 = tables_dir / "baseline_fpr.tex"
    out3.write_text("\n".join(lines3), encoding="utf-8")
    print(f"[table] {out3}")


# ──────────────────────────────────────────────────────────────────────────────
# Plots  (all use _savefig for PNG+PDF output)
# ──────────────────────────────────────────────────────────────────────────────

def _plot_collapse_rate_heatmap(ev: pd.DataFrame, plots_dir: Path):
    scheds = [s for s in SCHEDULE_ORDER if s in ev["schedule"].unique()]
    envs   = sorted(ev["env"].unique())
    data   = (ev.groupby(["env", "schedule"])["collapse_rate_shock"]
                .mean().unstack("schedule")
                .reindex(index=envs, columns=scheds).fillna(0))

    cmap = LinearSegmentedColormap.from_list("arcus", ["#E3F2FD", "#EF5350"], N=256)
    fig, ax = plt.subplots(
        figsize=(len(scheds) * 1.6 + 1.5, len(envs) * 0.75 + 1.2)
    )
    im = ax.imshow(data.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(scheds)))
    ax.set_xticklabels([SCHEDULE_SHORT.get(s, s) for s in scheds])
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels([_short_env(e) for e in envs])

    for i in range(len(envs)):
        for j in range(len(scheds)):
            v = data.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if v > 0.55 else "#222")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="collapse rate (shock phase)")
    out = plots_dir / "heatmap_collapse_rate.png"
    _savefig(fig, out)


def _plot_score_by_schedule(ev: pd.DataFrame, plots_dir: Path):
    envs   = sorted(ev["env"].unique())
    scheds = [s for s in SCHEDULE_ORDER if s in ev["schedule"].unique()]
    algos  = sorted(ev["algo"].unique())
    ncols  = 3
    nrows  = int(np.ceil(len(envs) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5, nrows * 3.8), sharey=False)
    axes = np.array(axes).flatten()

    for idx, env in enumerate(envs):
        ax  = axes[idx]
        sub = ev[ev["env"] == env]
        x   = np.arange(len(scheds))

        for algo in algos:
            asub  = sub[sub["algo"] == algo]
            means, stds = [], []
            for s in scheds:
                vals = asub[asub["schedule"] == s]["collapse_score_shock_mean"].to_numpy(dtype=float)
                means.append(np.nanmean(vals) if len(vals) else np.nan)
                stds.append(np.nanstd(vals)   if len(vals) else np.nan)
            means, stds = np.array(means), np.array(stds)
            mask = np.isfinite(means)
            if not mask.any():
                continue
            ax.errorbar(x[mask], means[mask], yerr=stds[mask],
                        label=algo, color=ALGO_COLORS.get(algo, "#555"),
                        marker=ALGO_MARKERS.get(algo, "o"),
                        linewidth=1.4, markersize=5, capsize=3, alpha=0.88)

        ax.set_title(_short_env(env), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SCHEDULE_SHORT.get(s, s) for s in scheds])
        ax.set_ylabel("collapse score")
        ax.set_ylim(0.4, 0.85)

    for idx in range(len(envs), len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(algos),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Collapse Score by Schedule — mean ± std across seeds",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _savefig(fig, plots_dir / "score_by_schedule_per_env.png")


def _plot_collapse_rate_by_algo(ev: pd.DataFrame, plots_dir: Path):
    envs   = sorted(ev["env"].unique())
    scheds = [s for s in SCHEDULE_ORDER
              if s != "baseline" and s in ev["schedule"].unique()]
    algos  = sorted(ev["algo"].unique())
    ncols  = 3
    nrows  = int(np.ceil(len(envs) / ncols))
    n_sch  = len(scheds)
    width  = 0.8 / n_sch

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5.5, nrows * 3.8))
    axes = np.array(axes).flatten()

    for idx, env in enumerate(envs):
        ax  = axes[idx]
        sub = ev[ev["env"] == env]
        x   = np.arange(len(algos))

        for si, sched in enumerate(scheds):
            ssub  = sub[sub["schedule"] == sched]
            means, stds = [], []
            for algo in algos:
                vals = ssub[ssub["algo"] == algo]["collapse_rate_shock"].to_numpy(dtype=float)
                means.append(np.nanmean(vals) if len(vals) else np.nan)
                stds.append(np.nanstd(vals)   if len(vals) else np.nan)
            offset = (si - n_sch / 2 + 0.5) * width
            ax.bar(x + offset, means, width=width * 0.9,
                   color=SCHEDULE_COLORS[sched],
                   label=SCHEDULE_SHORT[sched],
                   yerr=stds,
                   error_kw=dict(elinewidth=0.8, capsize=2),
                   alpha=0.88)

        ax.set_title(_short_env(env), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=30, ha="right")
        ax.set_ylabel("collapse rate")
        ax.set_ylim(0, 1.12)

    for idx in range(len(envs), len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="stressor", loc="lower center",
               ncol=n_sch, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Collapse Rate by Algo & Stressor — mean ± std across seeds",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _savefig(fig, plots_dir / "collapse_rate_by_algo.png")


def _plot_vulnerability_heatmap(ev: pd.DataFrame, plots_dir: Path):
    from matplotlib.patches import Patch

    all_scheds = ["concept_drift", "resource_constraint",
                  "trust_violation", "valence_inversion"]
    mid_scheds = ["concept_drift", "resource_constraint", "trust_violation"]
    colors_all = ["#42A5F5", "#FFA726", "#EF5350", "#AB47BC"]
    colors_mid = ["#42A5F5", "#FFA726", "#EF5350"]
    labels_all = ["CD", "RC", "TV", "VI"]
    labels_mid = ["CD", "RC", "TV"]

    envs  = sorted(ev["env"].unique())
    algos = sorted(ev["algo"].unique())

    def _build_matrix(scheds):
        sch_to_int = {s: i for i, s in enumerate(scheds)}
        sub   = ev[ev["schedule"].isin(scheds)]
        worst = (sub.groupby(["env", "algo", "schedule"])["collapse_score_shock_mean"]
                    .mean().reset_index())
        worst = worst.loc[
            worst.groupby(["env", "algo"])["collapse_score_shock_mean"].idxmax()
        ]
        mat = np.full((len(algos), len(envs)), -1, dtype=int)
        for _, row in worst.iterrows():
            if row["env"] in envs and row["algo"] in algos:
                mat[algos.index(row["algo"]),
                    envs.index(row["env"])] = sch_to_int.get(row["schedule"], -1)
        return mat

    mat_all = _build_matrix(all_scheds)
    mat_mid = _build_matrix(mid_scheds)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(len(envs) * 2.8 + 3, len(algos) * 0.8 + 2.8),
        gridspec_kw={"wspace": 0.48},
    )

    panels = [
        (axes[0], mat_all, colors_all, labels_all,
         "Most Damaging — ALL 4 stressors\n(CD / RC / TV / VI)"),
        (axes[1], mat_mid, colors_mid, labels_mid,
         "Most Damaging — middle 3 only\n(CD / RC / TV,  excl. VI)"),
    ]

    for ax, mat, colors, labels, title in panels:
        n    = len(labels)
        cmap = LinearSegmentedColormap.from_list("vuln", colors, N=n)
        ax.imshow(np.where(mat >= 0, mat, np.nan),
                  cmap=cmap, vmin=0, vmax=n - 1, aspect="auto")
        ax.set_xticks(range(len(envs)))
        ax.set_xticklabels([_short_env(e) for e in envs],
                           rotation=35, ha="right")
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos)
        ax.set_title(title, pad=8)

        for ai in range(len(algos)):
            for ei in range(len(envs)):
                v = mat[ai, ei]
                if v >= 0:
                    ax.text(ei, ai, labels[v], ha="center", va="center",
                            fontsize=8, fontweight="bold", color="white")

        legend_els = [Patch(facecolor=c, label=lb)
                      for c, lb in zip(colors, labels)]
        ax.legend(handles=legend_els, loc="lower center",
                  bbox_to_anchor=(0.5, -0.30), ncol=n,
                  framealpha=0.9)

    fig.suptitle("Algorithm Vulnerability Profiles across Environments",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    _savefig(fig, plots_dir / "vulnerability_heatmap.png")


def _plot_reward_vs_collapse(ev: pd.DataFrame, plots_dir: Path):
    from scipy import stats

    DISCRETE = {"CartPole-v1", "Acrobot-v1", "FrozenLake-v1", "MountainCar-v0"}

    bl = ev[ev["schedule"] == "baseline"].groupby(["env", "algo"])["reward_mean"].mean()
    vi = (ev[ev["schedule"] == "valence_inversion"]
            .groupby(["env", "algo"])["collapse_rate_shock"].mean())
    df = pd.DataFrame({"reward": bl, "vi_rate": vi}).dropna().reset_index()
    df["reward_norm"] = df.groupby("env")["reward"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                              gridspec_kw={"wspace": 0.38})

    for ax, xcol, xlabel, title_suffix in [
        (axes[0], "reward",      "Mean Reward (baseline, raw)",
         "Raw reward scale"),
        (axes[1], "reward_norm", "Normalised Reward (per-env min-max)",
         "Normalised — comparable across envs"),
    ]:
        x = df[xcol].to_numpy(dtype=float)
        y = df["vi_rate"].to_numpy(dtype=float)
        r, p = stats.pearsonr(x, y)

        slope, intercept, *_ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept

        rng = np.random.default_rng(42)
        y_boots = []
        for _ in range(500):
            idx  = rng.integers(0, len(x), len(x))
            s_, i_, *_ = stats.linregress(x[idx], y[idx])
            y_boots.append(s_ * x_line + i_)
        ax.fill_between(x_line,
                        np.percentile(y_boots, 2.5,  axis=0),
                        np.percentile(y_boots, 97.5, axis=0),
                        alpha=0.15, color="#666")
        ax.plot(x_line, y_line, color="#444", linewidth=1.2,
                linestyle="--",
                label=f"fit  (r={r:+.2f}, p={p:.3f})")

        for algo in sorted(df["algo"].unique()):
            sub = df[df["algo"] == algo]
            disc_mask = sub["env"].isin(DISCRETE)
            for mask, marker, ms in [
                (disc_mask,  "D", 52),
                (~disc_mask, "o", 60),
            ]:
                s = sub[mask]
                if len(s) == 0:
                    continue
                ax.scatter(s[xcol], s["vi_rate"],
                           color=ALGO_COLORS.get(algo, "#555"),
                           marker=marker, s=ms, zorder=3, alpha=0.88,
                           label=(f"{algo} ({'discrete' if marker=='D' else 'continuous'})"
                                  if xcol == "reward_norm" else "_nolegend_"))

        for _, row in df.iterrows():
            ax.annotate(_short_env(row["env"]),
                        (row[xcol], row["vi_rate"]),
                        fontsize=5.5, alpha=0.60,
                        xytext=(3, 3), textcoords="offset points")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Collapse Rate — valence inversion")
        ax.set_ylim(-0.05, 1.12)
        ax.set_title(
            f"Reward vs Identity Robustness  ({title_suffix})\n"
            f"r = {r:+.3f}  p = {p:.3f}  "
            f"{'(no significant correlation)' if p > 0.05 else '(significant)'}",
        )
        p_col = "#C62828" if p <= 0.05 else "#2E7D32"
        ax.annotate(
            f"Pearson r = {r:+.3f}\np = {p:.3f}",
            xy=(0.97, 0.05), xycoords="axes fraction",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=p_col, alpha=0.85),
        )

    axes[1].legend(ncol=2, loc="upper left", framealpha=0.85,
                   title="algo  (▲=discrete  ●=continuous)",
                   title_fontsize=8)
    fig.suptitle(
        "Does reward predict identity robustness under valence inversion?\n"
        "Each point = one (env, algo) pair averaged over 10 seeds × 2 eval modes",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    _savefig(fig, plots_dir / "reward_vs_collapse_scatter.png")


def _plot_seed_variance(ev: pd.DataFrame, plots_dir: Path):
    envs   = sorted(ev["env"].unique())
    scheds = [s for s in SCHEDULE_ORDER
              if s != "baseline" and s in ev["schedule"].unique()]
    algos  = sorted(ev["algo"].unique())
    ncols  = 3
    nrows  = int(np.ceil(len(envs) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5, nrows * 3.5))
    axes = np.array(axes).flatten()

    for idx, env in enumerate(envs):
        ax  = axes[idx]
        sub = ev[ev["env"] == env]
        data_by_sched, labels_by_sched, colors_by_sched = [], [], []

        for sched in scheds:
            vals = (sub[sub["schedule"] == sched]["collapse_rate_shock"]
                      .dropna().to_numpy(dtype=float))
            if len(vals):
                data_by_sched.append(vals)
                labels_by_sched.append(SCHEDULE_SHORT[sched])
                colors_by_sched.append(SCHEDULE_COLORS[sched])

        if not data_by_sched:
            axes[idx].set_visible(False)
            continue

        bp = ax.boxplot(data_by_sched, patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", linewidth=2),
                        flierprops=dict(marker=".", markersize=3, alpha=0.5))
        for patch, color in zip(bp["boxes"], colors_by_sched):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)

        ax.set_title(_short_env(env), fontweight="bold")
        ax.set_xticklabels(labels_by_sched)
        ax.set_ylabel("collapse rate shock")
        ax.set_ylim(-0.05, 1.1)

    for idx in range(len(envs), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Collapse Rate Seed Variance — box over seeds × algos × modes",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _savefig(fig, plots_dir / "seed_variance_boxplot.png")


def _plot_leaderboard_bar(agg: pd.DataFrame, plots_dir: Path):
    if "leaderboard_score" not in agg.columns:
        return

    sub    = agg[agg["eval_mode"] == "deterministic"]
    scheds = [s for s in SCHEDULE_ORDER if s in sub["schedule"].unique()]
    ncols  = len(scheds)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3.5, 6), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, sched in zip(axes, scheds):
        d      = sub[sub["schedule"] == sched].copy()
        d      = d.sort_values("leaderboard_score", ascending=True)
        labels = d["env"].map(_short_env) + "/" + d["algo"]
        colors = [ALGO_COLORS.get(a, "#555") for a in d["algo"]]
        ax.barh(range(len(d)), d["leaderboard_score"], color=colors, alpha=0.85)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(labels.tolist(), fontsize=7)
        ax.set_title(SCHEDULE_SHORT.get(sched, sched), fontweight="bold")
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="#999", linewidth=0.8, linestyle="--")

    fig.suptitle("Leaderboard Score by Stressor (deterministic)", fontsize=12)
    plt.tight_layout()
    _savefig(fig, plots_dir / "leaderboard_bar.png")


def _plot_stochastic_vs_deterministic(ev: pd.DataFrame, plots_dir: Path):
    stressors = [s for s in SCHEDULE_ORDER
                 if s != "baseline" and s in ev["schedule"].unique()]
    n = len(stressors)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.8, 3.8),
                              sharey=True, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, sched in zip(axes, stressors):
        sub  = ev[ev["schedule"] == sched]
        det  = (sub[sub["eval_mode"] == "deterministic"]
                  .groupby(["env", "algo"])["collapse_rate_shock"].mean())
        stoc = (sub[sub["eval_mode"] == "stochastic"]
                  .groupby(["env", "algo"])["collapse_rate_shock"].mean())
        df   = pd.DataFrame({"det": det, "stoc": stoc}).dropna().reset_index()

        for algo in sorted(df["algo"].unique()):
            s = df[df["algo"] == algo]
            ax.scatter(s["det"], s["stoc"],
                       color=ALGO_COLORS.get(algo, "#555"),
                       marker=ALGO_MARKERS.get(algo, "o"),
                       s=45, alpha=0.85, label=algo, zorder=3)

        ax.plot([0, 1], [0, 1], "--", color="#aaa", linewidth=1, zorder=1)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Deterministic")
        ax.set_ylabel("Stochastic" if sched == stressors[0] else "")
        ax.set_title(SCHEDULE_SHORT[sched], fontweight="bold")
        ax.set_aspect("equal")
        try:
            from scipy import stats as sp
            if len(df) > 2:
                r, _ = sp.pearsonr(df["det"], df["stoc"])
                ax.annotate(f"r={r:.2f}", xy=(0.05, 0.92),
                            xycoords="axes fraction")
        except Exception:
            pass

    handles = [
        plt.Line2D([0], [0], marker=ALGO_MARKERS.get(a, "o"),
                   color=ALGO_COLORS.get(a, "#555"),
                   linestyle="none", markersize=6, label=a)
        for a in sorted(ev["algo"].unique())
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.08), framealpha=0.9)
    fig.suptitle(
        "Deterministic vs Stochastic Eval — collapse rate per (env, algo)\n"
        "Points near diagonal: eval mode does not change ARCUS-H verdict",
        fontsize=11, y=1.04,
    )
    plt.tight_layout()
    _savefig(fig, plots_dir / "stochastic_vs_deterministic.png")


def _plot_collapse_by_action_space(ev: pd.DataFrame, plots_dir: Path):
    ev2 = ev.copy()
    ev2["action_space"] = ev2["env"].apply(
        lambda e: "Discrete" if e in DISCRETE_ENVS else "Continuous"
    )
    stressors  = [s for s in SCHEDULE_ORDER
                  if s != "baseline" and s in ev2["schedule"].unique()]
    colors_as  = {"Discrete": "#42A5F5", "Continuous": "#FFA726"}

    fig, axes = plt.subplots(1, len(stressors),
                              figsize=(len(stressors) * 3.2, 4.5),
                              sharey=True)
    if len(stressors) == 1:
        axes = [axes]

    for ax, sched in zip(axes, stressors):
        sub    = ev2[ev2["schedule"] == sched]
        data_d = sub[sub["action_space"] == "Discrete"  ]["collapse_rate_shock"].dropna().values
        data_c = sub[sub["action_space"] == "Continuous"]["collapse_rate_shock"].dropna().values

        parts = ax.violinplot([data_d, data_c], positions=[0, 1],
                              showmedians=True, showextrema=True)
        for body, color in zip(parts["bodies"],
                                [colors_as["Discrete"], colors_as["Continuous"]]):
            body.set_facecolor(color)
            body.set_alpha(0.6)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(2)

        rng = np.random.default_rng(42)
        for i, (data, color) in enumerate(
            zip([data_d, data_c], list(colors_as.values()))
        ):
            jitter = rng.uniform(-0.08, 0.08, len(data))
            ax.scatter(i + jitter, data, alpha=0.35, s=12,
                       color=color, zorder=3)

        try:
            from scipy import stats as sp
            if len(data_d) > 1 and len(data_c) > 1:
                _, p = sp.mannwhitneyu(data_d, data_c, alternative="two-sided")
                sig  = ("***" if p < 0.001 else "**" if p < 0.01
                        else "*" if p < 0.05 else "ns")
                ax.annotate(sig, xy=(0.5, 0.95), xycoords="axes fraction",
                            ha="center", fontsize=11, fontweight="bold",
                            color="#C62828" if p < 0.05 else "#aaa")
        except Exception:
            pass

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Discrete", "Continuous"])
        ax.set_title(SCHEDULE_SHORT[sched], fontweight="bold")
        ax.set_ylabel("Collapse Rate" if sched == stressors[0] else "")
        ax.set_ylim(-0.05, 1.15)

    fig.suptitle(
        "Collapse Rate: Discrete vs Continuous Action Space\n"
        r"* $p{<}0.05$  ** $p{<}0.01$  *** $p{<}0.001$  (Mann-Whitney U)",
        fontsize=11, y=1.04,
    )
    plt.tight_layout()
    _savefig(fig, plots_dir / "collapse_by_action_space.png")


def _plot_on_policy_vs_off_policy(ev: pd.DataFrame, plots_dir: Path):
    ev2 = ev.copy()
    ev2["policy_type"]  = ev2["algo"].apply(
        lambda a: "on-policy" if a in ON_POLICY_ALGOS else "off-policy"
    )
    ev2["action_space"] = ev2["env"].apply(
        lambda e: "discrete" if e in DISCRETE_ENVS else "continuous"
    )
    stressors    = [s for s in SCHEDULE_ORDER
                    if s != "baseline" and s in ev2["schedule"].unique()]
    action_types = ["discrete", "continuous"]
    colors_pt    = {"on-policy": "#43A047", "off-policy": "#FB8C00"}

    fig, axes = plt.subplots(1, 2,
                              figsize=(len(stressors) * 2.2 * 2, 4.5),
                              sharey=True)
    for ax, atype in zip(axes, action_types):
        sub = ev2[ev2["action_space"] == atype]
        x   = np.arange(len(stressors))
        width = 0.35
        for i, ptype in enumerate(["on-policy", "off-policy"]):
            means, cis = [], []
            for sched in stressors:
                vals = (sub[(sub["schedule"] == sched) & (sub["policy_type"] == ptype)]
                          ["collapse_rate_shock"].dropna().values)
                means.append(np.nanmean(vals) if len(vals) else np.nan)
                sem = np.nanstd(vals) / np.sqrt(max(len(vals), 1))
                cis.append(1.96 * sem)
            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width=width * 0.9,
                   color=colors_pt[ptype], label=ptype, alpha=0.85,
                   yerr=cis, error_kw=dict(elinewidth=1, capsize=3))

        ax.set_xticks(x)
        ax.set_xticklabels([SCHEDULE_SHORT[s] for s in stressors])
        ax.set_title(f"{atype.capitalize()} action space", fontweight="bold")
        ax.set_ylabel("Collapse Rate (shock)")
        ax.set_ylim(0, 1.15)
        ax.legend()

    fig.suptitle(
        "On-Policy vs Off-Policy Collapse Rates per Stressor\n"
        "Error bars = 95% CI across seeds × envs × eval modes",
        fontsize=11, y=1.04,
    )
    plt.tight_layout()
    _savefig(fig, plots_dir / "on_policy_vs_off_policy.png")


def _plot_per_seed_consistency(ev: pd.DataFrame, plots_dir: Path):
    if "seed" not in ev.columns:
        return
    stressors = [s for s in SCHEDULE_ORDER
                 if s != "baseline" and s in ev["schedule"].unique()]
    seeds = sorted(ev["seed"].dropna().unique())
    if len(seeds) < 2:
        return

    fig, axes = plt.subplots(1, len(stressors),
                              figsize=(len(stressors) * 3.5, 4),
                              sharey=True)
    if len(stressors) == 1:
        axes = [axes]

    for ax, sched in zip(axes, stressors):
        sub      = ev[ev["schedule"] == sched]
        per_seed = (sub.groupby("seed")["collapse_rate_shock"]
                      .agg(["mean", "std"]).reindex(seeds))
        means = per_seed["mean"].values
        stds  = per_seed["std"].values

        ax.fill_between(seeds, means - stds, means + stds,
                        alpha=0.18, color=SCHEDULE_COLORS[sched])
        ax.plot(seeds, means, "o-",
                color=SCHEDULE_COLORS[sched], linewidth=2, markersize=6)
        ax.axhline(np.nanmean(means), color=SCHEDULE_COLORS[sched],
                   linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xticks(seeds)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Collapse Rate" if sched == stressors[0] else "")
        ax.set_title(SCHEDULE_SHORT[sched], fontweight="bold")
        ax.set_ylim(-0.05, 1.1)
        cv = np.nanstd(means) / (np.nanmean(means) + 1e-8)
        ax.annotate(f"CV={cv:.2f}", xy=(0.97, 0.05),
                    xycoords="axes fraction", ha="right", color="#555")

    fig.suptitle(
        "Per-Seed Consistency — collapse rate per seed\n"
        "Averaged across all envs × algos × eval modes  |  CV = coeff. of variation",
        fontsize=11, y=1.04,
    )
    plt.tight_layout()
    _savefig(fig, plots_dir / "per_seed_consistency.png")


def _plot_identity_components_radar(ep_all, plots_dir: Path):
    if ep_all is None:
        return
    missing = [c for c in IDENTITY_COMPONENTS if c not in ep_all.columns]
    if missing:
        return
    shock     = ep_all[ep_all["stress_phase"] == "shock"]
    stressors = [s for s in SCHEDULE_ORDER
                 if s != "baseline" and s in shock["schedule"].unique()]
    if not stressors:
        return

    comp_means = shock.groupby("schedule")[IDENTITY_COMPONENTS].mean()
    bl_means   = ep_all[ep_all["stress_phase"] == "pre"][IDENTITY_COMPONENTS].mean()

    N      = len(IDENTITY_COMPONENTS)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    fig, axes = plt.subplots(1, len(stressors),
                              figsize=(len(stressors) * 3.5, 3.5),
                              subplot_kw=dict(polar=True))
    if len(stressors) == 1:
        axes = [axes]

    for ax, sched in zip(axes, stressors):
        bl_vals = ([float(bl_means[c]) for c in IDENTITY_COMPONENTS]
                   + [float(bl_means[IDENTITY_COMPONENTS[0]])])
        # Patch: fixed linestyle conflict warning
        ax.plot(angles, bl_vals, "o--", color="#aaa", linewidth=1.2,
                label="baseline")
        ax.fill(angles, bl_vals, alpha=0.08, color="#aaa")

        if sched not in comp_means.index:
            continue
        vals = ([float(comp_means.loc[sched, c]) for c in IDENTITY_COMPONENTS]
                + [float(comp_means.loc[sched, IDENTITY_COMPONENTS[0]])])
        color = SCHEDULE_COLORS.get(sched, "#333")
        ax.plot(angles, vals, "o-", color=color, linewidth=2,
                label=SCHEDULE_SHORT[sched])
        ax.fill(angles, vals, alpha=0.18, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c[:4] for c in IDENTITY_COMPONENTS])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
        ax.set_title(SCHEDULE_SHORT[sched], fontweight="bold", pad=12)
        ax.legend(fontsize=7, loc="upper right",
                  bbox_to_anchor=(1.35, 1.1))

    fig.suptitle(
        "Identity Component Breakdown per Stressor (shock phase)\n"
        "Grey dashed = baseline  |  "
        "comp=competence  cohe=coherence  cont=continuity  "
        "inte=integrity  mean=meaning",
        fontsize=9, y=1.05,
    )
    plt.tight_layout()
    _savefig(fig, plots_dir / "identity_components_radar.png")


def _plot_fpr_validation(ev: pd.DataFrame, plots_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax  = axes[0]
    bl  = ev[ev["schedule"] == "baseline"]
    fpr = bl.groupby(["env", "algo", "eval_mode"])["collapse_rate_pre"].mean().values
    ax.hist(fpr, bins=20, color="#42A5F5", alpha=0.7, edgecolor="white")
    ax.axvline(0.05, color="#EF5350", linewidth=2, linestyle="--",
               label="target α=0.05")
    ax.axvline(np.nanmean(fpr), color="#333", linewidth=1.5,
               label=f"mean={np.nanmean(fpr):.3f}")
    ax.set_xlabel("Baseline FPR (collapse_rate_pre)")
    ax.set_ylabel("Count")
    ax.set_title("False Positive Rate Distribution\n(should peak near α=0.05)")
    ax.legend()

    ax = axes[1]
    for sched in [s for s in SCHEDULE_ORDER
                  if s != "baseline" and s in ev["schedule"].unique()]:
        m = (ev[ev["schedule"] == sched]
               .groupby(["env", "algo", "eval_mode"])
               [["collapse_rate_pre", "collapse_rate_shock"]].mean()
               .reset_index())
        ax.scatter(m["collapse_rate_pre"], m["collapse_rate_shock"],
                   color=SCHEDULE_COLORS[sched], alpha=0.6, s=30,
                   label=SCHEDULE_SHORT[sched])

    ax.axhline(0.05, color="#aaa", linewidth=1, linestyle="--")
    ax.axvline(0.05, color="#aaa", linewidth=1, linestyle="--")
    ax.plot([0, 1], [0, 1], "--", color="#ddd", linewidth=1)
    ax.set_xlabel("Collapse Rate Pre (FPR)")
    ax.set_ylabel("Collapse Rate Shock")
    ax.set_title("FPR vs Shock Rate — stressors should push points above diagonal")
    ax.set_xlim(-0.02, 0.5)
    ax.set_ylim(-0.02, 1.05)
    ax.legend()

    fig.suptitle(
        "Scoring Validation: False Positive Rate ≈ α=0.05\n"
        "Each point = one (env, algo, eval_mode) run",
        fontsize=11, y=1.03,
    )
    plt.tight_layout()
    _savefig(fig, plots_dir / "fpr_validation.png")


def _plot_reward_degradation(ev: pd.DataFrame, plots_dir: Path):
    stressors = [s for s in SCHEDULE_ORDER
                 if s != "baseline" and s in ev["schedule"].unique()]
    envs   = sorted(ev["env"].unique())
    bl_rew = (ev[ev["schedule"] == "baseline"]
                .groupby(["env", "algo", "eval_mode"])["reward_mean"].mean())
    rows = []
    for sched in stressors:
        sh_rew    = (ev[ev["schedule"] == sched]
                       .groupby(["env", "algo", "eval_mode"])["reward_mean"].mean())
        drop      = bl_rew - sh_rew
        drop_norm = drop.groupby("env").transform(
            lambda x: x / (x.abs().max() + 1e-8)
        )
        rows.append(drop_norm.groupby("env").mean().rename(sched))

    data = (pd.DataFrame(rows).T
              .reindex(index=envs, columns=stressors).fillna(0))
    cmap = LinearSegmentedColormap.from_list("rdrop", ["#E3F2FD", "#B71C1C"], N=256)
    fig, ax = plt.subplots(
        figsize=(len(stressors) * 1.6 + 1.5, len(envs) * 0.75 + 1.2)
    )
    im = ax.imshow(data.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(stressors)))
    ax.set_xticklabels([SCHEDULE_SHORT[s] for s in stressors])
    ax.set_yticks(range(len(envs)))
    ax.set_yticklabels([_short_env(e) for e in envs])
    ax.set_title("Normalised Reward Drop per Stressor\n(1.0 = worst stressor for that env)",
                 pad=10)
    for i in range(len(envs)):
        for j in range(len(stressors)):
            v = data.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if v > 0.6 else "#222")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="normalised reward drop")
    plt.tight_layout()
    _savefig(fig, plots_dir / "reward_degradation_heatmap.png")


def _plot_mujoco_vs_classic(ev: pd.DataFrame, plots_dir: Path):
    """CI bars per suite per stressor. Patch: CoinRun and Breakout removed."""
    MUJOCO_ENVS  = {"HalfCheetah-v4", "Hopper-v4"}
    # Patch: IMAGE_ENVS contains only Pong now
    IMAGE_ENVS   = {"ALE/Pong-v5"}
    CLASSIC_ENVS = set(ev["env"].unique()) - MUJOCO_ENVS - IMAGE_ENVS
    stressors    = [s for s in SCHEDULE_ORDER
                    if s != "baseline" and s in ev["schedule"].unique()]

    # Patch: label updated to "Pong" only
    suites = [
        ("Classic\n(state, 6 envs)",       CLASSIC_ENVS, "#42A5F5"),
        ("MuJoCo\n(continuous, 2 envs)",    MUJOCO_ENVS,  "#FFA726"),
        ("Image\n(Pong)",                   IMAGE_ENVS,   "#AB47BC"),
    ]

    fig, ax = plt.subplots(figsize=(len(stressors) * 1.8 + 2, 4.5))
    x     = np.arange(len(stressors))
    width = 0.25

    for si, (label, env_set, color) in enumerate(suites):
        sub = ev[ev["env"].isin(env_set)]
        if sub.empty:
            continue
        means, cis = [], []
        for sched in stressors:
            vals = sub[sub["schedule"] == sched]["collapse_rate_shock"].dropna().values
            m    = np.nanmean(vals)
            ci   = 1.96 * np.nanstd(vals) / np.sqrt(max(len(vals), 1))
            means.append(m)
            cis.append(ci)
        offset = (si - 1) * width
        ax.bar(x + offset, means, width=width * 0.9,
               color=color, alpha=0.85, label=label,
               yerr=cis, error_kw=dict(elinewidth=1.2, capsize=4))

    ax.set_xticks(x)
    ax.set_xticklabels([SCHEDULE_SHORT[s] for s in stressors])
    ax.set_ylabel("Collapse Rate (shock)")
    ax.set_ylim(0, 1.15)
    ax.legend(framealpha=0.9)
    ax.set_title(
        "Collapse Rate by Suite and Stressor\n"
        "Error bars = 95% CI across seeds × algos × eval modes"
    )
    plt.tight_layout()
    _savefig(fig, plots_dir / "mujoco_vs_classic_depth.png")


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def _make_all_plots(ev: pd.DataFrame, agg: pd.DataFrame,
                    plots_dir: Path, ep_all=None):
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_collapse_rate_heatmap(ev, plots_dir)
    _plot_score_by_schedule(ev, plots_dir)
    _plot_collapse_rate_by_algo(ev, plots_dir)
    _plot_vulnerability_heatmap(ev, plots_dir)
    _plot_reward_vs_collapse(ev, plots_dir)
    _plot_seed_variance(ev, plots_dir)
    _plot_leaderboard_bar(agg, plots_dir)
    _plot_stochastic_vs_deterministic(ev, plots_dir)
    _plot_collapse_by_action_space(ev, plots_dir)
    _plot_on_policy_vs_off_policy(ev, plots_dir)
    _plot_per_seed_consistency(ev, plots_dir)
    _plot_identity_components_radar(ep_all, plots_dir)
    _plot_fpr_validation(ev, plots_dir)
    _plot_reward_degradation(ev, plots_dir)
    _plot_mujoco_vs_classic(ev, plots_dir)
    _write_latex_tables(ev, plots_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",       required=True)
    ap.add_argument("--print",      action="store_true")
    ap.add_argument("--write_csv",  action="store_true")
    ap.add_argument("--plots",      action="store_true")
    ap.add_argument("--leaderboard", action="store_true")
    ap.add_argument("--tables",     action="store_true",
                    help="Write LaTeX tables to plots/tables/ (auto-included with --plots)")
    ap.add_argument("--calib",      default=None)
    ap.add_argument("--calib_env",  default=None)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    calib = None
    if args.calib:
        p = Path(args.calib)
        if not p.exists():
            raise FileNotFoundError(f"--calib not found: {p}")
        calib = json.loads(p.read_text(encoding="utf-8"))

    calib_env_fallback = args.calib_env
    if calib and not calib_env_fallback:
        e0, _ = _pick_first_env_schedule(calib)
        calib_env_fallback = e0

    single_run = _is_single_run_root(root) and not args.leaderboard

    if single_run:
        ev  = _load_eval_csv(root)
        ep  = _load_per_episode_csv(root)
        group_cols  = ["env", "algo", "schedule", "eval_mode"]
        metric_cols = [c for c in ev.columns
                       if c not in group_cols and c not in ("seed",)]
        agg = ev.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
        agg = _flatten_cols(agg)
        agg = _add_reward_norm(agg)
        agg = _add_leaderboard_score(agg, calib, calib_env_fallback)

        if args.write_csv:
            out = root / "compare.csv"
            agg.to_csv(out, index=False)
            print(f"[OK] wrote: {out}")

        if args.print:
            _print_tables(ev, agg)

        plots_dir = root / "plots"
        if args.plots:
            _make_all_plots(ev, agg, plots_dir, ep_all=ep)
        elif args.tables:
            plots_dir.mkdir(parents=True, exist_ok=True)
            _write_latex_tables(ev, plots_dir)
        return

    # ── multi-run (leaderboard) mode ──────────────────────────────────────
    runs = _discover_runs(root)
    if not runs:
        raise FileNotFoundError(f"No runs found under {root}")

    eval_frames: List[pd.DataFrame] = []
    ep_frames:   List[pd.DataFrame] = []
    for run_dir in runs:
        try:
            df = _load_eval_csv(run_dir)
            df["run_dir"] = str(run_dir)
            eval_frames.append(df)
        except Exception as e:
            print(f"[WARN] {run_dir}: {e}")
        ep = _load_per_episode_csv(run_dir)
        if ep is not None:
            ep_frames.append(ep)

    if not eval_frames:
        raise FileNotFoundError(
            "Found runs but could not read any eval_results.csv"
        )

    ev     = pd.concat(eval_frames, axis=0, ignore_index=True)
    ep_all = (pd.concat(ep_frames, axis=0, ignore_index=True)
              if ep_frames else None)
    agg    = _aggregate_eval(ev)
    agg    = _add_reward_norm(agg)
    agg    = _add_leaderboard_score(agg, calib, calib_env_fallback)
    agg_sorted = agg.sort_values(
        ["env", "schedule", "eval_mode", "leaderboard_score"],
        ascending=[True, True, True, False],
    )

    if args.write_csv:
        out = root / "leaderboard.csv"
        agg_sorted.to_csv(out, index=False)
        print(f"[OK] wrote: {out}")

    if args.print:
        _print_tables(ev, agg_sorted)

    plots_dir = root / "plots"
    if args.plots:
        _make_all_plots(ev, agg_sorted, plots_dir, ep_all=ep_all)
    elif args.tables:
        plots_dir.mkdir(parents=True, exist_ok=True)
        _write_latex_tables(ev, plots_dir)


if __name__ == "__main__":
    main()
