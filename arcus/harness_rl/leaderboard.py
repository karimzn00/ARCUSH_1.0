from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _find_run_eval_files(runs_root: Path) -> List[Tuple[Path, Path]]:
    """
    Returns list of (eval_results.csv, per_episode.csv) pairs.
    """
    out: List[Tuple[Path, Path]] = []
    for p in runs_root.rglob("eval_results.csv"):
        if p.parent.name != "eval":
            continue
        per_ep = p.parent / "per_episode.csv"
        out.append((p, per_ep))
    return out


def _read_eval_results(paths: List[Tuple[Path, Path]]) -> pd.DataFrame:
    rows = []
    for eval_csv, _per in paths:
        try:
            df = pd.read_csv(eval_csv)
            df["run_dir"] = str(eval_csv.parents[1])  # .../runs/bench_xxx
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _read_per_episode(paths: List[Tuple[Path, Path]]) -> pd.DataFrame:
    rows = []
    for _eval_csv, per_csv in paths:
        if not per_csv.exists():
            continue
        try:
            df = pd.read_csv(per_csv)
            df["run_dir"] = str(per_csv.parents[1])
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _minmax_norm_group(df: pd.DataFrame, value_col: str, group_cols: List[str], out_col: str) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-12
    g = df.groupby(group_cols)[value_col]
    vmin = g.transform("min")
    vmax = g.transform("max")
    denom = (vmax - vmin).abs() + eps
    norm = (df[value_col] - vmin) / denom
    deg = (vmax - vmin).abs() < 1e-9
    norm = norm.where(~deg, 0.5)
    df[out_col] = norm.astype(float)
    return df


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "trust_violation_intensity_mean" not in df.columns and "trust_violation_intensity" in df.columns:
        df["trust_violation_intensity_mean"] = df["trust_violation_intensity"]
    return df


def build_leaderboard(eval_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if eval_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = eval_df.copy()

    # aggregate across seeds
    group = ["env", "algo", "eval_mode", "schedule"]
    metric_cols = [c for c in df.columns if c not in group + ["seed", "run_dir"]]
    agg = df.groupby(group)[metric_cols].mean(numeric_only=True).reset_index()

    # normalize reward per env+schedule+eval_mode across algos
    if "reward_mean" in agg.columns:
        agg = _minmax_norm_group(
            agg,
            value_col="reward_mean",
            group_cols=["env", "schedule", "eval_mode"],
            out_col="reward_norm",
        )
    else:
        agg["reward_norm"] = np.nan

    # robustness view uses stress schedules only
    stress_df = agg[agg["schedule"].astype(str).str.lower().isin(
        ["resource_constraint", "trust_violation", "valence_inversion"]
    )].copy()

    for col in ["identity_mean", "collapse_rate_shock", "collapse_score_shock_mean"]:
        if col not in stress_df.columns:
            stress_df[col] = np.nan

    # score weights: identity primary
    stress_df["robust_score"] = (
        0.55 * stress_df["identity_mean"].astype(float)
        + 0.30 * (1.0 - stress_df["collapse_rate_shock"].astype(float))
        + 0.15 * stress_df["reward_norm"].astype(float)
    )

    by_env = stress_df.groupby(["env", "algo", "eval_mode"]).agg(
        robust_score=("robust_score", "mean"),
        identity_mean=("identity_mean", "mean"),
        collapse_rate_shock=("collapse_rate_shock", "mean"),
        reward_norm=("reward_norm", "mean"),
    ).reset_index().sort_values(["env", "robust_score"], ascending=[True, False])

    overall = by_env.groupby(["algo", "eval_mode"]).agg(
        robust_score=("robust_score", "mean"),
        identity_mean=("identity_mean", "mean"),
        collapse_rate_shock=("collapse_rate_shock", "mean"),
        reward_norm=("reward_norm", "mean"),
        env_count=("env", "nunique"),
    ).reset_index().sort_values(["robust_score"], ascending=False)

    return by_env, overall


def _phase_boundaries(per_df: pd.DataFrame) -> Tuple[int | None, int | None]:
    if per_df.empty or "episode_idx" not in per_df.columns or "stress_phase" not in per_df.columns:
        return None, None
    tmp = per_df[["episode_idx", "stress_phase"]].drop_duplicates().sort_values("episode_idx")
    shock = tmp[tmp["stress_phase"].astype(str) == "shock"]
    post = tmp[tmp["stress_phase"].astype(str) == "post"]
    shock_start = int(shock["episode_idx"].min()) if len(shock) else None
    post_start = int(post["episode_idx"].min()) if len(post) else None
    return shock_start, post_start


def _plot_curve_group(per_df: pd.DataFrame, out_path: Path, title: str, y_col: str):
    if per_df.empty or y_col not in per_df.columns:
        return

    algos = sorted(per_df["algo"].astype(str).unique().tolist())

    plt.figure()
    for algo in algos:
        sub = per_df[per_df["algo"].astype(str) == algo].copy()
        if sub.empty:
            continue
        g = sub.groupby("episode_idx")[y_col]
        mean = g.mean()
        std = g.std()

        x = mean.index.to_numpy(dtype=int)
        y = mean.to_numpy(dtype=float)
        s = std.to_numpy(dtype=float)

        plt.plot(x, y, label=algo)
        plt.fill_between(x, y - s, y + s, alpha=0.2)

    shock_start, post_start = _phase_boundaries(per_df)
    if shock_start is not None:
        plt.axvline(shock_start, linestyle="--")
    if post_start is not None:
        plt.axvline(post_start, linestyle="--")

    plt.title(title)
    plt.xlabel("Episode index")
    plt.ylabel(y_col)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="runs", help="Root directory containing runs/bench_* folders")
    ap.add_argument("--out_dir", default="runs/_leaderboard", help="Output directory for merged tables and plots")
    ap.add_argument("--eval_mode", default="deterministic", choices=["deterministic", "stochastic", "both"])
    ap.add_argument("--schedules", default="baseline,resource_constraint,trust_violation,valence_inversion",
                    help="Comma-separated schedules to plot")

    # Compatibility flags (your command)
    ap.add_argument("--write_csv", action="store_true", help="Write CSV outputs (default: ON)")
    ap.add_argument("--no_write_csv", action="store_true", help="Disable writing CSV outputs")
    ap.add_argument("--plots", action="store_true", help="Generate curve plots if per_episode exists (default: ON)")
    ap.add_argument("--no_plots", action="store_true", help="Disable plots generation")

    args = ap.parse_args()

    do_write = bool(args.write_csv or (not args.no_write_csv))
    do_plots = bool(args.plots or (not args.no_plots))

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _find_run_eval_files(runs_root)
    eval_df = _read_eval_results(pairs)
    per_df = _read_per_episode(pairs)

    eval_df = _ensure_cols(eval_df)
    per_df = _ensure_cols(per_df)

    if eval_df.empty:
        raise SystemExit(f"[ERR] No eval_results.csv found under {runs_root}")

    if do_write:
        merged_eval_path = out_dir / "merged_eval_results.csv"
        eval_df.to_csv(merged_eval_path, index=False)
        print(f"[OK] wrote: {merged_eval_path}")

    by_env, overall = build_leaderboard(eval_df)

    if do_write:
        by_env_path = out_dir / "leaderboard_by_env.csv"
        overall_path = out_dir / "leaderboard_overall.csv"
        by_env.to_csv(by_env_path, index=False)
        overall.to_csv(overall_path, index=False)
        print(f"[OK] wrote: {by_env_path}")
        print(f"[OK] wrote: {overall_path}")

    if not do_plots:
        print("[INFO] plots disabled.")
        return

    if per_df.empty:
        print("[WARN] No per_episode.csv files found; skipping curve plots.")
        return

    schedules = [s.strip() for s in (args.schedules or "").split(",") if s.strip()]
    eval_modes = ["deterministic", "stochastic"] if args.eval_mode == "both" else [args.eval_mode]

    needed = {"env", "algo", "eval_mode", "schedule", "episode_idx", "stress_phase"}
    missing = needed - set(per_df.columns)
    if missing:
        raise SystemExit(f"[ERR] per_episode.csv missing required columns: {sorted(missing)}")

    if "episode_return" in per_df.columns:
        per_df = _minmax_norm_group(
            per_df,
            value_col="episode_return",
            group_cols=["env", "schedule", "eval_mode", "episode_idx"],
            out_col="reward_norm_episode",
        )

    for env in sorted(per_df["env"].astype(str).unique().tolist()):
        for eval_mode in eval_modes:
            for schedule in schedules:
                sel = per_df[
                    (per_df["env"].astype(str) == env)
                    & (per_df["eval_mode"].astype(str) == eval_mode)
                    & (per_df["schedule"].astype(str) == schedule)
                ].copy()
                if sel.empty:
                    continue

                base_name = f"{env}__{eval_mode}__{schedule}"

                if "identity" in sel.columns:
                    _plot_curve_group(
                        sel,
                        out_dir / "plots" / env / f"{base_name}__identity.png",
                        title=f"{env} | {eval_mode} | {schedule} | identity (mean±std across seeds)",
                        y_col="identity",
                    )

                if "collapse_score_episode" in sel.columns and sel["collapse_score_episode"].notna().any():
                    _plot_curve_group(
                        sel,
                        out_dir / "plots" / env / f"{base_name}__collapse_score.png",
                        title=f"{env} | {eval_mode} | {schedule} | collapse_score_episode (mean±std)",
                        y_col="collapse_score_episode",
                    )

                if "reward_norm_episode" in sel.columns and sel["reward_norm_episode"].notna().any():
                    _plot_curve_group(
                        sel,
                        out_dir / "plots" / env / f"{base_name}__reward_norm.png",
                        title=f"{env} | {eval_mode} | {schedule} | reward_norm_episode (mean±std)",
                        y_col="reward_norm_episode",
                    )

    print(f"[OK] plots -> {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
