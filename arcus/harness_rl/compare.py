from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_eval_csv(run_root: Path) -> pd.DataFrame:
    p = run_root / "eval" / "eval_results.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. You must run run_eval first to generate eval/eval_results.csv"
        )
    return pd.read_csv(p)


def _load_per_episode_csv(run_root: Path) -> pd.DataFrame | None:
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
    """
    Find all run folders under root that contain eval/eval_results.csv.
    """
    runs: List[Path] = []
    for p in root.rglob("eval/eval_results.csv"):
        run_dir = p.parent.parent
        runs.append(run_dir)
    uniq = []
    seen = set()
    for r in sorted(runs):
        if str(r) not in seen:
            uniq.append(r)
            seen.add(str(r))
    return uniq


def _aggregate_eval(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["env", "algo", "schedule", "eval_mode"]
    metric_cols = [c for c in df.columns if c not in group_cols and c not in ("seed", "run_dir")]

    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    agg = _flatten_cols(agg)

    if "reward_mean__mean" in agg.columns:
        agg["reward_mean"] = agg["reward_mean__mean"]
    if "identity_mean__mean" in agg.columns:
        agg["identity_mean"] = agg["identity_mean__mean"]
    if "collapse_rate_shock__mean" in agg.columns:
        agg["collapse_rate_shock"] = agg["collapse_rate_shock__mean"]

    return agg


def _add_reward_norm(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize reward within each (env, schedule, eval_mode) across algos to [0,1].
    This avoids comparing raw reward scales across different envs.
    """
    agg = agg.copy()
    if "reward_mean__mean" not in agg.columns:
        agg["reward_norm"] = np.nan
        return agg

    def _norm_group(g: pd.DataFrame) -> pd.Series:
        x = g["reward_mean__mean"].to_numpy(dtype=float)
        if not np.any(np.isfinite(x)):
            return pd.Series(np.full(len(g), np.nan), index=g.index)

        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))
        if not np.isfinite(xmin) or not np.isfinite(xmax) or abs(xmax - xmin) < 1e-12:
            return pd.Series(np.full(len(g), 0.5), index=g.index)

        y = (x - xmin) / (xmax - xmin)
        return pd.Series(y, index=g.index)

    agg["reward_norm"] = (
        agg.groupby(["env", "schedule", "eval_mode"], dropna=False, sort=False)
        .apply(_norm_group)
        .reset_index(level=[0, 1, 2], drop=True)
    )
    return agg


def _add_leaderboard_score(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Composite score for sorting the leaderboard.
    Tune weights as you like.
    """
    agg = agg.copy()

    ident = agg["identity_mean__mean"].to_numpy(dtype=float) if "identity_mean__mean" in agg.columns else np.full(len(agg), np.nan)
    crs = agg["collapse_rate_shock__mean"].to_numpy(dtype=float) if "collapse_rate_shock__mean" in agg.columns else np.full(len(agg), np.nan)
    rnorm = agg["reward_norm"].to_numpy(dtype=float) if "reward_norm" in agg.columns else np.full(len(agg), np.nan)

    # fill missing pieces safely
    ident = np.nan_to_num(ident, nan=0.0, posinf=0.0, neginf=0.0)
    crs = np.nan_to_num(crs, nan=0.0, posinf=0.0, neginf=0.0)
    rnorm = np.nan_to_num(rnorm, nan=0.5, posinf=0.5, neginf=0.5)

    score = 0.45 * ident + 0.45 * (1.0 - np.clip(crs, 0.0, 1.0)) + 0.10 * np.clip(rnorm, 0.0, 1.0)
    agg["leaderboard_score"] = score
    return agg


def _make_single_run_plots(run_root: Path, agg: pd.DataFrame):
    plots_dir = run_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if "identity_mean__mean" in agg.columns:
        for eval_mode in sorted(agg["eval_mode"].unique().tolist()):
            sub = agg[agg["eval_mode"] == eval_mode].copy()
            sub = sub.sort_values("identity_mean__mean", ascending=False)

            plt.figure()
            plt.bar(sub["schedule"].tolist(), sub["identity_mean__mean"].to_numpy())
            plt.xticks(rotation=30, ha="right")
            plt.title(f"Identity mean by schedule ({eval_mode})")
            plt.tight_layout()
            plt.savefig(plots_dir / f"identity_by_schedule_{eval_mode}.png")
            plt.close()

    if "reward_mean__mean" in agg.columns:
        for eval_mode in sorted(agg["eval_mode"].unique().tolist()):
            sub = agg[agg["eval_mode"] == eval_mode].copy()
            sub = sub.sort_values("reward_mean__mean", ascending=False)

            plt.figure()
            plt.bar(sub["schedule"].tolist(), sub["reward_mean__mean"].to_numpy())
            plt.xticks(rotation=30, ha="right")
            plt.title(f"Reward mean by schedule ({eval_mode})")
            plt.tight_layout()
            plt.savefig(plots_dir / f"reward_by_schedule_{eval_mode}.png")
            plt.close()


def _make_leaderboard_curves(root: Path, per_ep_all: pd.DataFrame):
    """
    Plot identity and reward curves per env/schedule/eval_mode comparing algos.
    Curves are mean across seeds. Episode axis is episode_idx (as emitted by run_eval).
    """
    plots_dir = root / "leaderboard_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    needed = {"env", "algo", "schedule", "eval_mode", "seed", "episode_idx", "identity", "episode_return"}
    missing = sorted(list(needed - set(per_ep_all.columns)))
    if missing:
        print(f"[WARN] per_episode.csv missing columns: {missing} -> skipping curve plots")
        return

    per_ep_all = per_ep_all.copy()
    per_ep_all["episode_idx"] = pd.to_numeric(per_ep_all["episode_idx"], errors="coerce")
    per_ep_all["identity"] = pd.to_numeric(per_ep_all["identity"], errors="coerce")
    per_ep_all["episode_return"] = pd.to_numeric(per_ep_all["episode_return"], errors="coerce")

    envs = sorted(per_ep_all["env"].dropna().unique().tolist())
    schedules = sorted(per_ep_all["schedule"].dropna().unique().tolist())
    eval_modes = sorted(per_ep_all["eval_mode"].dropna().unique().tolist())

    for env in envs:
        for schedule in schedules:
            for eval_mode in eval_modes:
                sub = per_ep_all[
                    (per_ep_all["env"] == env)
                    & (per_ep_all["schedule"] == schedule)
                    & (per_ep_all["eval_mode"] == eval_mode)
                ].copy()
                if sub.empty:
                    continue

                g = (
                    sub.groupby(["algo", "episode_idx"], dropna=True)[["identity", "episode_return"]]
                    .mean()
                    .reset_index()
                    .sort_values(["algo", "episode_idx"])
                )

                algos = sorted(g["algo"].dropna().unique().tolist())

                plt.figure()
                for algo in algos:
                    gg = g[g["algo"] == algo]
                    plt.plot(gg["episode_idx"].to_numpy(), gg["identity"].to_numpy(), label=str(algo))
                plt.title(f"Identity curves | {env} | {schedule} | {eval_mode}")
                plt.xlabel("Episode")
                plt.ylabel("Identity")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / f"identity_curve__{env}__{schedule}__{eval_mode}.png")
                plt.close()

                plt.figure()
                for algo in algos:
                    gg = g[g["algo"] == algo]
                    plt.plot(gg["episode_idx"].to_numpy(), gg["episode_return"].to_numpy(), label=str(algo))
                plt.title(f"Reward curves | {env} | {schedule} | {eval_mode}")
                plt.xlabel("Episode")
                plt.ylabel("Episode return")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / f"reward_curve__{env}__{schedule}__{eval_mode}.png")
                plt.close()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Single run dir (runs/bench_...) OR a folder containing many runs (e.g. runs/).")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--write_csv", action="store_true")
    ap.add_argument("--plots", action="store_true")

    ap.add_argument("--leaderboard", action="store_true", help="Force multi-run leaderboard mode even if root looks like a single run.")
    args = ap.parse_args()

    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    single_run = _is_single_run_root(root) and (not args.leaderboard)

    if single_run:
        df = _load_eval_csv(root)

        group_cols = ["env", "algo", "schedule", "eval_mode"]
        metric_cols = [c for c in df.columns if c not in group_cols and c not in ("seed",)]
        agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
        agg = _flatten_cols(agg)

        if args.write_csv:
            out_compare = root / "compare.csv"
            agg.to_csv(out_compare, index=False)
            print(f"[OK] wrote: {out_compare}")

            out_stats = root / "suite_stats.csv"
            agg.to_csv(out_stats, index=False)
            print(f"[OK] wrote: {out_stats}")
            print("[INFO] stats cols:", list(agg.columns))

        if args.print:
            cols = [c for c in [
                "env", "schedule", "eval_mode", "algo",
                "identity_mean__mean", "identity_mean__std",
                "collapse_rate_shock__mean",
                "reward_mean__mean",
            ] if c in agg.columns]
            print("\n=== STATS TABLE ===")
            print(agg[cols].to_string(index=False))

        if args.plots:
            _make_single_run_plots(root, agg)
            print(f"[OK] plots -> {root / 'plots'}")

        return

    runs = _discover_runs(root)
    if not runs:
        raise FileNotFoundError(f"No runs found under {root} (expected **/eval/eval_results.csv).")

    eval_frames: List[pd.DataFrame] = []
    per_ep_frames: List[pd.DataFrame] = []

    for run_dir in runs:
        try:
            df = _load_eval_csv(run_dir)
            df = df.copy()
            df["run_dir"] = str(run_dir)
            eval_frames.append(df)
        except Exception as e:
            print(f"[WARN] failed reading eval_results in {run_dir}: {e}")

        pep = _load_per_episode_csv(run_dir)
        if pep is not None and len(pep):
            pep = pep.copy()
            pep["run_dir"] = str(run_dir)
            per_ep_frames.append(pep)

    if not eval_frames:
        raise FileNotFoundError(f"Found runs but could not read any eval_results.csv under {root}")

    df_all = pd.concat(eval_frames, axis=0, ignore_index=True)

    agg = _aggregate_eval(df_all)
    agg = _add_reward_norm(agg)
    agg = _add_leaderboard_score(agg)
    agg_sorted = agg.sort_values(["env", "schedule", "eval_mode", "leaderboard_score"], ascending=[True, True, True, False])

    if args.write_csv:
        out_long = root / "leaderboard_long.csv"
        df_all.to_csv(out_long, index=False)
        print(f"[OK] wrote: {out_long}")

        out_lb = root / "leaderboard.csv"
        agg_sorted.to_csv(out_lb, index=False)
        print(f"[OK] wrote: {out_lb}")

    if args.print:
        cols = [c for c in [
            "env", "schedule", "eval_mode", "algo",
            "leaderboard_score",
            "identity_mean__mean",
            "collapse_rate_shock__mean",
            "reward_mean__mean",
            "reward_norm",
        ] if c in agg_sorted.columns]
        print("\n=== LEADERBOARD (TOP ROWS PER ENV/SCHEDULE/EVAL_MODE) ===")

        shown = (
            agg_sorted.groupby(["env", "schedule", "eval_mode"], dropna=False)
            .head(5)[cols]
        )
        print(shown.to_string(index=False))

    if args.plots and per_ep_frames:
        per_ep_all = pd.concat(per_ep_frames, axis=0, ignore_index=True)
        _make_leaderboard_curves(root, per_ep_all)
        print(f"[OK] leaderboard curve plots -> {root / 'leaderboard_plots'}")
    elif args.plots:
        print("[WARN] no per_episode.csv found under runs -> skipping curve plots")

    if args.write_csv:
        out_stats = root / "leaderboard_stats.csv"
        agg_sorted.to_csv(out_stats, index=False)
        print(f"[OK] wrote: {out_stats}")


if __name__ == "__main__":
    main()
