from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: List[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _parse_seeds(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return [0]
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--envs", default="CartPole-v1")
    ap.add_argument("--algos", default="ppo")
    ap.add_argument("--timesteps", type=int, default=100_000)

    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", default="0-9")

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--progress_bar", action="store_true")

    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--recovery_ratio", type=float, default=0.70)

    ap.add_argument("--collapse_rel_floor", type=float, default=0.35)
    ap.add_argument("--collapse_abs_identity", type=float, default=0.12)
    ap.add_argument("--collapse_abs_integrity", type=float, default=0.06)
    ap.add_argument("--collapse_abs_meaning", type=float, default=0.06)
    ap.add_argument("--collapse_streak_shock", type=int, default=3)
    ap.add_argument("--collapse_streak_post", type=int, default=2)
    ap.add_argument("--collapse_use_components", action="store_true")

    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--skip_compare", action="store_true")

    ap.add_argument("--compare_print", action="store_true")
    ap.add_argument("--infer_action_space", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--write_csv", action="store_true")

    ap.add_argument("--csv", default=None)
    ap.add_argument("--plots_dir", default=None)
    ap.add_argument("--suite_write_csv", action="store_true")
    ap.add_argument("--suite_plots", action="store_true")

    args = ap.parse_args()

    envs = _split_csv(args.envs)
    algos = _split_csv(args.algos)

    if args.seed is not None:
        seeds = [int(args.seed)]
    else:
        seeds = _parse_seeds(args.seeds)

    out_root = Path(args.out_root)
    ensure_dir(out_root)

    run_id = args.run_id
    if not run_id:
        from datetime import datetime
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"[OK] suite stamp/run_id = {run_id}")
    print(f"[OK] envs={envs}")
    print(f"[OK] algos={algos}")
    print(f"[OK] seeds={seeds}")

    for env_id in envs:
        for algo in algos:
            run_dir = out_root / f"bench_{env_id}_{algo}_{run_id}"
            ensure_dir(run_dir)

            if not args.skip_train:
                for seed in seeds:
                    seed_dir = run_dir / f"seed_{seed}"
                    ensure_dir(seed_dir)

                    cmd_train = [
                        "python", "-m", "arcus.harness_rl.run_train",
                        "--env", env_id,
                        "--algo", algo,
                        "--timesteps", str(args.timesteps),
                        "--seed", str(seed),
                        "--out_dir", str(seed_dir),
                        "--run_id", "train_none",
                        "--device", str(args.device),
                        "--log_interval", str(args.log_interval),
                    ]
                    run_cmd(cmd_train)

            if not args.skip_eval:
                cmd_eval = [
                    "python", "-m", "arcus.harness_rl.run_eval",
                    "--run_dir", str(run_dir),
                    "--env", env_id,
                    "--algo", algo,
                    "--seeds", ",".join(str(s) for s in seeds),
                    "--episodes", str(args.episodes),
                    "--workers", str(args.workers),
                    "--recovery_ratio", str(args.recovery_ratio),
                    "--collapse_rel_floor", str(args.collapse_rel_floor),
                    "--collapse_abs_identity", str(args.collapse_abs_identity),
                    "--collapse_abs_integrity", str(args.collapse_abs_integrity),
                    "--collapse_abs_meaning", str(args.collapse_abs_meaning),
                    "--collapse_streak_shock", str(args.collapse_streak_shock),
                    "--collapse_streak_post", str(args.collapse_streak_post),
                    "--device", str(args.device),
                ]
                if args.both:
                    cmd_eval.append("--both")
                if args.collapse_use_components:
                    cmd_eval.append("--collapse_use_components")
                run_cmd(cmd_eval)

            if not args.skip_compare:
                cmd_cmp = [
                    "python", "-m", "arcus.harness_rl.compare",
                    "--root", str(run_dir),
                ]
                if args.compare_print:
                    cmd_cmp.append("--print")
                if args.infer_action_space:
                    cmd_cmp.append("--infer_action_space")
                if args.plots:
                    cmd_cmp.append("--plots")
                if args.write_csv:
                    cmd_cmp.append("--write_csv")
                if args.csv:
                    cmd_cmp += ["--csv", str(args.csv)]
                if args.plots_dir:
                    cmd_cmp += ["--plots_dir", str(args.plots_dir)]
                if args.suite_write_csv:
                    cmd_cmp.append("--suite_write_csv")
                if args.suite_plots:
                    cmd_cmp.append("--suite_plots")

                run_cmd(cmd_cmp)

    print("[DONE] benchmark suite complete.")


if __name__ == "__main__":
    main()
