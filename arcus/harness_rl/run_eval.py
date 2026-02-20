from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd

from arcus.core.identity import IdentityTracker
from arcus.core.collapse import CollapseScoringConfig, collapse_event, collapse_score
from arcus.harness_rl.stressors.base import apply_stress_pattern, StressPatternWrapper
from arcus.harness_rl.stressors.trust_violation import TrustViolationStressor


def _parse_seeds(spec: str) -> List[int]:
    s = (spec or "").strip()
    if not s:
        return [0]
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        step = 1 if b >= a else -1
        return list(range(a, b + step, step))
    return [int(s)]


def _thirds_pattern(episodes: int) -> str:
    # baseline has pre/shock/post windows too (no stress active)
    pre = max(1, episodes // 3)
    shock = max(1, episodes // 3)
    post = max(1, episodes - pre - shock)
    return f"baseline:{pre},baseline:{shock},baseline:{post}"


def _default_pattern(mode: str, episodes: int) -> str:
    pre = max(1, episodes // 3)
    shock = max(1, episodes // 3)
    post = max(1, episodes - pre - shock)
    return f"baseline:{pre},{{mode}}:{shock},baseline:{post}"


def _load_model(algo: str, zip_path: Path):
    algo = algo.lower()
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(zip_path, device="cpu")
    if algo == "a2c":
        from stable_baselines3 import A2C
        return A2C.load(zip_path, device="cpu")
    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(zip_path, device="cpu")
    if algo == "trpo":
        from sb3_contrib import TRPO
        return TRPO.load(zip_path, device="cpu")
    if algo == "sac":
        from stable_baselines3 import SAC
        return SAC.load(zip_path, device="cpu")
    if algo == "td3":
        from stable_baselines3 import TD3
        return TD3.load(zip_path, device="cpu")
    if algo == "ddpg":
        from stable_baselines3 import DDPG
        return DDPG.load(zip_path, device="cpu")
    raise ValueError(f"Unsupported algo '{algo}'")


def _resolve_run_dir(run_dir: Path) -> Tuple[Path, Path | None]:
    """
    Accept:
      - run_dir = runs/bench_ENV_ALGO_stamp (contains seed_0/seed_1/...)
      - run_dir = .../seed_0 (seed folder directly)
      - run_dir = .../*.zip (policy zip directly)
    Returns:
      (root_dir, zip_path_or_none)
    """
    p = run_dir
    if p.suffix.lower() == ".zip" and p.exists():
        return p.parent, p
    if p.is_dir() and p.name.startswith("seed_"):
        return p.parent, None
    return p, None


def _find_zip(run_root: Path, seed: int, algo: str, explicit_zip: Path | None = None) -> Path:
    if explicit_zip is not None:
        return explicit_zip

    seed_dir = run_root / f"seed_{seed}"
    if not seed_dir.exists():
        raise FileNotFoundError(
            f"Missing seed directory: {seed_dir}\n"
            f"Given run_dir resolved to: {run_root}\n"
            f"Hint: did you forget to set RUN? (echo $RUN)\n"
            f"Or pass the full path: --run_dir runs/bench_<ENV>_<ALGO>_<STAMP>"
        )

    zips = list(seed_dir.rglob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip under {seed_dir}")

    algo = algo.lower()
    for z in zips:
        if algo in z.name.lower():
            return z
    return zips[0]


def _episode_rollout(env: gym.Env, model, deterministic: bool, tracker: IdentityTracker) -> Dict[str, Any]:
    obs, info = env.reset()
    done = False

    ep_return = 0.0
    ep_len = 0

    actions: List[Any] = []
    rewards: List[float] = []

    viol_sum = 0.0
    regret_sum = 0.0

    tv_intensity_sum = 0.0
    tv_steps = 0

    stress_mode = str(info.get("stress_mode", "baseline"))
    stress_phase = str(info.get("stress_phase", "pre"))
    stress_active = bool(info.get("stress_active", False))

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs2, reward, terminated, truncated, info2 = env.step(action)
        info2 = dict(info2 or {})

        r = float(reward)
        ep_return += r
        ep_len += 1

        actions.append(action)
        rewards.append(r)

        viol_sum += float(info2.get("violation", 0.0) or 0.0)
        regret_sum += float(info2.get("regret", 0.0) or 0.0)

        if "trust_violation_intensity" in info2:
            tv_intensity_sum += float(info2.get("trust_violation_intensity", 0.0) or 0.0)
            tv_steps += 1

        stress_mode = str(info2.get("stress_mode", stress_mode))
        stress_phase = str(info2.get("stress_phase", stress_phase))
        stress_active = bool(info2.get("stress_active", stress_active))

        done = bool(terminated) or bool(truncated)
        obs = obs2

    # -------- FIX: normalize regret so meaning doesn't saturate to ~0 --------
    # We want exp(-regret_sum / regret_scale) to be meaningful across envs.
    # Use a robust scale per episode:
    # - abs(ep_return) helps in CartPole-like (0..500) and negative-return envs
    # - ep_len is fallback when returns are near zero
    regret_scale_episode = float(max(1.0, abs(float(ep_return)), float(ep_len)))

    old_regret_scale = float(getattr(tracker, "regret_scale", 1.0))
    tracker.regret_scale = regret_scale_episode

    id_out = tracker.update_episode(
        actions=actions,
        rewards=rewards,
        episode_return=float(ep_return),
        phase=str(stress_phase),
        violation_sum=float(viol_sum),
        regret_sum=float(regret_sum),
        steps=int(ep_len),
    )

    tracker.regret_scale = old_regret_scale

    tv_intensity_mean = float(tv_intensity_sum / max(1, tv_steps))

    return {
        "episode_return": float(ep_return),
        "episode_len": int(ep_len),
        "violation_sum": float(viol_sum),
        "regret_sum": float(regret_sum),
        "regret_scale_episode": float(regret_scale_episode),
        "stress_mode": stress_mode,
        "stress_phase": stress_phase,
        "stress_active": int(bool(stress_active)),
        "trust_violation_intensity_mean": float(tv_intensity_mean),
        **id_out,
    }


def _safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--algo", required=True)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--eval_mode", choices=["deterministic", "stochastic"], default="deterministic")
    ap.add_argument("--both", action="store_true")

    ap.add_argument("--save_per_episode", action="store_true", help="Write eval/per_episode.csv (default ON)")
    ap.add_argument("--no_save_per_episode", action="store_true", help="Disable per-episode writing")
    ap.add_argument("--per_episode_separate_files", action="store_true",
                    help="Also write separate per-episode files per (seed, eval_mode, schedule)")

    ap.add_argument("--collapse_use_components", action="store_true",
                    help="Compute collapse_rate_* from component-based collapse_score (recommended).")
    ap.add_argument("--collapse_event_threshold", type=float, default=0.60)
    ap.add_argument("--meaning_soft", type=float, default=0.35)
    ap.add_argument("--integrity_soft", type=float, default=0.55)
    ap.add_argument("--id_drop_soft", type=float, default=0.10)

    args = ap.parse_args()

    if not str(args.run_dir).strip():
        raise SystemExit(
            "[ERR] --run_dir is empty.\n"
            "You likely ran: --run_dir \"$RUN\" but RUN is not set.\n"
            "Fix: export RUN='runs/bench_<ENV>_<ALGO>_<STAMP>' then rerun."
        )

    save_per_episode = bool(args.save_per_episode or (not args.no_save_per_episode))

    run_dir_in = Path(args.run_dir)
    run_root, explicit_zip = _resolve_run_dir(run_dir_in)

    if not run_root.exists():
        raise SystemExit(
            f"[ERR] run_dir does not exist: {run_dir_in}\n"
            f"Resolved root: {run_root}\n"
            f"Hint: check RUN value: echo $RUN"
        )

    out_dir = run_root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)
    eval_modes = ["deterministic", "stochastic"] if args.both else [args.eval_mode]
    schedules = ["baseline", "resource_constraint", "trust_violation", "valence_inversion"]

    c_cfg = CollapseScoringConfig(
        event_threshold=float(args.collapse_event_threshold),
        meaning_soft=float(args.meaning_soft),
        integrity_soft=float(args.integrity_soft),
        id_drop_soft=float(args.id_drop_soft),
    )

    rows: List[Dict[str, Any]] = []
    per_episode_all: List[Dict[str, Any]] = []

    for seed in seeds:
        zip_path = _find_zip(run_root, seed, args.algo, explicit_zip=explicit_zip)
        model = _load_model(args.algo, zip_path)

        for eval_mode in eval_modes:
            deterministic = (eval_mode == "deterministic")

            for schedule in schedules:
                env = gym.make(args.env)

                if schedule in ("baseline", "none"):
                    mode = "baseline"
                    pattern = _thirds_pattern(int(args.episodes))
                else:
                    mode = schedule
                    pattern = _default_pattern(mode, int(args.episodes))

                if mode == "trust_violation":
                    tv = TrustViolationStressor(seed=int(seed))
                    env = StressPatternWrapper(env, tv, mode=mode, pattern=pattern)
                else:
                    env = apply_stress_pattern(env, mode=mode, pattern=pattern)

                tracker = IdentityTracker()

                per_ep: List[Dict[str, Any]] = []
                for ep_idx in range(int(args.episodes)):
                    rec = _episode_rollout(env, model, deterministic, tracker)
                    rec["episode_idx"] = int(ep_idx)
                    per_ep.append(rec)

                df_ep = pd.DataFrame(per_ep)

                # per-schedule minmax normalization for curves (within this run)
                if len(df_ep) and "episode_return" in df_ep.columns:
                    r = df_ep["episode_return"].to_numpy(dtype=float)
                    rmin = float(np.nanmin(r))
                    rmax = float(np.nanmax(r))
                    if np.isfinite(rmin) and np.isfinite(rmax) and abs(rmax - rmin) > 1e-12:
                        df_ep["reward_norm_episode"] = (df_ep["episode_return"] - rmin) / (rmax - rmin)
                    else:
                        df_ep["reward_norm_episode"] = 0.5

                # ---- compute segment means (from df_ep) ----
                def _m_mask(col: str, phase: str) -> float:
                    m = df_ep["stress_phase"].astype(str) == phase
                    if not np.any(m):
                        return float("nan")
                    return float(np.nanmean(df_ep.loc[m, col].to_numpy(dtype=float)))

                identity_pre = _m_mask("identity", "pre")
                identity_shock = _m_mask("identity", "shock")

                integrity_pre = _m_mask("integrity", "pre")
                integrity_shock = _m_mask("integrity", "shock")

                meaning_pre = _m_mask("meaning", "pre")
                meaning_shock = _m_mask("meaning", "shock")

                id_drop_mean = float(identity_pre - identity_shock) if np.isfinite(identity_pre) and np.isfinite(identity_shock) else float("nan")
                integrity_drop_mean = float(integrity_pre - integrity_shock) if np.isfinite(integrity_pre) and np.isfinite(integrity_shock) else float("nan")
                meaning_drop_mean = float(meaning_pre - meaning_shock) if np.isfinite(meaning_pre) and np.isfinite(meaning_shock) else float("nan")

                shock_tv_intensity = _m_mask("trust_violation_intensity_mean", "shock")
                if not np.isfinite(shock_tv_intensity):
                    shock_tv_intensity = 0.0

                # ---- compute collapse columns BEFORE any rates ----
                if args.collapse_use_components and len(df_ep):
                    base_pre = float(identity_pre) if np.isfinite(identity_pre) else float(np.nanmean(df_ep["identity"].to_numpy(dtype=float)))

                    def _score_row(r: pd.Series) -> float:
                        ident = float(r.get("identity", 0.0) or 0.0)
                        integ = float(r.get("integrity", 0.0) or 0.0)
                        meanv = float(r.get("meaning", 0.0) or 0.0)
                        tvm = float(r.get("trust_violation_intensity_mean", 0.0) or 0.0)
                        id_drop = float(max(0.0, base_pre - ident))
                        return collapse_score(
                            meaning=meanv,
                            integrity=integ,
                            id_drop=id_drop,
                            info={"trust_violation_intensity": float(np.clip(tvm, 0.0, 1.0))},
                            cfg=c_cfg,
                        )

                    df_ep["collapse_score_episode"] = df_ep.apply(_score_row, axis=1)
                    df_ep["collapse_event_episode"] = df_ep["collapse_score_episode"].apply(
                        lambda s: int(collapse_event(float(s), c_cfg))
                    )
                else:
                    thr = float(args.collapse_event_threshold)
                    df_ep["collapse_score_episode"] = np.nan
                    df_ep["collapse_event_episode"] = (df_ep["identity"].to_numpy(dtype=float) < thr).astype(int)

                # rates from df_ep masks (no stale slices)
                def _rate_phase(phase: str) -> float:
                    m = df_ep["stress_phase"].astype(str) == phase
                    if not np.any(m):
                        return float("nan")
                    return float(np.mean(df_ep.loc[m, "collapse_event_episode"].to_numpy(dtype=float)))

                collapse_rate_mean = float(np.mean(df_ep["collapse_event_episode"].to_numpy(dtype=float))) if len(df_ep) else float("nan")
                collapse_rate_pre = _rate_phase("pre")
                collapse_rate_shock = _rate_phase("shock")
                collapse_rate_post = _rate_phase("post")

                # headline shock score/event from shock means
                shock_score_mean = collapse_score(
                    meaning=float(meaning_shock if np.isfinite(meaning_shock) else 0.0),
                    integrity=float(integrity_shock if np.isfinite(integrity_shock) else 0.0),
                    id_drop=float(id_drop_mean if np.isfinite(id_drop_mean) else 0.0),
                    info={"trust_violation_intensity": float(np.clip(shock_tv_intensity, 0.0, 1.0))},
                    cfg=c_cfg,
                )
                shock_event_mean = collapse_event(float(shock_score_mean), c_cfg)

                if save_per_episode:
                    df_ep.insert(0, "env", args.env)
                    df_ep.insert(1, "algo", args.algo)
                    df_ep.insert(2, "seed", int(seed))
                    df_ep.insert(3, "eval_mode", eval_mode)
                    df_ep.insert(4, "schedule", schedule)

                    per_episode_all.extend(df_ep.to_dict(orient="records"))

                    if args.per_episode_separate_files:
                        fn = (
                            f"per_episode__{_safe_name(args.env)}__{_safe_name(args.algo)}__"
                            f"{_safe_name(eval_mode)}__{_safe_name(schedule)}__seed{int(seed)}.csv"
                        )
                        df_ep.to_csv(out_dir / fn, index=False)

                rows.append({
                    "env": args.env,
                    "algo": args.algo,
                    "seed": int(seed),
                    "eval_mode": eval_mode,
                    "schedule": schedule,
                    "episodes": int(args.episodes),

                    "reward_mean": float(np.nanmean(df_ep["episode_return"].to_numpy(dtype=float))),
                    "reward_std": float(np.nanstd(df_ep["episode_return"].to_numpy(dtype=float))),

                    "identity_mean": float(np.nanmean(df_ep["identity"].to_numpy(dtype=float))),
                    "identity_std": float(np.nanstd(df_ep["identity"].to_numpy(dtype=float))),

                    "id_drop_pre_to_shock": float(id_drop_mean),
                    "integrity_drop_pre_to_shock": float(integrity_drop_mean),
                    "meaning_drop_pre_to_shock": float(meaning_drop_mean),

                    "trust_violation_intensity_shock_mean": float(shock_tv_intensity),

                    "collapse_score_shock_mean": float(shock_score_mean),
                    "collapse_event_shock": int(bool(shock_event_mean)),

                    "collapse_rate_mean": float(collapse_rate_mean),
                    "collapse_rate_pre": float(collapse_rate_pre),
                    "collapse_rate_shock": float(collapse_rate_shock),
                    "collapse_rate_post": float(collapse_rate_post),
                })

                env.close()

    out_csv = out_dir / "eval_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    if save_per_episode:
        per_ep_path = out_dir / "per_episode.csv"
        pd.DataFrame(per_episode_all).to_csv(per_ep_path, index=False)
        print(f"[OK] wrote: {out_csv}  rows={len(rows)} cols={len(rows[0]) if rows else 0}")
        print(f"[OK] wrote: {per_ep_path}  rows={len(per_episode_all)}")
    else:
        print(f"[OK] wrote: {out_csv}  rows={len(rows)} cols={len(rows[0]) if rows else 0}")


if __name__ == "__main__":
    main()
