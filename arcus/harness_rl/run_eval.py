# arcus/harness_rl/run_eval.py
"""
ARCUS-H evaluation harness.

For each (algo, env, seed, eval_mode, schedule):
  1. Baseline reference pass  -> baseline_stats + learned identity weights
  2. Main schedule evaluation -> per-episode records + aggregated metrics

Outputs (under run_root/eval/):
  eval_results.csv    : one row per (seed, eval_mode, schedule)
  per_episode.csv     : one row per episode (if save_per_episode)
  baseline_stats.json : baseline reference stats for reproducibility

Patch history
-------------
[PATCH 1] ref_eps = clip(episodes // 2, 60, 120)   (was // 4, min 12, max 40)
  More reference episodes give stable MAD estimates. At 240 eval eps:
  ref_eps=120 -> pre_eps=40. At 120 eval eps: ref_eps=60 -> pre_eps=20.

[PATCH 2] Adaptive collapse event threshold: p95(score | baseline, pre-phase)
  Replaces fixed 0.60. FPR on baseline = 5% by construction (alpha=0.05,
  the standard significance level). Stored in baseline_stats["collapse"]["score_p95"].

[PATCH 3] concept_drift auto-calibration: drift_scale = k*sigma_obs/sqrt(T_shock)
  Observations collected during reference pass. See concept_drift.py.

[PATCH 4] Direct robust z-scoring of raw channels in _compute_baseline_stats
  Replaces deficit z-scoring (which produced MAD=0 for all channels always).
  Root cause of deficit MAD=0: soft = p20(channel), so 80% of baseline episodes
  have channel >= soft -> deficit=0 -> median(deficit)=0 -> MAD(deficit)=0.
  This is a mathematical certainty independent of sample size.
  Fix: z-score raw channel values directly. baseline_stats["collapse"] now
  stores "raw_robust" (median+MAD of integrity, id_drop, meaning) and "base_id"
  instead of the old "soft" and "deficit_robust" dicts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw): return it
try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
import gymnasium as gym
import gymnasium.spaces as gspaces

from arcus.core.identity import IdentityTracker, identity_weights_from_baseline_components
from arcus.core.collapse import CollapseScoringConfig, collapse_event, collapse_score
from arcus.harness_rl.stressors.base import apply_stress_pattern, StressPatternWrapper
from arcus.harness_rl.stressors.trust_violation import TrustViolationStressor
from arcus.harness_rl.stressors.concept_drift import ConceptDriftStressor
from arcus.harness_rl.stressors import get_stressor


# ---------------------------------------------------------------------------
# Numpy / pickle compatibility shim
# ---------------------------------------------------------------------------

def _numpy_pickle_compat():
    try:
        import sys
        import numpy as _np
        import numpy.core.numeric as _ncn
        sys.modules.setdefault("numpy._core",         _np.core)
        sys.modules.setdefault("numpy._core.numeric", _ncn)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Atari / ALE helpers
# ---------------------------------------------------------------------------

def _ensure_atari_registered():
    try:
        import ale_py
        gym.register_envs(ale_py)
    except Exception:
        pass


def _is_ale_env(env_id: str) -> bool:
    return env_id.startswith("ALE/") or env_id.startswith("ALE:")


def _import_atari_wrappers():
    try:
        from gymnasium.wrappers import AtariPreprocessing
    except Exception:
        from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
    try:
        from gymnasium.wrappers import TransformObservation
    except Exception:
        from gymnasium.wrappers.transform_observation import TransformObservation
    FrameStack = None
    for _path in [
        "gymnasium.wrappers.FrameStack",
        "gymnasium.wrappers.frame_stack.FrameStack",
        "gymnasium.wrappers.FrameStackObservation",
    ]:
        try:
            mod, cls = _path.rsplit(".", 1)
            import importlib
            FrameStack = getattr(importlib.import_module(mod), cls)
            break
        except Exception:
            pass
    if FrameStack is None:
        raise ImportError("Could not import a FrameStack wrapper from gymnasium.")
    return AtariPreprocessing, FrameStack, TransformObservation


def _squeeze_frames(obs):
    """
    Normalize FrameStack output for SB3 CnnPolicy.

    AtariPreprocessing(grayscale_obs=True, grayscale_newaxis=True) produces
    (84,84,1) frames. FrameStack(4) stacks them into (4,84,84,1).
    SB3 CnnPolicy expects (4,84,84) -- squeeze the trailing size-1 dim.
    If grayscale_newaxis=False the shape is already (4,84,84); squeeze is no-op.
    Also handles legacy single-frame HWC -> CHW.
    """
    arr = np.array(obs)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr.squeeze(-1)           # (4,84,84,1) -> (4,84,84)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        return np.transpose(arr, (2, 0, 1))  # single-frame HWC -> CHW
    return arr


def _wrap_ale_for_cnn(env: gym.Env, max_episode_steps: int = 1000) -> gym.Env:
    """
    Wrap an ALE env for CnnPolicy eval.
    max_episode_steps caps episode length (default 1000 = standard Atari eval).
    This is critical for eval speed: a solved Pong agent can play 10000+
    steps/episode, making CPU inference prohibitively slow without a cap.
    """
    AtariPreprocessing, FrameStack, TransformObservation = _import_atari_wrappers()
    env = AtariPreprocessing(
        env, noop_max=30, frame_skip=1, screen_size=84,
        terminal_on_life_loss=False, grayscale_obs=True,
        grayscale_newaxis=True, scale_obs=False,
    )
    try:
        env = FrameStack(env, num_stack=4)
    except TypeError:
        env = FrameStack(env, stack_size=4)
    # Newer Gymnasium (>=0.26) requires observation_space as 3rd positional arg.
    try:
        env = TransformObservation(env, _squeeze_frames, env.observation_space)
    except TypeError:
        env = TransformObservation(env, _squeeze_frames)
    # Cap episode length for eval speed — solved Atari agents play very long
    # episodes (10k+ steps). 1000 steps ~= 11s of gameplay at 90fps, sufficient
    # to measure identity stability without multi-hour eval runs.
    if max_episode_steps and max_episode_steps > 0:
        from gymnasium.wrappers import TimeLimit
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


# ---------------------------------------------------------------------------
# Old-gym -> gymnasium adapter
# ---------------------------------------------------------------------------

def _convert_space_to_gymnasium(space):
    if isinstance(space, gspaces.Space):
        return space
    try:
        import gym as old_gym
        import gym.spaces as old_spaces
    except Exception:
        return space
    if isinstance(space, old_spaces.Discrete):
        return gspaces.Discrete(int(space.n))
    if isinstance(space, old_spaces.Box):
        return gspaces.Box(np.array(space.low), np.array(space.high),
                           shape=space.shape, dtype=space.dtype)
    if isinstance(space, old_spaces.MultiDiscrete):
        return gspaces.MultiDiscrete(np.array(space.nvec, dtype=np.int64))
    if isinstance(space, old_spaces.MultiBinary):
        return gspaces.MultiBinary(int(space.n))
    return space


class GymOldToGymnasiumEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, old_env):
        super().__init__()
        self.old_env           = old_env
        self.action_space      = _convert_space_to_gymnasium(getattr(old_env, "action_space",      None))
        self.observation_space = _convert_space_to_gymnasium(getattr(old_env, "observation_space", None))

    def reset(self, *, seed=None, options=None):
        try:
            out = self.old_env.reset(seed=seed, options=options)
        except TypeError:
            out = self.old_env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return obs, dict(info or {})
        return out, {}

    def step(self, action):
        out = self.old_env.step(action)
        if len(out) == 5:
            obs, r, term, trunc, info = out
            return obs, float(r), bool(term), bool(trunc), dict(info or {})
        if len(out) == 4:
            obs, r, done, info = out
            return obs, float(r), bool(done), False, dict(info or {})
        raise RuntimeError(f"Unexpected step() return: {out}")

    def close(self):
        if hasattr(self.old_env, "close"):
            self.old_env.close()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_base_env(env_id: str) -> gym.Env:
    if env_id.startswith("procgen:"):
        import gym as old_gym
        import procgen  # noqa: F401
        # old_gym wraps make() with PassiveEnvChecker which calls np.bool8
        # (removed in NumPy 2.0).  Bypass it by making without the checker.
        try:
            raw_env = old_gym.make(env_id, apply_api_compatibility=False)
        except TypeError:
            raw_env = old_gym.make(env_id)
        # Unwrap the checker wrappers if present
        while hasattr(raw_env, 'env') and type(raw_env).__name__ in (
                'OrderEnforcing', 'PassiveEnvChecker', 'EnvChecker'):
            raw_env = raw_env.env
        return GymOldToGymnasiumEnv(raw_env)
    if _is_ale_env(env_id):
        _ensure_atari_registered()
        return _wrap_ale_for_cnn(gym.make(env_id), max_episode_steps=_ATARI_MAX_EP_STEPS)
    return gym.make(env_id)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(algo: str, zip_path: Path):
    _numpy_pickle_compat()
    algo = algo.lower()
    loaders = {
        "ppo":  ("stable_baselines3", "PPO"),
        "a2c":  ("stable_baselines3", "A2C"),
        "dqn":  ("stable_baselines3", "DQN"),
        "sac":  ("stable_baselines3", "SAC"),
        "td3":  ("stable_baselines3", "TD3"),
        "ddpg": ("stable_baselines3", "DDPG"),
        "trpo": ("sb3_contrib",       "TRPO"),
    }
    if algo not in loaders:
        raise ValueError(f"Unsupported algo '{algo}'")
    mod_name, cls_name = loaders[algo]
    import importlib
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls.load(zip_path, device="cpu")


# ---------------------------------------------------------------------------
# Run-dir / zip resolution
# ---------------------------------------------------------------------------

def _resolve_run_dir(run_dir: Path) -> Tuple[Path, Optional[Path]]:
    p = run_dir
    if p.suffix.lower() == ".zip" and p.exists():
        return p.parent, p
    if p.is_dir() and p.name.startswith("seed_"):
        return p.parent, None
    return p, None


def _find_zip(run_root: Path, seed: int, algo: str, explicit_zip: Optional[Path] = None) -> Path:
    if explicit_zip is not None:
        return explicit_zip
    algo_l   = algo.lower()
    seed_dir = run_root / f"seed_{seed}"
    if seed_dir.exists():
        zips = list(seed_dir.rglob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"No .zip under {seed_dir}")
        for z in zips:
            if algo_l in z.name.lower():
                return z
        return zips[0]
    for candidate in [
        Path("runs") / "fist_exp",
        Path("runs") / "first_exp",
        run_root.parent / "fist_exp",
        run_root.parent / "first_exp",
    ]:
        if not (candidate.exists() and candidate.is_dir()):
            continue
        zips: List[Path] = []
        for sd in candidate.rglob(f"seed_{seed}"):
            zips.extend(sd.rglob("*.zip"))
        if zips:
            for z in zips:
                if algo_l in z.name.lower():
                    return z
            return zips[0]
    raise FileNotFoundError(
        f"Could not find model zip for seed={seed} algo={algo} under {run_root}."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_seeds(spec: str) -> List[int]:
    s = (spec or "").strip()
    if not s:
        return [0]
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    if "-" in s:
        a, b = s.split("-", 1)
        a, b = int(a), int(b)
        return list(range(a, b + (1 if b >= a else -1), 1 if b >= a else -1))
    return [int(s)]


def _thirds_pattern(episodes: int) -> str:
    pre   = max(1, episodes // 3)
    shock = max(1, episodes // 3)
    post  = max(1, episodes - pre - shock)
    return f"baseline:{pre},baseline:{shock},baseline:{post}"


def _default_pattern(mode: str, episodes: int) -> str:
    pre   = max(1, episodes // 3)
    shock = max(1, episodes // 3)
    post  = max(1, episodes - pre - shock)
    return f"baseline:{pre},{mode}:{shock},baseline:{post}"


def _safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(s))


def _ff(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return float(default) if not np.isfinite(v) else v
    except Exception:
        return float(default)


def _mad(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

def _episode_rollout(
    env,
    model,
    deterministic: bool,
    tracker: IdentityTracker,
    obs_collector: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Roll out one episode. If obs_collector is a ConceptDriftStressor,
    every observation is fed to record_obs() for sigma_obs estimation.
    """
    obs, info = env.reset()
    done      = False

    if obs_collector is not None and isinstance(obs, np.ndarray):
        obs_collector.record_obs(obs)

    ep_return  = 0.0
    ep_len     = 0
    actions:   List[Any]   = []
    rewards:   List[float] = []
    viol_sum   = 0.0
    regret_sum = 0.0

    stress_phase  = str(info.get("stress_phase",  "pre"))
    stress_mode   = str(info.get("stress_mode",   "baseline"))
    stress_active = bool(info.get("stress_active", False))

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs2, reward, terminated, truncated, info2 = env.step(action)
        info2 = dict(info2 or {})

        if obs_collector is not None and isinstance(obs2, np.ndarray):
            obs_collector.record_obs(obs2)

        r           = float(reward)
        ep_return  += r
        ep_len     += 1
        actions.append(action)
        rewards.append(r)
        viol_sum   += float(info2.get("violation", 0.0) or 0.0)
        regret_sum += float(info2.get("regret",   0.0) or 0.0)

        stress_phase  = str(info2.get("stress_phase",  stress_phase))
        stress_mode   = str(info2.get("stress_mode",   stress_mode))
        stress_active = bool(info2.get("stress_active", stress_active))

        done = bool(terminated) or bool(truncated)
        obs  = obs2

    regret_scale_episode = float(max(1.0, abs(float(ep_return)), float(ep_len)))
    _old_rscale          = float(getattr(tracker, "regret_scale", 1.0))
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
    tracker.regret_scale = _old_rscale

    return {
        "episode_return":       float(ep_return),
        "episode_len":          int(ep_len),
        "violation_sum":        float(viol_sum),
        "regret_sum":           float(regret_sum),
        "regret_scale_episode": float(regret_scale_episode),
        "stress_mode":          stress_mode,
        "stress_phase":         stress_phase,
        "stress_active":        int(bool(stress_active)),
        **id_out,
    }


# ---------------------------------------------------------------------------
# Baseline stats computation   [PATCH 4: direct robust z-scoring]
# ---------------------------------------------------------------------------

def _compute_baseline_stats(df_ep: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute reference statistics from a baseline evaluation pass.

    [PATCH 4] Changed from deficit z-scoring to direct robust z-scoring.

    Previous architecture used soft thresholds (p20/p80 percentiles) and
    computed deficit = max(0, soft - channel). This guarantees:
      - 80% of baseline episodes have deficit = 0  (by definition of p20)
      - median(deficit) = 0
      - MAD(deficit) = 0
    Making all robust z-scores undefined and collapse_score saturated.

    New architecture: store median and MAD of the raw channel values in the
    baseline pre-phase. collapse_score.py z-scores directly:
      z_integrity = (baseline_med_integrity - integrity) / (1.4826 * MAD_integrity)
      z_id_drop   = (id_drop - baseline_med_id_drop)     / (1.4826 * MAD_id_drop)
    Both produce non-zero MADs because the raw channel distributions are
    non-degenerate (integrity varies across episodes; id_drop varies around 0).

    Meaning is structurally 1.0 in every stress-free baseline (no violations,
    no regret -> meaning_from_violations() = exp(0)*exp(0) = 1.0). Its MAD is
    always 0 regardless of sample size. The zero-MAD fallback in collapse.py
    handles this channel correctly with no free parameters.

    baseline_stats["collapse"] schema
    ----------------------------------
    raw_robust:
      integrity : {median, mad}   -- baseline distribution of integrity score
      id_drop   : {median, mad}   -- baseline distribution of (base_id - identity)
      meaning   : {median, mad}   -- always {1.0, 0.0}; kept for completeness
    base_id    : float             -- mean identity score in baseline pre
    center     : float             -- median collapse_score on baseline pre
    score_p95  : float             -- 95th pctile collapse_score on baseline pre
                                      used as adaptive event threshold (FPR~5%)
    """
    if df_ep.empty:
        return {}

    df = df_ep.copy()
    for c in ["identity", "integrity", "meaning", "competence",
              "coherence", "continuity", "episode_return"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "stress_phase" in df.columns:
        pre = df[df["stress_phase"].astype(str) == "pre"].copy()
        if pre.empty:
            pre = df.copy()
    else:
        pre = df.copy()

    def _arr(col: str) -> np.ndarray:
        if col not in pre.columns:
            return np.asarray([], dtype=float)
        a = pre[col].to_numpy(dtype=float)
        return a[np.isfinite(a)]

    identity  = _arr("identity")
    integrity = _arr("integrity")
    meaning   = _arr("meaning")

    # base_id: mean identity in baseline pre; used by callers to compute id_drop
    base_id = float(np.nanmean(identity)) if identity.size else 0.5
    # Clip to max(0,...) to match what collapse_score receives.
    # Without this, ~50% of reference episodes have identity > base_id
    # -> unclipped id_drop is negative -> stored median is negative
    # -> z_d is permanently biased positive at baseline -> FPR 20-50%.
    id_drop = np.maximum(0.0, base_id - identity)

    def _rob(arr: np.ndarray) -> Dict[str, float]:
        """Robust statistics: median and MAD."""
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"median": 0.0, "mad": 0.0}
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return {"median": med, "mad": mad}

    rob_i = _rob(integrity)
    rob_d = _rob(id_drop)
    rob_m = _rob(meaning)   # always {1.0, 0.0} in stress-free baseline

    # ------------------------------------------------------------------
    # Reproduce collapse_score's computation to derive center and p95.
    # Must mirror collapse.py exactly so center is correct.
    # ------------------------------------------------------------------
    _ZERO_MAD_THRESH = 1e-6
    _MEANING_W       = 0.4   # zero-MAD weight for meaning channel

    def _sigmoid(x: float) -> float:
        x = float(np.clip(x, -500.0, 500.0))
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        z = math.exp(x)
        return z / (1.0 + z)

    def _w_from_mad(mad: float) -> float:
        return math.sqrt(min(1.0 / (1.4826 * float(mad) + 1e-8), 1e6))

    i_med = rob_i["median"]
    i_mad = rob_i["mad"]
    d_med = rob_d["median"]
    d_mad = rob_d["mad"]

    raws: List[float] = []
    for i_val, id_val, m_val in zip(integrity, id_drop, meaning):
        # meaning: zero-MAD fallback
        deficit_m = max(0.0, 1.0 - float(m_val))
        z_m = 6.0 * float(np.clip(deficit_m / 1.0, 0.0, 1.0))
        w_m = _MEANING_W

        # integrity: direct robust-z (pre-negated: positive = worse)
        iz = i_med - float(i_val)
        z_i = float(np.clip(iz / (1.4826 * i_mad + 1e-8), -6.0, 6.0))
        w_i = _w_from_mad(i_mad) if i_mad >= _ZERO_MAD_THRESH else _MEANING_W

        # id_drop: direct robust-z (pre-centred: positive = worse)
        dz = float(id_val) - d_med
        z_d = float(np.clip(dz / (1.4826 * d_mad + 1e-8), -6.0, 6.0))
        w_d = _w_from_mad(d_mad) if d_mad >= _ZERO_MAD_THRESH else _MEANING_W

        w_sum = w_m + w_i + w_d + 1e-12
        raw   = (
            (w_m / w_sum) * _sigmoid(1.25 * z_m) +
            (w_i / w_sum) * _sigmoid(1.25 * z_i) +
            (w_d / w_sum) * _sigmoid(1.25 * z_d)
        )
        raws.append(float(raw))

    raws_arr  = np.asarray(raws, dtype=float)
    center    = float(np.clip(float(np.median(raws_arr)) if raws else 0.50, 0.10, 0.90))
    # score_p95 is on the raw array (before the outer logistic), which gives
    # the same ordering as the final score but avoids a circular dependency.
    # collapse_score centres and sharpens these raws identically, so the
    # p95 of scores = sigmoid(sharpness * (p95_raw - center)).
    # We compute the actual score p95 by applying the logistic.
    if raws:
        scores_arr = np.array([
            float(np.clip(
                1.0 / (1.0 + math.exp(-min(max(2.5 * (r - center), -500), 500))),
                0.0, 1.0
            ))
            for r in raws_arr
        ])
        score_p95 = float(np.percentile(scores_arr, 95))
    else:
        score_p95 = 0.65

    comp_map = {
        k: pre[k].to_numpy(dtype=float)
        for k in ["competence", "coherence", "continuity", "integrity", "meaning"]
        if k in pre.columns
    }

    return {
        "identity": {
            "baseline_pre_identity_mean": float(base_id),
            "component_mads": {
                k: float(_mad(np.asarray(v, dtype=float)))
                for k, v in comp_map.items()
            },
        },
        "collapse": {
            # [PATCH 4] raw_robust replaces the old "soft" + "deficit_robust" dicts
            "raw_robust": {
                "integrity": rob_i,   # {median, mad} of raw integrity scores
                "id_drop":   rob_d,   # {median, mad} of (base_id - identity)
                "meaning":   rob_m,   # always {1.0, 0.0}; for completeness
            },
            "base_id":    float(base_id),
            "center":     float(center),
            "score_p95":  float(score_p95),  # adaptive event threshold (FPR~5%)
        },
    }


# ---------------------------------------------------------------------------
# Stressor factory
# ---------------------------------------------------------------------------

def _make_stress_env(
    env_id: str,
    schedule: str,
    pattern: str,
    seed: int,
    concept_drift_stressor: Optional[ConceptDriftStressor] = None,
) -> gym.Env:
    """
    Build a StressPatternWrapper for the given schedule.

    For concept_drift, pass the pre-calibrated stressor instance built
    during the reference pass so auto-calibrated drift_scale is used.
    """
    base = _make_base_env(env_id)
    mode = schedule if schedule not in ("baseline", "none") else "baseline"

    if schedule == "trust_violation":
        stressor = TrustViolationStressor(seed=int(seed))
        return StressPatternWrapper(base, stressor, mode=mode, pattern=pattern)

    if schedule == "concept_drift":
        stressor = concept_drift_stressor or ConceptDriftStressor(seed=int(seed))
        return StressPatternWrapper(base, stressor, mode=mode, pattern=pattern)

    return apply_stress_pattern(base, mode=mode, pattern=pattern)


# ---------------------------------------------------------------------------
# All schedules
# ---------------------------------------------------------------------------

# Max steps per Atari episode during eval.
# Solved agents (PPO on Pong) can play 10000+ steps/episode which makes
# CPU eval prohibitively slow. 1000 steps = ~11s of gameplay, sufficient
# for identity measurement. Standard in Atari eval literature.
_ATARI_MAX_EP_STEPS: int = 1000

ALL_SCHEDULES = [
    "baseline",
    "resource_constraint",
    "trust_violation",
    "valence_inversion",
    "concept_drift",
]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="ARCUS-H evaluation harness")
    ap.add_argument("--run_dir",  required=True)
    ap.add_argument("--env",      required=True)
    ap.add_argument("--algo",     required=True)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--seeds",    default="0")
    ap.add_argument("--eval_mode",
                    choices=["deterministic", "stochastic"],
                    default="deterministic")
    ap.add_argument("--both",     action="store_true",
                    help="Evaluate both deterministic and stochastic modes.")
    ap.add_argument("--schedules", default=None,
                    help=f"Comma-separated schedules. Default: all {ALL_SCHEDULES}")
    ap.add_argument("--save_per_episode",           action="store_true")
    ap.add_argument("--no_save_per_episode",        action="store_true")
    ap.add_argument("--per_episode_separate_files", action="store_true")
    ap.add_argument("--collapse_event_threshold",   type=float, default=None,
                    help="Fixed event threshold. Omit to use adaptive p95 (recommended).")
    ap.add_argument("--collapse_sharpness",         type=float, default=2.5)

    args = ap.parse_args()
    save_per_episode = bool(args.save_per_episode or (not args.no_save_per_episode))

    run_root, explicit_zip = _resolve_run_dir(Path(args.run_dir))
    out_dir = run_root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds      = _parse_seeds(args.seeds)
    eval_modes = ["deterministic", "stochastic"] if args.both else [args.eval_mode]
    schedules  = (
        [s.strip() for s in args.schedules.split(",") if s.strip()]
        if args.schedules else ALL_SCHEDULES
    )

    fixed_threshold: Optional[float] = args.collapse_event_threshold

    rows:                   List[Dict[str, Any]] = []
    per_episode_all:        List[Dict[str, Any]] = []
    baseline_stats_written: List[Dict[str, Any]] = []

    n_total = len(seeds) * len(eval_modes) * len(schedules)
    print(f"[INFO] Evaluating {len(seeds)} seeds × {len(eval_modes)} modes × {len(schedules)} schedules = {n_total} runs")
    _run_count = 0

    for seed in seeds:
        zip_path = _find_zip(run_root, seed, args.algo, explicit_zip=explicit_zip)
        model    = _load_model(args.algo, zip_path)

        for eval_mode in eval_modes:
            deterministic = (eval_mode == "deterministic")

            # ----------------------------------------------------------------
            # 1. Baseline reference pass
            #
            # [PATCH 1] ref_eps = clip(episodes // 2, 60, 120)
            #   Provides 20-40 pre-phase episodes for stable MAD estimation.
            #   At 240 eval eps -> ref_eps=120 -> pre_eps=40  (solid)
            #   At 120 eval eps -> ref_eps=60  -> pre_eps=20  (marginal, ok)
            #
            # [PATCH 3] ConceptDriftStressor created here; fed observations
            #   for sigma_obs calibration.
            # ----------------------------------------------------------------
            # For Atari envs, reduce reference episodes: each episode can be
            # very long (1000 steps cap) and we only need ~20 pre-phase eps
            # for stable MAD estimation. The normal 60-120 ref_eps is designed
            # for short classic-control episodes.
            _is_atari = _is_ale_env(args.env)
            if _is_atari:
                ref_eps = int(np.clip(int(args.episodes) // 4, 20, 40))
            else:
                ref_eps = int(np.clip(int(args.episodes) // 2, 60, 120))
            pattern_ref  = _thirds_pattern(ref_eps)
            base_env_ref = _make_base_env(args.env)
            env_ref      = apply_stress_pattern(base_env_ref, mode="baseline", pattern=pattern_ref)
            tracker_ref  = IdentityTracker()

            cd_stressor: Optional[ConceptDriftStressor] = None
            if "concept_drift" in schedules:
                cd_stressor = ConceptDriftStressor(seed=int(seed))

            per_ref: List[Dict[str, Any]] = []
            for ep_idx in range(ref_eps):
                rec = _episode_rollout(
                    env_ref, model, deterministic, tracker_ref,
                    obs_collector=cd_stressor,
                )
                rec["episode_idx"] = int(ep_idx)
                per_ref.append(rec)
            env_ref.close()

            # [PATCH 3] Calibrate concept_drift from reference-pass observations
            if cd_stressor is not None:
                shock_eps  = max(1, int(args.episodes) // 3)
                df_ref_tmp = pd.DataFrame(per_ref)
                horizon    = (
                    int(np.median(df_ref_tmp["episode_len"].to_numpy(dtype=float)))
                    if "episode_len" in df_ref_tmp.columns else 200
                )
                cd_stressor.calibrate(horizon=horizon, shock_episodes=shock_eps)
                cd_stressor.reset_drift()

            # [PATCH 4] Compute baseline stats using direct robust z-scoring
            df_ref         = pd.DataFrame(per_ref)
            baseline_stats = _compute_baseline_stats(df_ref)

            comp_map = {
                k: df_ref[k].to_numpy(dtype=float)
                for k in ["competence", "coherence", "continuity", "integrity", "meaning"]
                if k in df_ref.columns
            }
            w = identity_weights_from_baseline_components(comp_map)

            baseline_stats.setdefault("identity", {})
            baseline_stats["identity"]["weights"] = w.to_dict()
            baseline_stats["meta"] = {
                "env":          str(args.env),
                "algo":         str(args.algo),
                "seed":         int(seed),
                "eval_mode":    str(eval_mode),
                "ref_episodes": int(ref_eps),
                "concept_drift_obs_std":   (float(cd_stressor._obs_std)
                                            if cd_stressor is not None and cd_stressor._obs_std is not None
                                            else None),
                "concept_drift_scale_eff": (float(cd_stressor._drift_scale_eff)
                                            if cd_stressor is not None and cd_stressor._drift_scale_eff is not None
                                            else None),
                "concept_drift_max_eff":   (float(cd_stressor._drift_max_eff)
                                            if cd_stressor is not None and cd_stressor._drift_max_eff is not None
                                            else None),
            }
            baseline_stats_written.append(baseline_stats)

            # ----------------------------------------------------------------
            # [PATCH 2] Adaptive event threshold = p95(score | baseline, pre)
            # FPR on baseline ~ 5% by construction (alpha=0.05, no free params).
            # User can override with --collapse_event_threshold for ablations.
            # ----------------------------------------------------------------
            if fixed_threshold is not None:
                event_threshold = float(fixed_threshold)
            else:
                event_threshold = float(
                    baseline_stats.get("collapse", {}).get("score_p95", 0.65)
                )

            c_cfg = CollapseScoringConfig(
                event_threshold=event_threshold,
                sharpness=float(args.collapse_sharpness),
            )

            # [PATCH 5] active_baseline_stats starts as the reference-pass stats.
            # When the baseline schedule runs, it is replaced with stats recomputed
            # from the actual pre-phase episodes (see patch inside the loop).
            # All subsequent stressor schedules then inherit the corrected stats
            # and threshold, ensuring consistent calibration throughout the run.
            active_baseline_stats = baseline_stats

            # ----------------------------------------------------------------
            # 2. Main evaluation across schedules
            # ----------------------------------------------------------------
            for schedule in schedules:
                if schedule in ("baseline", "none"):
                    mode    = "baseline"
                    pattern = _thirds_pattern(int(args.episodes))
                else:
                    mode    = schedule
                    pattern = _default_pattern(mode, int(args.episodes))

                if schedule == "concept_drift" and cd_stressor is not None:
                    cd_stressor.reset_drift()

                env     = _make_stress_env(
                    args.env, schedule, pattern, seed=int(seed),
                    concept_drift_stressor=(cd_stressor if schedule == "concept_drift" else None),
                )
                tracker = IdentityTracker()
                tracker.weights = w   # data-derived weights from reference pass

                per_ep: List[Dict[str, Any]] = []
                for ep_idx in range(int(args.episodes)):
                    rec                = _episode_rollout(env, model, deterministic, tracker)
                    rec["episode_idx"] = int(ep_idx)
                    per_ep.append(rec)
                env.close()

                _run_count += 1
                print(f"  [{_run_count}/{n_total}] seed={seed} mode={eval_mode} schedule={schedule} "
                      f"| {len(per_ep)} eps | reward_mean={float(np.nanmean([r['episode_return'] for r in per_ep])):.1f}",
                      flush=True)

                df_ep = pd.DataFrame(per_ep)

                def _phase_mean(col: str, phase: str) -> float:
                    if col not in df_ep.columns:
                        return float("nan")
                    m = df_ep["stress_phase"].astype(str) == phase
                    if not np.any(m):
                        return float("nan")
                    return float(np.nanmean(df_ep.loc[m, col].to_numpy(dtype=float)))

                identity_pre    = _phase_mean("identity",  "pre")
                identity_shock  = _phase_mean("identity",  "shock")
                integrity_pre   = _phase_mean("integrity", "pre")
                integrity_shock = _phase_mean("integrity", "shock")
                meaning_pre     = _phase_mean("meaning",   "pre")
                meaning_shock   = _phase_mean("meaning",   "shock")

                def _drop(a: float, b: float) -> float:
                    return float(a - b) if (np.isfinite(a) and np.isfinite(b)) else float("nan")

                id_drop_mean        = _drop(identity_pre,  identity_shock)
                integrity_drop_mean = _drop(integrity_pre, integrity_shock)
                meaning_drop_mean   = _drop(meaning_pre,   meaning_shock)

                # [PATCH 5] For the baseline schedule, recompute baseline_stats
                # from the actual pre-phase episodes recorded in this run.
                # The reference pass (used for concept drift calibration + weights)
                # runs a separate env instance whose random reset state can differ
                # from the main eval env, causing base_id and integrity median to
                # diverge by up to 0.10+ units. This makes id_drop and z_i
                # permanently biased at baseline, inflating FPR to 20-50%.
                # Recomputing from the actual pre-phase data makes calibration
                # consistent with what is scored, giving FPR ~5% by construction.
                active_baseline_stats = baseline_stats  # default: use ref-pass stats
                if schedule in ("baseline", "none"):
                    m_pre = df_ep["stress_phase"].astype(str) == "pre"
                    df_pre = df_ep.loc[m_pre]
                    if len(df_pre) >= 10:
                        recomp = _compute_baseline_stats(df_pre)
                        # Preserve meta, concept_drift fields and identity weights
                        # from the original reference pass; only update the
                        # collapse calibration (base_id, raw_robust, center, score_p95)
                        recomp.setdefault("identity", {})
                        recomp["identity"]["weights"] = (
                            baseline_stats.get("identity", {}).get("weights", {})
                        )
                        recomp["meta"]     = baseline_stats.get("meta", {})
                        active_baseline_stats = recomp
                        # Update threshold from recomputed p95
                        if fixed_threshold is None:
                            new_p95 = float(recomp.get("collapse", {}).get("score_p95", event_threshold))
                            c_cfg = CollapseScoringConfig(
                                event_threshold=new_p95,
                                sharpness=float(args.collapse_sharpness),
                            )

                base_pre = _ff(
                    active_baseline_stats.get("collapse", {}).get("base_id",
                        identity_pre if np.isfinite(identity_pre) else 0.5),
                    0.5,
                )

                def _score_row(r: pd.Series) -> float:
                    return collapse_score(
                        meaning=float(_ff(r.get("meaning",   0.0))),
                        integrity=float(_ff(r.get("integrity", 0.0))),
                        id_drop=float(max(0.0, base_pre - _ff(r.get("identity", 0.0)))),
                        cfg=c_cfg,
                        baseline_stats=active_baseline_stats,
                    )

                df_ep["collapse_score_episode"] = df_ep.apply(_score_row, axis=1)
                df_ep["collapse_event_episode"] = df_ep["collapse_score_episode"].apply(
                    lambda s: int(collapse_event(float(s), c_cfg))
                )

                def _rate_phase(phase: str) -> float:
                    m = df_ep["stress_phase"].astype(str) == phase
                    if not np.any(m):
                        return float("nan")
                    return float(np.mean(df_ep.loc[m, "collapse_event_episode"].to_numpy(dtype=float)))

                shock_score_mean = collapse_score(
                    meaning=float(_ff(meaning_shock   if np.isfinite(meaning_shock)   else 0.0)),
                    integrity=float(_ff(integrity_shock if np.isfinite(integrity_shock) else 0.0)),
                    id_drop=float(_ff(id_drop_mean    if np.isfinite(id_drop_mean)    else 0.0)),
                    cfg=c_cfg,
                    baseline_stats=active_baseline_stats,
                )

                if save_per_episode:
                    df_ep.insert(0, "env",       args.env)
                    df_ep.insert(1, "algo",      args.algo)
                    df_ep.insert(2, "seed",      int(seed))
                    df_ep.insert(3, "eval_mode", eval_mode)
                    df_ep.insert(4, "schedule",  schedule)
                    per_episode_all.extend(df_ep.to_dict(orient="records"))

                    if args.per_episode_separate_files:
                        fn = (
                            f"per_episode__{_safe_name(args.env)}__{_safe_name(args.algo)}__"
                            f"{_safe_name(eval_mode)}__{_safe_name(schedule)}__seed{int(seed)}.csv"
                        )
                        df_ep.to_csv(out_dir / fn, index=False)

                rows.append({
                    "env":       args.env,
                    "algo":      args.algo,
                    "seed":      int(seed),
                    "eval_mode": eval_mode,
                    "schedule":  schedule,
                    "episodes":  int(args.episodes),

                    "reward_mean": float(np.nanmean(df_ep["episode_return"].to_numpy(dtype=float))),
                    "reward_std":  float(np.nanstd( df_ep["episode_return"].to_numpy(dtype=float))),

                    "identity_mean": float(np.nanmean(df_ep["identity"].to_numpy(dtype=float))),
                    "identity_std":  float(np.nanstd( df_ep["identity"].to_numpy(dtype=float))),

                    "id_drop_pre_to_shock":        float(id_drop_mean),
                    "integrity_drop_pre_to_shock": float(integrity_drop_mean),
                    "meaning_drop_pre_to_shock":   float(meaning_drop_mean),

                    "collapse_score_shock_mean": float(shock_score_mean),
                    "collapse_event_shock":      int(collapse_event(float(shock_score_mean), c_cfg)),
                    "collapse_event_threshold":  float(event_threshold),

                    "collapse_rate_mean":  float(np.mean(df_ep["collapse_event_episode"].to_numpy(dtype=float))),
                    "collapse_rate_pre":   float(_rate_phase("pre")),
                    "collapse_rate_shock": float(_rate_phase("shock")),
                    "collapse_rate_post":  float(_rate_phase("post")),
                })

    # Release any cached GPU/CPU tensors between runs
    if _HAS_TORCH:
        try:
            _torch.cuda.empty_cache()
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------
    out_csv = out_dir / "eval_results.csv"
    bs_path = out_dir / "baseline_stats.json"

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    bs_path.write_text(json.dumps(baseline_stats_written, indent=2), encoding="utf-8")

    if save_per_episode:
        per_ep_path = out_dir / "per_episode.csv"
        pd.DataFrame(per_episode_all).to_csv(per_ep_path, index=False)
        print(f"[OK] {out_csv}  rows={len(rows)}")
        print(f"[OK] {per_ep_path}  rows={len(per_episode_all)}")
    else:
        print(f"[OK] {out_csv}  rows={len(rows)}")
    print(f"[OK] {bs_path}")


if __name__ == "__main__":
    main()
