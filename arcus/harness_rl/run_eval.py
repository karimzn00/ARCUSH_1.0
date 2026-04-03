from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

import gymnasium as gym
import gymnasium.spaces as gspaces

from arcus.core.identity import (
    IdentityTracker,
    identity_score,
    identity_weights_from_baseline_components,
)
from arcus.core.collapse import CollapseScoringConfig, collapse_event, collapse_score
from arcus.core.meaning_proxy import MeaningProxyTracker
from arcus.harness_rl.stressors.base import apply_stress_pattern, StressPatternWrapper
from arcus.harness_rl.stressors.trust_violation import TrustViolationStressor
from arcus.harness_rl.stressors.valence_inversion import ValenceInversionStressor
from arcus.harness_rl.stressors.resource_constraint import ResourceConstraintStressor
from arcus.harness_rl.stressors.concept_drift import ConceptDriftStressor
from arcus.harness_rl.stressors.observation_noise import ObservationNoiseStressor
from arcus.harness_rl.stressors.sensor_blackout import SensorBlackoutStressor
from arcus.harness_rl.stressors.reward_noise import RewardNoiseStressor
from arcus.harness_rl.stressors import get_stressor

class RunningMeanStdWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, warmup: int = 500, clip: float = 5.0):
        super().__init__(env)
        self._warmup  = int(warmup)
        self._clip    = float(clip)
        self._n       = 0
        self._mean:  Optional[np.ndarray] = None
        self._M2:    Optional[np.ndarray] = None
        self._obs_shape = None

    def _lazy_init(self, obs: np.ndarray):
        if self._mean is not None:
            return
        shape = obs.shape
        self._obs_shape = shape
        self._mean      = np.zeros(shape, dtype=np.float64)
        self._M2        = np.zeros(shape, dtype=np.float64)
        lo = np.full(shape, -self._clip, dtype=np.float32)
        hi = np.full(shape,  self._clip, dtype=np.float32)
        self.observation_space = gspaces.Box(lo, hi, shape=shape, dtype=np.float32)

    def _update(self, obs: np.ndarray):
        x = obs.astype(np.float64).reshape(self._mean.shape)
        self._n   += 1
        delta      = x - self._mean
        self._mean += delta / self._n
        self._M2   += delta * (x - self._mean)

    @property
    def _var(self) -> np.ndarray:
        return self._M2 / max(self._n - 1, 1)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self._n < self._warmup:
            return obs.astype(np.float32)
        x = obs.astype(np.float64).reshape(self._mean.shape)
        z = (x - self._mean) / (np.sqrt(self._var) + 1e-8)
        return np.clip(z, -self._clip, self._clip).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        arr = np.asarray(obs, dtype=np.float64)
        self._lazy_init(arr)
        self._update(arr)
        return self._normalize(arr), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        arr = np.asarray(obs, dtype=np.float64)
        self._lazy_init(arr)
        self._update(arr)
        return self._normalize(arr), reward, terminated, truncated, info

def _free_memory():
    gc.collect()
    if _HAS_TORCH:
        try:
            _torch.cuda.empty_cache()
        except Exception:
            pass


def _del_model(model):
    try:
        del model
    except Exception:
        pass
    _free_memory()

def _numpy_pickle_compat():
    try:
        import sys
        import numpy as _np
        import numpy.core.numeric as _ncn
        sys.modules.setdefault("numpy._core",         _np.core)
        sys.modules.setdefault("numpy._core.numeric", _ncn)
    except Exception:
        pass

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
    arr = np.array(obs)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr.squeeze(-1)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        return np.transpose(arr, (2, 0, 1))
    return arr


def _wrap_ale_for_cnn(env: gym.Env, max_episode_steps: int = 1000) -> gym.Env:
    AtariPreprocessing, FrameStack, TransformObservation = _import_atari_wrappers()
    env = AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=84,
        terminal_on_life_loss=True, grayscale_obs=True,
        grayscale_newaxis=True, scale_obs=False,
    )
    try:
        env = FrameStack(env, num_stack=4)
    except TypeError:
        env = FrameStack(env, stack_size=4)
    try:
        env = TransformObservation(env, _squeeze_frames, env.observation_space)
    except TypeError:
        env = TransformObservation(env, _squeeze_frames)
    if max_episode_steps and max_episode_steps > 0:
        from gymnasium.wrappers import TimeLimit
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

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

_ATARI_MAX_EP_STEPS: int = 1000


def _make_base_env(env_id: str, obs_normalize: bool = False) -> gym.Env:
    if env_id.startswith("procgen:"):
        import gym as old_gym
        import procgen  # noqa: F401
        try:
            raw_env = old_gym.make(env_id, apply_api_compatibility=False)
        except TypeError:
            raw_env = old_gym.make(env_id)
        while hasattr(raw_env, "env") and type(raw_env).__name__ in (
                "OrderEnforcing", "PassiveEnvChecker", "EnvChecker"):
            raw_env = raw_env.env
        return GymOldToGymnasiumEnv(raw_env)

    if _is_ale_env(env_id):
        _ensure_atari_registered()
        raw = gym.make(env_id, frameskip=1, repeat_action_probability=0.0)
        env = _wrap_ale_for_cnn(raw, max_episode_steps=_ATARI_MAX_EP_STEPS)
        if obs_normalize:
            env = RunningMeanStdWrapper(env)
        return env

    env = gym.make(env_id)
    if obs_normalize:
        env = RunningMeanStdWrapper(env)
    return env

def _load_model(algo: str, zip_path: Path):
    _numpy_pickle_compat()
    algo = algo.lower()
    loaders = {
        "ppo":   ("stable_baselines3", "PPO"),
        "a2c":   ("stable_baselines3", "A2C"),
        "dqn":   ("stable_baselines3", "DQN"),
        "sac":   ("stable_baselines3", "SAC"),
        "td3":   ("stable_baselines3", "TD3"),
        "ddpg":  ("stable_baselines3", "DDPG"),
        "trpo":  ("sb3_contrib",       "TRPO"),
        "qrdqn": ("sb3_contrib",       "QRDQN"),
    }
    if algo not in loaders:
        raise ValueError(f"Unsupported algo '{algo}'. Supported: {list(loaders)}")
    mod_name, cls_name = loaders[algo]
    import importlib
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls.load(zip_path, device="cpu")

def _resolve_run_dir(run_dir: Path) -> Tuple[Path, Optional[Path]]:
    p = run_dir
    if p.suffix.lower() == ".zip" and p.exists():
        return p.parent, p
    if p.is_dir() and p.name.startswith("seed_"):
        return p.parent, None
    return p, None


def _find_zip(run_root: Path, seed: int, algo: str,
              explicit_zip: Optional[Path] = None) -> Path:
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
    zips = list(run_root.rglob("*.zip"))
    if zips:
        for z in zips:
            if algo_l in z.name.lower():
                return z
        return zips[0]
    raise FileNotFoundError(
        f"No model zip found for seed={seed} algo={algo} under {run_root}"
    )

def _parse_seeds(spec: str) -> List[int]:
    s = (spec or "").strip()
    if not s:
        return [0]
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
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

def _episode_rollout(
    env,
    model,
    deterministic: bool,
    tracker: IdentityTracker,
    meaning_proxy: Optional[MeaningProxyTracker] = None,
    obs_collector: Optional[Any] = None,
) -> Dict[str, Any]:
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

    if meaning_proxy is not None and viol_sum == 0.0 and regret_sum == 0.0:
        proxy_meaning = meaning_proxy.compute(actions, stress_phase)
        id_out        = dict(id_out)
        id_out["meaning"] = float(proxy_meaning)
        id_out["identity"] = float(identity_score(
            competence=float(id_out["competence"]),
            coherence=float(id_out["coherence"]),
            continuity=float(id_out["continuity"]),
            integrity=float(id_out["integrity"]),
            meaning=float(proxy_meaning),
            weights=tracker.weights,
        ))

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

def _compute_baseline_stats(df_ep: pd.DataFrame) -> Dict[str, Any]:
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

    base_id = float(np.nanmean(identity)) if identity.size else 0.5
    id_drop = np.maximum(0.0, base_id - identity)

    def _rob(arr: np.ndarray) -> Dict[str, float]:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"median": 0.0, "mad": 0.0}
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return {"median": med, "mad": mad}

    rob_i = _rob(integrity)
    rob_d = _rob(id_drop)
    rob_m = _rob(meaning)

    eps   = 1e-8
    i_med = rob_i["median"];  i_mad = rob_i["mad"]
    d_med = rob_d["median"];  d_mad = rob_d["mad"]
    m_mad = rob_m["mad"]

    w_sum = i_mad + d_mad + m_mad + eps
    w_i   = i_mad / w_sum
    w_d   = d_mad / w_sum
    w_m   = m_mad / w_sum

    raws: List[float] = []
    for i_val, id_val, m_val in zip(integrity, id_drop, meaning):
        z_i = float(np.clip((i_med - float(i_val)) / (1.4826 * i_mad + eps), -6.0, 6.0))
        z_d = float(np.clip((float(id_val) - d_med) / (1.4826 * d_mad + eps), -6.0, 6.0))
        deficit_m = max(0.0, 1.0 - float(m_val))
        if m_mad > 1e-6:
            z_m = float(np.clip(
                (deficit_m - (1.0 - rob_m["median"])) / (1.4826 * m_mad + eps), 0.0, 6.0))
        else:
            z_m = float(np.clip(deficit_m / 0.1, 0.0, 6.0))
        raw = math.sqrt(w_i * z_i**2 + w_d * z_d**2 + w_m * z_m**2)
        raws.append(float(raw))

    raws_arr = np.asarray(raws, dtype=float)
    center   = float(np.clip(
        float(np.median(raws_arr)) if len(raws_arr) > 0 else 0.5, 0.0, 10.0))

    if len(raws_arr) > 0:
        scores_arr = np.array([
            float(np.clip(1.0 / (1.0 + math.exp(
                -min(max(2.5 * (r - center), -500), 500))), 0.0, 1.0))
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

    channel_mads = {k: float(_mad(np.asarray(v, dtype=float))) for k, v in comp_map.items()}

    return {
        "identity": {
            "baseline_pre_identity_mean": float(base_id),
            "component_mads": {
                k: float(_mad(np.asarray(v, dtype=float)))
                for k, v in comp_map.items()
            },
        },
        "collapse": {
            "raw_robust": {"integrity": rob_i, "id_drop": rob_d, "meaning": rob_m},
            "base_id":   float(base_id),
            "center":    float(center),
            "score_p95": float(score_p95),
        },
        "channel_mads": channel_mads,
    }

def _make_stress_env(
    env_id: str,
    schedule: str,
    pattern: str,
    seed: int,
    concept_drift_stressor: Optional[ConceptDriftStressor] = None,
    observation_noise_stressor: Optional[ObservationNoiseStressor] = None,
    sensor_blackout_stressor: Optional[SensorBlackoutStressor] = None,
    reward_noise_stressor: Optional[RewardNoiseStressor] = None,
    obs_normalize: bool = False,
) -> gym.Env:
    base = _make_base_env(env_id, obs_normalize=obs_normalize)
    mode = schedule if schedule not in ("baseline", "none") else "baseline"

    if schedule == "resource_constraint":
        return StressPatternWrapper(base, ResourceConstraintStressor(),
                                    mode=mode, pattern=pattern)
    if schedule == "valence_inversion":
        return StressPatternWrapper(base, ValenceInversionStressor(),
                                    mode=mode, pattern=pattern)
    if schedule == "trust_violation":
        return StressPatternWrapper(base, TrustViolationStressor(seed=int(seed)),
                                    mode=mode, pattern=pattern)
    if schedule == "concept_drift":
        st = concept_drift_stressor or ConceptDriftStressor(seed=int(seed))
        return StressPatternWrapper(base, st, mode=mode, pattern=pattern)
    if schedule == "observation_noise":
        st = observation_noise_stressor or ObservationNoiseStressor(seed=int(seed))
        return StressPatternWrapper(base, st, mode=mode, pattern=pattern)
    if schedule == "sensor_blackout":
        st = sensor_blackout_stressor or SensorBlackoutStressor(seed=int(seed))
        return StressPatternWrapper(base, st, mode=mode, pattern=pattern)
    if schedule == "reward_noise":
        st = reward_noise_stressor or RewardNoiseStressor(seed=int(seed))
        return StressPatternWrapper(base, st, mode=mode, pattern=pattern)

    return apply_stress_pattern(base, mode=mode, pattern=pattern)

ALL_SCHEDULES = [
    "baseline",
    "resource_constraint",
    "trust_violation",
    "valence_inversion",
    "concept_drift",
    "observation_noise",
    "sensor_blackout",
    "reward_noise",
]

REWARD_STRESSOR_SET = {"valence_inversion", "reward_noise"}

def _cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Conditional Value-at-Risk at level alpha.
    Returns the mean of the bottom alpha-fraction of returns.
    Defined so that lower (more negative) CVaR = worse tail performance.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(alpha * arr.size)))
    return float(np.mean(np.sort(arr)[:k]))

def _eval_seed_mode(
    seed: int,
    eval_mode: str,
    schedules: List[str],
    args,
    run_root: Path,
    explicit_zip: Optional[Path],
    out_dir: Path,
    fixed_threshold: Optional[float],
    _done_keys: set,
    save_per_episode: bool,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:

    rows:                   List[Dict[str, Any]] = []
    per_episode_all:        List[Dict[str, Any]] = []
    baseline_stats_written: List[Dict[str, Any]] = []

    deterministic = (eval_mode == "deterministic")

    pending = [s for s in schedules
               if (int(seed), str(eval_mode), str(s)) not in _done_keys]
    if not pending:
        print(f"  [RESUME] seed={seed} mode={eval_mode} — all done, skipping.")
        return rows, per_episode_all, baseline_stats_written

    zip_path = _find_zip(run_root, seed, args.algo, explicit_zip=explicit_zip)
    model    = _load_model(args.algo, zip_path)

    try:
        _is_atari = _is_ale_env(args.env)
        ref_eps   = int(np.clip(int(args.episodes) // 4 if _is_atari
                                else int(args.episodes) // 2, 20 if _is_atari else 60,
                                40  if _is_atari else 120))

        obs_normalize  = bool(getattr(args, "obs_normalize", False))
        fpr_target     = float(getattr(args, "fpr_target", 0.05))
        fpr_percentile = float(np.clip((1.0 - fpr_target) * 100.0, 50.0, 99.0))
        pattern_ref  = _thirds_pattern(ref_eps)
        base_env_ref = _make_base_env(args.env, obs_normalize=obs_normalize)
        env_ref      = apply_stress_pattern(base_env_ref, mode="baseline", pattern=pattern_ref)
        tracker_ref  = IdentityTracker()
        mp_ref = MeaningProxyTracker()
        cd_stressor: Optional[ConceptDriftStressor] = None
        if "concept_drift" in schedules:
            cd_stressor = ConceptDriftStressor(seed=int(seed))

        on_stressor: Optional[ObservationNoiseStressor] = None
        if "observation_noise" in schedules:
            on_stressor = ObservationNoiseStressor(seed=int(seed))

        sb_stressor: Optional[SensorBlackoutStressor] = None
        if "sensor_blackout" in schedules:
            sb_stressor = SensorBlackoutStressor(seed=int(seed))

        rn_stressor: Optional[RewardNoiseStressor] = None
        if "reward_noise" in schedules:
            rn_stressor = RewardNoiseStressor(seed=int(seed))

        per_ref: List[Dict[str, Any]] = []
        try:
            for ep_idx in range(ref_eps):
                rec = _episode_rollout(
                    env_ref, model, deterministic, tracker_ref,
                    meaning_proxy=mp_ref,
                    obs_collector=cd_stressor,
                )
                rec["episode_idx"] = int(ep_idx)
                per_ref.append(rec)
        finally:
            env_ref.close()

        if cd_stressor is not None:
            shock_eps  = max(1, int(args.episodes) // 3)
            df_ref_tmp = pd.DataFrame(per_ref)
            horizon    = (int(np.median(df_ref_tmp["episode_len"].to_numpy(dtype=float)))
                          if "episode_len" in df_ref_tmp.columns else 200)
            cd_stressor.calibrate(horizon=horizon, shock_episodes=shock_eps)
            cd_stressor.reset_drift()

        if on_stressor is not None:
            if cd_stressor is not None and cd_stressor._obs_std is not None:
                on_stressor.calibrate(obs_std=float(cd_stressor._obs_std))
            else:
                _tmp = _make_base_env(args.env)
                on_stressor.calibrate(obs_std=None, observation_space=_tmp.observation_space)
                _tmp.close()
            on_stressor.reset_rng()

        if sb_stressor is not None:
            df_ref_tmp = pd.DataFrame(per_ref)
            horizon = (int(np.median(df_ref_tmp["episode_len"].to_numpy(dtype=float)))
                       if "episode_len" in df_ref_tmp.columns else 200)
            sb_stressor.calibrate(horizon=horizon)
            sb_stressor.reset_rng()

        if rn_stressor is not None:
            df_ref_tmp = pd.DataFrame(per_ref)
            if "episode_return" in df_ref_tmp.columns and "episode_len" in df_ref_tmp.columns:
                returns     = df_ref_tmp["episode_return"].to_numpy(dtype=float)
                lengths     = np.where(df_ref_tmp["episode_len"].to_numpy(dtype=float) > 0,
                                       df_ref_tmp["episode_len"].to_numpy(dtype=float), 1.0)
                step_r      = returns / lengths
                reward_std  = float(np.std(step_r))
                reward_min  = float(np.min(step_r))
                reward_max  = float(np.max(step_r))
            else:
                reward_std = reward_min = reward_max = None
            rn_stressor.calibrate(reward_std=reward_std,
                                  reward_min=reward_min, reward_max=reward_max)
            rn_stressor.reset_rng()

        df_ref         = pd.DataFrame(per_ref)
        baseline_stats = _compute_baseline_stats(df_ref)

        comp_map = {k: df_ref[k].to_numpy(dtype=float)
                    for k in ["competence", "coherence", "continuity", "integrity", "meaning"]
                    if k in df_ref.columns}
        w = identity_weights_from_baseline_components(comp_map)

        fpr_target     = float(getattr(args, "fpr_target", 0.05))
        fpr_percentile = float(np.clip((1.0 - fpr_target) * 100.0, 50.0, 99.0))

        baseline_stats.setdefault("identity", {})
        baseline_stats["identity"]["weights"] = w.to_dict()
        baseline_stats["meta"] = {
            "env":               str(args.env),
            "algo":              str(args.algo),
            "seed":              int(seed),
            "eval_mode":         str(eval_mode),
            "ref_episodes":      int(ref_eps),
            "arcus_h_version":   "1.4",
            "is_atari":          bool(_is_ale_env(args.env)),
            "meaning_proxy":     True,
            "obs_normalize":     obs_normalize,
            "fpr_target":        fpr_target,
            "concept_drift_obs_std":      (float(cd_stressor._obs_std)
                                           if cd_stressor is not None
                                           and cd_stressor._obs_std is not None else None),
            "concept_drift_scale_eff":    (float(cd_stressor._drift_scale_eff)
                                           if cd_stressor is not None
                                           and cd_stressor._drift_scale_eff is not None else None),
            "concept_drift_max_eff":      (float(cd_stressor._drift_max_eff)
                                           if cd_stressor is not None
                                           and cd_stressor._drift_max_eff is not None else None),
            "observation_noise_std":      (float(on_stressor.effective_noise_std)
                                           if on_stressor is not None
                                           and on_stressor.effective_noise_std is not None else None),
            "observation_noise_is_image": (bool(on_stressor.is_image)
                                           if on_stressor is not None else None),
            "sensor_blackout_prob":       (float(sb_stressor.effective_blackout_prob)
                                           if sb_stressor is not None else None),
            "sensor_blackout_fraction":   (float(sb_stressor.expected_blackout_fraction)
                                           if sb_stressor is not None else None),
            "reward_noise_std":           (float(rn_stressor.effective_noise_std)
                                           if rn_stressor is not None
                                           and rn_stressor.effective_noise_std is not None else None),
        }
        baseline_stats_written.append(baseline_stats)

        if fixed_threshold is not None:
            event_threshold = float(fixed_threshold)
        else:
            raw_p95 = float(baseline_stats.get("collapse", {}).get("score_p95", 0.65))
            if fpr_target != 0.05:
                fpr_05_pct = 95.0
                scale = fpr_percentile / fpr_05_pct
                event_threshold = float(np.clip(raw_p95 * scale, 0.5, 0.99))
            else:
                event_threshold = raw_p95

        c_cfg = CollapseScoringConfig(
            event_threshold=event_threshold,
            sharpness=float(args.collapse_sharpness),
        )
        active_baseline_stats = baseline_stats

        for i_sched, schedule in enumerate(schedules):
            _key = (int(seed), str(eval_mode), str(schedule))
            if _key in _done_keys:
                print(f"    SKIP seed={seed} mode={eval_mode} schedule={schedule}")
                continue

            if schedule in ("baseline", "none"):
                mode    = "baseline"
                pattern = _thirds_pattern(int(args.episodes))
            else:
                mode    = schedule
                pattern = _default_pattern(mode, int(args.episodes))

            if schedule == "concept_drift"    and cd_stressor is not None:
                cd_stressor.reset_drift()
            if schedule == "observation_noise" and on_stressor is not None:
                on_stressor.reset_rng()
            if schedule == "sensor_blackout"   and sb_stressor is not None:
                sb_stressor.reset_rng()
            if schedule == "reward_noise"      and rn_stressor is not None:
                rn_stressor.reset_rng()

            env = _make_stress_env(
                args.env, schedule, pattern, seed=int(seed),
                concept_drift_stressor=(cd_stressor    if schedule == "concept_drift"    else None),
                observation_noise_stressor=(on_stressor if schedule == "observation_noise" else None),
                sensor_blackout_stressor=(sb_stressor   if schedule == "sensor_blackout"   else None),
                reward_noise_stressor=(rn_stressor       if schedule == "reward_noise"       else None),
                obs_normalize=obs_normalize,
            )
            tracker = IdentityTracker()
            tracker.weights = w
            mp_sched = MeaningProxyTracker()

            per_ep: List[Dict[str, Any]] = []
            try:
                for ep_idx in range(int(args.episodes)):
                    rec = _episode_rollout(
                        env, model, deterministic, tracker,
                        meaning_proxy=mp_sched,
                    )
                    rec["episode_idx"] = int(ep_idx)
                    per_ep.append(rec)
            finally:
                env.close()

            print(f"    [{i_sched+1}/{len(schedules)}] seed={seed} mode={eval_mode} "
                  f"schedule={schedule} | {len(per_ep)} eps | "
                  f"rew={float(np.nanmean([r['episode_return'] for r in per_ep])):.1f}",
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

            active_baseline_stats = baseline_stats
            if schedule in ("baseline", "none"):
                m_pre  = df_ep["stress_phase"].astype(str) == "pre"
                df_pre = df_ep.loc[m_pre]
                if len(df_pre) >= 10:
                    recomp = _compute_baseline_stats(df_pre)
                    recomp.setdefault("identity", {})
                    recomp["identity"]["weights"] = (
                        baseline_stats.get("identity", {}).get("weights", {}))
                    recomp["meta"]     = baseline_stats.get("meta", {})
                    active_baseline_stats = recomp
                    if fixed_threshold is None:
                        new_p95 = float(recomp.get("collapse", {}).get("score_p95", event_threshold))
                        c_cfg   = CollapseScoringConfig(
                            event_threshold=new_p95,
                            sharpness=float(args.collapse_sharpness))

            base_pre = _ff(
                active_baseline_stats.get("collapse", {}).get(
                    "base_id", identity_pre if np.isfinite(identity_pre) else 0.5),
                0.5)

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
                lambda s: int(collapse_event(float(s), c_cfg)))

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
                    fn = (f"per_episode__{_safe_name(args.env)}__{_safe_name(args.algo)}__"
                          f"{_safe_name(eval_mode)}__{_safe_name(schedule)}__seed{seed}.csv")
                    df_ep.to_csv(out_dir / fn, index=False)

            m_shock       = df_ep["stress_phase"].astype(str) == "shock"
            shock_returns = df_ep.loc[m_shock, "episode_return"].to_numpy(dtype=float) \
                            if m_shock.any() else np.array([], dtype=float)
            cvar_shock_05  = _cvar(shock_returns, alpha=0.05)
            cvar_shock_25  = _cvar(shock_returns, alpha=0.25)

            ch_drop = {}
            for ch in ("competence", "coherence", "continuity", "integrity", "meaning"):
                pre_val   = _phase_mean(ch, "pre")
                shock_val = _phase_mean(ch, "shock")
                ch_drop[f"{ch}_pre"]   = pre_val
                ch_drop[f"{ch}_shock"] = shock_val
                ch_drop[f"{ch}_drop"]  = _drop(pre_val, shock_val)

            ch_mads = active_baseline_stats.get("channel_mads", {})

            rows.append({
                "env":       args.env,
                "algo":      args.algo,
                "seed":      int(seed),
                "eval_mode": eval_mode,
                "schedule":  schedule,
                "is_reward_stressor": int(schedule in REWARD_STRESSOR_SET),
                "episodes":  int(args.episodes),
                "reward_mean":  float(np.nanmean(df_ep["episode_return"].to_numpy(dtype=float))),
                "reward_std":   float(np.nanstd( df_ep["episode_return"].to_numpy(dtype=float))),
                "cvar_shock_05":   float(cvar_shock_05),
                "cvar_shock_25":   float(cvar_shock_25),
                "identity_mean": float(np.nanmean(df_ep["identity"].to_numpy(dtype=float))),
                "identity_std":  float(np.nanstd( df_ep["identity"].to_numpy(dtype=float))),
                "id_drop_pre_to_shock":        float(id_drop_mean),
                "integrity_drop_pre_to_shock": float(integrity_drop_mean),
                "meaning_drop_pre_to_shock":   float(meaning_drop_mean),
                **ch_drop,
                "mad_competence":  float(ch_mads.get("competence",  float("nan"))),
                "mad_coherence":   float(ch_mads.get("coherence",   float("nan"))),
                "mad_continuity":  float(ch_mads.get("continuity",  float("nan"))),
                "mad_integrity":   float(ch_mads.get("integrity",   float("nan"))),
                "mad_meaning":     float(ch_mads.get("meaning",     float("nan"))),
                "collapse_score_shock_mean":   float(shock_score_mean),
                "collapse_event_shock":        int(collapse_event(float(shock_score_mean), c_cfg)),
                "collapse_event_threshold":    float(event_threshold),
                "collapse_rate_mean":  float(np.mean(df_ep["collapse_event_episode"].to_numpy(dtype=float))),
                "collapse_rate_pre":   float(_rate_phase("pre")),
                "collapse_rate_shock": float(_rate_phase("shock")),
                "collapse_rate_post":  float(_rate_phase("post")),
                "fpr_actual":   float(_rate_phase("pre")),
                "fpr_target":   float(fpr_target),
                "obs_normalize": int(obs_normalize),
            })

    finally:
        _del_model(model)

    return rows, per_episode_all, baseline_stats_written

def main():
    ap = argparse.ArgumentParser(description="ARCUS-H evaluation harness")
    ap.add_argument("--run_dir",  required=True)
    ap.add_argument("--env",      required=True)
    ap.add_argument("--algo",     required=True)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--seeds",    default="0",
                    help="e.g. '0', '0,1,2', '0-9'. Pass all seeds to one process.")
    ap.add_argument("--eval_mode", choices=["deterministic", "stochastic"],
                    default="deterministic")
    ap.add_argument("--both",  action="store_true",
                    help="Evaluate both deterministic and stochastic modes.")
    ap.add_argument("--schedules", default=None,
                    help=f"Comma-separated schedules. Default: all {ALL_SCHEDULES}")
    ap.add_argument("--save_per_episode",           action="store_true")
    ap.add_argument("--no_save_per_episode",        action="store_true")
    ap.add_argument("--per_episode_separate_files", action="store_true")
    ap.add_argument("--collapse_event_threshold",   type=float, default=None)
    ap.add_argument("--collapse_sharpness",         type=float, default=2.5)
    ap.add_argument("--eval_subdir", default=None,
                    help="Write to <run_dir>/eval/<eval_subdir>/ instead of <run_dir>/eval/")
    ap.add_argument("--resume",  action="store_true",
                    help="Skip already-completed triples; auto-enabled on --schedules subset.")
    ap.add_argument("--fpr_target",    type=float, default=0.05,
                    help="Target false-positive rate for adaptive threshold. Default 0.05.")
    ap.add_argument("--obs_normalize", action="store_true",
                    help="Wrap base env in running mean-std normaliser before stressors. "
                         "Use for Atari runs to remove wrapper asymmetry.")

    args = ap.parse_args()
    save_per_episode = bool(args.save_per_episode or (not args.no_save_per_episode))

    run_root, explicit_zip = _resolve_run_dir(Path(args.run_dir))
    out_dir = (run_root / "eval" / args.eval_subdir
               if args.eval_subdir else run_root / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds      = _parse_seeds(args.seeds)
    eval_modes = ["deterministic", "stochastic"] if args.both else [args.eval_mode]
    schedules  = ([s.strip() for s in args.schedules.split(",") if s.strip()]
                  if args.schedules else ALL_SCHEDULES)

    fixed_threshold: Optional[float] = args.collapse_event_threshold
    out_csv     = out_dir / "eval_results.csv"
    per_ep_path = out_dir / "per_episode.csv"

    _done_keys: set = set()
    all_rows:        List[Dict[str, Any]] = []
    all_per_episode: List[Dict[str, Any]] = []
    all_bs_stats:    List[Dict[str, Any]] = []
    bs_path = out_dir / "baseline_stats.json"
    
    if out_csv.exists():
        try:
            _existing = pd.read_csv(out_csv)
            all_rows  = _existing.to_dict(orient="records")
            for _, r in _existing.iterrows():
                _done_keys.add((int(r.get("seed", -1)),
                                str(r.get("eval_mode", "")),
                                str(r.get("schedule",  ""))))
            print(f"[RESUME] Loaded {len(all_rows)} rows from {out_csv.name} | "
                  f"{len(_done_keys)} (seed, mode, schedule) triples will be skipped")
        except Exception as e:
            print(f"[RESUME] Warning: could not read {out_csv.name} ({e}) — starting fresh")
            all_rows = []; _done_keys = set()

    if save_per_episode and per_ep_path.exists():
        try:
            all_per_episode = pd.read_csv(per_ep_path).to_dict(orient="records")
            print(f"[RESUME] Loaded {len(all_per_episode)} per-episode rows")
        except Exception:
            all_per_episode = []

    if bs_path.exists():
        try:
            _existing_bs = json.loads(bs_path.read_text(encoding="utf-8"))
            if isinstance(_existing_bs, list):
                all_bs_stats = _existing_bs
            elif isinstance(_existing_bs, dict):
                all_bs_stats = [_existing_bs]
            print(f"[RESUME] Loaded {len(all_bs_stats)} existing baseline_stats entries")
        except Exception as e:
            print(f"[RESUME] Warning: could not read baseline_stats.json ({e})")

    n_total = len(seeds) * len(eval_modes) * len(schedules)
    n_skip  = len(_done_keys)
    print(f"[INFO] {len(seeds)} seeds × {len(eval_modes)} modes × "
          f"{len(schedules)} schedules = {n_total} runs"
          + (f"  ({n_total-n_skip} remaining, {n_skip} skipped)" if n_skip else ""))
    print(f"[INFO] Seeds run SERIALLY — do NOT use & parallelism")

    for seed in seeds:
        for eval_mode in eval_modes:
            print(f"\n[SEED {seed} | {eval_mode}]", flush=True)
            rows, per_ep, bs = _eval_seed_mode(
                seed=seed, eval_mode=eval_mode, schedules=schedules,
                args=args, run_root=run_root, explicit_zip=explicit_zip,
                out_dir=out_dir, fixed_threshold=fixed_threshold,
                _done_keys=_done_keys, save_per_episode=save_per_episode,
            )
            all_rows.extend(rows)
            all_per_episode.extend(per_ep)
            all_bs_stats.extend(bs)

            for r in rows:
                _done_keys.add((int(r["seed"]), str(r["eval_mode"]), str(r["schedule"])))
            if out_csv.exists() and _done_keys:
                shutil.copy2(out_csv, out_csv.with_suffix(".csv.bak"))
            pd.DataFrame(all_rows).to_csv(out_csv, index=False)
            if save_per_episode and all_per_episode:
                pd.DataFrame(all_per_episode).to_csv(per_ep_path, index=False)
            if all_bs_stats:
                bs_path.write_text(json.dumps(all_bs_stats, indent=2), encoding="utf-8")

            _free_memory()
            print(f"  [MEM] GC done after seed={seed} mode={eval_mode}", flush=True)

    if out_csv.exists():
        shutil.copy2(out_csv, out_csv.with_suffix(".csv.bak"))
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    bs_path.write_text(json.dumps(all_bs_stats, indent=2), encoding="utf-8")

    if save_per_episode:
        pd.DataFrame(all_per_episode).to_csv(per_ep_path, index=False)
        print(f"\n[OK] {out_csv}  rows={len(all_rows)}")
        print(f"[OK] {per_ep_path}  rows={len(all_per_episode)}")
    else:
        print(f"\n[OK] {out_csv}  rows={len(all_rows)}")
    print(f"[OK] {bs_path}")


if __name__ == "__main__":
    main()
