# arcus/harness_rl/shocks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    import gym


@dataclass
class Segment:
    tag: str
    length: int


def parse_schedule_spec(spec: str, mode: str) -> List[Segment]:
    """
    Example: "none:20,{mode}:40,none:60" with mode="invert"
    """
    spec = spec.replace("{mode}", mode).strip()
    if not spec:
        return [Segment("none", 0)]
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    segs: List[Segment] = []
    for p in parts:
        if ":" not in p:
            raise ValueError(f"Bad schedule segment '{p}', expected 'tag:len'")
        tag, ln = p.split(":", 1)
        tag = tag.strip()
        ln_i = int(ln.strip())
        segs.append(Segment(tag, ln_i))
    return segs


class ShockScheduler:
    """
    Drives 'in_shock' based on episode step index.
    """
    def __init__(self, schedule: List[Segment]):
        self.schedule = schedule
        self.total = sum(s.length for s in schedule)

    def tag_at(self, t: int) -> str:
        if self.total <= 0:
            return "none"
        t = int(t)
        acc = 0
        for s in self.schedule:
            if acc <= t < acc + s.length:
                return s.tag
            acc += s.length
        return self.schedule[-1].tag if self.schedule else "none"

    def is_shock(self, t: int) -> bool:
        return self.tag_at(t) != "none"


class ShockWrapper(gym.Wrapper):
    """
    Applies shocks to observations and/or dynamics.
    Reward shock is optional but not the only effect.
    """

    def __init__(
        self,
        env: gym.Env,
        mode: str,
        schedule_spec: str,
        seed: int,
        obs_sigma: float = 0.05,
        betray_p: float = 0.35,
        scarcity_action_scale: float = 0.5,
        reward_scale_in_shock: float = 1.0,   # keep 1.0 by default to avoid "reward-only" framing
        reward_bias_in_shock: float = 0.0,
    ):
        super().__init__(env)
        self.mode = mode
        self.schedule_spec = schedule_spec
        self._rng = np.random.default_rng(seed)

        self._sched = ShockScheduler(parse_schedule_spec(schedule_spec, mode))
        self._t = 0

        self.obs_sigma = float(obs_sigma)
        self.betray_p = float(betray_p)
        self.scarcity_action_scale = float(scarcity_action_scale)
        self.reward_scale_in_shock = float(reward_scale_in_shock)
        self.reward_bias_in_shock = float(reward_bias_in_shock)

    @property
    def in_shock(self) -> bool:
        return self._sched.is_shock(self._t)

    @property
    def shock_tag(self) -> str:
        return self._sched.tag_at(self._t)

    def reset(self, **kwargs):
        self._t = 0
        obs, info = self.env.reset(**kwargs)
        obs = self._shock_obs(obs, force_tag="none")  # no shock on reset
        return obs, info

    def step(self, action):
        # Dynamics shock = modify action before stepping
        a = self._shock_action(action)

        obs, reward, terminated, truncated, info = self.env.step(a)

        # Observation shock = perturb/transform observation
        obs = self._shock_obs(obs)

        # Reward shock = optional (default neutral)
        reward = self._shock_reward(reward)

        self._t += 1

        info = dict(info)
        info["arcus_shock_tag"] = self.shock_tag
        info["arcus_in_shock"] = self.in_shock
        return obs, reward, terminated, truncated, info

    def _shock_action(self, action):
        if not self.in_shock:
            return action

        # Convert to np array for safe ops, then back to original type
        a = np.array(action, dtype=np.float32)

        if self.mode == "invert":
            a = -a  # flip policy effect on dynamics
        elif self.mode == "betrayal":
            # with some probability, "betray" by flipping sign or injecting noise
            if self._rng.random() < self.betray_p:
                if self._rng.random() < 0.5:
                    a = -a
                else:
                    a = a + self._rng.normal(0.0, 0.25, size=a.shape).astype(np.float32)
        elif self.mode == "scarcity":
            # damp actions (like reduced actuator authority)
            a = a * self.scarcity_action_scale
            # also occasional dropout
            if self._rng.random() < 0.15:
                a = a * 0.0

        # clip if action space has bounds
        if hasattr(self.env, "action_space") and isinstance(self.env.action_space, gym.spaces.Box):
            lo = self.env.action_space.low
            hi = self.env.action_space.high
            a = np.clip(a, lo, hi)

        return a

    def _shock_obs(self, obs, force_tag: Optional[str] = None):
        tag = force_tag if force_tag is not None else self.shock_tag
        if tag == "none":
            return obs

        x = np.array(obs, dtype=np.float32)

        if self.mode == "invert":
            x = -x
        elif self.mode == "betrayal":
            # noisy + occasional sign flip on some dims
            x = x + self._rng.normal(0.0, self.obs_sigma, size=x.shape).astype(np.float32)
            if self._rng.random() < 0.25:
                mask = self._rng.random(size=x.shape) < 0.5
                x = np.where(mask, -x, x)
        elif self.mode == "scarcity":
            # quantize / coarsen + noise
            step = 0.05
            x = np.round(x / step) * step
            x = x + self._rng.normal(0.0, self.obs_sigma * 0.75, size=x.shape).astype(np.float32)

        return x

    def _shock_reward(self, reward):
        if not self.in_shock:
            return reward
        return (reward * self.reward_scale_in_shock) + self.reward_bias_in_shock


def l2_mean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.linalg.norm(a, axis=-1))) if a.ndim > 1 else float(np.mean(np.abs(a)))


def trajectory_divergence(
    base_actions: List[np.ndarray],
    test_actions: List[np.ndarray],
    base_obs: List[np.ndarray],
    test_obs: List[np.ndarray],
) -> Dict[str, float]:
    """
    Simple, stable "behavioral divergence" signals:
      - mean L2 action difference
      - action cosine similarity (mean)
      - mean L2 obs difference
    """
    n = min(len(base_actions), len(test_actions), len(base_obs), len(test_obs))
    if n <= 0:
        return {
            "action_l2_mean": float("nan"),
            "action_cos_mean": float("nan"),
            "obs_l2_mean": float("nan"),
        }

    A = np.stack([np.asarray(x, dtype=np.float32).ravel() for x in base_actions[:n]], axis=0)
    B = np.stack([np.asarray(x, dtype=np.float32).ravel() for x in test_actions[:n]], axis=0)
    Oa = np.stack([np.asarray(x, dtype=np.float32).ravel() for x in base_obs[:n]], axis=0)
    Ob = np.stack([np.asarray(x, dtype=np.float32).ravel() for x in test_obs[:n]], axis=0)

    diff_a = np.linalg.norm(A - B, axis=1)
    diff_o = np.linalg.norm(Oa - Ob, axis=1)

    # cosine similarity for actions
    denom = (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)) + 1e-8
    cos = np.sum(A * B, axis=1) / denom

    return {
        "action_l2_mean": float(np.mean(diff_a)),
        "action_cos_mean": float(np.mean(cos)),
        "obs_l2_mean": float(np.mean(diff_o)),
    }
