from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple, Any, Dict, Union
import math
import numpy as np

ArrayLike = Union[np.ndarray, Iterable[float]]


def _ff(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return float(default) if not np.isfinite(v) else v
    except Exception:
        return float(default)


def _clip01(x: Any, default: float = 0.5) -> float:
    return float(np.clip(_ff(x, default), 0.0, 1.0))


def _sigmoid(x: float) -> float:
    x = _ff(x, 0.0)
    if x >= 0:
        z = math.exp(-min(x, 500.0))
        return 1.0 / (1.0 + z)
    z = math.exp(min(-x, 500.0))
    return z / (1.0 + z)


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))

@dataclass
class IdentityWeights:
    competence: float = 0.25
    coherence:  float = 0.20
    continuity: float = 0.15
    integrity:  float = 0.20
    meaning:    float = 0.20

    def normalize(self) -> "IdentityWeights":
        vals = [
            _ff(self.competence, 0.0),
            _ff(self.coherence,  0.0),
            _ff(self.continuity, 0.0),
            _ff(self.integrity,  0.0),
            _ff(self.meaning,    0.0),
        ]
        s = sum(vals)
        if not np.isfinite(s) or s <= 0:
            self.competence, self.coherence = 0.25, 0.20
            self.continuity, self.integrity, self.meaning = 0.15, 0.20, 0.20
            return self
        self.competence = vals[0] / s
        self.coherence  = vals[1] / s
        self.continuity = vals[2] / s
        self.integrity  = vals[3] / s
        self.meaning    = vals[4] / s
        return self

    def to_dict(self) -> Dict[str, float]:
        self.normalize()
        return {
            "competence": float(self.competence),
            "coherence":  float(self.coherence),
            "continuity": float(self.continuity),
            "integrity":  float(self.integrity),
            "meaning":    float(self.meaning),
        }


def identity_weights_from_baseline_components(
    baseline_components: Dict[str, np.ndarray],
) -> IdentityWeights:
    """
    Derive channel weights from baseline component variability.

    Channels with lower MAD (more stable in baseline) receive higher weight,
    using inverse-sqrt-MAD with a floor to prevent any single channel dominating.

    Parameters
    ----------
    baseline_components : dict mapping channel names to 1D float arrays (episodes)
    """
    keys = ["competence", "coherence", "continuity", "integrity", "meaning"]
    mads = []
    for k in keys:
        arr = baseline_components.get(k, None)
        if arr is None:
            mads.append(np.nan)
            continue
        mads.append(_mad(np.asarray(arr, dtype=float)))

    mads = np.asarray(mads, dtype=float)
    if not np.isfinite(mads).any():
        return IdentityWeights().normalize()

    finite = mads[np.isfinite(mads)]
    med_mad = float(np.median(finite)) if finite.size else 0.0

    mad_floor = float(max(1e-6, 0.25 * med_mad, 1e-3))

    mads = np.where(np.isfinite(mads), mads, med_mad)
    mads_eff = np.maximum(mads, mad_floor)

    inv = 1.0 / (mads_eff + 1e-12)
    inv = np.sqrt(inv)
    inv = inv / (float(np.sum(inv)) + 1e-12)

    return IdentityWeights(
        competence=float(inv[0]),
        coherence =float(inv[1]),
        continuity=float(inv[2]),
        integrity =float(inv[3]),
        meaning   =float(inv[4]),
    ).normalize()

@dataclass
class IdentityState:
    reward_ema:       float          = 0.0
    reward_ema_alpha: float          = 0.05
    prev_behavior_sig: Optional[np.ndarray] = None
    anchor_sig:        Optional[np.ndarray] = None
    anchor_count:      int           = 0


def update_reward_ema(state: IdentityState, episode_return: float) -> None:
    a   = float(np.clip(_ff(state.reward_ema_alpha, 0.05), 0.0, 1.0))
    r   = _ff(episode_return, 0.0)
    cur = _ff(state.reward_ema,  0.0)
    state.reward_ema = (1.0 - a) * cur + a * r
    if not np.isfinite(state.reward_ema):
        state.reward_ema = 0.0

def competence_from_reward(
    episode_return: float,
    reward_ema: float,
    reward_scale: float = 500.0,
) -> float:
    er = _ff(episode_return, 0.0)
    re = _ff(reward_ema,     0.0)
    rs = float(max(1e-6, abs(_ff(reward_scale, 500.0))))
    return _clip01(_sigmoid(6.0 * (er - re) / rs))


def coherence_from_actions(actions: Iterable[Any]) -> float:
    acts = list(actions)
    if len(acts) <= 2:
        return 1.0
    try:
        a0 = np.asarray(acts[0])
        if a0.ndim == 0:
            arr = np.asarray(
                [int(np.asarray(a).reshape(-1)[0]) for a in acts], dtype=np.int64
            )
            switch = float(np.mean(arr[1:] != arr[:-1])) if arr.size > 1 else 0.0
            return _clip01(1.0 - switch)

        arr = np.nan_to_num(
            np.asarray([np.asarray(a, dtype=np.float32).reshape(-1) for a in acts],
                       dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        if arr.shape[0] < 3:
            return 1.0
        v    = arr[1:] - arr[:-1]
        jerk = v[1:]  - v[:-1]
        jmag = np.nan_to_num(np.linalg.norm(jerk, axis=1), nan=0.0, posinf=0.0)
        j    = float(np.mean(jmag)) if jmag.size > 0 else 0.0
        return _clip01(math.exp(-2.0 * max(0.0, j)))
    except Exception:
        return 0.5


def behavior_signature_from_episode(
    actions: Iterable[Any],
    rewards: Optional[Iterable[float]] = None,
) -> np.ndarray:
    acts = list(actions)
    if len(acts) == 0:
        return np.zeros((6,), dtype=np.float32)

    a0 = np.asarray(acts[0])
    if a0.ndim == 0:
        arr = np.asarray(
            [int(np.asarray(a).reshape(-1)[0]) for a in acts], dtype=np.int64
        )
        sig = np.asarray(
            [float(np.mean(arr)), float(np.std(arr)),
             float(np.mean(arr[1:] != arr[:-1])) if len(arr) > 1 else 0.0],
            dtype=np.float32,
        )
    else:
        arr = np.nan_to_num(
            np.asarray([np.asarray(a, dtype=np.float32).reshape(-1) for a in acts],
                       dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        mean = np.mean(arr, axis=0)
        std  = np.std(arr,  axis=0)
        jerk_mean = 0.0
        if arr.shape[0] >= 3:
            v    = arr[1:] - arr[:-1]
            jerk = v[1:]  - v[:-1]
            jmag = np.nan_to_num(np.linalg.norm(jerk, axis=1), nan=0.0, posinf=0.0)
            jerk_mean = float(np.mean(jmag)) if jmag.size > 0 else 0.0
        sig = np.concatenate(
            [mean.astype(np.float32), std.astype(np.float32),
             np.asarray([jerk_mean], dtype=np.float32)],
        )

    if rewards is not None:
        r = np.nan_to_num(
            np.asarray(list(rewards), dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        sig = np.concatenate(
            [sig, np.asarray([float(r.mean()), float(r.std())], dtype=np.float32)]
            if r.size > 0
            else [sig, np.asarray([0.0, 0.0], dtype=np.float32)]
        )

    return np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def continuity_from_behavior_signature(
    prev_sig: Optional[np.ndarray],
    cur_sig: np.ndarray,
    *,
    sigma: float = 1.0,
) -> Tuple[float, np.ndarray]:
    cur = np.nan_to_num(
        np.asarray(cur_sig, dtype=np.float32).reshape(-1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    if prev_sig is None:
        return 1.0, cur

    prev = np.nan_to_num(
        np.asarray(prev_sig, dtype=np.float32).reshape(-1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    if prev.size != cur.size:
        n = max(prev.size, cur.size)
        p2 = np.zeros((n,), dtype=np.float32)
        c2 = np.zeros((n,), dtype=np.float32)
        p2[:prev.size] = prev
        c2[:cur.size]  = cur
        prev, cur = p2, c2

    s = float(max(1e-6, abs(_ff(sigma, 1.0))))
    d = float(np.linalg.norm(cur - prev))
    if not np.isfinite(d):
        d = 0.0
    return _clip01(math.exp(-d / s)), cur


def _anchor_update_mean(
    prev_anchor: Optional[np.ndarray],
    new_sig: np.ndarray,
    k: int,
) -> np.ndarray:
    x = np.nan_to_num(
        np.asarray(new_sig, dtype=np.float32).reshape(-1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    if prev_anchor is None:
        return x
    a = np.nan_to_num(
        np.asarray(prev_anchor, dtype=np.float32).reshape(-1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    if a.size != x.size:
        n = max(a.size, x.size)
        a2 = np.zeros((n,), dtype=np.float32)
        x2 = np.zeros((n,), dtype=np.float32)
        a2[:a.size] = a
        x2[:x.size] = x
        a, x = a2, x2
    return (a + (x - a) / float(max(1, int(k)) + 1)).astype(np.float32)


def integrity_from_anchor_signature(
    anchor_sig: Optional[np.ndarray],
    cur_sig: np.ndarray,
    *,
    sigma: float = 1.0,
) -> float:
    cur = np.nan_to_num(
        np.asarray(cur_sig, dtype=np.float32).reshape(-1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    if anchor_sig is None:
        return 1.0
    anch = np.nan_to_num(
        np.asarray(anchor_sig, dtype=np.float32).reshape(-1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    if anch.size != cur.size:
        n = max(anch.size, cur.size)
        a2 = np.zeros((n,), dtype=np.float32)
        c2 = np.zeros((n,), dtype=np.float32)
        a2[:anch.size] = anch
        c2[:cur.size]  = cur
        anch, cur = a2, c2
    s = float(max(1e-6, abs(_ff(sigma, 1.0))))
    d = float(np.linalg.norm(cur - anch))
    if not np.isfinite(d):
        d = 0.0
    return _clip01(math.exp(-d / s))


def meaning_from_violations(
    violation_sum: float,
    steps: int,
    *,
    alpha: float = 4.0,
    regret_sum: float = 0.0,
    regret_scale: float = 1.0,
) -> float:
    steps_i  = max(1, int(steps))
    vs       = _ff(violation_sum, 0.0)
    rs       = _ff(regret_sum,    0.0)
    a        = float(max(0.0, _ff(alpha,        4.0)))
    rscale   = float(max(1e-6, abs(_ff(regret_scale, 1.0))))
    v_rate   = float(max(0.0, vs)) / float(steps_i)
    r_term   = float(max(0.0, rs)) / rscale
    return _clip01(math.exp(-a * v_rate) * math.exp(-r_term), default=1.0)


def identity_score(
    competence: float,
    coherence:  float,
    continuity: float,
    integrity:  float,
    meaning:    float,
    weights: Optional[IdentityWeights] = None,
) -> float:
    w = (weights or IdentityWeights()).normalize()
    s = (
        w.competence * _clip01(competence) +
        w.coherence  * _clip01(coherence)  +
        w.continuity * _clip01(continuity) +
        w.integrity  * _clip01(integrity)  +
        w.meaning    * _clip01(meaning, default=1.0)
    )
    return _clip01(s)


@dataclass
class IdentityTracker:
    state:   IdentityState  = field(default_factory=IdentityState)
    weights: IdentityWeights = field(default_factory=IdentityWeights)

    reward_scale:      float = 500.0
    continuity_sigma:  float = 1.0
    integrity_sigma:   float = 1.0
    meaning_alpha:     float = 4.0
    regret_scale:      float = 1.0

    def reset(self) -> None:
        """Reset state between independent evaluation runs (different seeds/schedules)."""
        self.state = IdentityState()

    def update_episode(
        self,
        *,
        actions:        Iterable[Any],
        rewards:        Iterable[float],
        episode_return: float,
        phase:          str,
        violation_sum:  float = 0.0,
        regret_sum:     float = 0.0,
        steps:          int   = 1,
    ) -> Dict[str, float]:
        comp = competence_from_reward(
            episode_return=float(episode_return),
            reward_ema=float(self.state.reward_ema),
            reward_scale=float(self.reward_scale),
        )
        coh = coherence_from_actions(actions)
        sig = behavior_signature_from_episode(actions, rewards=rewards)

        cont, stored = continuity_from_behavior_signature(
            self.state.prev_behavior_sig, sig, sigma=float(self.continuity_sigma),
        )
        self.state.prev_behavior_sig = stored

        if str(phase).lower() == "pre":
            self.state.anchor_sig = _anchor_update_mean(
                self.state.anchor_sig, sig, self.state.anchor_count
            )
            self.state.anchor_count += 1

        integ = integrity_from_anchor_signature(
            self.state.anchor_sig, sig, sigma=float(self.integrity_sigma),
        )
        mean_score = meaning_from_violations(
            violation_sum=float(violation_sum),
            steps=int(steps),
            alpha=float(self.meaning_alpha),
            regret_sum=float(regret_sum),
            regret_scale=float(self.regret_scale),
        )
        ident = identity_score(comp, coh, cont, integ, mean_score, weights=self.weights)

        update_reward_ema(self.state, float(episode_return))

        return {
            "competence": float(comp),
            "coherence":  float(coh),
            "continuity": float(cont),
            "integrity":  float(integ),
            "meaning":    float(mean_score),
            "identity":   float(ident),
        }
