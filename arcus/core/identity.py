from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional, Tuple, Any, Dict, Union
import json
import math
import numpy as np


ArrayLike = Union[np.ndarray, Iterable[float]]

def _finite_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _finite01(x: Any, default: float = 0.5) -> float:
    v = _finite_float(x, default=default)
    return float(np.clip(v, 0.0, 1.0))


def _sigmoid(x: float) -> float:
    x = _finite_float(x, default=0.0)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

@dataclass
class IdentityWeights:
    competence: float = 0.25
    coherence: float = 0.20
    continuity: float = 0.15
    integrity: float = 0.20
    meaning: float = 0.20

    def normalize(self) -> "IdentityWeights":
        self.competence = _finite_float(self.competence, default=0.0)
        self.coherence = _finite_float(self.coherence, default=0.0)
        self.continuity = _finite_float(self.continuity, default=0.0)
        self.integrity = _finite_float(self.integrity, default=0.0)
        self.meaning = _finite_float(self.meaning, default=0.0)

        s = float(self.competence + self.coherence + self.continuity + self.integrity + self.meaning)
        if not np.isfinite(s) or s <= 0:
            self.competence, self.coherence, self.continuity, self.integrity, self.meaning = 0.25, 0.20, 0.15, 0.20, 0.20
            return self

        self.competence /= s
        self.coherence /= s
        self.continuity /= s
        self.integrity /= s
        self.meaning /= s
        return self


@dataclass
class IdentityState:
    reward_ema: float = 0.0
    reward_ema_alpha: float = 0.05
    prev_behavior_sig: Optional[np.ndarray] = None
    anchor_sig: Optional[np.ndarray] = None
    anchor_count: int = 0


def update_reward_ema(state: IdentityState, episode_return: float) -> None:
    a = _finite_float(state.reward_ema_alpha, default=0.05)
    a = float(np.clip(a, 0.0, 1.0))
    r = _finite_float(episode_return, default=0.0)
    cur = _finite_float(state.reward_ema, default=0.0)

    state.reward_ema = (1.0 - a) * cur + a * r
    if not np.isfinite(state.reward_ema):
        state.reward_ema = 0.0


def competence_from_reward(
    episode_return: float,
    reward_ema: float,
    reward_scale: float = 500.0,
) -> float:
    er = _finite_float(episode_return, default=0.0)
    re = _finite_float(reward_ema, default=0.0)
    rs = _finite_float(reward_scale, default=500.0)
    rs = float(max(1e-6, abs(rs)))

    x = (er - re) / rs
    return _finite01(_sigmoid(6.0 * x), default=0.5)


def coherence_from_actions(actions: Iterable[Any]) -> float:
    acts = list(actions)
    if len(acts) <= 2:
        return 1.0

    try:
        a0 = np.asarray(acts[0])
        if a0.ndim == 0:
            arr = np.asarray([int(np.asarray(a).reshape(-1)[0]) for a in acts], dtype=np.int64)
            switches = float(np.mean(arr[1:] != arr[:-1])) if arr.size > 1 else 0.0
            return _finite01(1.0 - switches, default=0.5)

        arr = np.asarray([np.asarray(a, dtype=np.float32).reshape(-1) for a in acts], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if arr.shape[0] < 3:
            return 1.0

        v = arr[1:] - arr[:-1]
        jerk = v[1:] - v[:-1]
        jerk_mag = np.linalg.norm(jerk, axis=1)
        jerk_mag = np.nan_to_num(jerk_mag, nan=0.0, posinf=0.0, neginf=0.0)

        j = float(np.mean(jerk_mag)) if jerk_mag.size > 0 else 0.0
        val = math.exp(-2.0 * float(max(0.0, j)))
        return _finite01(val, default=0.5)
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
        arr = np.asarray([int(np.asarray(a).reshape(-1)[0]) for a in acts], dtype=np.int64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        switch = float(np.mean(arr[1:] != arr[:-1])) if len(arr) > 1 else 0.0
        sig = np.asarray([mean, std, switch], dtype=np.float32)
    else:
        arr = np.asarray([np.asarray(a, dtype=np.float32).reshape(-1) for a in acts], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)

        if arr.shape[0] >= 3:
            v = arr[1:] - arr[:-1]
            jerk = v[1:] - v[:-1]
            jerk_mag = np.linalg.norm(jerk, axis=1)
            jerk_mag = np.nan_to_num(jerk_mag, nan=0.0, posinf=0.0, neginf=0.0)
            jerk_mean = float(np.mean(jerk_mag)) if jerk_mag.size > 0 else 0.0
        else:
            jerk_mean = 0.0

        sig = np.concatenate(
            [mean.astype(np.float32), std.astype(np.float32), np.asarray([jerk_mean], dtype=np.float32)],
            axis=0,
        )

    if rewards is not None:
        r = np.asarray(list(rewards), dtype=np.float32)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        if r.size > 0:
            sig = np.concatenate([sig, np.asarray([float(r.mean()), float(r.std())], dtype=np.float32)], axis=0)
        else:
            sig = np.concatenate([sig, np.asarray([0.0, 0.0], dtype=np.float32)], axis=0)

    return np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)



def continuity_from_behavior_signature(
    prev_sig: Optional[np.ndarray],
    cur_sig: np.ndarray,
    *,
    sigma: float = 1.0,
) -> Tuple[float, np.ndarray]:
    cur = np.asarray(cur_sig, dtype=np.float32).reshape(-1)
    cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)

    if prev_sig is None:
        return 1.0, cur

    prev = np.asarray(prev_sig, dtype=np.float32).reshape(-1)
    prev = np.nan_to_num(prev, nan=0.0, posinf=0.0, neginf=0.0)

    if prev.size != cur.size:
        n = int(max(prev.size, cur.size))
        p = np.zeros((n,), dtype=np.float32)
        c = np.zeros((n,), dtype=np.float32)
        p[: prev.size] = prev
        c[: cur.size] = cur
        prev, cur = p, c

    s = float(max(1e-6, abs(_finite_float(sigma, default=1.0))))
    d = float(np.linalg.norm(cur - prev))
    if not np.isfinite(d):
        d = 0.0

    cont = math.exp(-d / s)
    return _finite01(cont, default=0.5), cur



def _anchor_update_mean(prev_anchor: Optional[np.ndarray], new_sig: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(new_sig, dtype=np.float32).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if prev_anchor is None:
        return x

    a = np.asarray(prev_anchor, dtype=np.float32).reshape(-1)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    if a.size != x.size:
        n = int(max(a.size, x.size))
        aa = np.zeros((n,), dtype=np.float32)
        xx = np.zeros((n,), dtype=np.float32)
        aa[: a.size] = a
        xx[: x.size] = x
        a, x = aa, xx

    k = max(1, int(k))
    return (a + (x - a) / float(k + 1)).astype(np.float32)


def integrity_from_anchor_signature(
    anchor_sig: Optional[np.ndarray],
    cur_sig: np.ndarray,
    *,
    sigma: float = 1.0,
) -> float:
    cur = np.asarray(cur_sig, dtype=np.float32).reshape(-1)
    cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)

    if anchor_sig is None:
        return 1.0

    anch = np.asarray(anchor_sig, dtype=np.float32).reshape(-1)
    anch = np.nan_to_num(anch, nan=0.0, posinf=0.0, neginf=0.0)

    if anch.size != cur.size:
        n = int(max(anch.size, cur.size))
        a = np.zeros((n,), dtype=np.float32)
        c = np.zeros((n,), dtype=np.float32)
        a[: anch.size] = anch
        c[: cur.size] = cur
        anch, cur = a, c

    s = float(max(1e-6, abs(_finite_float(sigma, default=1.0))))
    d = float(np.linalg.norm(cur - anch))
    if not np.isfinite(d):
        d = 0.0

    val = math.exp(-d / s)
    return _finite01(val, default=0.5)


# ----------------------------
# Meaning / constraint-respect channel
# ----------------------------

def meaning_from_violations(
    violation_sum: float,
    steps: int,
    *,
    alpha: float = 4.0,
    regret_sum: float = 0.0,
    regret_scale: float = 1.0,
) -> float:
    steps_i = max(1, int(steps))
    vs = _finite_float(violation_sum, default=0.0)
    rs = _finite_float(regret_sum, default=0.0)
    a = float(max(0.0, _finite_float(alpha, default=4.0)))
    rscale = float(max(1e-6, abs(_finite_float(regret_scale, default=1.0))))

    violation_rate = float(max(0.0, vs)) / float(steps_i)
    regret_term = float(max(0.0, rs)) / rscale

    v_part = math.exp(-a * violation_rate)
    r_part = math.exp(-regret_term)

    return _finite01(v_part * r_part, default=1.0)



def identity_score(
    competence: float,
    coherence: float,
    continuity: float,
    integrity: float,
    meaning: float,
    weights: Optional[IdentityWeights] = None,
) -> float:
    w = (weights or IdentityWeights()).normalize()

    c = _finite01(competence, default=0.5)
    h = _finite01(coherence, default=0.5)
    t = _finite01(continuity, default=0.5)
    i = _finite01(integrity, default=0.5)
    m = _finite01(meaning, default=1.0)

    s = (
        w.competence * c
        + w.coherence * h
        + w.continuity * t
        + w.integrity * i
        + w.meaning * m
    )
    return _finite01(s, default=0.5)



@dataclass
class IdentityTracker:
    state: IdentityState = field(default_factory=IdentityState)
    weights: IdentityWeights = field(default_factory=IdentityWeights)

    reward_scale: float = 500.0
    continuity_sigma: float = 1.0
    integrity_sigma: float = 1.0

    meaning_alpha: float = 4.0
    regret_scale: float = 1.0

    def update_episode(
        self,
        *,
        actions: Iterable[Any],
        rewards: Iterable[float],
        episode_return: float,
        phase: str,
        violation_sum: float = 0.0,
        regret_sum: float = 0.0,
        steps: int = 1,
    ) -> Dict[str, float]:
        comp = competence_from_reward(
            episode_return=float(episode_return),
            reward_ema=float(self.state.reward_ema),
            reward_scale=float(self.reward_scale),
        )

        coh = coherence_from_actions(actions)

        sig = behavior_signature_from_episode(actions, rewards=rewards)

        cont, stored = continuity_from_behavior_signature(
            self.state.prev_behavior_sig,
            sig,
            sigma=float(self.continuity_sigma),
        )
        self.state.prev_behavior_sig = stored

        if str(phase).lower() == "pre":
            self.state.anchor_sig = _anchor_update_mean(self.state.anchor_sig, sig, self.state.anchor_count)
            self.state.anchor_count += 1

        integ = integrity_from_anchor_signature(
            self.state.anchor_sig,
            sig,
            sigma=float(self.integrity_sigma),
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
            "coherence": float(coh),
            "continuity": float(cont),
            "integrity": float(integ),
            "meaning": float(mean_score),
            "identity": float(ident),
        }



@dataclass
class EpisodeLog:
    episode: int
    reward: float
    steps: int
    stress: str
    competence: float
    coherence: float
    continuity: float
    integrity: float
    meaning: float
    identity: float
    collapse: bool
    notes: str = ""


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any] | EpisodeLog]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            if isinstance(r, EpisodeLog):
                r = asdict(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
