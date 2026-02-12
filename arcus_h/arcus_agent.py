from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

from .metrics import IdentityState
from .narratives import NarrativeState


ACTIONS = ("REST", "WORK", "PROBE")


@dataclass
class ArcusConfig:
    # action value weights
    w_reward: float = 1.0
    w_identity: float = 2.2
    w_meaning: float = 1.4
    w_regret: float = 0.9

    # meaning-based action gating
    meaning_floor: float = 0.22
    trust_floor: float = 0.18

    # identity update rates
    lr_competence: float = 0.09
    lr_integrity: float = 0.06
    lr_coherence: float = 0.08
    lr_continuity: float = 0.05

    # counterfactual lookahead
    cf_horizon: int = 2
    cf_samples: int = 12

    # exploration
    temperature: float = 0.65
    eps: float = 0.04


class ArcusHV4:
    """ARCUS-H v4 (toy scaffold)."""

    def __init__(self, seed: int = 0, config: Optional[ArcusConfig] = None):
        self.rng = np.random.default_rng(seed)
        self.cfg = config or ArcusConfig()

        self.identity = IdentityState(competence=0.60, integrity=0.62, coherence=0.58, continuity=0.60)
        self.narrative = NarrativeState()

        self.prev_obs: Optional[np.ndarray] = None
        self.prev_action: Optional[int] = None
        self.prev_pred_values: Optional[np.ndarray] = None

        self.self_centroid = None
        self.steps = 0

    def reset(self):
        self.identity = IdentityState(competence=0.60, integrity=0.62, coherence=0.58, continuity=0.60)
        self.narrative = NarrativeState()
        self.prev_obs = None
        self.prev_action = None
        self.prev_pred_values = None
        self.self_centroid = None
        self.steps = 0

    def _predict_action_outcomes(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        meaning, energy, boredom, trust, novelty, backlog = map(float, obs)

        exp_reward = np.zeros(3, dtype=np.float64)
        exp_d_meaning = np.zeros(3, dtype=np.float64)
        exp_d_id = np.zeros(3, dtype=np.float64)

        # REST
        exp_reward[0] = -0.05 + 0.02 * (energy - 0.5) - 0.15 * boredom
        exp_d_meaning[0] = -0.012
        exp_d_id[0] = +0.003 * max(0.0, 0.4 - boredom)

        # WORK
        hollow = max(0.0, 0.55 - novelty)
        exp_prog = (0.12) * (0.6 + 0.4 * energy) * (0.6 + 0.4 * trust)
        exp_complete = np.clip(0.05 + 0.65 * exp_prog + 0.25 * backlog, 0.0, 0.95)
        exp_reward[1] = 6.0 * exp_complete + 1.8 * exp_prog - 0.9 * (boredom + 0.04) - 0.2 * max(0.0, 0.3 - (energy - 0.09))
        exp_d_meaning[1] = 0.05 * exp_prog + 0.20 * exp_complete - 0.08 * hollow
        exp_d_id[1] = 0.02 * exp_prog + 0.04 * exp_complete - 0.03 * hollow

        # PROBE
        exp_reward[2] = 0.10 - 0.15 * max(0.0, 0.25 - energy) - 0.25 * boredom + 0.08 * novelty
        exp_d_meaning[2] = 0.04 + 0.05 * max(0.0, 0.65 - meaning)
        exp_d_id[2] = 0.010 + 0.015 * (novelty - 0.5)

        return exp_reward, exp_d_id, exp_d_meaning

    def _meaning_gate(self, obs: np.ndarray, exp_d_meaning: np.ndarray) -> np.ndarray:
        meaning, energy, boredom, trust, novelty, backlog = map(float, obs)
        mask = np.ones(3, dtype=np.float64)

        if meaning < self.cfg.meaning_floor + 0.05:
            for a in range(3):
                if float(exp_d_meaning[a]) < -0.002:
                    mask[a] = 0.0

        if trust < self.cfg.trust_floor + 0.05:
            mask[1] *= 0.2

        if energy < 0.12:
            mask[1] = 0.0

        return mask

    # public wrapper (for your experiments)
    def meaning_gate(self, obs: np.ndarray, exp_d_meaning: np.ndarray) -> np.ndarray:
        return self._meaning_gate(obs, exp_d_meaning)

    def _counterfactual_narrative_regret(self, obs: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        penalties = np.zeros(3, dtype=np.float64)
        base_strength = self.narrative.strength

        for a0 in range(3):
            drops = []
            for _ in range(cfg.cf_samples):
                m, e, b, tr, nv, bl = map(float, obs)
                strength = base_strength

                for h in range(cfg.cf_horizon):
                    exp_r, exp_d_id, exp_dm = self._predict_action_outcomes(np.array([m, e, b, tr, nv, bl]))
                    if h == 0:
                        a = a0
                    else:
                        a = int(np.argmax(exp_r + 0.5 * exp_dm))

                    dm = float(exp_dm[a])
                    if a == 0:
                        e = min(1.0, e + 0.10); b = max(0.0, b - 0.06); nv = max(0.0, nv - 0.01)
                    elif a == 1:
                        e = max(0.0, e - 0.09); b = min(1.0, b + 0.04); nv = max(0.0, nv - 0.04); bl = max(0.0, bl - 0.05)
                    else:
                        e = max(0.0, e - 0.04); b = max(0.0, b - 0.02); nv = min(1.0, nv + 0.10); bl = min(1.0, bl + 0.01)

                    m = float(np.clip(m + dm, 0.0, 1.0))
                    tr = float(np.clip(tr + 0.01 * (nv - 0.5) - 0.01 * max(0.0, b - 0.5), 0.0, 1.0))

                    growth = 0.6 * (m - 0.4) + 0.2 * (nv - 0.5)
                    survival = 0.7 * (0.5 - m) + 0.4 * (0.45 - tr)
                    repair = 0.8 * max(0.0, 0.45 - tr) + 0.2 * max(0.0, m - 0.35)

                    best = max(growth, survival, repair, 0.0)
                    strength = 0.8 * strength + 0.2 * float(np.clip(0.5 + best, 0.0, 1.0))

                drops.append(max(0.0, base_strength - strength))

            penalties[a0] = float(np.mean(drops))

        return penalties

    def act(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float64)

        exp_r, exp_d_id, exp_dm = self._predict_action_outcomes(obs)
        gate = self._meaning_gate(obs, exp_dm)
        cf_pen = self._counterfactual_narrative_regret(obs)

        v = (self.cfg.w_reward * exp_r + self.cfg.w_identity * exp_d_id + self.cfg.w_meaning * exp_dm - 1.3 * cf_pen) * gate

        if self.rng.random() < self.cfg.eps:
            a = int(self.rng.integers(0, 3))
        else:
            temp = max(1e-6, self.cfg.temperature)
            z = v / temp
            z = z - np.max(z)
            p = np.exp(z) * gate
            if p.sum() <= 0:
                a = int(np.argmax(v))
            else:
                p = p / p.sum()
                a = int(self.rng.choice([0, 1, 2], p=p))

        self.prev_obs = obs.copy()
        self.prev_action = a
        self.prev_pred_values = v.copy()
        return a

    def observe(self, obs: np.ndarray, reward: float, info: Dict) -> None:
        obs = np.asarray(obs, dtype=np.float64)
        meaning, energy, boredom, trust, novelty, backlog = map(float, obs)

        progress = float(info.get("progress", 0.0))
        completed = float(info.get("task_completed", False))

        d_comp = (0.12 * progress + 0.10 * completed) - 0.10 * max(0.0, 0.25 - energy)
        d_int = 0.06 * (trust - 0.5) - 0.10 * max(0.0, 0.22 - meaning)
        coh_bonus = self.narrative.coherence_bonus()
        d_coh = 0.04 * (meaning - 0.5) + coh_bonus - 0.05 * max(0.0, boredom - 0.6)

        x = obs.astype(np.float64)
        if self.self_centroid is None:
            self.self_centroid = x.copy()
        else:
            self.self_centroid = 0.97 * self.self_centroid + 0.03 * x
        dist = float(np.linalg.norm(x - self.self_centroid))
        d_cont = 0.03 * (0.5 - dist)

        self.identity.competence = float(np.clip(self.identity.competence + self.cfg.lr_competence * d_comp, 0.0, 1.0))
        self.identity.integrity = float(np.clip(self.identity.integrity + self.cfg.lr_integrity * d_int, 0.0, 1.0))
        self.identity.coherence = float(np.clip(self.identity.coherence + self.cfg.lr_coherence * d_coh, 0.0, 1.0))
        self.identity.continuity = float(np.clip(self.identity.continuity + self.cfg.lr_continuity * d_cont, 0.01, 1.0))

        self.narrative.update(meaning=meaning, progress=progress, trust=trust, novelty=novelty)

        if self.prev_pred_values is not None and self.prev_action is not None:
            v = self.prev_pred_values
            chosen = float(v[self.prev_action])
            best = float(np.max(v))
            regret = max(0.0, best - chosen)

            id_loss_factor = 1.0 + 2.0 * max(0.0, 0.30 - meaning) + 1.5 * max(0.0, 0.25 - trust)
            shaped = self.cfg.w_regret * regret * id_loss_factor
            self.identity.coherence = float(np.clip(self.identity.coherence - 0.03 * shaped, 0.0, 1.0))

        self.steps += 1

    def identity_overall(self) -> float:
        return self.identity.overall()

    def identity_components(self) -> Dict[str, float]:
        return self.identity.as_dict()

    def narrative_summary(self) -> Dict[str, float]:
        return {"kind": str(self.narrative.kind.value), "strength": float(self.narrative.strength)}
