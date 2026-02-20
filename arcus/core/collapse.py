from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np


@dataclass
class CollapseScoringConfig:
    meaning_soft: float = 0.35
    integrity_soft: float = 0.55
    id_drop_soft: float = 0.10

    # weights
    w_meaning: float = 1.2
    w_integrity: float = 0.9
    w_id_drop: float = 0.8
    w_trust: float = 1.0 

    sharpness: float = 3.0

    event_threshold: float = 0.60


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def collapse_score(
    *,
    meaning: float,
    integrity: float,
    id_drop: float,
    info: Dict[str, Any] | None,
    cfg: CollapseScoringConfig,
) -> float:
    """
    Returns score in [0,1]. Higher = more collapsed.
    Designed to be discriminative (avoid saturating).
    """
    meaning = float(np.clip(float(meaning), 0.0, 1.0))
    integrity = float(np.clip(float(integrity), 0.0, 1.0))
    id_drop = float(max(float(id_drop), 0.0))

    trust_intensity = 0.0
    if info:
        trust_intensity = float(info.get("trust_violation_intensity", 0.0) or 0.0)
    trust_intensity = float(np.clip(trust_intensity, 0.0, 1.0))

    d_mean = max(0.0, float(cfg.meaning_soft) - meaning)
    d_int = max(0.0, float(cfg.integrity_soft) - integrity)
    d_idd = max(0.0, id_drop - float(cfg.id_drop_soft))

    raw = (
        float(cfg.w_meaning) * d_mean
        + float(cfg.w_integrity) * d_int
        + float(cfg.w_id_drop) * d_idd
        + float(cfg.w_trust) * trust_intensity
    )

    raw_n = raw / (raw + 1.0)

    center = 0.35
    s = _sigmoid(float(cfg.sharpness) * (raw_n - center))
    return float(np.clip(s, 0.0, 1.0))


def collapse_event(score: float, cfg: CollapseScoringConfig) -> bool:
    return bool(float(score) >= float(cfg.event_threshold))
