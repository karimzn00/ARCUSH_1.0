from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
@dataclass
class CollapseScoringConfig:
    event_threshold: Optional[float] = None
    sharpness:       float            = 2.5
    center:          Optional[float]  = None
    eps:             float            = 1e-8

def collapse_score(
    meaning:        float,
    integrity:      float,
    id_drop:        float,
    cfg:            CollapseScoringConfig,
    baseline_stats: Optional[Dict[str, Any]] = None,
) -> float:
    eps = float(cfg.eps)
    if baseline_stats is not None:
        collapse_b = baseline_stats.get("collapse", {})
        raw_robust = collapse_b.get("raw_robust", {})

        i_stat = raw_robust.get("integrity", {})
        d_stat = raw_robust.get("id_drop",   {})
        m_stat = raw_robust.get("meaning",   {})

        i_med = float(i_stat.get("median", 0.5))
        i_mad = float(i_stat.get("mad",    0.1))
        d_med = float(d_stat.get("median", 0.0))
        d_mad = float(d_stat.get("mad",    0.1))
        m_med = float(m_stat.get("median", 1.0))
        m_mad = float(m_stat.get("mad",    0.0))

        center = float(collapse_b.get("center", 0.5))
    else:
        i_med, i_mad = 0.5, 0.1
        d_med, d_mad = 0.0, 0.1
        m_med, m_mad = 1.0, 0.0
        center = 0.5


    if cfg.center is not None:
        center = float(cfg.center)
    w_i = i_mad
    w_d = d_mad
    w_m = m_mad

    w_sum = w_i + w_d + w_m + eps
    w_i   = w_i / w_sum
    w_d   = w_d / w_sum
    w_m   = w_m / w_sum

    z_i = (i_med - float(integrity)) / (1.4826 * i_mad + eps)
    z_i = float(np.clip(z_i, -6.0, 6.0))

    z_d = (float(id_drop) - d_med) / (1.4826 * d_mad + eps)
    z_d = float(np.clip(z_d, -6.0, 6.0))

    deficit_m = max(0.0, 1.0 - float(meaning))
    if m_mad > 1e-6:
        z_m = (deficit_m - (1.0 - m_med)) / (1.4826 * m_mad + eps)
        z_m = float(np.clip(z_m, -6.0, 6.0))
    else:
        z_m = deficit_m / 0.1
        z_m = float(np.clip(z_m, 0.0, 6.0))

    raw = math.sqrt(w_i * z_i**2 + w_d * z_d**2 + w_m * z_m**2)

    beta  = float(cfg.sharpness)
    x     = beta * (raw - center)
    x     = float(np.clip(x, -500.0, 500.0))
    score = 1.0 / (1.0 + math.exp(-x))

    return float(np.clip(score, 0.0, 1.0))

def collapse_event(score: float, cfg: CollapseScoringConfig) -> bool:
    if cfg.event_threshold is None:
        raise ValueError(
            "collapse_event() requires cfg.event_threshold to be set. "
            "Use the adaptive p95 threshold from _compute_baseline_stats()."
        )
    return float(score) >= float(cfg.event_threshold)
