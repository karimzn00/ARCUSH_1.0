from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CollapseScoringConfig:
    event_threshold: float = 0.60
    sharpness: float = 2.5

_MEANING_REF_SCALE   : float = 1.0
_MEANING_ZERO_MAD_W  : float = 0.4
_ZERO_MAD_THRESH     : float = 1e-6


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _ff(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return float(default) if not np.isfinite(v) else v
    except Exception:
        return float(default)


def _sigmoid(x: float) -> float:
    x = float(np.clip(_ff(x), -500.0, 500.0))
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _robust_z(val: float, med: float, mad: float) -> float:
    """Standard robust z-score: (val - median) / (1.4826 * MAD)."""
    return (float(val) - float(med)) / (1.4826 * float(mad) + 1e-8)


def _weight_from_mad(mad: float) -> float:
    """Channel weight = sqrt(precision) = sqrt(1 / scale), capped for stability."""
    return math.sqrt(min(1.0 / (1.4826 * float(mad) + 1e-8), 1e6))


# ---------------------------------------------------------------------------
# Baseline stats accessors
# ---------------------------------------------------------------------------

def _get_raw_robust(
    baseline_stats: Optional[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Extract raw_robust stats from baseline_stats.
    Returns {'integrity': {median, mad}, 'id_drop': {median, mad}, 'meaning': {median, mad}}
    """
    _def_i = {"median": 0.6,  "mad": 0.10}   
    _def_d = {"median": 0.0,  "mad": 0.03}   
    _def_m = {"median": 1.0,  "mad": 0.0}    
    try:
        rr = baseline_stats["collapse"]["raw_robust"]
        return {
            "integrity": {
                "median": _ff(rr["integrity"]["median"], _def_i["median"]),
                "mad":    _ff(rr["integrity"]["mad"],    _def_i["mad"]),
            },
            "id_drop": {
                "median": max(0.0, _ff(rr["id_drop"]["median"], _def_d["median"])),
                "mad":    _ff(rr["id_drop"]["mad"],    _def_d["mad"]),
            },
            "meaning": {
                "median": _ff(rr["meaning"]["median"], 1.0),
                "mad":    _ff(rr["meaning"]["mad"],    0.0),
            },
        }
    except Exception:
        return {"integrity": dict(_def_i), "id_drop": dict(_def_d), "meaning": dict(_def_m)}


def _get_base_id(baseline_stats: Optional[Dict[str, Any]]) -> float:
    try:
        return _ff(baseline_stats["collapse"]["base_id"], 0.5)
    except Exception:
        return 0.5


def _get_center(baseline_stats: Optional[Dict[str, Any]]) -> float:
    try:
        c = _ff(baseline_stats["collapse"]["center"], 0.50)
        return float(np.clip(c, 0.10, 0.90))
    except Exception:
        return 0.50


# ---------------------------------------------------------------------------
# Per-channel z-score
# ---------------------------------------------------------------------------

def _channel_z_and_weight(
    channel: str,
    val: float,
    med: float,
    mad: float,
) -> tuple[float, float]:
    """
    Returns (z, weight) for one channel.

    For integrity: val = integrity score, worse = lower, so z = (med - val) / scale
    For id_drop:   val = id_drop,         worse = higher, so z = (val - med) / scale
    Both are already oriented so positive z = worse than baseline.

    For meaning (MAD=0): zero-MAD fallback maps meaning deficit in [0,1] -> z in [0,6].
    """
    if channel == "meaning":
        deficit = max(0.0, 1.0 - float(val))
        z = 6.0 * float(np.clip(deficit / _MEANING_REF_SCALE, 0.0, 1.0))
        return z, _MEANING_ZERO_MAD_W

    if mad >= _ZERO_MAD_THRESH:
        z = _robust_z(val, med, mad)
        z = float(np.clip(z, -6.0, 6.0))
        w = _weight_from_mad(mad)
    else:
        z = float(np.clip(val, -6.0, 6.0))
        w = _MEANING_ZERO_MAD_W
    return z, w

def collapse_score(
    *,
    meaning: float,
    integrity: float,
    id_drop: float,
    cfg: CollapseScoringConfig,
    baseline_stats: Optional[Dict[str, Any]] = None,
    **_ignored,
) -> float:
    meaning   = float(np.clip(_ff(meaning,   0.0), 0.0, 1.0))
    integrity = float(np.clip(_ff(integrity, 0.0), 0.0, 1.0))
    id_drop   = float(max(0.0, _ff(id_drop, 0.0)))

    rr     = _get_raw_robust(baseline_stats)
    center = _get_center(baseline_stats)


    z_m, w_m = _channel_z_and_weight(
        "meaning", meaning,
        med=rr["meaning"]["median"], mad=rr["meaning"]["mad"]
    )

    i_med = rr["integrity"]["median"]
    i_mad = rr["integrity"]["mad"]
    z_i, w_i = _channel_z_and_weight(
        "integrity", i_med - integrity,
        med=0.0, mad=i_mad
    )

    d_med = rr["id_drop"]["median"]
    d_mad = rr["id_drop"]["mad"]
    z_d, w_d = _channel_z_and_weight(
        "id_drop", id_drop - d_med, 
        med=0.0, mad=d_mad
    )

    w_sum = w_m + w_i + w_d + 1e-12
    raw   = (
        (w_m / w_sum) * _sigmoid(1.25 * z_m) +
        (w_i / w_sum) * _sigmoid(1.25 * z_i) +
        (w_d / w_sum) * _sigmoid(1.25 * z_d)
    )

    score = _sigmoid(float(cfg.sharpness) * (float(raw) - center))
    return float(np.clip(score, 0.0, 1.0))


def collapse_event(score: float, cfg: CollapseScoringConfig) -> bool:
    """Binary collapse event: score >= event_threshold."""
    return bool(_ff(score, 0.0) >= _ff(cfg.event_threshold, 0.60))
