from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

SCHEMA_VERSION = "1.0"

STANDARD_KEYS = [
    "schema_version",
    "run_id",
    "env_id",
    "algo",
    "seed",
    "schedule",
    "stress",
    "episode",
    "reward",
    "steps",
    "competence",
    "integrity",
    "stability",
    "continuity",
    "coherence",
    "arcus",
    "arcus_vol",
    "identity_vol",
    "collapsed",
    "collapse_type",
    "recovery_flag",
]

def normalize_row(row: Dict[str, Any], *, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ensure required keys exist, with schema_version and standardized naming."""
    defaults = defaults or {}
    out = dict(defaults)
    out.update(row)
    out["schema_version"] = SCHEMA_VERSION
    if "collapse" in out and "collapsed" not in out:
        out["collapsed"] = bool(out["collapse"])
    if "identity" in out and "stability" not in out:
        out["stability"] = float(out["identity"])
    if "episode" not in out and "ep" in out:
        out["episode"] = int(out["ep"])
    for k in STANDARD_KEYS:
        out.setdefault(k, None)
    return out
