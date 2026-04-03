from __future__ import annotations

from typing import Any, Dict, Tuple
from gymnasium import spaces

from .base import BaseStressor


class BaselineStressor(BaseStressor):
    name = "baseline"

    def transform_step(
        self,
        action: Any,
        obs: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
        *,
        action_space: spaces.Space,
        active: bool,
        phase: str,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        info.setdefault("violation",     0.0)
        info.setdefault("regret",        0.0)
        info["stress_applied"] = 0
        return obs, float(reward), bool(terminated), bool(truncated), info
