from __future__ import annotations

import numpy as np


def build_schedule_mask(*, episodes: int, schedule_tag: str, schedule_spec: str) -> np.ndarray:
    """
    Returns a boolean mask of length `episodes` telling whether each episode is under stress.

    schedule_spec examples:
      "none"
      "none:20,invert:20,none:60"
    """
    mask = np.zeros((episodes,), dtype=bool)

    if schedule_spec.strip().lower() == "none":
        return mask

    parts = [p.strip() for p in schedule_spec.split(",") if p.strip()]
    idx = 0
    for part in parts:
        name, dur = part.split(":")
        name = name.strip()
        dur = int(dur.strip())
        if dur <= 0:
            continue
        j = min(episodes, idx + dur)
        if name != "none":
            mask[idx:j] = True
        idx = j
        if idx >= episodes:
            break
    return mask
