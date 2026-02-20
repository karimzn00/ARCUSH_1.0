from __future__ import annotations

from pathlib import Path
import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def detect_device(prefer: str = "auto") -> str:
    """Return 'cuda' if available else 'cpu' when prefer='auto'."""
    prefer = (prefer or "auto").lower()
    if prefer in ("cpu", "cuda"):
        return prefer
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def flatten_params(policy_or_model) -> np.ndarray:
    """Accept SB3 model (has .policy) or policy itself; return flat params."""
    if torch is None:
        raise RuntimeError("torch is required for flatten_params()")

    obj = policy_or_model
    if hasattr(obj, "policy"):
        obj = obj.policy

    params = []
    for p in obj.parameters():
        params.append(p.detach().flatten().cpu())
    if not params:
        return np.zeros((0,), dtype=np.float32)
    return torch.cat(params).numpy()


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
