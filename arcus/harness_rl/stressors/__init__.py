from __future__ import annotations
from typing import Any, Dict, Type

from .base import apply_stress_pattern, BaseStressor
from .baseline import BaselineStressor
from .resource_constraint import ResourceConstraintStressor
from .trust_violation import TrustViolationStressor, TrustViolationConfig
from .valence_inversion import ValenceInversionStressor
from .concept_drift import ConceptDriftStressor, ConceptDriftConfig
from .observation_noise import ObservationNoiseStressor
from .sensor_blackout import SensorBlackoutStressor
from .reward_noise import RewardNoiseStressor

_REGISTRY: Dict[str, tuple[Type[BaseStressor], bool]] = {
    "none":                (BaselineStressor,          False),
    "baseline":            (BaselineStressor,          False),
    "resource_constraint": (ResourceConstraintStressor, False),
    "trust_violation":     (TrustViolationStressor,     True),
    "valence_inversion":   (ValenceInversionStressor,   False),
    "concept_drift":       (ConceptDriftStressor,       True),
    "observation_noise":   (ObservationNoiseStressor,   True),
    "sensor_blackout":     (SensorBlackoutStressor,     True),
    "reward_noise":        (RewardNoiseStressor,        True),
}


def available_stressors() -> list[str]:
    """Return sorted list of registered stressor names."""
    return sorted(_REGISTRY.keys())


def get_stressor(name: str, *, seed: int = 0, **kwargs: Any) -> BaseStressor:
    key = (name or "baseline").strip().lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown stressor '{name}'. Available: {available_stressors()}"
        )
    cls, accepts_seed = _REGISTRY[key]
    if accepts_seed:
        return cls(seed=int(seed), **kwargs)
    return cls(**kwargs)
