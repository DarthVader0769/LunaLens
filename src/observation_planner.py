"""Observation policies with an explicit, immutable information boundary.

Planners receive only an initial map, a reconstruction containing already
revealed observations, initial descriptors, tile geometry/cost, and acquisition
history. Ground truth, reconstruction errors and evaluation history are absent.
This API prevents accidental scientific leakage; arbitrary Python plug-ins are
trusted code, not programs executing in a security sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from .feature_extraction import FEATURE_NAMES
from .preprocessing import validate_image
from .segmentation import Region, _positive_int

DEFAULT_WEIGHTS = {
    "variance": 0.10, "edge_density": 0.15, "gradient": 0.15,
    "texture": 0.20, "entropy": 0.15, "contrast": 0.10, "information": 0.15,
}


def _immutable_copy(values: np.ndarray, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Copy into a bytes-backed array whose writable flag cannot be re-enabled."""
    copy = np.array(values, dtype=dtype, order="C", copy=True)
    return np.frombuffer(copy.tobytes(), dtype=copy.dtype).reshape(copy.shape)


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


@dataclass(frozen=True, slots=True)
class PlannerState:
    """Detached read-only snapshot of the information a policy may use.

    ``features`` contains normalized initial-map descriptors, not features of
    unrevealed high-resolution tiles. ``history`` contains region IDs only.
    ``observed_mask[k]`` and ``features[k]`` refer to ``regions[k]``.
    """

    initial_map: np.ndarray
    current_map: np.ndarray
    regions: tuple[Region, ...]
    features: np.ndarray
    observed_mask: np.ndarray
    remaining_budget: float
    history: tuple[int, ...]

    def __post_init__(self) -> None:
        initial = validate_image(self.initial_map)
        current = validate_image(self.current_map)
        if initial.shape != current.shape:
            raise ValueError("initial_map and current_map must have the same shape")
        regions = tuple(self.regions)
        if not all(isinstance(region, Region) for region in regions):
            raise ValueError("regions must contain Region objects")
        ids = [region.id for region in regions]
        if len(set(ids)) != len(ids):
            raise ValueError("region IDs must be unique")
        for region in regions:
            if region.y0 + region.height > initial.shape[0] or region.x0 + region.width > initial.shape[1]:
                raise ValueError("region extends outside the available image")
        features = np.asarray(self.features)
        if (features.shape != (len(regions), len(FEATURE_NAMES))
                or not np.issubdtype(features.dtype, np.number) or np.iscomplexobj(features)
                or not np.all(np.isfinite(features)) or np.any(features < 0) or np.any(features > 1)):
            raise ValueError("features must be a finite normalized (N, 7) matrix in [0, 1]")
        observed = np.asarray(self.observed_mask)
        if observed.shape != (len(regions),) or observed.dtype != np.dtype(bool):
            raise ValueError("observed_mask must be a boolean vector of length N")
        history = tuple(_positive_int(item, "history region ID", allow_zero=True) for item in self.history)
        if len(set(history)) != len(history):
            raise ValueError("history cannot repeat a region ID")
        if set(history) != {region.id for region, seen in zip(regions, observed) if seen}:
            raise ValueError("history must describe exactly the observed region IDs")
        object.__setattr__(self, "initial_map", _immutable_copy(initial, np.float32))
        object.__setattr__(self, "current_map", _immutable_copy(current, np.float32))
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "features", _immutable_copy(features, np.float64))
        object.__setattr__(self, "observed_mask", _immutable_copy(observed, bool))
        object.__setattr__(self, "remaining_budget", _finite_nonnegative(self.remaining_budget, "remaining_budget"))
        object.__setattr__(self, "history", history)


@runtime_checkable
class Strategy(Protocol):
    name: str

    def scores(self, state: PlannerState) -> np.ndarray: ...

    def choose_next(self, state: PlannerState, rng: np.random.Generator) -> int | None: ...


def affordable(cost: float | np.ndarray, remaining_budget: float) -> bool | np.ndarray:
    """Relative round-off tolerance: a zero budget never buys a positive tile."""
    costs = np.asarray(cost, dtype=np.float64)
    tolerance = 16 * np.finfo(np.float64).eps * np.maximum(np.abs(costs), abs(remaining_budget))
    result = costs <= remaining_budget + tolerance
    return bool(result) if result.ndim == 0 else result


def eligible_mask(state: PlannerState) -> np.ndarray:
    return ~state.observed_mask & affordable(np.array([r.cost for r in state.regions]), state.remaining_budget)


def _weight_vector(weights: Mapping[str, float] | None) -> tuple[np.ndarray, Mapping[str, float]]:
    supplied = DEFAULT_WEIGHTS if weights is None else weights
    if not isinstance(supplied, Mapping):
        raise ValueError("weights must map feature names to nonnegative numbers")
    unknown = set(supplied) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown feature weight(s): {', '.join(sorted(map(str, unknown)))}")
    vector = np.array([_finite_nonnegative(supplied.get(name, 0.0), f"weight {name}") for name in FEATURE_NAMES])
    if not np.any(vector > 0):
        raise ValueError("at least one feature weight must be positive")
    # Scale before summing so individually finite large weights cannot overflow.
    vector /= vector.max()
    vector /= vector.sum()
    normalized = MappingProxyType(dict(zip(FEATURE_NAMES, map(float, vector))))
    return _immutable_copy(vector), normalized


def _mask_scores(values: np.ndarray, state: PlannerState) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if result.shape != (len(state.regions),):
        raise ValueError("strategy scores must have one value per region")
    result[~eligible_mask(state)] = -np.inf
    return result


class _RankedStrategy:
    """Select the largest finite eligible score; break ties by smallest ID."""

    def choose_next(self, state: PlannerState, rng: np.random.Generator) -> int | None:
        del rng  # Deterministic policies do not advance the caller's RNG.
        scores = np.asarray(self.scores(state), dtype=np.float64)
        if scores.shape != (len(state.regions),) or np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
            raise ValueError("strategy scores must be length N and contain finite values or -inf")
        candidates = np.flatnonzero(eligible_mask(state) & np.isfinite(scores))
        if candidates.size == 0:
            return None
        maximum = np.max(scores[candidates])
        return min(state.regions[index].id for index in candidates if scores[index] == maximum)


class RandomStrategy:
    """Uniform sampling without replacement over currently affordable tiles."""

    name = "random"

    def scores(self, state: PlannerState) -> np.ndarray:
        return _mask_scores(np.ones(len(state.regions)), state)

    def choose_next(self, state: PlannerState, rng: np.random.Generator) -> int | None:
        candidates = np.flatnonzero(eligible_mask(state))
        if candidates.size == 0:
            return None
        return state.regions[int(rng.choice(candidates))].id


class SystematicStrategy(_RankedStrategy):
    """Visit the smallest unobserved affordable region ID (row-major tiling)."""

    name = "systematic"

    def scores(self, state: PlannerState) -> np.ndarray:
        return _mask_scores(-np.array([region.id for region in state.regions], dtype=np.float64), state)


class FeatureStrategy(_RankedStrategy):
    """Rank initial-map weighted feature utility per unit observation cost.

    Supplied weights replace the default mapping. Omitted feature names receive
    zero weight; weights are normalized to sum one. Descriptors and utility are
    fixed at the initial map, while affordability and availability change.
    """

    name = "feature"

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self._weights, self.weights = _weight_vector(weights)

    def scores(self, state: PlannerState) -> np.ndarray:
        costs = np.array([region.cost for region in state.regions], dtype=np.float64)
        return _mask_scores((state.features @ self._weights) / costs, state)


def make_strategy(name: str, weights: Mapping[str, float] | None = None) -> Strategy:
    """Construct a baseline or the isolated editable custom algorithm."""
    if not isinstance(name, str) or name not in ("random", "systematic", "feature", "custom"):
        raise ValueError("strategy must be 'random', 'systematic', 'feature', or 'custom'")
    # Validate supplied weights even for methods that do not use them, so a UI
    # cannot silently accept a misspelled or invalid experimental parameter.
    if weights is not None:
        _weight_vector(weights)
    if name == "random":
        return RandomStrategy()
    if name == "systematic":
        return SystematicStrategy()
    if name == "feature":
        return FeatureStrategy(weights)
    from .custom_strategy import CustomStrategy
    return CustomStrategy(weights)
