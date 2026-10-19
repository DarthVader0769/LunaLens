"""A budgeted image-reveal simulator, not a spacecraft dynamics model.

Ground truth is private to this function's reconstruction and evaluator. Each
policy call receives detached immutable copies through PlannerState, never the
truth or metric history. This is an experimental information boundary for
trusted Python strategies; it does not sandbox hostile plug-in code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .evaluation import evaluate_reconstruction
from .feature_extraction import extract_features, normalise_features
from .observation_planner import (
    PlannerState, Strategy, _finite_nonnegative, _immutable_copy, affordable, make_strategy,
)
from .preprocessing import degrade_image, validate_image
from .segmentation import Region, _positive_int, segment_image


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    region_size: int = 32
    downsample_factor: int = 4
    blur_sigma: float = 1.0
    budget: float = 16.0
    seed: int = 0
    cost_mode: str = "uniform"

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_size", _positive_int(self.region_size, "region_size"))
        object.__setattr__(self, "downsample_factor", _positive_int(self.downsample_factor, "downsample_factor"))
        object.__setattr__(self, "blur_sigma", _finite_nonnegative(self.blur_sigma, "blur_sigma"))
        object.__setattr__(self, "budget", _finite_nonnegative(self.budget, "budget"))
        object.__setattr__(self, "seed", _positive_int(self.seed, "seed", allow_zero=True))
        if self.cost_mode not in ("uniform", "area"):
            raise ValueError("cost_mode must be 'uniform' or 'area'")


@dataclass(frozen=True, slots=True)
class SimulationResult:
    strategy_name: str
    config: SimulationConfig
    regions: tuple[Region, ...]
    initial_map: np.ndarray
    reconstructed_map: np.ndarray
    selected_ids: tuple[int, ...]
    history: list[dict]
    raw_features: np.ndarray
    features: np.ndarray
    initial_scores: np.ndarray


def _initial_scores(strategy: Strategy, state: PlannerState) -> np.ndarray:
    try:
        raw = np.asarray(strategy.scores(state))
        if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
            raise ValueError("scores must be real numbers")
        scores = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("strategy scores must be a real numeric vector of length N") from exc
    if scores.shape != (len(state.regions),) or np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise ValueError("strategy scores must have length N and contain finite values or -inf")
    return scores


def run_simulation(
    ground_truth: np.ndarray, strategy: str | Strategy = "feature",
    config: SimulationConfig | None = None, weights: Mapping[str, float] | None = None,
    initial_map: np.ndarray | None = None,
) -> SimulationResult:
    """Reveal selected tiles exactly, evaluating after each affordable observation.

    An explicit ``initial_map`` supports controlled counterfactual experiments:
    it bypasses degradation and must match the truth shape. Strategies can use
    revealed current-map pixels but never see error feedback. The simulator
    independently checks strategy outputs, repeated IDs and total cost.

    Budgets above the cost of all tiles are permitted; acquisition ends after
    full coverage. Fractional budgets may leave an unusable remainder. A custom
    strategy may deliberately return None to stop early.
    """
    if config is None:
        config = SimulationConfig()
    if not isinstance(config, SimulationConfig):
        raise ValueError("config must be a SimulationConfig")
    truth = np.array(validate_image(ground_truth), dtype=np.float32, order="C", copy=True)
    truth.setflags(write=False)
    if initial_map is None:
        available = degrade_image(truth, factor=config.downsample_factor, blur_sigma=config.blur_sigma)
    else:
        available = validate_image(initial_map)
        if available.shape != truth.shape:
            raise ValueError("initial_map and ground_truth must have the same shape")
    available = np.array(available, dtype=np.float32, order="C", copy=True)
    regions = segment_image(truth.shape, config.region_size, config.cost_mode)
    raw_features = extract_features(available, regions)
    features = normalise_features(raw_features)
    if isinstance(strategy, str):
        policy = make_strategy(strategy, weights)
    else:
        if weights is not None:
            raise ValueError("supply weights when constructing a strategy object, not to run_simulation")
        if (not isinstance(getattr(strategy, "name", None), str) or not strategy.name
                or not callable(getattr(strategy, "scores", None))
                or not callable(getattr(strategy, "choose_next", None))):
            raise ValueError("strategy object must define name, scores(state), and choose_next(state, rng)")
        policy = strategy

    rng = np.random.default_rng(config.seed)
    reconstructed = available.copy()
    observed = np.zeros(len(regions), dtype=bool)
    selected: list[int] = []
    costs: list[float] = []
    total_cost, covered_pixels = 0.0, 0
    residual = truth.astype(np.float64) - available.astype(np.float64)
    initial_mse = float(np.mean(residual * residual))
    history = [evaluate_reconstruction(
        truth, reconstructed, initial_mse=initial_mse, observations=0, total_cost=0.0, coverage=0.0,
    )]

    def remaining_budget() -> float:
        if config.cost_mode == "area":
            # Area costs have an exact integer-pixel numerator. Subtracting
            # accumulated fractional costs from a large budget can incorrectly
            # refuse the last partial tile. Work in pixel units instead, and
            # snap only round-off-sized differences from an integer pixel.
            unit_area = config.region_size**2
            pixel_budget = config.budget * unit_area
            if np.isfinite(pixel_budget):
                nearest_pixel = round(pixel_budget)
                tolerance = 16 * np.finfo(float).eps * abs(pixel_budget)
                if abs(pixel_budget - nearest_pixel) <= tolerance:
                    pixel_budget = nearest_pixel
                return max(0.0, (pixel_budget - covered_pixels) / unit_area)
        return max(0.0, config.budget - total_cost)

    def snapshot() -> PlannerState:
        return PlannerState(
            initial_map=available, current_map=reconstructed, regions=regions,
            features=features, observed_mask=observed,
            remaining_budget=remaining_budget(), history=tuple(selected),
        )

    state = snapshot()
    initial_scores = _initial_scores(policy, state)
    # A bounded loop guarantees a defective strategy cannot run indefinitely.
    for _ in range(len(regions)):
        selected_id = policy.choose_next(state, rng)
        if selected_id is None:
            break
        selected_id = _positive_int(selected_id, "selected region ID", allow_zero=True)
        if selected_id >= len(regions):
            raise ValueError(f"strategy selected unknown region ID {selected_id}")
        if observed[selected_id]:
            raise ValueError(f"strategy selected already observed region {selected_id}")
        region = regions[selected_id]
        remaining = remaining_budget()
        if not affordable(region.cost, remaining):
            raise ValueError(f"strategy selected unaffordable region {selected_id}")
        next_total = math.fsum([*costs, region.cost])
        if not affordable(next_total, config.budget):
            raise ValueError("strategy would exceed the observation budget")
        reconstructed[region.slices] = truth[region.slices]
        observed[selected_id] = True
        selected.append(selected_id)
        costs.append(region.cost)
        total_cost = next_total
        covered_pixels += region.area
        history.append(evaluate_reconstruction(
            truth, reconstructed, initial_mse=initial_mse, observations=len(selected),
            total_cost=total_cost, coverage=covered_pixels / truth.size,
        ))
        if len(selected) < len(regions):
            state = snapshot()

    return SimulationResult(
        strategy_name=policy.name, config=config, regions=regions,
        initial_map=_immutable_copy(available), reconstructed_map=_immutable_copy(reconstructed),
        selected_ids=tuple(selected), history=history, raw_features=_immutable_copy(raw_features),
        features=_immutable_copy(features), initial_scores=_immutable_copy(initial_scores),
    )
