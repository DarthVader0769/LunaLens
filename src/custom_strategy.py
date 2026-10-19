"""Editable experimental policy: feature utility, novelty, and spatial spread.

This module is the intended starting point for the student's own contribution.
Keep the ``scores(state)`` and ``choose_next(state, rng)`` interface unchanged.
The initial example is a heuristic, not a claimed research novelty or a learned
information-gain model. Change it, predeclare comparisons, and test honestly.

For an eligible region i, with normalized initial descriptor f_i and centre p_i:

  u_i = sum_j w_j f_ij
  n_i = min_{k already observed} ||f_i-f_k||_2 / sqrt(7)
  d_i = min_{k already observed} ||p_i-p_k||_2 / image_diagonal
  score_i = (alpha*u_i + beta*n_i + gamma*d_i) / cost_i

Novelty and distance are zero before any observation. Coefficients are scaled
to sum one. No descriptors are fitted to hidden or later-revealed truth: the
history provides only which initial descriptors/positions have been sampled.
This deliberately separates coverage/exploration from baseline feature ranking.

Where it may fail: diversity can spend budget on smooth or unusual-but-already
well-reconstructed tiles; illumination and compression can look informative;
the low-resolution map may erase the fine structures that need observations.
Nothing in this policy guarantees better MAE, MSE, or scientific coverage.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.spatial.distance import cdist

from .observation_planner import (
    PlannerState, _RankedStrategy, _finite_nonnegative, _mask_scores, _weight_vector,
)


class CustomStrategy(_RankedStrategy):
    name = "custom"

    def __init__(
        self, weights: Mapping[str, float] | None = None, *,
        utility_weight: float = 0.60, novelty_weight: float = 0.25,
        spatial_weight: float = 0.15,
    ) -> None:
        self._weights, self.weights = _weight_vector(weights)
        coefficients = np.array([
            _finite_nonnegative(utility_weight, "utility_weight"),
            _finite_nonnegative(novelty_weight, "novelty_weight"),
            _finite_nonnegative(spatial_weight, "spatial_weight"),
        ])
        if not np.any(coefficients > 0):
            raise ValueError("at least one custom coefficient must be positive")
        coefficients /= coefficients.max()
        coefficients /= coefficients.sum()
        self.utility_weight, self.novelty_weight, self.spatial_weight = map(float, coefficients)

    def scores(self, state: PlannerState) -> np.ndarray:
        count = len(state.regions)
        utility = state.features @ self._weights
        novelty = np.zeros(count, dtype=np.float64)
        dispersion = np.zeros(count, dtype=np.float64)
        if count and np.any(state.observed_mask):
            observed_features = state.features[state.observed_mask]
            novelty = cdist(state.features, observed_features).min(axis=1) / np.sqrt(state.features.shape[1])
            centers = np.array([
                [region.y0 + region.height / 2.0, region.x0 + region.width / 2.0]
                for region in state.regions
            ])
            diagonal = float(np.hypot(*state.initial_map.shape))
            dispersion = cdist(centers, centers[state.observed_mask]).min(axis=1) / diagonal
        costs = np.array([region.cost for region in state.regions], dtype=np.float64)
        scores = (self.utility_weight * utility + self.novelty_weight * novelty
                  + self.spatial_weight * dispersion) / costs
        return _mask_scores(scores, state)
