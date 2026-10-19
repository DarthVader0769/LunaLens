"""Rankings, affordability, parameter validation and copied planner state."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.custom_strategy import CustomStrategy
from src.feature_extraction import FEATURE_NAMES
from src.observation_planner import PlannerState, eligible_mask, make_strategy
from src.segmentation import Region, segment_image


def state_with_features(features, *, observed=(), budget=10.0, regions=None):
    features = np.asarray(features, dtype=float)
    if regions is None:
        regions = segment_image((4, 4 * len(features)), 4)
    height = max(r.y0 + r.height for r in regions)
    width = max(r.x0 + r.width for r in regions)
    image = np.zeros((height, width), dtype=np.float32)
    seen = np.array([region.id in observed for region in regions], dtype=bool)
    return PlannerState(image, image, regions, features, seen, budget, tuple(observed))


def test_weighted_ranking_matches_known_numeric_example_and_cost():
    features = np.zeros((3, 7))
    features[:, 0], features[:, 4] = [0.2, 0.8, 0.4], [0.8, 0.2, 0.1]
    policy = make_strategy("feature", {"variance": 3.0, "entropy": 1.0})
    state = state_with_features(features)
    np.testing.assert_allclose(policy.scores(state), [0.35, 0.65, 0.325])
    assert policy.choose_next(state, np.random.default_rng(0)) == 1
    regions = tuple(Region(i, 0, i, 0, 4 * i, 4, 4, cost) for i, cost in enumerate([1, 2, 0.25]))
    costs_state = state_with_features(features, regions=regions)
    np.testing.assert_allclose(policy.scores(costs_state), [0.35, 0.325, 1.3])
    assert policy.choose_next(costs_state, np.random.default_rng(0)) == 2


def test_lowest_region_id_breaks_ties_even_when_regions_are_reordered():
    regions = (Region(8, 0, 0, 0, 0, 4, 4), Region(2, 0, 1, 0, 4, 4, 4), Region(5, 0, 2, 0, 8, 4, 4))
    state = state_with_features(np.ones((3, 7)), regions=regions)
    for name in ("systematic", "feature", "custom"):
        assert make_strategy(name).choose_next(state, np.random.default_rng(0)) == 2


@pytest.mark.parametrize("name", ["random", "systematic", "feature", "custom"])
def test_all_policies_exclude_observed_and_unaffordable_regions(name):
    regions = tuple(Region(i, 0, i, 0, i * 4, 4, 4, cost) for i, cost in enumerate([1, 0.4, 0.2]))
    state = state_with_features(np.ones((3, 7)), observed=(2,), budget=0.5, regions=regions)
    policy = make_strategy(name)
    np.testing.assert_array_equal(eligible_mask(state), [False, True, False])
    assert policy.choose_next(state, np.random.default_rng(10)) == 1
    assert np.isneginf(policy.scores(state)[[0, 2]]).all()
    empty = state_with_features(np.ones((3, 7)), budget=0.1, regions=regions)
    assert policy.choose_next(empty, np.random.default_rng(10)) is None


def test_custom_novelty_and_spread_change_priorities_after_an_observation():
    features = np.zeros((3, 7))
    features[1] = 0.05
    features[2] = 1.0
    before = state_with_features(features)
    after = state_with_features(features, observed=(0,))
    novelty = CustomStrategy(utility_weight=0, novelty_weight=1, spatial_weight=0)
    np.testing.assert_array_equal(novelty.scores(before), 0)
    np.testing.assert_allclose(novelty.scores(after)[1:], [0.05, 1])
    assert novelty.choose_next(after, np.random.default_rng(0)) == 2
    dispersion = CustomStrategy(utility_weight=0, novelty_weight=0, spatial_weight=1)
    assert dispersion.scores(after)[2] > dispersion.scores(after)[1] > 0


def test_custom_utility_only_is_equivalent_to_feature_baseline():
    features = np.random.default_rng(8).random((4, 7))
    state = state_with_features(features, observed=(0,))
    np.testing.assert_array_equal(
        make_strategy("feature").scores(state),
        CustomStrategy(utility_weight=1, novelty_weight=0, spatial_weight=0).scores(state),
    )


def test_state_arrays_are_copied_and_cannot_be_made_writable():
    image = np.zeros((4, 8), dtype=np.float32)
    features = np.zeros((2, 7))
    observed = np.zeros(2, dtype=bool)
    state = PlannerState(image, image, segment_image(image.shape, 4), features, observed, 2, ())
    for view, source in ((state.initial_map, image), (state.current_map, image),
                         (state.features, features), (state.observed_mask, observed)):
        assert not np.shares_memory(view, source)
        assert not view.flags.writeable
        with pytest.raises(ValueError):
            view.setflags(write=True)
        with pytest.raises(ValueError):
            view.flat[0] = 1
    image[:] = 1
    features[:] = 1
    observed[:] = True
    assert not state.initial_map.any()
    assert not state.current_map.any()
    assert not state.features.any()
    assert not state.observed_mask.any()
    with pytest.raises(FrozenInstanceError):
        state.remaining_budget = 100


@pytest.mark.parametrize("weights", [{}, {"texture": -1}, {"texture": np.nan}, {"entropy": np.inf},
                                          {"edge_densitty": 1}, {"variance": True}, {"contrast": "1"}, [1] * 7])
def test_bad_weights_fail_clearly_even_for_random_policy(weights):
    for name in ("feature", "custom", "random"):
        with pytest.raises(ValueError):
            make_strategy(name, weights)


def test_large_finite_weights_normalize_without_overflow():
    policy = make_strategy("feature", {name: 1e308 for name in FEATURE_NAMES})
    state = state_with_features(np.ones((2, 7)))
    np.testing.assert_allclose(policy.scores(state), 1)


@pytest.mark.parametrize("name", ["unknown", "Feature", None, 2])
def test_unknown_strategy_rejected(name):
    with pytest.raises(ValueError, match="strategy"):
        make_strategy(name)


@pytest.mark.parametrize("kwargs", [
    {"utility_weight": -1}, {"novelty_weight": np.inf}, {"spatial_weight": np.nan},
    {"utility_weight": True}, {"utility_weight": 0, "novelty_weight": 0, "spatial_weight": 0},
])
def test_custom_coefficients_are_validated(kwargs):
    with pytest.raises(ValueError):
        CustomStrategy(**kwargs)


@pytest.mark.parametrize("field,value", [
    ("features", np.ones((2, 6))), ("features", np.full((2, 7), np.nan)),
    ("features", np.full((2, 7), 2.0)), ("observed_mask", np.zeros(2)),
    ("observed_mask", np.zeros(3, dtype=bool)), ("remaining_budget", -1),
    ("current_map", np.zeros((4, 4), dtype=np.float32)), ("history", (0,)),
])
def test_inconsistent_planner_state_is_rejected(field, value):
    image = np.zeros((4, 8), dtype=np.float32)
    arguments = dict(initial_map=image, current_map=image, regions=segment_image(image.shape, 4),
                     features=np.zeros((2, 7)), observed_mask=np.zeros(2, dtype=bool), remaining_budget=2, history=())
    arguments[field] = value
    with pytest.raises(ValueError):
        PlannerState(**arguments)
