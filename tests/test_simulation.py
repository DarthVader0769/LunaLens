"""Executable budget/reconstruction invariants and counterfactual leakage tests."""

from dataclasses import fields
import math

import numpy as np
import pytest

from src.observation_planner import PlannerState, make_strategy
from src.simulator import SimulationConfig, run_simulation

METHODS = ("random", "systematic", "feature", "custom")


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("budget,expected", [(0, 0), (0.9, 0), (2.5, 2), (50, 12)])
def test_uniform_budget_enforcement_and_complete_history(name, budget, expected):
    truth = np.random.default_rng(5).random((12, 16), dtype=np.float32)
    result = run_simulation(truth, name, SimulationConfig(region_size=4, budget=budget))
    assert len(result.selected_ids) == expected
    assert len(set(result.selected_ids)) == expected
    assert len(result.history) == expected + 1
    assert [record["observations"] for record in result.history] == list(range(expected + 1))
    assert result.history[-1]["total_cost"] <= budget
    assert all(record["coverage"] == pytest.approx(record["observations"] / 12) for record in result.history)
    # Exact replacement guarantees nonincreasing squared error; SSIM need not be monotone.
    assert all(after["mse"] <= before["mse"] + 1e-15 for before, after in zip(result.history, result.history[1:]))
    if expected == 12:
        np.testing.assert_array_equal(result.reconstructed_map, truth)
        assert result.history[-1]["mse"] == 0
        assert math.isinf(result.history[-1]["psnr"])


def test_area_cost_can_skip_expensive_tiles_and_observe_a_partial_boundary_region():
    truth = np.random.default_rng(1).random((5, 7), dtype=np.float32)
    result = run_simulation(truth, "systematic", SimulationConfig(region_size=4, cost_mode="area", budget=0.3))
    assert result.selected_ids == (2,)
    assert result.history[-1]["total_cost"] == 0.25
    assert result.history[-1]["coverage"] == pytest.approx(4 / 35)
    full = run_simulation(truth, "feature", SimulationConfig(region_size=4, cost_mode="area", budget=35 / 16))
    np.testing.assert_array_equal(full.reconstructed_map, truth)
    assert full.history[-1]["coverage"] == 1


def test_zero_budget_never_spends_even_on_extremely_small_fractional_cost():
    truth = np.ones((1, 1), dtype=np.float32)
    initial = np.zeros_like(truth)
    none = run_simulation(truth, "systematic", SimulationConfig(region_size=1_000_000, cost_mode="area", budget=0), initial_map=initial)
    assert none.selected_ids == ()
    one = run_simulation(truth, "systematic", SimulationConfig(region_size=1_000_000, cost_mode="area", budget=1e-12), initial_map=initial)
    assert one.selected_ids == (0,)
    np.testing.assert_array_equal(one.reconstructed_map, truth)


@pytest.mark.parametrize("name", METHODS)
def test_full_fractional_budget_does_not_lose_the_last_tile_to_subtraction_roundoff(name):
    # 228 / 25 is represented just below 9.12. Naive budget - spent can leave
    # 0.11999999999999922 for a final tile costing 0.12.
    truth = np.random.default_rng(16).random((6, 38), dtype=np.float32)
    config = SimulationConfig(region_size=5, cost_mode="area", budget=228 / 25, seed=0)
    result = run_simulation(truth, name, config)
    assert len(result.selected_ids) == len(result.regions) == 16
    np.testing.assert_array_equal(result.reconstructed_map, truth)
    assert result.history[-1]["total_cost"] == pytest.approx(config.budget)


def test_reconstruction_replaces_only_selected_pixels_and_preserves_inputs():
    truth = np.random.default_rng(1).random((11, 17), dtype=np.float32)
    initial = np.full_like(truth, 0.2)
    truth_before, initial_before = truth.copy(), initial.copy()
    result = run_simulation(truth, "feature", SimulationConfig(region_size=5, budget=4), initial_map=initial)
    revealed = np.zeros(truth.shape, dtype=bool)
    for region_id in result.selected_ids:
        revealed[result.regions[region_id].slices] = True
    np.testing.assert_array_equal(result.reconstructed_map[revealed], truth[revealed])
    np.testing.assert_array_equal(result.reconstructed_map[~revealed], initial[~revealed])
    np.testing.assert_array_equal(truth, truth_before)
    np.testing.assert_array_equal(initial, initial_before)
    assert result.history[-1]["coverage"] == pytest.approx(revealed.mean())
    expected_mse = np.mean((truth.astype(float) - result.reconstructed_map) ** 2)
    assert result.history[-1]["mse"] == pytest.approx(expected_mse)
    assert result.history[-1]["mse_reduction"] == pytest.approx(result.history[0]["mse"] - expected_mse)


@pytest.mark.parametrize("name", METHODS)
def test_fixed_seed_repeats_decisions_reconstruction_and_metrics(name):
    truth = np.random.default_rng(7).random((20, 24), dtype=np.float32)
    config = SimulationConfig(region_size=4, budget=8, seed=42)
    a, b = run_simulation(truth, name, config), run_simulation(truth, name, config)
    assert a.selected_ids == b.selected_ids
    assert a.history == b.history
    np.testing.assert_array_equal(a.reconstructed_map, b.reconstructed_map)
    if name == "random":
        other = run_simulation(truth, name, SimulationConfig(region_size=4, budget=8, seed=43))
        assert a.selected_ids != other.selected_ids


@pytest.mark.parametrize("name", METHODS)
def test_initial_decision_cannot_depend_on_completely_different_unrevealed_truth(name):
    initial = np.random.default_rng(4).random((16, 24), dtype=np.float32)
    config = SimulationConfig(region_size=4, budget=1, seed=19)
    a = run_simulation(np.zeros_like(initial), name, config, initial_map=initial)
    b = run_simulation(np.ones_like(initial), name, config, initial_map=initial)
    assert a.selected_ids == b.selected_ids
    np.testing.assert_array_equal(a.raw_features, b.raw_features)
    np.testing.assert_array_equal(a.features, b.features)
    np.testing.assert_array_equal(a.initial_scores, b.initial_scores)
    # The hidden maps really produce different private evaluation values.
    assert a.history[0]["mse"] != b.history[0]["mse"]


@pytest.mark.parametrize("name", METHODS)
def test_counterfactual_unrevealed_changes_leave_all_decisions_identical_until_feedback(name):
    rng = np.random.default_rng(20)
    truth = rng.random((16, 24), dtype=np.float32)
    initial = rng.random(truth.shape, dtype=np.float32)
    config = SimulationConfig(region_size=4, budget=3, seed=8)
    reference = run_simulation(truth, name, config, initial_map=initial)
    unobserved = next(region for region in reference.regions if region.id not in reference.selected_ids)
    counterfactual = truth.copy()
    counterfactual[unobserved.slices] = 1 - counterfactual[unobserved.slices]
    comparison = run_simulation(counterfactual, name, config, initial_map=initial)
    assert comparison.selected_ids == reference.selected_ids
    np.testing.assert_array_equal(comparison.reconstructed_map, reference.reconstructed_map)


def test_spy_strategy_sees_only_permitted_pixels_and_immutable_detached_snapshots():
    truth = np.random.default_rng(2).random((8, 12), dtype=np.float32)
    initial = np.full_like(truth, 0.25)
    snapshots = []
    expected_fields = {"initial_map", "current_map", "regions", "features", "observed_mask", "remaining_budget", "history"}

    class SpyStrategy:
        name = "information_boundary_spy"

        def scores(self, state):
            return np.zeros(len(state.regions))

        def choose_next(self, state, rng):
            assert isinstance(state, PlannerState)
            assert {field.name for field in fields(state)} == expected_fields
            assert not hasattr(state, "ground_truth")
            assert not hasattr(state, "metrics")
            assert all(isinstance(region_id, int) for region_id in state.history)
            allowed = initial.copy()
            for region_id in state.history:
                region = state.regions[region_id]
                allowed[region.slices] = truth[region.slices]
            np.testing.assert_array_equal(state.current_map, allowed)
            np.testing.assert_array_equal(state.initial_map, initial)
            for array in (state.initial_map, state.current_map, state.features, state.observed_mask):
                assert not array.flags.writeable
                assert not np.shares_memory(array, truth)
                assert not np.shares_memory(array, initial)
                with pytest.raises(ValueError):
                    array.setflags(write=True)
            snapshots.append(state)
            return len(state.history) if state.remaining_budget >= 1 else None

    result = run_simulation(truth, SpyStrategy(), SimulationConfig(region_size=4, budget=3), initial_map=initial)
    assert result.selected_ids == (0, 1, 2)
    assert [snapshot.history for snapshot in snapshots] == [(), (0,), (0, 1), (0, 1, 2)]
    # Later observations cannot mutate an earlier snapshot retained by a policy.
    np.testing.assert_array_equal(snapshots[0].current_map, initial)
    assert snapshots[0].remaining_budget == 3
    assert not snapshots[0].observed_mask.any()


@pytest.mark.parametrize("selected_id", [-1, 999, 1.0, True, "0"])
def test_simulator_rejects_invalid_strategy_output(selected_id):
    class BadStrategy:
        name = "invalid"

        def scores(self, state):
            return np.zeros(len(state.regions))

        def choose_next(self, state, rng):
            return selected_id

    with pytest.raises(ValueError):
        run_simulation(np.zeros((8, 8), dtype=np.float32), BadStrategy(), SimulationConfig(region_size=4, budget=2))


def test_simulator_independently_rejects_repeat_and_unaffordable_selections():
    class Repeat:
        name = "ignores_constraints"

        def scores(self, state):
            return np.ones(len(state.regions))

        def choose_next(self, state, rng):
            return 0

    truth = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="already observed"):
        run_simulation(truth, Repeat(), SimulationConfig(region_size=4, budget=2))
    with pytest.raises(ValueError, match="unaffordable"):
        run_simulation(truth, Repeat(), SimulationConfig(region_size=4, budget=0.5))


@pytest.mark.parametrize("score", [np.nan, np.inf, "invalid", 1 + 2j])
def test_simulator_rejects_invalid_initial_score_vectors(score):
    class BadScores:
        name = "bad_scores"

        def scores(self, state):
            return np.full(len(state.regions), score)

        def choose_next(self, state, rng):
            return None

    with pytest.raises(ValueError, match="scores"):
        run_simulation(np.zeros((8, 8), dtype=np.float32), BadScores())


@pytest.mark.parametrize("kwargs", [
    {"budget": -1}, {"budget": np.nan}, {"budget": np.inf}, {"budget": True}, {"budget": "2"},
    {"seed": -1}, {"seed": 1.2}, {"seed": True}, {"seed": "2"},
    {"region_size": 0}, {"region_size": 2.5}, {"region_size": True},
    {"downsample_factor": 0}, {"downsample_factor": 1.5},
    {"blur_sigma": -1}, {"blur_sigma": np.inf}, {"cost_mode": "unknown"},
])
def test_invalid_simulation_parameters_fail_before_running(kwargs):
    with pytest.raises(ValueError):
        SimulationConfig(**kwargs)


def test_initial_map_shape_and_strategy_object_parameters_are_validated():
    truth = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        run_simulation(truth, initial_map=np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="strategy object"):
        run_simulation(truth, object())
    with pytest.raises(ValueError, match="constructing"):
        run_simulation(truth, make_strategy("feature"), weights={"texture": 1})


def test_perfect_initial_image_has_defined_zero_improvement_metrics():
    truth = np.full((8, 8), 0.5, dtype=np.float32)
    result = run_simulation(truth, "custom", SimulationConfig(region_size=4, budget=2), initial_map=truth)
    for record in result.history:
        assert record["mse"] == 0
        assert record["mse_reduction"] == 0
        assert record["relative_mse_reduction"] == 0
        assert record["improvement_per_observation"] == 0
        assert record["ssim"] == 1
