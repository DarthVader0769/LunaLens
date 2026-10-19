"""Known numerical cases and evaluation integrity checks."""

import numpy as np
import pytest
from skimage.metrics import structural_similarity

from src.evaluation import compute_metrics, evaluate_reconstruction


def test_constant_known_errors_use_unit_data_range():
    truth = np.ones((9, 11), dtype=np.float32)
    reconstruction = np.full_like(truth, 0.5)
    values = compute_metrics(truth, reconstruction)
    assert values["mae"] == 0.5
    assert values["mse"] == 0.25
    assert values["psnr"] == pytest.approx(6.020599913279624)
    expected_ssim = (2 * 0.5 + 0.01**2) / (1.0 + 0.5**2 + 0.01**2)
    assert values["ssim"] == pytest.approx(expected_ssim, abs=1e-6)


def test_exact_reconstruction_has_zero_error_infinite_psnr():
    image = np.random.default_rng(42).random((16, 23), dtype=np.float32)
    values = compute_metrics(image, image.copy())
    assert values == {"mae": 0.0, "mse": 0.0, "psnr": float("inf"), "ssim": 1.0}


@pytest.mark.parametrize("shape", [(1, 1), (2, 20), (20, 2)])
def test_ssim_is_unavailable_only_for_too_small_images(shape):
    values = compute_metrics(np.zeros(shape), np.ones(shape))
    assert np.isnan(values["ssim"])
    assert values["mse"] == 1.0
    assert values["psnr"] == 0.0


@pytest.mark.parametrize("shape,window", [((3, 8), 3), ((4, 7), 3), ((6, 10), 5), ((7, 10), 7)])
def test_ssim_uses_largest_valid_window(shape, window):
    rng = np.random.default_rng(23)
    truth = rng.random(shape, dtype=np.float32)
    reconstruction = rng.random(shape, dtype=np.float32)
    observed = compute_metrics(truth, reconstruction)["ssim"]
    expected = structural_similarity(truth, reconstruction, data_range=1.0, win_size=window)
    assert observed == pytest.approx(expected)


def test_cumulative_improvement_formulas():
    values = evaluate_reconstruction(
        np.ones((8, 8)), np.full((8, 8), 0.5),
        initial_mse=0.5, observations=2, total_cost=1.5, coverage=0.25,
    )
    assert values["mse_reduction"] == 0.25
    assert values["relative_mse_reduction"] == 0.5
    assert values["improvement_per_observation"] == 0.125
    assert values["observations"] == 2
    assert values["total_cost"] == 1.5
    assert values["coverage"] == 0.25


def test_zero_error_and_zero_budget_are_defined():
    image = np.zeros((8, 8))
    values = evaluate_reconstruction(
        image, image, initial_mse=0.0, observations=0, total_cost=0.0, coverage=0.0,
    )
    assert values["mse_reduction"] == 0.0
    assert values["relative_mse_reduction"] == 0.0
    assert values["improvement_per_observation"] == 0.0


@pytest.mark.parametrize("bad", [np.array([[np.nan]]), np.array([[np.inf]]), np.array([[1.01]]), np.array([[-0.01]]), np.zeros((2, 2, 3)), np.array([[1 + 2j]])])
def test_bad_intensities_rejected(bad):
    with pytest.raises(ValueError):
        compute_metrics(bad, bad)


def test_shapes_and_empty_images_rejected():
    with pytest.raises(ValueError):
        compute_metrics(np.zeros((4, 5)), np.zeros((5, 4)))
    with pytest.raises(ValueError):
        compute_metrics(np.zeros((0, 3)), np.zeros((0, 3)))


@pytest.mark.parametrize("override", [{"observations": -1}, {"observations": 1.5}, {"observations": True}, {"coverage": 1.1}, {"initial_mse": -0.1}, {"total_cost": float("inf")}])
def test_bad_history_metadata_rejected(override):
    kwargs = dict(initial_mse=0.0, observations=0, total_cost=0.0, coverage=0.0)
    kwargs.update(override)
    with pytest.raises(ValueError):
        evaluate_reconstruction(np.zeros((4, 4)), np.zeros((4, 4)), **kwargs)


def test_budget_prefix_never_rounds_up():
    from experiments.run_experiments import record_at_budget
    history = [{"observations": count, "total_cost": float(count)} for count in range(11)]
    quarter = record_at_budget(history, 0.25, 10.0)
    assert quarter["observations"] == 2
    assert quarter["budget_fraction"] == 0.25
    assert quarter["actual_budget_fraction"] == 0.2
    assert record_at_budget(history, 1.0, 10.0)["observations"] == 10


def test_aggregation_does_not_pseudoreplicate_random_seeds():
    from experiments.run_experiments import aggregate_images, aggregate_rows
    # Image A has three random seeds, image B only one. The image means are 2
    # and 10; equal image weighting must yield 6, not pooled-run mean 4.
    rows = [
        {"image_id": image_id, "kind": "real", "strategy": "random", "budget_fraction": 0.25, "mse": error}
        for image_id, error in [("A", 1.0), ("A", 2.0), ("A", 3.0), ("B", 10.0)]
    ]
    per_image = aggregate_rows(rows, ("image_id", "kind", "strategy", "budget_fraction"), metrics=("mse",))
    per_kind = aggregate_images(per_image, metrics=("mse",))
    assert [row["n_runs"] for row in per_image] == [3, 1]
    assert per_image[0]["mse_mean"] == 2.0
    assert per_image[1]["mse_sd"] is None
    assert per_kind[0]["mse_mean"] == 6.0
    assert per_kind[0]["n_images"] == 2
    assert per_kind[0]["mse_sd"] == pytest.approx(np.std([2.0, 10.0], ddof=1))


def test_report_json_is_strict_with_documented_infinity_representation(tmp_path):
    import json
    from experiments.run_experiments import write_json
    path = tmp_path / "summary.json"
    write_json(path, {"psnr": np.inf, "ssim": np.nan, "mse": 0.0, "n": np.int64(2)})
    text = path.read_text()
    assert "Infinity" not in text and "NaN" not in text
    assert json.loads(text) == {"psnr": None, "ssim": None, "mse": 0.0, "n": 2}


def test_visualisation_overlay_and_png_export():
    from src.segmentation import segment_image
    from src.visualisation import figure_to_png_bytes, plot_feature_map, values_to_image
    image = np.zeros((5, 7), dtype=np.float32)
    regions = segment_image(image.shape, 4)
    heatmap = values_to_image(image.shape, regions, [0.0, 0.25, 0.5, 1.0])
    assert heatmap[0, 0] == 0.0
    assert heatmap[4, 6] == 1.0
    assert np.isfinite(heatmap).all()
    figure = plot_feature_map(image, regions, [0.0, 0.25, 0.5, 1.0])
    png = figure_to_png_bytes(figure, dpi=60)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 2000


@pytest.mark.parametrize("strategy", ["random", "systematic", "feature", "custom"])
def test_current_strategies_full_scan_prefixes_equal_independent_caps(strategy):
    from src.simulator import SimulationConfig, run_simulation
    image = np.random.default_rng(81).random((16, 16), dtype=np.float32)
    full = run_simulation(image, strategy, SimulationConfig(region_size=4, budget=16, seed=2))
    for cap in (0, 3, 8):
        independent = run_simulation(image, strategy, SimulationConfig(region_size=4, budget=cap, seed=2))
        assert independent.selected_ids == full.selected_ids[:cap]
        for metric in ("mae", "mse", "ssim"):
            assert independent.history[-1][metric] == pytest.approx(full.history[cap][metric])


def test_budget_dependent_custom_policy_cannot_be_evaluated_by_full_scan_prefix():
    from src.simulator import SimulationConfig, run_simulation

    class HorizonAwareExample:
        name = "horizon_aware_test"

        def scores(self, state):
            values = np.arange(len(state.regions), dtype=float)
            if state.remaining_budget <= 3:
                values *= -1
            values[state.observed_mask] = -np.inf
            return values

        def choose_next(self, state, rng):
            if state.remaining_budget < 1 or np.all(state.observed_mask):
                return None
            return int(np.argmax(self.scores(state)))

    image = np.random.default_rng(7).random((8, 8), dtype=np.float32)
    full = run_simulation(image, HorizonAwareExample(), SimulationConfig(region_size=2, budget=16))
    short = run_simulation(image, HorizonAwareExample(), SimulationConfig(region_size=2, budget=3))
    assert full.selected_ids[:3] != short.selected_ids


def test_independent_budget_cli_uses_actual_horizon_for_endpoint_tables(tmp_path, monkeypatch):
    import csv
    from PIL import Image
    import experiments.run_experiments as experiment
    from src.simulator import run_simulation as real_simulation

    class HorizonAwareExample:
        name = "custom"

        def scores(self, state):
            values = np.arange(len(state.regions), dtype=float)
            if state.remaining_budget <= 3:
                values *= -1
            values[state.observed_mask] = -np.inf
            return values

        def choose_next(self, state, rng):
            if state.remaining_budget < 1 or np.all(state.observed_mask):
                return None
            return int(np.argmax(self.scores(state)))

    source = tmp_path / "fixture.png"
    pixels = np.random.default_rng(21).integers(0, 256, (16, 16), dtype=np.uint8)
    Image.fromarray(pixels).save(source)
    monkeypatch.setattr(experiment, "list_sample_images", lambda: [{"id": "fixture", "name": "Fixture", "kind": "synthetic", "path": str(source)}])
    monkeypatch.setattr(experiment, "_generate_figures", lambda *args, **kwargs: [])

    def controlled_simulation(truth, method, *, config, weights, initial_map):
        if method == "custom":
            return real_simulation(truth, HorizonAwareExample(), config=config, initial_map=initial_map)
        return real_simulation(truth, method, config=config, weights=weights, initial_map=initial_map)

    monkeypatch.setattr(experiment, "run_simulation", controlled_simulation)
    output = tmp_path / "executed"
    args = experiment.build_parser().parse_args(["--max-dimension", "16", "--region-size", "4", "--seeds", "0",
                                                "--no-ablations", "--independent-budgets", "--output-dir", str(output)])
    summary = experiment.run_benchmark(args)
    assert summary["baseline_runs"] == 4
    assert summary["independent_endpoint_runs"] == 16
    with (output / "results" / "selection_history.csv").open(newline="") as handle:
        selections = list(csv.DictReader(handle))
    actual_quarter = [int(row["region_id"]) for row in selections if row["strategy"] == "custom" and row["variant"] == "independent_budget_0.25"]
    trajectory = [int(row["region_id"]) for row in selections if row["strategy"] == "custom" and row["variant"] == "baseline"]
    assert actual_quarter == [15, 0, 1, 2]
    assert actual_quarter != trajectory[:4]
    assert (output / "results" / "independent_endpoint_history.csv").is_file()
