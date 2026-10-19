"""Execute the actual Streamlit UI, uploads, simulator and evidence exports.

These tests use Streamlit 1.66's supported AppTest file-uploader interface.
They validate app state and generated download payloads; they do not claim
browser screenshot, responsive layout or OS download-dialog verification.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
import pytest
from streamlit.testing.v1 import AppTest

from app import comparison_archive, comparison_tables, image_digest, run_archive, source_hashes
from src.image_loader import list_sample_images, load_image
from src.simulator import SimulationConfig, run_simulation


ROOT = Path(__file__).resolve().parents[1]


def new_app() -> AppTest:
    app = AppTest.from_file(ROOT / "app.py", default_timeout=40).run()
    assert not app.exception
    app.select_slider(key="max_dimension").set_value(128).run()
    assert not app.exception
    return app


def test_dashboard_configuration_run_exports_and_stale_input_clear():
    app = new_app()
    assert [tab.label for tab in app.tabs] == [
        "Overview", "Image Explorer", "Observation Planner", "Results", "Algorithm Explorer"]
    app.selectbox(key="strategy").select("custom")
    app.selectbox(key="cost_mode").select("area")
    app.select_slider(key="downsample_factor").set_value(8)
    app.slider(key="blur_sigma").set_value(.5)
    app.number_input(key="seed").set_value(41)
    app.number_input(key="weight_texture").set_value(.9)
    app.run()
    app.button(key="run_simulation").click().run()
    assert not app.exception
    result = app.session_state["simulation_result"]
    assert result.strategy_name == "custom"
    assert result.initial_map.shape == (128, 128)
    assert result.config.downsample_factor == 8
    assert result.config.blur_sigma == .5
    assert result.config.seed == 41
    assert result.config.cost_mode == "area"
    assert result.config.budget == 4
    assert len(result.selected_ids) == 4
    assert result.history[-1]["mse"] <= result.history[0]["mse"]
    assert {widget.key for widget in app.download_button} >= {
        "download_history", "download_reconstruction", "download_run_bundle"}
    assert all(widget.proto.url for widget in app.download_button)
    previous_fingerprint = app.session_state["active_input_fingerprint"]
    app.number_input(key="weight_variance").set_value(.8).run()
    assert not app.exception
    assert "simulation_result" not in app.session_state
    assert app.session_state["active_input_fingerprint"] != previous_fingerprint
    assert "download_history" not in {widget.key for widget in app.download_button}
    app.button(key="run_simulation").click().run()
    assert "simulation_result" in app.session_state
    other_image = next(entry["id"] for entry in list_sample_images()
                       if entry["id"] != app.selectbox(key="sample_image").value)
    app.selectbox(key="sample_image").select(other_image).run()
    assert not app.exception
    assert "simulation_result" not in app.session_state


def test_dashboard_comparison_uses_real_shared_budget_runs_and_random_repeats():
    app = new_app()
    app.button(key="compare_methods").click().run()
    assert not app.exception
    results = app.session_state["comparison_results"]
    assert len(results) == 6
    assert {result.strategy_name for result in results} == {"random", "systematic", "feature", "custom"}
    assert [result.config.seed for result in results if result.strategy_name == "random"] == [7, 8, 9]
    for result in results:
        assert result.config.budget == 4
        assert len(result.history) == 5
        assert result.history[-1]["total_cost"] == 4
        np.testing.assert_array_equal(result.initial_map, results[0].initial_map)
    assert {widget.key for widget in app.download_button} >= {
        "download_comparison_metrics", "download_comparison_curve", "download_comparison_bundle"}
    app.select_slider(key="random_repetitions").set_value(1).run()
    assert not app.exception
    assert "comparison_results" not in app.session_state
    app.button(key="compare_methods").click().run()
    assert len(app.session_state["comparison_results"]) == 4
    app.slider(key="budget_percent").set_value(50).run()
    assert not app.exception
    assert "comparison_results" not in app.session_state


@pytest.mark.parametrize("file_format,suffix,mime", [
    ("PNG", "png", "image/png"), ("JPEG", "jpg", "image/jpeg"), ("TIFF", "tiff", "image/tiff")])
def test_dashboard_uploads_and_simulates_png_jpeg_tiff(file_format, suffix, mime):
    app = new_app()
    app.selectbox(key="image_source").select("Upload image").run()
    assert not app.exception
    pixels = np.random.default_rng(23).integers(0, 256, (64, 80), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format=file_format)
    app.file_uploader(key="image_upload").set_value((f"test_image.{suffix}", buffer.getvalue(), mime)).run()
    assert not app.exception
    app.selectbox(key="region_size").select(16).run()
    app.button(key="run_simulation").click().run()
    assert not app.exception
    result = app.session_state["simulation_result"]
    assert result.initial_map.shape == (64, 80)
    assert len(result.selected_ids) == 5
    expected_truth = load_image(buffer.getvalue())
    for region in result.regions:
        if region.id in result.selected_ids:
            np.testing.assert_array_equal(result.reconstructed_map[region.slices], expected_truth[region.slices])
    # Replacing bytes under the same uploaded filename also invalidates a run.
    pixels = 255 - pixels
    replacement = io.BytesIO()
    Image.fromarray(pixels).save(replacement, format=file_format)
    app.file_uploader(key="image_upload").set_value((f"test_image.{suffix}", replacement.getvalue(), mime)).run()
    assert not app.exception
    assert "simulation_result" not in app.session_state


def test_dashboard_invalid_upload_is_readable_and_zero_weights_disable_actions():
    app = new_app()
    for widget in app.number_input:
        if widget.key and widget.key.startswith("weight_"):
            widget.set_value(0.0)
    app.run()
    assert not app.exception
    assert app.button(key="run_simulation").disabled
    assert app.button(key="compare_methods").disabled
    app.selectbox(key="image_source").select("Upload image").run()
    app.file_uploader(key="image_upload").set_value(("broken.png", b"not a valid image", "image/png")).run()
    assert not app.exception
    assert any("could not be prepared" in error.value for error in app.error)


def test_dashboard_source_change_clears_results_and_requires_server_restart():
    app = new_app()
    app.button(key="run_simulation").click().run()
    assert not app.exception
    assert "simulation_result" in app.session_state
    # Simulate a session that loaded an older custom implementation without
    # editing any project source files or relying on automatic module reloads.
    prior_sources = dict(app.session_state["loaded_source_sha256"])
    prior_sources["src/custom_strategy.py"] = "0" * 64
    app.session_state["loaded_source_sha256"] = prior_sources
    app.run()
    assert not app.exception
    assert "simulation_result" not in app.session_state
    assert not app.button
    assert any("src/custom_strategy.py" in error.value and "restart" in error.value for error in app.error)


def test_export_archives_preserve_actual_metrics_configuration_arrays_and_figures():
    reference = np.random.default_rng(8).random((32, 40), dtype=np.float32)
    config = SimulationConfig(region_size=8, budget=5, seed=8)
    result = run_simulation(reference, "feature", config)
    metadata = {"image_sha256": image_digest(reference), "source_sha256": source_hashes()}
    payload = run_archive(result, reference, metadata)
    with ZipFile(io.BytesIO(payload)) as archive:
        exported_config = json.loads(archive.read("configuration.json"))
        assert exported_config["simulation"]["budget"] == config.budget
        assert exported_config["selected_ids"] == list(result.selected_ids)
        assert exported_config["source_sha256"]["src/custom_strategy.py"] == source_hashes()["src/custom_strategy.py"]
        history = pd.read_csv(io.BytesIO(archive.read("history.csv")))
        assert history.iloc[-1].mse == pytest.approx(result.history[-1]["mse"])
        arrays = np.load(io.BytesIO(archive.read("maps_float32.npz")))
        np.testing.assert_array_equal(arrays["reference"], reference)
        np.testing.assert_array_equal(arrays["reconstructed"], result.reconstructed_map)
        for name in ("reconstruction_comparison.png", "selected_regions.png", "performance_mse.png"):
            image = Image.open(io.BytesIO(archive.read(name)))
            assert image.width >= 900
            image.verify()
    results = [result, run_simulation(reference, "random", config)]
    endpoints, summary, histories = comparison_tables(results)
    assert summary.runs.sum() == 2
    assert len(histories) == sum(len(item.history) for item in results)
    with ZipFile(io.BytesIO(comparison_archive(results, metadata))) as archive:
        actual = pd.read_csv(io.BytesIO(archive.read("endpoint_runs.csv")))
        np.testing.assert_allclose(actual.mse, endpoints.mse)
        assert len(json.loads(archive.read("configuration.json"))["runs"]) == 2
        assert archive.read("performance_mse.png").startswith(b"\x89PNG\r\n\x1a\n")
