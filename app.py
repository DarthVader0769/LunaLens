"""LunaLens: an offline, executable observation-planning research dashboard.

Run with ``streamlit run app.py`` from the repository root.  The UI calls the
same tested simulator as the experiment runner; no results are hard-coded.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import streamlit as st

import src as source_package
from src.evaluation import compute_metrics
from src.custom_strategy import CustomStrategy
from src.feature_extraction import (
    FEATURE_NAMES,
    extract_features,
    normalise_features,
    terrain_proxies,
)
from src.image_loader import list_sample_images, load_image
from src.observation_planner import DEFAULT_WEIGHTS, PlannerState, make_strategy
from src.preprocessing import degrade_image
from src.segmentation import segment_image
from src.simulator import SimulationConfig, SimulationResult, run_simulation


ROOT = Path(__file__).resolve().parent


def source_hashes() -> dict[str, str]:
    """Identify the complete input/planning/evaluation pipeline on disk."""
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "src").glob("*.py"))}


# The imported package survives ordinary Streamlit script reruns and is shared
# by sessions in this server. A new script execution must not silently redefine
# what code was loaded. A session token below additionally catches source edits
# if Streamlit's module watcher reloads this package. Restarting the server is
# the supported way to load edits to a strategy or another pipeline module.
if not hasattr(source_package, "_lunalens_dashboard_loaded_sources"):
    source_package._lunalens_dashboard_loaded_sources = source_hashes()
LOADED_SOURCE_SHA256 = dict(source_package._lunalens_dashboard_loaded_sources)


METHOD_NAMES = {
    "random": "Random selection",
    "systematic": "Systematic grid scan",
    "feature": "Weighted feature selection",
    "custom": "Custom: utility + novelty + dispersion",
}
METHOD_COLORS = {
    "random": "#737b8d",
    "systematic": "#de7c32",
    "feature": "#167aaf",
    "custom": "#9656aa",
}
METRIC_NAMES = {
    "mse": "Mean squared error (lower is better)",
    "mae": "Mean absolute error (lower is better)",
    "psnr": "PSNR, dB (higher is better)",
    "ssim": "SSIM (higher is better)",
    "relative_mse_reduction": "Fraction of initial MSE removed (higher is better)",
    "improvement_per_observation": "MSE reduction per observation (higher is better)",
    "coverage": "Observed pixel coverage",
}


def image_digest(image: np.ndarray) -> str:
    """Hash exactly the standardized image used in the experiment."""
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode())
    digest.update(str(image.dtype).encode())
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def input_fingerprint(image_hash: str, settings: dict[str, Any]) -> str:
    payload = json.dumps({"image_sha256": image_hash, **settings}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@st.cache_data(show_spinner=False, max_entries=8)
def prepare_map(
    reference: np.ndarray,
    region_size: int,
    factor: int,
    sigma: float,
    cost_mode: str,
) -> tuple:
    """Feature extraction is deliberately restricted to the degraded image."""
    initial = degrade_image(reference, factor=factor, blur_sigma=sigma)
    regions = segment_image(reference.shape, region_size, cost_mode=cost_mode)
    raw = extract_features(initial, regions)
    return initial, regions, raw, normalise_features(raw)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_terrain_proxies(initial: np.ndarray) -> dict:
    return terrain_proxies(initial)


def priority_scores(initial, regions, features, weights, budget) -> np.ndarray:
    state = PlannerState(
        initial_map=_readonly(initial),
        current_map=_readonly(initial),
        regions=regions,
        features=_readonly(features),
        observed_mask=_readonly(np.zeros(len(regions), dtype=bool)),
        remaining_budget=float(budget),
        history=(),
    )
    return make_strategy("feature", weights=weights).scores(state)


def feature_table(regions, raw, features, scores=None, selected_ids=()) -> pd.DataFrame:
    order = {region_id: i + 1 for i, region_id in enumerate(selected_ids)}
    rows = []
    for index, region in enumerate(regions):
        record = {
            "region_id": region.id,
            "row": region.row,
            "col": region.col,
            "y0": region.y0,
            "x0": region.x0,
            "height": region.height,
            "width": region.width,
            "cost": region.cost,
            "selected": region.id in order,
            "selection_order": order.get(region.id),
        }
        for column, name in enumerate(FEATURE_NAMES):
            record[f"raw_{name}"] = float(raw[index, column])
            record[f"normalised_{name}"] = float(features[index, column])
        if scores is not None:
            value = float(scores[index])
            record["initial_priority"] = value if np.isfinite(value) else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def _png(figure: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return buffer.getvalue()


def region_figure(image, regions, selected_ids=(), *, title="Observation regions", grid=True) -> bytes:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(image, cmap="gray", vmin=0, vmax=1)
    selected = set(selected_ids)
    for region in regions:
        is_selected = region.id in selected
        if grid or is_selected:
            axis.add_patch(Rectangle(
                (region.x0 - .5, region.y0 - .5), region.width, region.height,
                facecolor="#28c4ce" if is_selected else "none",
                alpha=.22 if is_selected else .45,
                edgecolor="#19aebe" if is_selected else "#f2a541",
                linewidth=1.15 if is_selected else .45,
            ))
    if selected_ids and len(selected_ids) <= 48:
        by_id = {region.id: region for region in regions}
        for order, region_id in enumerate(selected_ids, start=1):
            region = by_id[region_id]
            axis.text(
                region.x0 + (region.width - 1) / 2,
                region.y0 + (region.height - 1) / 2,
                str(order), ha="center", va="center", color="white", fontsize=7,
                bbox={"facecolor": "black", "alpha": .55, "pad": 1, "edgecolor": "none"},
            )
    axis.set_title(title)
    axis.set_axis_off()
    return _png(figure)


def heatmap_figure(shape, regions, values, title, *, normalised=True) -> bytes:
    heatmap = np.full(shape, np.nan, dtype=float)
    for region, value in zip(regions, values):
        heatmap[region.slices] = value if np.isfinite(value) else np.nan
    figure, axis = plt.subplots(figsize=(7, 6))
    kwargs = {"vmin": 0, "vmax": 1} if normalised else {}
    display = axis.imshow(heatmap, cmap="viridis", **kwargs)
    figure.colorbar(display, ax=axis, shrink=.8, label="Normalised feature" if normalised else "Score per unit cost")
    axis.set_title(title)
    axis.set_axis_off()
    return _png(figure)


def reconstruction_figure(reference, initial, reconstructed, title="Reconstruction comparison") -> bytes:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, array, label in zip(
        axes, (reference, initial, reconstructed),
        ("Reference (evaluation only)", "Initial degraded map", "Reconstructed map"),
    ):
        axis.imshow(array, cmap="gray", vmin=0, vmax=1)
        axis.set_title(label)
        axis.set_axis_off()
    figure.suptitle(title)
    return _png(figure)


def performance_figure(results: Iterable[SimulationResult], metric="mse") -> bytes:
    figure, axis = plt.subplots(figsize=(9, 5))
    labels_seen = set()
    for result in results:
        name = result.strategy_name
        label = METHOD_NAMES.get(name, name) if name not in labels_seen else None
        history = pd.DataFrame(result.history)
        values = history[metric].to_numpy(dtype=float)
        # An infinite PSNR is scientifically valid but has no finite plot height.
        values[~np.isfinite(values)] = np.nan
        axis.step(history.total_cost, values, where="post", label=label,
                  color=METHOD_COLORS.get(name), alpha=.55 if name == "random" else 1,
                  linestyle="--" if name == "random" else "-", linewidth=1.8)
        labels_seen.add(name)
    axis.set_xlabel("Cumulative observation cost")
    axis.set_ylabel(METRIC_NAMES[metric])
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    return _png(figure)


def performance_chart(results: Iterable[SimulationResult], metric="mse") -> go.Figure:
    figure = go.Figure()
    for result in results:
        history = pd.DataFrame(result.history)
        name = result.strategy_name
        label = METHOD_NAMES.get(name, name)
        if name == "random":
            label += f" (seed {result.config.seed})"
        values = history[metric].to_numpy(dtype=float)
        values[~np.isfinite(values)] = np.nan
        figure.add_trace(go.Scatter(
            x=history.total_cost, y=values, mode="lines+markers", name=label,
            line={"shape": "hv", "color": METHOD_COLORS.get(name),
                  "dash": "dash" if name == "random" else "solid"},
            opacity=.65 if name == "random" else 1,
            marker={"size": 3}, customdata=history[["observations", "coverage"]],
            hovertemplate="Cost %{x:.3g}<br>Value %{y:.6g}<br>Observations %{customdata[0]}"
                          "<br>Coverage %{customdata[1]:.1%}<extra>%{fullData.name}</extra>",
        ))
    figure.update_layout(
        height=420, xaxis_title="Cumulative observation cost",
        yaxis_title=METRIC_NAMES[metric], margin={"l": 15, "r": 15, "t": 25, "b": 15},
        legend={"orientation": "h", "y": -0.25},
    )
    return figure


def comparison_tables(results: Iterable[SimulationResult]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    endpoints, histories = [], []
    for result in results:
        endpoints.append({"strategy": result.strategy_name, "seed": result.config.seed, **result.history[-1]})
        histories.extend({"strategy": result.strategy_name, "seed": result.config.seed, **row}
                         for row in result.history)
    endpoint_frame = pd.DataFrame(endpoints)
    summary = []
    for name, group in endpoint_frame.groupby("strategy", sort=False):
        row = {"strategy": name, "runs": len(group)}
        for metric in ("observations", "total_cost", "coverage", "mae", "mse", "psnr", "ssim",
                       "relative_mse_reduction", "improvement_per_observation"):
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
        row["mse_seed_sd"] = float(np.std(group.mse.to_numpy(dtype=float), ddof=0))
        summary.append(row)
    return endpoint_frame, pd.DataFrame(summary), pd.DataFrame(histories)


def _array_png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.rint(np.clip(array, 0, 1) * 65535).astype(np.uint16)).save(buffer, format="PNG")
    return buffer.getvalue()


def run_archive(result: SimulationResult, reference: np.ndarray, metadata: dict) -> bytes:
    """Portable run evidence, including exact float arrays and display figures."""
    buffer = io.BytesIO()
    config = {**metadata, "simulation": asdict(result.config), "strategy": result.strategy_name,
              "selected_ids": list(result.selected_ids),
              "array_format": "float32 intensity in [0,1]; PNG maps quantized to uint16"}
    arrays = io.BytesIO()
    np.savez_compressed(arrays, reference=reference, initial=result.initial_map,
                        reconstructed=result.reconstructed_map)
    features = feature_table(result.regions, result.raw_features, result.features,
                             result.initial_scores, result.selected_ids)
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("configuration.json", json.dumps(config, indent=2, allow_nan=False))
        archive.writestr("history.csv", pd.DataFrame(result.history).to_csv(index=False))
        archive.writestr("regions_and_features.csv", features.to_csv(index=False))
        archive.writestr("maps_float32.npz", arrays.getvalue())
        archive.writestr("reconstructed_map.png", _array_png(result.reconstructed_map))
        archive.writestr("reconstruction_comparison.png", reconstruction_figure(
            reference, result.initial_map, result.reconstructed_map, METHOD_NAMES.get(result.strategy_name, result.strategy_name)))
        archive.writestr("selected_regions.png", region_figure(
            result.initial_map, result.regions, result.selected_ids, title="Selected regions on the initial map"))
        archive.writestr("performance_mse.png", performance_figure([result]))
        archive.writestr("README.txt", "Generated from an executed LunaLens simulation.\n"
                         "Reference imagery is available only to the simulator and evaluator.\n"
                         "The planner sees degraded initial features and already observed tiles.\n"
                         "CSV preserves inf PSNR for exact recovery and missing SSIM when undefined.\n"
                         "This is a simplified image-acquisition model, not an orbital mission simulation.\n")
    return buffer.getvalue()


def comparison_archive(results: list[SimulationResult], metadata: dict) -> bytes:
    endpoints, summary, histories = comparison_tables(results)
    buffer = io.BytesIO()
    configs = [{"strategy": result.strategy_name, **asdict(result.config)} for result in results]
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("configuration.json", json.dumps({**metadata, "runs": configs}, indent=2, allow_nan=False))
        archive.writestr("endpoint_runs.csv", endpoints.to_csv(index=False))
        archive.writestr("summary.csv", summary.to_csv(index=False))
        archive.writestr("histories.csv", histories.to_csv(index=False))
        archive.writestr("performance_mse.png", performance_figure(results))
        for result in results:
            archive.writestr(f"maps/{result.strategy_name}_seed_{result.config.seed}.png",
                             _array_png(result.reconstructed_map))
    return buffer.getvalue()


@st.cache_data(show_spinner=False, max_entries=3)
def cached_run_archive(result, reference, metadata):
    return run_archive(result, reference, metadata)


@st.cache_data(show_spinner=False, max_entries=2)
def cached_comparison_archive(results, metadata):
    return comparison_archive(results, metadata)


def _format_metric(value: float, *, percentage=False) -> str:
    if np.isnan(value):
        return "Undefined"
    if np.isposinf(value):
        return "∞ (exact recovery)"
    return f"{value:.1%}" if percentage else f"{value:.6g}"


def _metrics_panel(result: SimulationResult) -> None:
    endpoint = result.history[-1]
    columns = st.columns(4)
    for column, label, key in zip(columns, ("MSE", "MAE", "PSNR (dB)", "SSIM"), ("mse", "mae", "psnr", "ssim")):
        column.metric(label, _format_metric(float(endpoint[key])))
    columns = st.columns(4)
    columns[0].metric("Observations", int(endpoint["observations"]))
    columns[1].metric("Cost used / budget", f"{endpoint['total_cost']:.3g} / {result.config.budget:.3g}")
    columns[2].metric("Observed pixel coverage", _format_metric(float(endpoint["coverage"]), percentage=True))
    columns[3].metric("Initial MSE removed", _format_metric(float(endpoint["relative_mse_reduction"]), percentage=True))
    st.caption(f"MSE reduction per observation: {endpoint['improvement_per_observation']:.6g}. "
               "Initial MSE removed is zero when the initial map already equals the reference.")


def _render_overview(reference, regions, initial, source_info):
    st.subheader("Can a satellite improve a map with fewer observations?")
    st.write("LunaLens tests whether visual features in an initially degraded image help choose "
             "which regions to observe at higher resolution. Each observation replaces one tile "
             "with its reference pixels; the rest of the map retains the initial information.")
    cols = st.columns(3)
    cols[0].metric("Working image", f"{reference.shape[1]} × {reference.shape[0]} px")
    cols[1].metric("Observation regions", len(regions))
    cols[2].metric("Initial map MSE", _format_metric(compute_metrics(reference, initial)["mse"]))
    st.dataframe(pd.DataFrame([
        {"Method": METHOD_NAMES["random"], "Selection rule": "Uniform random draw from affordable, unobserved regions."},
        {"Method": METHOD_NAMES["systematic"], "Selection rule": "Deterministic row-major spatial scan."},
        {"Method": METHOD_NAMES["feature"], "Selection rule": "Weighted initial-image feature utility per unit cost."},
        {"Method": METHOD_NAMES["custom"], "Selection rule": "Feature utility, descriptor novelty and spatial dispersion."},
    ]), hide_index=True)
    st.info("Information boundary: the planner receives the degraded initial map, initial features, "
            "region geometry, cost, budget, observation history and the map revealed so far. "
            "The full-resolution reference and evaluation errors are never passed to a strategy.")
    st.write("Start in **Image Explorer**, configure and run a method in **Observation Planner**, "
             "then execute the equal-budget comparison in **Results**. **Algorithm Explorer** explains "
             "the scoring and the editable custom strategy.")
    st.caption("This is a software research model of image acquisition. It does not model orbits, "
               "illumination changes, instrument noise, pointing, communication or spacecraft operations. "
               "No strategy is assumed to outperform the others.")


def _render_explorer(reference, initial, regions, raw_features, features, source_info):
    st.subheader(source_info["name"])
    kind = str(source_info.get("kind", "user upload"))
    if "synthetic" in kind.lower():
        st.warning("Synthetic lunar-like texture. This image is procedurally generated and is not NASA imagery.")
    elif kind == "user upload":
        st.info("User-provided image. LunaLens has not verified its provenance or scientific calibration.")
    else:
        st.info(f"Bundled real lunar imagery · {kind}. The working image may be a crop or resized derivative.")
    details = [source_info.get(key) for key in ("mission", "instrument", "credit") if source_info.get(key)]
    if details:
        st.caption(" · ".join(str(value) for value in details))
    if source_info.get("source_url"):
        st.markdown(f"[Original image source and attribution]({source_info['source_url']})")
    if source_info.get("description"):
        st.caption(str(source_info["description"]))
    columns = st.columns(2)
    columns[0].image(reference, caption="Reference image — evaluation and simulated reveals only", clamp=True, width="stretch")
    columns[1].image(initial, caption="Initial degraded image — available to the planner", clamp=True, width="stretch")
    if st.checkbox("Show observation grid", key="show_grid"):
        st.image(region_figure(initial, regions), width="stretch")
    st.subheader("Features extracted from the initial map")
    selected_feature = st.selectbox("Feature heatmap", list(FEATURE_NAMES), key="feature_heatmap")
    index = FEATURE_NAMES.index(selected_feature)
    figure = heatmap_figure(initial.shape, regions, features[:, index],
                            f"Initial-map {selected_feature.replace('_', ' ')}")
    st.image(figure, width="stretch")
    st.caption("Each colour represents one region. Features are min-max scaled across this initial "
               "image; a constant feature is zero. Colours cannot be compared as physical quantities "
               "across images. Raw feature values are included in the CSV.")
    columns = st.columns(2)
    columns[0].download_button("Download features (CSV)",
                              feature_table(regions, raw_features, features).to_csv(index=False),
                              "lunalens_features.csv", "text/csv", key="download_features", on_click="ignore")
    columns[1].download_button("Download heatmap (PNG)", figure, "lunalens_feature_heatmap.png",
                              "image/png", key="download_heatmap", on_click="ignore")
    with st.expander("Inspect raw and normalised feature values"):
        st.dataframe(feature_table(regions, raw_features, features), hide_index=True)
    if st.checkbox("Show exploratory terrain proxies", key="show_terrain"):
        with st.spinner("Computing image-derived masks and circular candidates…"):
            proxies = cached_terrain_proxies(initial)
        st.warning("These are visual proxies, not confirmed craters, measured slope, elevation or physical hazards. "
                   "Dark pixels may be shadows or low albedo; circular edges can be non-crater structures.")
        columns = st.columns(3)
        for column, key, title in zip(columns,
            ("shadow_mask", "boundary_mask", "rough_texture_mask"),
            ("Dark-pixel / shadow proxy", "Sharp boundary proxy", "Rough-looking texture proxy")):
            column.image(np.asarray(proxies[key], dtype=np.uint8) * 255, caption=title, width="stretch")
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.imshow(initial, cmap="gray", vmin=0, vmax=1)
        for circle in proxies.get("candidate_circles", []):
            axis.add_patch(Circle((circle["x"], circle["y"]), circle["radius"],
                                  fill=False, edgecolor="#29d0d0", linewidth=1))
        axis.set_title(f"Circular-edge candidates: {len(proxies.get('candidate_circles', []))}")
        axis.set_axis_off()
        st.image(_png(figure), width="stretch")
        st.caption("All proxies above use only the initial degraded map. Candidate counts are exploratory, "
                   "not an inventory validated by a planetary scientist.")


def _render_algorithms():
    st.subheader("What the planner is allowed to know")
    st.write("The simulator keeps the full-resolution image private. It creates a read-only state "
             "containing the initial map, the reconstruction so far, region descriptors, observation "
             "status, remaining budget and selected region IDs. Strategies return an affordable "
             "unobserved region ID. The simulator independently validates every choice.")
    st.subheader("The seven initial-image features")
    st.dataframe(pd.DataFrame([
        {"Feature": "Variance", "Definition": "Population variance of pixel intensities", "Intuition": "Intensity spread may indicate unresolved structure."},
        {"Feature": "Edge density", "Definition": "Fraction of Sobel gradient magnitudes > 0.04", "Intuition": "Many image boundaries can signal visual complexity."},
        {"Feature": "Gradient", "Definition": "Mean Sobel magnitude; kernels scaled by 1/8", "Intuition": "Strong intensity transitions may benefit from detail."},
        {"Feature": "Texture", "Definition": "RMS residual after Gaussian smoothing at σ = 1 px", "Intuition": "High-frequency appearance can indicate intricate texture."},
        {"Feature": "Entropy", "Definition": "Shannon entropy of a fixed 32-bin intensity histogram", "Intuition": "A broad intensity distribution has higher entropy."},
        {"Feature": "Contrast", "Definition": "95th minus 5th intensity percentile", "Intuition": "Tonal separation can highlight local structure."},
        {"Feature": "Information", "Definition": "(entropy / 5) × (contrast + edge density) / 2", "Intuition": "A correlated composite heuristic, not measured future information gain."},
    ]), hide_index=True)
    st.latex(r"z_{ij}=\begin{cases}(f_{ij}-\min_k f_{kj})/(\max_k f_{kj}-\min_k f_{kj}) & \max f_j>\min f_j\\0 & \mathrm{otherwise}\end{cases}")
    st.subheader("A · Random selection")
    st.write("Draw uniformly among affordable, unobserved regions with a seeded NumPy generator. "
             "The seed makes a run repeatable. Use multiple seeds to see how much luck changes the outcome.")
    st.subheader("B · Systematic grid scan")
    st.write("Visit tiles from left to right, then top to bottom. When a region is unaffordable, "
             "scan onward for an affordable one. This baseline is deterministic and depends on image orientation.")
    st.subheader("C · Weighted feature selection")
    st.latex(r"u_i=\sum_j \frac{w_j}{\sum_k w_k}z_{ij},\qquad s_i=\frac{u_i}{c_i}")
    st.write("Compute each descriptor from the initial map, scale each feature across regions, form "
             "a nonnegative weighted average, and divide by observation cost. Select the highest "
             "affordable score; ties use the lowest region ID. The baseline scores stay fixed, "
             "because no new evidence about unrevealed tiles becomes available.")
    st.subheader("D · Editable custom strategy")
    st.latex(r"s_i=\frac{0.60u_i+0.25n_i+0.15d_i}{c_i}")
    st.write("Novelty nᵢ is the nearest distance to an observed tile's initial feature vector, "
             "divided by √7. Dispersion dᵢ is the nearest distance to an observed tile's centroid, "
             "divided by the image diagonal. Both are zero before the first observation. Recompute "
             "these terms after each selection so the plan reacts to what has already been selected.")
    with st.expander("Read the actual custom scoring implementation"):
        st.code(inspect.getsource(CustomStrategy.scores), language="python")
    st.caption("This code is read from the working src/custom_strategy.py module. Keep scores(state) "
               "and choose_next(state, rng) compatible when editing it. The constructor exposes "
               "utility_weight, novelty_weight and spatial_weight; inherited choose_next applies "
               "affordability, observation-status and deterministic tie rules.")
    st.info("After editing src/custom_strategy.py or another src module, stop and restart the Streamlit "
            "server before rerunning experiments. LunaLens records source hashes and blocks the current "
            "session when source files change, so old measurements cannot be attributed to an edited algorithm.")
    st.subheader("Where these methods can fail")
    st.write("Visible low-resolution complexity need not predict missing high-resolution detail. "
             "A tile can look smooth after degradation while concealing fine texture; another can "
             "look striking but already be represented accurately. Histogram-based features are "
             "correlated, and diversity may spend budget on visually different tiles with little "
             "reconstruction error. The experiment measures these consequences rather than assuming "
             "that a high priority score equals useful information gain.")
    st.write("Your strongest contribution can be a precise utility hypothesis, an improved acquisition "
             "or observation-cost model, and a held-out experiment that might disprove your hypothesis. "
             "Change one component at a time, keep budgets matched, repeat random seeds, and inspect "
             "failures on real and synthetic images separately.")


def main() -> None:
    st.set_page_config(page_title="LunaLens · Observation Planning", page_icon="🌙", layout="wide")
    st.title("LunaLens")
    st.caption("An AI that knows where a satellite should look next? An executable experiment, with measured answers.")
    current_source_hashes = source_hashes()
    loaded_sources = st.session_state.setdefault("loaded_source_sha256", LOADED_SOURCE_SHA256)
    if loaded_sources != current_source_hashes:
        changed = sorted(name for name in set(loaded_sources) | set(current_source_hashes)
                         if loaded_sources.get(name) != current_source_hashes.get(name))
        for key in ("simulation_result", "comparison_results", "comparison_fingerprint", "active_input_fingerprint"):
            st.session_state.pop(key, None)
        st.error("Source files changed while this app was running: " + ", ".join(changed) + ". "
                 "Previous results were cleared. Stop this Streamlit server and restart it with "
                 "`streamlit run app.py` before continuing, so the edited algorithm modules are loaded.")
        st.stop()
    with st.sidebar:
        st.header("Image and map settings")
        source = st.selectbox("Image source", ("Bundled sample", "Upload image"), key="image_source")
        sample_entries = list_sample_images()
        if source == "Bundled sample":
            if not sample_entries:
                st.error("No bundled images found. Select Upload image, or restore data/sample_images and its manifest.")
                st.stop()
            by_id = {str(entry["id"]): entry for entry in sample_entries}
            sample_id = st.selectbox("Sample image", list(by_id),
                                     format_func=lambda key: f"{by_id[key]['name']} · {by_id[key].get('kind', 'sample')}",
                                     key="sample_image")
            source_info = dict(by_id[sample_id])
            image_path = Path(source_info["path"])
            image_source = image_path if image_path.is_absolute() else ROOT / image_path
            source_key = str(sample_id)
        else:
            upload = st.file_uploader("Upload PNG, JPEG or TIFF", type=["png", "jpg", "jpeg", "tif", "tiff"], key="image_upload")
            if upload is None:
                st.info("Choose an image to begin. Uploaded imagery stays within this running app.")
                st.stop()
            if upload.size > 32 * 1024 * 1024:
                st.error("Please use an image smaller than 32 MB.")
                st.stop()
            image_source = upload.getvalue()
            source_key = upload.name
            source_info = {"id": "upload", "name": upload.name, "kind": "user upload", "credit": "User-provided image"}
        max_dimension = st.select_slider("Maximum working image dimension (px)", (128, 256, 384, 512, 768), value=512, key="max_dimension")
        region_size = st.selectbox("Region size (px)", (8, 16, 32, 64, 128), index=2, key="region_size")
        factor = st.select_slider("Downsample factor", (1, 2, 4, 8, 16), value=4, key="downsample_factor")
        sigma = st.slider("Gaussian blur σ (px)", 0.0, 4.0, 1.0, .25, key="blur_sigma")
        cost_mode = st.selectbox("Observation cost model", ("uniform", "area"),
                                format_func=lambda mode: "One unit per region" if mode == "uniform" else "Proportional to tile area",
                                key="cost_mode")
        st.caption("Working images are capped at 768 px on the longest side. Edge tiles retain "
                   "their actual dimensions. All processing runs locally without network requests.")
    try:
        reference = load_image(image_source, max_dimension=max_dimension)
        with st.spinner("Preparing the degraded map and extracting initial-image features…"):
            initial, regions, raw_features, features = prepare_map(reference, region_size, factor, sigma, cost_mode)
    except (ValueError, OSError, TypeError) as error:
        st.error(f"The image could not be prepared: {error}")
        st.stop()
    digest = image_digest(reference)
    total_cost = float(sum(region.cost for region in regions))
    overview, explorer, planner, results_tab, algorithms = st.tabs(
        ["Overview", "Image Explorer", "Observation Planner", "Results", "Algorithm Explorer"])
    with overview:
        _render_overview(reference, regions, initial, source_info)
    with explorer:
        _render_explorer(reference, initial, regions, raw_features, features, source_info)
    with planner:
        st.subheader("Configure and run a real observation plan")
        strategy = st.selectbox("Observation strategy", ["feature", "random", "systematic", "custom"],
                                format_func=lambda method: METHOD_NAMES[method], key="strategy")
        columns = st.columns(2)
        budget_percent = columns[0].slider("Observation budget (% of total cost)", 0, 100, 25, 5, key="budget_percent")
        seed = int(columns[1].number_input("Random seed", min_value=0, max_value=1_000_000_000, value=7, step=1, key="seed"))
        with st.expander("Feature weights — used by feature and custom methods", expanded=False):
            columns = st.columns(4)
            weights = {name: float(columns[index % 4].number_input(
                name.replace("_", " ").title(), min_value=0.0, max_value=5.0,
                value=float(DEFAULT_WEIGHTS[name]), step=.05, key=f"weight_{name}"))
                for index, name in enumerate(FEATURE_NAMES)}
            st.caption("Weights are normalised to sum to one. These controls change the same feature "
                       "weights for the single run and all compared methods. Set at least one weight above zero.")
        budget = total_cost * budget_percent / 100
        config = SimulationConfig(region_size=region_size, downsample_factor=factor, blur_sigma=sigma,
                                  budget=budget, seed=seed, cost_mode=cost_mode)
        settings = {"source": source_key, "max_dimension": max_dimension, "strategy": strategy,
                    "source_sha256": current_source_hashes,
                    "weights": weights, "budget_percent": budget_percent, **asdict(config)}
        fingerprint = input_fingerprint(digest, settings)
        previous = st.session_state.get("active_input_fingerprint")
        if previous is not None and previous != fingerprint:
            st.session_state.pop("simulation_result", None)
            st.session_state.pop("comparison_results", None)
            st.session_state.pop("comparison_fingerprint", None)
            st.caption("Inputs changed. Previous simulation and comparison results were cleared.")
        st.session_state["active_input_fingerprint"] = fingerprint
        metadata = {"image": {key: str(value) for key, value in source_info.items()},
                    "image_sha256": digest, "working_shape": list(reference.shape), "settings": settings,
                    "source_sha256": {**current_source_hashes,
                                      "app.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
        st.write(f"**{len(regions)} regions · {budget:.3g} cost units available out of {total_cost:.3g} "
                 f"({budget_percent}%) · {METHOD_NAMES[strategy]}**")
        st.caption("The budget is a hard upper bound. A fractional remainder may be unused when no "
                   "remaining tile is affordable. Pixel coverage can differ from budget fraction for edge tiles.")
        valid_weights = sum(weights.values()) > 0
        if not valid_weights:
            st.warning("Set at least one feature weight above zero to run the simulation or comparison.")
        if valid_weights:
            scores = priority_scores(initial, regions, features, weights, budget)
            with st.expander("Preview weighted feature priority on the initial map"):
                st.image(heatmap_figure(initial.shape, regions, scores,
                                        "Weighted feature score per cost (initial map)", normalised=False), width="stretch")
                st.caption("This is the feature baseline's priority, before any observations. Random selection "
                           "and systematic scanning do not use these scores. Unaffordable tiles may be blank.")
        if len(regions) > 1024:
            st.caption("This configuration has many small regions. Full-budget comparisons can take several "
                       "minutes because image-quality metrics are evaluated after every observation.")
        if st.button("Run simulation", type="primary", key="run_simulation", disabled=not valid_weights):
            with st.spinner("Selecting regions, revealing observations and evaluating every reconstruction…"):
                result = run_simulation(reference, strategy=strategy, config=config, weights=weights, initial_map=initial)
            st.session_state["simulation_result"] = result
            st.success(f"Completed {len(result.selected_ids)} observations. Open Results for metrics and exports.")
        result = st.session_state.get("simulation_result")
        if result is not None:
            st.image(region_figure(initial, result.regions, result.selected_ids,
                                    title="Selected observations on the initial map"), width="stretch")
            st.caption("Cyan regions received high-resolution reveals. Numbers show observation order when "
                       "48 or fewer regions were selected. Unselected regions remain at initial quality.")
    with results_tab:
        st.subheader("Current simulation")
        if result is None:
            st.info("Run a simulation in Observation Planner to see reconstruction metrics and export its results. "
                    "You can also run the four-method comparison below directly.")
        else:
            _metrics_panel(result)
            run_figure = reconstruction_figure(reference, initial, result.reconstructed_map, METHOD_NAMES[strategy])
            st.image(run_figure, width="stretch")
            metric = st.selectbox("Simulation curve metric", list(METRIC_NAMES), format_func=lambda key: METRIC_NAMES[key], key="run_metric")
            st.plotly_chart(performance_chart([result], metric), width="stretch", key="single_performance")
            if metric == "psnr":
                st.caption("Infinite PSNR means exact recovery and has no finite plot height; those points are omitted from the curve.")
            with st.expander("Per-observation measurements"):
                st.dataframe(pd.DataFrame(result.history), hide_index=True)
            columns = st.columns(3)
            columns[0].download_button("Download history (CSV)", pd.DataFrame(result.history).to_csv(index=False),
                                      "lunalens_history.csv", "text/csv", key="download_history", on_click="ignore")
            columns[1].download_button("Download reconstruction figure (PNG)", run_figure,
                                      "lunalens_reconstruction.png", "image/png", key="download_reconstruction", on_click="ignore")
            with st.spinner("Preparing the reproducible run export…"):
                archive = cached_run_archive(result, reference, metadata)
            columns[2].download_button("Download run bundle (ZIP)", archive, "lunalens_run.zip",
                                      "application/zip", key="download_run_bundle", on_click="ignore")
            st.caption("The run bundle includes configuration, image hash, selected regions, raw and normalised "
                       "features, every metric, exact float32 maps, 16-bit reconstruction PNG and publication-size figures.")
        st.divider()
        st.subheader("Compare all four methods under the same budget")
        repeats = st.select_slider("Random repetitions", (1, 3, 5, 10), value=3, key="random_repetitions")
        st.caption(f"The deterministic methods run once each. Random selection runs {repeats} time(s), "
                   f"with consecutive seeds starting at {seed}. All runs share this image, degradation, "
                   "tile geometry, weights, cost model and budget. No weights are tuned using these results.")
        comparison_fingerprint = f"{fingerprint}:{repeats}"
        if st.session_state.get("comparison_fingerprint") != comparison_fingerprint:
            st.session_state.pop("comparison_results", None)
        if st.button("Run equal-budget comparison", type="primary", key="compare_methods", disabled=not valid_weights):
            jobs = [(method, seed) for method in ("systematic", "feature", "custom")]
            jobs += [("random", seed + repeat) for repeat in range(repeats)]
            completed = []
            progress = st.progress(0, text="Starting actual simulations…")
            for index, (method, run_seed) in enumerate(jobs):
                progress.progress(index / len(jobs), text=f"Running {METHOD_NAMES[method]} (seed {run_seed}) · {index + 1}/{len(jobs)}")
                completed.append(run_simulation(reference, strategy=method,
                    config=replace(config, seed=run_seed), weights=weights, initial_map=initial))
            progress.progress(1.0, text=f"Completed {len(jobs)} simulations.")
            st.session_state["comparison_results"] = completed
            st.session_state["comparison_fingerprint"] = comparison_fingerprint
        comparison = st.session_state.get("comparison_results")
        if comparison:
            endpoints, summary, histories = comparison_tables(comparison)
            display = summary.rename(columns={
                "strategy": "Method", "runs": "Runs", "observations_mean": "Observations (mean)",
                "total_cost_mean": "Cost (mean)", "coverage_mean": "Coverage (mean)",
                "mae_mean": "MAE (mean)", "mse_mean": "MSE (mean)", "mse_seed_sd": "MSE seed SD",
                "psnr_mean": "PSNR (mean)", "ssim_mean": "SSIM (mean)",
                "relative_mse_reduction_mean": "Initial MSE removed (mean)",
                "improvement_per_observation_mean": "MSE reduction / observation (mean)",
            })
            display["Method"] = display["Method"].map(METHOD_NAMES)
            st.dataframe(display, hide_index=True, column_config={
                column: st.column_config.NumberColumn(format="%.6g")
                for column in display.columns if column not in ("Method", "Runs")})
            st.caption("Each random seed is a separate run; the summary reports seed means and population "
                       "standard deviation of MSE. Deterministic methods have one run, so their displayed "
                       "seed SD is zero. This comparison describes one input and is not an estimate of "
                       "performance across lunar terrains.")
            feature_mse = float(summary.loc[summary.strategy == "feature", "mse_mean"].iloc[0])
            random_mse = float(summary.loc[summary.strategy == "random", "mse_mean"].iloc[0])
            relation = "equal to" if np.isclose(feature_mse, random_mse, rtol=1e-6, atol=1e-12) else "lower than" if feature_mse < random_mse else "higher than"
            st.write(f"On this image and budget, weighted feature MSE is **{feature_mse:.6g}**, "
                     f"{relation} the random-seed mean of **{random_mse:.6g}**.")
            comparison_metric = st.selectbox("Comparison curve metric", list(METRIC_NAMES),
                format_func=lambda key: METRIC_NAMES[key], key="comparison_metric")
            st.plotly_chart(performance_chart(comparison, comparison_metric), width="stretch", key="comparison_performance")
            st.caption("Curves show every observed state. Each dashed random curve is one seed. "
                       "Steps represent discrete observations; lines are not interpolated future measurements.")
            if comparison_metric == "psnr":
                st.caption("Infinite PSNR is shown as exact recovery in tables and omitted from the finite-height plot.")
            with st.expander("Reconstructed maps for each method", expanded=True):
                for index, method in enumerate(("random", "systematic", "feature", "custom")):
                    if index % 2 == 0:
                        columns = st.columns(2)
                    representative = next(item for item in comparison if item.strategy_name == method)
                    caption = METHOD_NAMES[method]
                    if method == "random":
                        caption += f" · representative seed {representative.config.seed}"
                    columns[index % 2].image(representative.reconstructed_map, caption=caption, clamp=True, width="stretch")
            with st.expander("Individual run endpoint measurements"):
                st.dataframe(endpoints, hide_index=True)
            columns = st.columns(3)
            columns[0].download_button("Download comparison metrics (CSV)", endpoints.to_csv(index=False),
                                      "lunalens_comparison_metrics.csv", "text/csv", key="download_comparison_metrics", on_click="ignore")
            columns[1].download_button("Download comparison curve (PNG)", performance_figure(comparison, comparison_metric),
                                      "lunalens_comparison_curve.png", "image/png", key="download_comparison_curve", on_click="ignore")
            with st.spinner("Preparing comparison figures and numeric exports…"):
                archive = cached_comparison_archive(comparison, metadata)
            columns[2].download_button("Download comparison bundle (ZIP)", archive, "lunalens_comparison.zip",
                                      "application/zip", key="download_comparison_bundle", on_click="ignore")
            st.caption("Comparison bundle: all histories, individual endpoints, summary statistics, exact "
                       "run configurations, MSE comparison PNG and 16-bit reconstruction PNGs for every run.")
    with algorithms:
        _render_algorithms()


if __name__ == "__main__":
    main()
