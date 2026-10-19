"""Readable, exportable figures generated from measured arrays and histories.

Functions return Matplotlib ``Figure`` objects without creating pyplot-managed
windows. The same figures can be displayed by Streamlit or exported to PNG.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


METHOD_LABELS = {
    "random": "Random", "systematic": "Systematic", "feature": "Feature weighted",
    "custom": "Custom diversity",
}
METHOD_COLORS = {
    "random": "#727c8f", "systematic": "#d88432", "feature": "#008f98", "custom": "#8856ba",
}


def _new_figure(*, width: float = 10.0, height: float = 5.0) -> Figure:
    figure = Figure(figsize=(width, height), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _show_image(axis, array: np.ndarray, title: str) -> None:
    axis.imshow(array, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    axis.set_title(title, fontsize=11, pad=9)
    axis.set_axis_off()


def image_comparison(
    truth: np.ndarray,
    initial: np.ndarray,
    reconstruction: np.ndarray | None = None,
    *,
    title: str = "Ground truth and initially available map",
) -> Figure:
    """Compare images on one fixed grayscale scale, without contrast stretching."""
    count = 2 if reconstruction is None else 3
    figure = _new_figure(width=5 * count, height=5.2)
    axes = figure.subplots(1, count)
    _show_image(axes[0], truth, "Reference image (evaluation only)")
    _show_image(axes[1], initial, "Initial degraded map (planner input)")
    if reconstruction is not None:
        _show_image(axes[2], reconstruction, "Map after selected observations")
    figure.suptitle(title, fontsize=14)
    return figure


def plot_segmentation(image: np.ndarray, regions: Sequence, *, title: str = "Observation regions") -> Figure:
    figure = _new_figure(width=7.5, height=7.5)
    axis = figure.subplots()
    _show_image(axis, image, f"{len(regions)} regions; boundary tiles retained")
    for region in regions:
        axis.add_patch(Rectangle((region.x0 - 0.5, region.y0 - 0.5), region.width, region.height,
                                 fill=False, linewidth=0.45, edgecolor="#22d0d1", alpha=0.85))
    figure.suptitle(title, fontsize=14)
    return figure


def values_to_image(shape: tuple[int, int], regions: Sequence, values: Iterable[float]) -> np.ndarray:
    """Expand one scalar per tile into its exact pixel footprint."""
    array = np.asarray(list(values), dtype=float)
    if array.shape != (len(regions),):
        raise ValueError("Exactly one scalar is required per observation region.")
    heatmap = np.full(shape, np.nan, dtype=float)
    for region, value in zip(regions, array):
        heatmap[region.slices] = value
    return heatmap


def plot_feature_map(
    image: np.ndarray,
    regions: Sequence,
    values: Iterable[float],
    *,
    title: str = "Initial-map feature values",
    colorbar_label: str = "Normalised descriptor (0–1)",
    cmap: str = "viridis",
) -> Figure:
    figure = _new_figure(width=8.3, height=7)
    axis = figure.subplots()
    artist = axis.imshow(values_to_image(image.shape, regions, values), cmap=cmap, interpolation="nearest")
    axis.set_axis_off()
    axis.set_title("Computed from the initial degraded map only", fontsize=10)
    figure.colorbar(artist, ax=axis, shrink=0.8, label=colorbar_label)
    figure.suptitle(title, fontsize=14)
    return figure


def plot_priority_map(image: np.ndarray, regions: Sequence, scores: Iterable[float], *, title: str = "Initial observation priority") -> Figure:
    return plot_feature_map(image, regions, scores, title=title,
                            colorbar_label="Initial feature utility per observation cost", cmap="magma")


def _overlay_selected(axis, image: np.ndarray, regions: Sequence, selected_ids: Sequence[int], title: str) -> None:
    _show_image(axis, image, title)
    selected = set(selected_ids)
    for region in regions:
        if region.id in selected:
            axis.add_patch(Rectangle((region.x0 - 0.5, region.y0 - 0.5), region.width, region.height,
                                     linewidth=0.55, edgecolor="#29e4cf", facecolor="#29e4cf", alpha=0.27))


def plot_selected_regions(
    image: np.ndarray, regions: Sequence, selected_ids: Sequence[int], *, title: str = "Selected observation regions"
) -> Figure:
    figure = _new_figure(width=7.5, height=7.5)
    axis = figure.subplots()
    _overlay_selected(axis, image, regions, selected_ids, f"{len(selected_ids)} observed regions (teal)")
    figure.suptitle(title, fontsize=14)
    return figure


def plot_selection_comparison(
    image: np.ndarray, regions: Sequence, selections: Mapping[str, Sequence[int]], *, title: str
) -> Figure:
    figure = _new_figure(width=11.5, height=11.0)
    axes = np.asarray(figure.subplots(2, 2)).ravel()
    for axis, (method, selected) in zip(axes, selections.items()):
        _overlay_selected(axis, image, regions, selected, f"{METHOD_LABELS.get(method, method)} · {len(selected)} regions")
    for axis in axes[len(selections):]:
        axis.set_axis_off()
    figure.suptitle(title, fontsize=15)
    return figure


def plot_reconstruction_comparison(
    truth: np.ndarray, initial: np.ndarray, reconstructions: Mapping[str, np.ndarray], *, title: str
) -> Figure:
    figure = _new_figure(width=15, height=10.4)
    axes = np.asarray(figure.subplots(2, 3)).ravel()
    _show_image(axes[0], truth, "Reference image (evaluation only)")
    _show_image(axes[1], initial, "Initial degraded map")
    for axis, (method, reconstruction) in zip(axes[2:], reconstructions.items()):
        _show_image(axis, reconstruction, METHOD_LABELS.get(method, method))
    figure.suptitle(title, fontsize=15)
    return figure


def plot_performance_curves(
    records: Sequence[Mapping], *, metric: str = "mse", x_key: str = "budget_fraction",
    title: str = "Reconstruction error versus observation budget", show_spread: bool = False,
) -> Figure:
    """Plot aggregates whose random seeds were already averaged within image.

    If enabled, shaded bands show ± one sample SD *across images*, not a
    confidence interval and not variation over deterministic reruns.
    """
    kinds = list(dict.fromkeys(str(row["kind"]) for row in records))
    figure = _new_figure(width=max(7.0, 6.2 * len(kinds)), height=5.3)
    axes = np.atleast_1d(figure.subplots(1, len(kinds)))
    labels = {"mse": "Mean squared error (lower is better)", "mae": "Mean absolute error",
              "ssim": "SSIM", "relative_mse_reduction": "Reduction in initial MSE (%)",
              "improvement_per_observation": "Cumulative MSE gain / observation"}
    for axis, kind in zip(axes, kinds):
        subset = [row for row in records if row["kind"] == kind]
        for method in METHOD_LABELS:
            rows = sorted((row for row in subset if row["strategy"] == method), key=lambda row: row[x_key])
            if not rows:
                continue
            scale_x = 100.0 if x_key == "budget_fraction" else 1.0
            scale_y = 100.0 if metric == "relative_mse_reduction" else 1.0
            x = np.asarray([row[x_key] for row in rows], dtype=float) * scale_x
            y = np.asarray([row[f"{metric}_mean"] for row in rows], dtype=float) * scale_y
            axis.plot(x, y, label=METHOD_LABELS[method], color=METHOD_COLORS[method], linewidth=1.9)
            if show_spread:
                std = np.asarray([row.get(f"{metric}_sd") or 0.0 for row in rows], dtype=float) * scale_y
                axis.fill_between(x, y - std, y + std, color=METHOD_COLORS[method], alpha=0.08)
        n_images = max((int(row.get("n_images", 1)) for row in subset), default=0)
        axis.set_title(f"{'Real lunar imagery' if kind == 'real' else 'Synthetic imagery'} · {n_images} images")
        axis.set_xlabel("Allowed observation budget (% of full scan)" if x_key == "budget_fraction" else "Mean observation count per image")
        axis.set_ylabel(labels.get(metric, metric))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    figure.suptitle(title, fontsize=14)
    return figure


def _formatted(value: float | None, digits: int = 4) -> str:
    if value is None or np.isnan(float(value)):
        return "—"
    if np.isposinf(float(value)):
        return "∞"
    return f"{float(value):.{digits}f}"


def plot_metrics_table(records: Sequence[Mapping], *, title: str, footnote: str = "") -> Figure:
    figure = _new_figure(width=14.0, height=max(4.1, 0.45 * len(records) + 2.2))
    axis = figure.subplots()
    axis.set_axis_off()
    columns = ["Dataset", "Method", "Images", "MAE ↓", "MSE ↓", "PSNR dB ↑", "SSIM ↑", "MSE gain % ↑"]
    rows = [[str(row["kind"]).capitalize(), METHOD_LABELS.get(row["strategy"], row["strategy"]),
             str(row.get("n_images", 1)), _formatted(row["mae_mean"]), _formatted(row["mse_mean"], 6),
             _formatted(row["psnr_mean"], 2), _formatted(row["ssim_mean"]),
             _formatted(100 * row["relative_mse_reduction_mean"], 2)] for row in records]
    table = axis.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#dce2e8")
        if row == 0:
            cell.set_facecolor("#17384a")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eef5f7")
    figure.suptitle(title, fontsize=15)
    if footnote:
        figure.text(0.5, 0.02, footnote, fontsize=9, ha="center", va="bottom", wrap=True)
    return figure


def plot_feature_ablations(records: Sequence[Mapping], *, title: str = "Feature drop ablation at 25% budget") -> Figure:
    kinds = list(dict.fromkeys(row["kind"] for row in records))
    variants = list(dict.fromkeys(row["dropped_feature"] for row in records if row["dropped_feature"] != "none"))
    figure = _new_figure(width=11.5, height=5.8)
    axis = figure.subplots()
    y = np.arange(len(variants), dtype=float)
    height = 0.7 / max(1, len(kinds))
    for index, kind in enumerate(kinds):
        by_variant = {row["dropped_feature"]: row for row in records if row["kind"] == kind}
        values = [by_variant[feature]["relative_mse_change_vs_full_mean"] * 100 for feature in variants]
        positions = y + (index - (len(kinds) - 1) / 2) * height
        axis.barh(positions, values, height=height, label=f"{str(kind).capitalize()} images", color=("#008f98", "#8856ba")[index % 2])
    axis.axvline(0, color="#4a5565", linewidth=1)
    axis.set_yticks(y, [f"Drop {name.replace('_', ' ')}" for name in variants])
    axis.invert_yaxis()
    axis.set_xlabel("MSE change relative to all-feature baseline (%) · positive means removal hurt")
    axis.grid(axis="x", alpha=0.2)
    axis.legend()
    figure.suptitle(title, fontsize=14)
    return figure


def figure_to_png_bytes(figure: Figure, dpi: int = 180) -> bytes:
    """Render a figure for an export button or report attachment."""
    output = BytesIO()
    figure.savefig(output, format="png", dpi=dpi, facecolor="white", bbox_inches="tight")
    return output.getvalue()


def save_figure(figure: Figure, path: str | Path, *, dpi: int = 180) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(figure_to_png_bytes(figure, dpi=dpi))
    return destination
