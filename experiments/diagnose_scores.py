"""Offline analysis of executed benchmark diagnostics; never imported by policies.

Run after the benchmark: python -m experiments.diagnose_scores
Creates a dataset gallery, score/error scatter plots, and a strictly evaluation-
only error-optimal bound. Reading true errors is legitimate here because all
planning decisions have already been made. None of this is a runnable strategy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.image_loader import PROJECT_ROOT, list_sample_images, load_image
from src.visualisation import METHOD_COLORS, METHOD_LABELS, save_figure


def observations_to_targets(results: Path, figures: Path) -> None:
    """First passage to error-reduction targets along completed trajectories.

    For the shipped prefix-invariant policies this also answers how many
    observations were needed. A future horizon-dependent policy needs its
    separately executed budget endpoints; a trajectory is not a budget curve.
    """
    history = pd.read_csv(results / "history.csv")
    records = []
    keys = ["image_id", "kind", "strategy", "seed"]
    for key, group in history.groupby(keys, dropna=False, sort=False):
        group = group.sort_values("observations")
        initially_exact = float(group.iloc[0].mse) == 0.0
        for target in (.25, .50, .75, .90):
            reached = group if initially_exact else group[group.relative_mse_reduction >= target]
            hit = reached.iloc[0] if not reached.empty else None
            records.append({
                **dict(zip(keys, key)), "target_relative_mse_reduction": target,
                "observations": int(hit.observations) if hit is not None else np.nan,
                "total_cost": float(hit.total_cost) if hit is not None else np.nan,
                "reached": hit is not None,
                "interpretation": "First passage along full-scan trajectory; current policies are prefix-invariant.",
            })
    frame = pd.DataFrame(records)
    frame.to_csv(results / "observations_to_targets_by_run.csv", index=False)
    per_image = frame.groupby(["image_id", "kind", "strategy", "target_relative_mse_reduction"], sort=False,
                              as_index=False)[["observations", "total_cost"]].mean()
    per_image.to_csv(results / "observations_to_targets_by_image.csv", index=False)
    per_kind = per_image.groupby(["kind", "strategy", "target_relative_mse_reduction"], sort=False,
                                as_index=False)[["observations", "total_cost"]].mean()
    per_kind.to_csv(results / "observations_to_targets_by_kind.csv", index=False)
    quarter = per_image[np.isclose(per_image.target_relative_mse_reduction, .50)]
    identifiers = list(quarter.image_id.unique())
    names = {entry["id"]: entry["name"].replace("LRO — ", "").replace("SYNTHETIC — ", "Synthetic: ")
             for entry in list_sample_images()}
    figure = Figure(figsize=(12, 5.8), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    x = np.arange(len(identifiers))
    for index, method in enumerate(METHOD_LABELS):
        data = quarter[quarter.strategy == method].set_index("image_id").loc[identifiers]
        axis.bar(x + (index - 1.5)*.20, data.observations, width=.19,
                 color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    axis.set_xticks(x, [names.get(identifier, identifier).replace(" ", "\n", 1) for identifier in identifiers], fontsize=9)
    axis.set_ylabel("Observations until initial MSE is halved (lower is better)")
    axis.grid(axis="y", alpha=.2)
    axis.set_axisbelow(True)
    axis.legend(ncols=2, fontsize=9)
    figure.suptitle("Same reconstruction target, different observation counts", fontsize=15)
    axis.set_title("256 regions per image; random averages five seeds; full-scan first-passage counts", fontsize=10)
    save_figure(figure, figures / "observations_to_half_error.png", dpi=180)


def dataset_gallery(destination: Path) -> None:
    entries = list_sample_images()
    rows = (len(entries) + 2) // 3
    figure = Figure(figsize=(12, 4.4 * rows + .8), facecolor="white")
    FigureCanvasAgg(figure)
    axes = np.atleast_1d(figure.subplots(rows, 3)).ravel()
    for axis, entry in zip(axes, entries):
        axis.imshow(load_image(entry["path"]), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(entry["name"], fontsize=10, pad=10)
        axis.set_axis_off()
    for axis in axes[len(entries):]:
        axis.set_axis_off()
    figure.subplots_adjust(left=.025, right=.975, top=.9, bottom=.055, hspace=.18, wspace=.09)
    figure.suptitle("LunaLens offline dataset — LROC images and procedural fixtures", fontsize=15, y=.975)
    figure.text(.5, .02, "Real samples: NASA / GSFC / Arizona State University, fixed 512-pixel center crops.\nSynthetic samples: software fixtures, no scientific calibration.", ha="center", fontsize=9)
    save_figure(figure, destination, dpi=180)


def analyse(output: Path) -> pd.DataFrame:
    results = output / "results"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    # All values come from the completed experiment; no reranking is passed back.
    regions = pd.read_csv(results / "region_diagnostics.csv")
    correlations = pd.read_csv(results / "score_error_correlations.csv").set_index("image_id")
    endpoints = pd.read_csv(results / "summary_by_image.csv")
    names = {entry["id"]: entry["name"] for entry in list_sample_images()}
    image_ids = list(regions["image_id"].unique())
    figure = Figure(figsize=(13, 4 * ((len(image_ids) + 2) // 3) + .4), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    axes = np.atleast_1d(figure.subplots((len(image_ids) + 2) // 3, 3)).ravel()
    bound_rows = []
    for axis, identifier in zip(axes, image_ids):
        subset = regions[regions.image_id == identifier]
        rho = correlations.loc[identifier, "spearman_score_vs_true_tile_mse"]
        axis.scatter(subset.initial_feature_score, subset.evaluation_only_true_tile_mse,
                     s=13, alpha=.60, color="#008f98", edgecolors="none")
        axis.set_title(f"{names.get(identifier, identifier)}\nSpearman rho = {rho:.3f}", fontsize=10)
        axis.set_xlabel("Initial feature utility (planner-accessible)", fontsize=9)
        axis.set_ylabel("True tile MSE (evaluation only)", fontsize=9)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        axis.grid(alpha=.18)
        total_error = float(subset.evaluation_only_global_mse_gain.sum())
        count = len(subset)
        for fraction in (0.10, .25, .50, .75, 1.0):
            budget = int(np.floor(fraction * count + 1e-9))
            optimal_gain = float(np.sort(subset.evaluation_only_global_mse_gain.to_numpy())[::-1][:budget].sum())
            for method in ("random", "systematic", "feature", "custom"):
                endpoint = endpoints[(endpoints.image_id == identifier) & (endpoints.strategy == method)
                                     & np.isclose(endpoints.budget_fraction, fraction)].iloc[0]
                bound_rows.append({
                    "image_id": identifier, "kind": subset.kind.iloc[0], "strategy": method,
                    "budget_fraction": fraction, "observations": budget,
                    "measured_relative_mse_reduction": float(endpoint.relative_mse_reduction_mean),
                    "evaluation_only_optimal_relative_mse_reduction": optimal_gain / total_error if total_error else 0,
                    "exact_uniform_random_expected_relative_mse_reduction": budget / count if total_error else 0,
                    "note": "Equal-cost, disjoint, exact reveals only; bound sees truth and is NOT a deployable strategy.",
                })
    for axis in axes[len(image_ids):]:
        axis.set_axis_off()
    figure.suptitle("Does visible structure predict recoverable error? — evaluation-only diagnostic", fontsize=14)
    save_figure(figure, figures / "score_error_diagnostics.png", dpi=180)
    table = pd.DataFrame(bound_rows)
    table.to_csv(results / "evaluation_only_bounds.csv", index=False)
    dataset_gallery(figures / "dataset_gallery.png")
    observations_to_targets(results, figures)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="evidence")
    args = parser.parse_args()
    destination = Path(args.output_dir)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    table = analyse(destination)
    quarter = table[(table.strategy == "feature") & (table.budget_fraction == .25)]
    print(quarter[["image_id", "measured_relative_mse_reduction", "evaluation_only_optimal_relative_mse_reduction"]].to_string(index=False))
    print(f"Saved diagnostic gallery/plots and {len(table)} bound rows under {destination}.")


if __name__ == "__main__":
    main()
