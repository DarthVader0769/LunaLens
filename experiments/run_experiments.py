"""Execute fair observation-planning comparisons and produce portfolio evidence.

Run from the repository root::

    python -m experiments.run_experiments

By default each baseline scans the complete image once. Prefix histories give
valid smaller-budget results for the four current, tested policies: their
relative priorities do not depend on the remaining horizon. A future custom
policy may use remaining_budget and change its earlier decisions; evaluate
such a policy with --independent-budgets, which runs each endpoint at its actual
budget cap. Curves always describe full-scan trajectories, explicitly labelled.
Deterministic methods run once per image/configuration. Random seeds are
averaged *within* each image before real/synthetic summaries are calculated.
No weights are tuned using these results. This is a simplified image simulator.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from src import __version__
from src.custom_strategy import CustomStrategy
from src.evaluation import compute_metrics
from src.feature_extraction import FEATURE_NAMES
from src.image_loader import list_sample_images, load_image
from src.observation_planner import DEFAULT_WEIGHTS
from src.preprocessing import degrade_image
from src.segmentation import segment_image
from src.simulator import SimulationConfig, run_simulation
from src.visualisation import (
    image_comparison, plot_feature_ablations, plot_feature_map, plot_metrics_table,
    plot_performance_curves, plot_priority_map, plot_reconstruction_comparison,
    plot_segmentation, plot_selection_comparison, save_figure,
)


ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("random", "systematic", "feature", "custom")
BUDGET_FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
METRICS = (
    "observations", "total_cost", "coverage", "mae", "mse", "psnr", "ssim",
    "mse_reduction", "relative_mse_reduction", "improvement_per_observation", "actual_budget_fraction",
)


def _finite_json(value: Any) -> Any:
    """Strict JSON: nonfinite metrics become null, with their meaning documented."""
    if isinstance(value, Mapping):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_finite_json(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def record_at_budget(history: Sequence[Mapping], fraction: float, full_cost: float) -> dict:
    """Take the latest affordable prefix; do not round up a discrete budget.

    This corresponds to a real run with that budget for the uniform-cost
    benchmark. Boundary tile area costs are supported by the simulator but are
    not used for these equal-observation-count comparisons.
    """
    if not history or not 0 <= fraction <= 1 or full_cost <= 0:
        raise ValueError("A nonempty history, fraction in [0, 1], and positive full cost are required.")
    limit = fraction * full_cost
    affordable = [row for row in history if float(row["total_cost"]) <= limit + 1e-9]
    if not affordable:
        raise ValueError("History must include its zero-observation record.")
    result = dict(affordable[-1])
    result["budget_fraction"] = float(fraction)
    result["actual_budget_fraction"] = float(result["total_cost"] / full_cost)
    return result


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if np.any(np.isnan(array)):
        return float("nan")
    return float(np.mean(array))


def _sample_sd(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    if len(array) < 2 or not np.all(np.isfinite(array)):
        return None
    return float(np.std(array, ddof=1))


def aggregate_rows(
    rows: Sequence[Mapping], group_keys: Sequence[str], *, metrics: Sequence[str] = METRICS,
    count_name: str = "n_runs",
) -> list[dict]:
    """Aggregate one explicitly chosen sampling level; never pool seeds/images."""
    groups: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    result = []
    for key, members in groups.items():
        record = dict(zip(group_keys, key))
        record[count_name] = len(members)
        for metric in metrics:
            values = [member[metric] for member in members]
            record[f"{metric}_mean"] = _mean(values)
            record[f"{metric}_sd"] = _sample_sd(values)
        result.append(record)
    return result


def aggregate_images(per_image: Sequence[Mapping], *, metrics: Sequence[str] = METRICS,
                     extra_keys: Sequence[str] = ()) -> list[dict]:
    """Give every source image equal weight after averaging its random seeds."""
    image_means = []
    for row in per_image:
        image_means.append({
            "kind": row["kind"], "strategy": row["strategy"], "budget_fraction": row["budget_fraction"],
            **{key: row[key] for key in extra_keys},
            **{metric: row[f"{metric}_mean"] for metric in metrics},
        })
    return aggregate_rows(image_means, ("kind", "strategy", "budget_fraction", *extra_keys),
                          metrics=metrics, count_name="n_images")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    values = {"python": platform.python_version(), "lunalens": __version__}
    for package in ("numpy", "scipy", "scikit-image", "matplotlib", "pandas", "Pillow", "streamlit", "plotly"):
        try:
            values[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            values[package] = "not installed"
    return values


def _code_hashes() -> dict[str, str]:
    sources = sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "experiments").glob("*.py"))
    for filename in ("requirements.txt", "requirements-lock.txt", "pyproject.toml"):
        if (ROOT / filename).is_file():
            sources.append(ROOT / filename)
    return {str(path.relative_to(ROOT)): _sha256(path) for path in sources}


def _entry_path(entry: Mapping) -> Path:
    path = Path(entry["path"])
    return path if path.is_absolute() else ROOT / path


def _reconstruct_prefix(truth: np.ndarray, result, observations: int) -> np.ndarray:
    reconstruction = result.initial_map.copy()
    by_id = {region.id: region for region in result.regions}
    for region_id in result.selected_ids[:observations]:
        region = by_id[region_id]
        reconstruction[region.slices] = truth[region.slices]
    return reconstruction


def _selection_rows(entry: Mapping, result, seed: int | None, *, variant: str = "baseline") -> list[dict]:
    by_id = {region.id: region for region in result.regions}
    records = []
    for index, region_id in enumerate(result.selected_ids, start=1):
        region = by_id[region_id]
        records.append({
            "image_id": entry["id"], "kind": entry["kind"], "strategy": result.strategy_name,
            "seed": seed, "variant": variant, "observation": index, "region_id": region_id,
            "row": region.row, "col": region.col, "y0": region.y0, "x0": region.x0,
            "height": region.height, "width": region.width, "cost": region.cost,
            "initial_score": float(result.initial_scores[region_id]),
        })
    return records


def _region_diagnostics(entry: Mapping, truth: np.ndarray, result) -> tuple[list[dict], dict]:
    """Evaluation-only relation between initial descriptors and true reveal value.

    These errors are calculated after plans are complete and never enter any
    strategy, state object, score or weight. They help explain success/failure.
    """
    squared_error = np.square(np.subtract(truth, result.initial_map, dtype=np.float64))
    rows = []
    for region in result.regions:
        values = {
            "image_id": entry["id"], "kind": entry["kind"], "region_id": region.id,
            "evaluation_only_true_tile_mse": float(squared_error[region.slices].mean()),
            "evaluation_only_global_mse_gain": float(squared_error[region.slices].sum() / truth.size),
            "initial_feature_score": float(result.initial_scores[region.id]),
        }
        for index, name in enumerate(FEATURE_NAMES):
            values[f"normalised_{name}"] = float(result.features[region.id, index])
            values[f"raw_{name}"] = float(result.raw_features[region.id, index])
        rows.append(values)
    errors = [row["evaluation_only_true_tile_mse"] for row in rows]
    scores = [row["initial_feature_score"] for row in rows]
    if len(set(scores)) > 1 and len(set(errors)) > 1:
        correlation = float(spearmanr(scores, errors).statistic)
    else:
        correlation = float("nan")
    return rows, {"image_id": entry["id"], "kind": entry["kind"], "spearman_score_vs_true_tile_mse": correlation}


def _comparisons_at_quarter(per_image: Sequence[Mapping], per_kind: Sequence[Mapping]) -> list[dict]:
    comparisons = []
    quarters = [row for row in per_image if row["budget_fraction"] == 0.25]
    by_image = defaultdict(dict)
    for row in quarters:
        by_image[(row["kind"], row["image_id"])][row["strategy"]] = row
    grouped = {(row["kind"], row["strategy"]): row for row in per_kind if row["budget_fraction"] == 0.25}
    for kind in dict.fromkeys(row["kind"] for row in quarters):
        for method in ("feature", "custom"):
            for baseline in ("random", "systematic"):
                changes, wins, ties, losses = [], 0, 0, 0
                for (image_kind, image_id), rows in by_image.items():
                    if image_kind != kind:
                        continue
                    control, candidate = rows[baseline]["mse_mean"], rows[method]["mse_mean"]
                    if control > 0:
                        changes.append((control - candidate) / control)
                    if math.isclose(candidate, control, rel_tol=1e-9, abs_tol=1e-14):
                        ties += 1
                    elif candidate < control:
                        wins += 1
                    else:
                        losses += 1
                baseline_mse = grouped[kind, baseline]["mse_mean"]
                method_mse = grouped[kind, method]["mse_mean"]
                comparisons.append({
                    "kind": kind, "method": method, "baseline": baseline, "budget_fraction": 0.25,
                    "n_images": wins + ties + losses, "wins": wins, "ties": ties, "losses": losses,
                    "relative_reduction_in_mean_mse": (baseline_mse - method_mse) / baseline_mse if baseline_mse else None,
                    "mean_of_per_image_relative_mse_reductions": _mean(changes) if changes else None,
                })
    return comparisons


def _generate_figures(output: Path, representative: Mapping, kind_summaries: Sequence[Mapping],
                      kind_curves: Sequence[Mapping], ablation_kinds: Sequence[Mapping], dpi: int) -> list[str]:
    figure_dir = output / "figures"
    truth, initial, result = representative["truth"], representative["initial"], representative["feature_result"]
    name = representative["entry"]["name"]
    figures = {
        "01_original_and_degraded.png": image_comparison(truth, initial, title=f"LunaLens · {name}"),
        "02_observation_segmentation.png": plot_segmentation(initial, result.regions, title=f"Observation tiling · {name}"),
        "03_information_heatmap.png": plot_feature_map(initial, result.regions, result.features[:, FEATURE_NAMES.index("information")],
                                                       title="Estimated information descriptor · initial map"),
        "04_observation_priorities.png": plot_priority_map(initial, result.regions, result.initial_scores),
        "05_selected_observation_regions.png": plot_selection_comparison(initial, result.regions, representative["selections"],
                                                                         title="Selected regions at 25% budget · random uses first listed seed"),
        "06_reconstructed_maps.png": plot_reconstruction_comparison(truth, initial, representative["reconstructions"],
                                                                   title="Reconstructed maps at the same 25% observation budget"),
        "07_error_vs_budget.png": plot_performance_curves(kind_curves, title="Full-scan trajectories · reconstruction error versus prefix budget"),
        "08_measured_metrics_table.png": plot_metrics_table([row for row in kind_summaries if row["budget_fraction"] == 0.25],
                                                              title="Measured reconstruction quality · 25% observation budget",
                                                              footnote="Random: seeds averaged within each image. All methods: equal-weight image means. Real and synthetic results kept separate."),
        "10_improvement_vs_observation_count.png": plot_performance_curves(kind_curves, metric="relative_mse_reduction", x_key="observations_mean",
                                                                           title="Full-scan trajectories · improvement versus observation count"),
        "11_gain_per_observation.png": plot_performance_curves(kind_curves, metric="improvement_per_observation",
                                                              title="Full-scan trajectories · cumulative gain per observation"),
    }
    if ablation_kinds:
        figures["09_feature_drop_ablation.png"] = plot_feature_ablations(ablation_kinds)
    saved = []
    for filename, figure in sorted(figures.items()):
        destination = save_figure(figure, figure_dir / filename, dpi=dpi)
        saved.append(str(destination.relative_to(output)))
        figure.clear()
    return saved


def run_benchmark(args: argparse.Namespace) -> dict:
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    results_dir = output / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    entries = list_sample_images()
    if args.image_ids:
        known = {entry["id"] for entry in entries}
        unknown = set(args.image_ids) - known
        if unknown:
            raise ValueError(f"Unknown image IDs: {sorted(unknown)}. Available: {sorted(known)}")
        entries = [entry for entry in entries if entry["id"] in args.image_ids]
    if args.limit is not None:
        entries = entries[:args.limit]
    if not entries:
        raise ValueError("No sample images found. Check data/manifest.json.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Random seeds must be distinct; repeated seeds are not independent runs.")
    if args.region_size < 1 or args.max_dimension < 1 or args.downsample_factor < 1 or args.blur_sigma < 0 or not math.isfinite(args.blur_sigma):
        raise ValueError("Invalid region size, image size, degradation factor or blur sigma.")

    started = time.perf_counter()
    custom_preset = CustomStrategy(dict(DEFAULT_WEIGHTS))
    custom_coefficients = {name: getattr(custom_preset, name) for name in ("utility_weight", "novelty_weight", "spatial_weight")}
    run_metadata = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": ["python", "-m", "experiments.run_experiments", *sys.argv[1:]],
        "arguments": vars(args), "versions": _versions(), "platform": platform.platform(),
        "source_sha256": _code_hashes(), "default_weights_fixed_before_results": dict(DEFAULT_WEIGHTS),
        "custom_coefficients_fixed_before_results": custom_coefficients,
        "strategies": list(STRATEGIES), "random_seeds": list(args.seeds),
        "deterministic_runs_per_image_per_strategy": 1,
        "budget_fractions": BUDGET_FRACTIONS, "cost_mode": "uniform",
        "endpoint_evaluation": "Independent simulations at each actual budget cap." if args.independent_budgets else "Prefixes of full scans; equivalence to independently capped runs tested for all four current policies.",
        "curve_interpretation": "Full-scan trajectories, not independently replanned curves. Future policies whose priorities depend on remaining_budget require --independent-budgets for valid endpoint comparisons.",
        "aggregation": "Average seeds within each image, then average image-level means equally within real or synthetic kind. SD in kind tables is sample SD across image means; it is not a confidence interval.",
        "nonfinite_metrics": "CSV uses inf/nan. Strict JSON uses null: PSNR at zero MSE is +infinity; SSIM only undefined for images narrower than 3 pixels. SD is null when fewer than 2 samples or values nonfinite.",
        "hypothesis": "Feature utility and descriptor/spatial diversity may improve early map reconstruction; results are measured, and no weights are tuned on benchmark outcomes.",
        "ablation": "At 25% budget, drop each feature's weight to zero while retaining the other preset weights. Deterministic, one run per image per feature, no tuning." if not args.no_ablations else "Disabled by explicit CLI option.",
        "inputs": [],
    }
    write_json(results_dir / "reproducibility.json", run_metadata)
    write_json(results_dir / "summary.json", {
        "status": "running", "started_utc": run_metadata["started_utc"],
        "message": "The experiment is executing. A completed summary is written only after all comparisons and figures finish.",
    })

    histories, endpoint_runs, curve_runs, selections, diagnostics, correlations = [], [], [], [], [], []
    ablation_runs, ablation_histories, independent_histories = [], [], []
    representative = None
    baseline_run_count = ablation_run_count = independent_run_count = 0
    curve_fractions = sorted(set(np.linspace(0.0, 1.0, 257).tolist()) | set(BUDGET_FRACTIONS))
    for image_index, entry in enumerate(entries, start=1):
        path = _entry_path(entry)
        truth = load_image(path, max_dimension=args.max_dimension)
        initial = degrade_image(truth, factor=args.downsample_factor, blur_sigma=args.blur_sigma)
        regions = segment_image(truth.shape, args.region_size, cost_mode="uniform")
        full_cost = float(sum(region.cost for region in regions))
        quarter_count = int(math.floor(0.25 * len(regions) + 1e-9))
        metadata_entry = {**dict(entry), "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                          "file_sha256": _sha256(path), "decoded_array_sha256": _array_sha256(truth),
                          "initial_map_sha256": _array_sha256(initial), "shape": list(truth.shape),
                          "region_count": len(regions), "full_observation_cost": full_cost,
                          "initial_metrics": compute_metrics(truth, initial)}
        run_metadata["inputs"].append(metadata_entry)
        print(f"[{image_index}/{len(entries)}] {entry['id']} ({entry['kind']}), {truth.shape}, {len(regions)} regions", flush=True)
        quarter_results = {}
        reference_run = None
        for method in STRATEGIES:
            seeds = args.seeds if method == "random" else [None]
            for seed in seeds:
                config = SimulationConfig(region_size=args.region_size, downsample_factor=args.downsample_factor,
                                          blur_sigma=args.blur_sigma, budget=full_cost, seed=seed if seed is not None else 0,
                                          cost_mode="uniform")
                result = run_simulation(truth, method, config=config, weights=dict(DEFAULT_WEIGHTS), initial_map=initial)
                baseline_run_count += 1
                if result.history[-1]["mse"] != 0.0 or len(result.selected_ids) != len(regions):
                    raise RuntimeError(f"Full-budget reconstruction failed for {entry['id']} / {method}.")
                prefix = {"image_id": entry["id"], "kind": entry["kind"], "strategy": method, "seed": seed}
                histories.extend({**prefix, **row, "budget_fraction": row["total_cost"] / full_cost} for row in result.history)
                for fraction in BUDGET_FRACTIONS:
                    endpoint_result = result
                    if args.independent_budgets and 0.0 < fraction < 1.0:
                        endpoint_config = SimulationConfig(region_size=args.region_size, downsample_factor=args.downsample_factor,
                                                           blur_sigma=args.blur_sigma, budget=fraction * full_cost,
                                                           seed=seed if seed is not None else 0, cost_mode="uniform")
                        endpoint_result = run_simulation(truth, method, config=endpoint_config,
                                                         weights=dict(DEFAULT_WEIGHTS), initial_map=initial)
                        independent_run_count += 1
                        independent_histories.extend({**prefix, "allowed_budget_fraction": fraction, **row}
                                                     for row in endpoint_result.history)
                        selections.extend(_selection_rows(entry, endpoint_result, seed, variant=f"independent_budget_{fraction:g}"))
                    endpoint_runs.append({**prefix, **record_at_budget(endpoint_result.history, fraction, full_cost)})
                    if fraction == 0.25 and method not in quarter_results:
                        quarter_results[method] = endpoint_result
                curve_runs.extend({**prefix, **record_at_budget(result.history, fraction, full_cost)} for fraction in curve_fractions)
                selections.extend(_selection_rows(entry, result, seed))
                if method == "feature":
                    reference_run = result
                print(f"  {method:10s} seed={seed if seed is not None else 'deterministic'} · full scan verified", flush=True)
        diagnostic_rows, diagnostic_summary = _region_diagnostics(entry, truth, reference_run)
        diagnostics.extend(diagnostic_rows)
        correlations.append(diagnostic_summary)
        if representative is None or (representative["entry"]["kind"] != "real" and entry["kind"] == "real"):
            representative = {"entry": dict(entry), "truth": truth.copy(), "initial": initial.copy(),
                              "feature_result": reference_run,
                              "selections": {name: result.selected_ids[:quarter_count] for name, result in quarter_results.items()},
                              "reconstructions": {name: _reconstruct_prefix(truth, result, quarter_count) for name, result in quarter_results.items()}}
        if not args.no_ablations:
            full_feature = record_at_budget(quarter_results["feature"].history, 0.25, full_cost)
            for feature in ("none", *FEATURE_NAMES):
                if feature == "none":
                    values = full_feature
                else:
                    weights = dict(DEFAULT_WEIGHTS)
                    weights[feature] = 0.0
                    config = SimulationConfig(region_size=args.region_size, downsample_factor=args.downsample_factor,
                                              blur_sigma=args.blur_sigma, budget=0.25 * full_cost, seed=0, cost_mode="uniform")
                    result = run_simulation(truth, "feature", config=config, weights=weights, initial_map=initial)
                    ablation_run_count += 1
                    values = record_at_budget(result.history, 0.25, full_cost)
                    ablation_histories.extend({"image_id": entry["id"], "kind": entry["kind"], "dropped_feature": feature, **row}
                                               for row in result.history)
                    selections.extend(_selection_rows(entry, result, None, variant=f"drop_{feature}"))
                ablation_runs.append({"image_id": entry["id"], "kind": entry["kind"], "strategy": "feature",
                                      "dropped_feature": feature, **values,
                                      "relative_mse_change_vs_full": (values["mse"] - full_feature["mse"]) / full_feature["mse"] if full_feature["mse"] else 0.0})
            print("  Feature-drop ablations complete (7 features; fixed weights)", flush=True)
        # Save completed work after each image so interruption never looks like a completed run.
        write_csv(results_dir / "history.csv", histories)
        write_csv(results_dir / "selection_history.csv", selections)
        write_csv(results_dir / "ablation_history.csv", ablation_histories)
        if args.independent_budgets:
            write_csv(results_dir / "independent_endpoint_history.csv", independent_histories)
        write_json(results_dir / "reproducibility.json", run_metadata)

    image_keys = ("image_id", "kind", "strategy", "budget_fraction")
    image_summary = aggregate_rows(endpoint_runs, image_keys)
    kind_summary = aggregate_images(image_summary)
    image_curves = aggregate_rows(curve_runs, image_keys)
    kind_curves = aggregate_images(image_curves)
    ablation_metrics = (*METRICS, "relative_mse_change_vs_full")
    ablation_images = aggregate_rows(ablation_runs, (*image_keys, "dropped_feature"), metrics=ablation_metrics) if ablation_runs else []
    ablation_kinds = aggregate_images(ablation_images, metrics=ablation_metrics, extra_keys=("dropped_feature",)) if ablation_images else []
    tables = {
        "endpoints_by_run.csv": endpoint_runs,
        "summary_by_image.csv": image_summary, "summary_by_kind.csv": kind_summary,
        "curves_by_image.csv": image_curves, "curves_by_kind.csv": kind_curves,
        "ablations_by_image.csv": ablation_images, "ablations_by_kind.csv": ablation_kinds,
        "region_diagnostics.csv": diagnostics, "score_error_correlations.csv": correlations,
    }
    for filename, rows in tables.items():
        write_csv(results_dir / filename, rows)
    figure_files = _generate_figures(output, representative, kind_summary, kind_curves, ablation_kinds, args.dpi)
    summary = {
        "status": "complete", "project": "LunaLens", "version": __version__,
        "n_images": len(entries), "n_real_images": sum(entry["kind"] == "real" for entry in entries),
        "n_synthetic_images": sum(entry["kind"] == "synthetic" for entry in entries),
        "baseline_runs": baseline_run_count, "ablation_runs": ablation_run_count,
        "independent_endpoint_runs": independent_run_count,
        "random_seeds": list(args.seeds), "representative_image": representative["entry"]["id"],
        "representative_random_seed": args.seeds[0],
        "config": {"max_dimension": args.max_dimension, "region_size": args.region_size,
                   "downsample_factor": args.downsample_factor, "blur_sigma": args.blur_sigma,
                   "cost_mode": "uniform", "weights": dict(DEFAULT_WEIGHTS),
                   "custom_coefficients": custom_coefficients, "independent_budgets": args.independent_budgets},
        "endpoint_evaluation": run_metadata["endpoint_evaluation"],
        "curve_interpretation": run_metadata["curve_interpretation"],
        "aggregation": run_metadata["aggregation"],
        "quarter_budget_results": [row for row in kind_summary if row["budget_fraction"] == 0.25],
        "quarter_budget_comparisons": _comparisons_at_quarter(image_summary, kind_summary),
        "score_error_correlations_evaluation_only": correlations,
        "ablations_at_quarter": ablation_kinds,
        "full_budget_reconstruction_verified_for_every_run": True,
        "figures": figure_files,
        "limitations": ["Small convenience sample; source images are publication renderings, not calibrated sensor observations.",
                        "No train/test tuning split because fixed features/weights were not fitted. These data are not an independent validation set for future tuning.",
                        "Exact noiseless tile replacement; no orbit geometry, pointing dynamics, bandwidth model or radiometric uncertainty.",
                        "Image texture and candidate shapes do not measure elevation, slope, geological truth or physical hazards.",
                        "Full-scan prefix curves are valid capped-budget comparisons for the current policies. Future horizon-aware custom priorities require independent endpoint runs.",
                        "No statistical significance claim; image-level sample SD describes heterogeneity."],
    }
    run_metadata.update({"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                         "elapsed_seconds": time.perf_counter() - started, "baseline_runs": baseline_run_count,
                         "ablation_runs": ablation_run_count, "independent_endpoint_runs": independent_run_count,
                         "history_rows": len(histories)})
    write_json(results_dir / "reproducibility.json", run_metadata)
    write_json(results_dir / "summary.json", summary)
    print(f"Completed {baseline_run_count} baseline + {ablation_run_count} ablation + {independent_run_count} independent endpoint runs in {run_metadata['elapsed_seconds']:.1f}s.", flush=True)
    print(f"Numeric results: {results_dir}\nFigures: {output / 'figures'}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="evidence", help="Directory for figures/ and results/ (default: evidence)")
    parser.add_argument("--image-ids", nargs="+", help="Manifest IDs to include; default includes all bundled images")
    parser.add_argument("--limit", type=int, help="Use only the first N selected images (useful for smoke runs)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Distinct random seeds; deterministic methods always run once")
    parser.add_argument("--max-dimension", type=int, default=512)
    parser.add_argument("--region-size", type=int, default=32)
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--no-ablations", action="store_true", help="Skip feature-drop ablations for a quick smoke run")
    parser.add_argument("--independent-budgets", action="store_true", help="Rerun each nontrivial endpoint at its actual budget cap; required for future horizon-aware policies. Curves remain full-scan trajectories; this mode costs additional runtime.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG export resolution")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.dpi < 50:
        parser.error("--dpi must be at least 50")
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    try:
        run_benchmark(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Experiment configuration error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
