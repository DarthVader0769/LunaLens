"""Pure full-reference reconstruction metrics on a fixed [0, 1] scale.

This module has no access to a planner and never imports the simulator. Ground
truth is used here strictly *after* decisions, to measure their consequences.
SSIM uses scikit-image's standard uniform-window implementation, a data range
of one, sample covariance, and the largest valid odd window up to seven pixels.
Images narrower than three pixels cannot support that estimator; their SSIM is
NaN. PSNR is positive infinity for an exact reconstruction.
"""

from __future__ import annotations

from math import isfinite

import numpy as np
from skimage.metrics import structural_similarity


def _metric_images(truth: np.ndarray, reconstruction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate without renormalising intensities or changing the evaluation target."""
    reference = np.asarray(truth)
    estimate = np.asarray(reconstruction)
    if reference.ndim != 2 or estimate.ndim != 2:
        raise ValueError("Metrics require two-dimensional grayscale images.")
    if reference.shape != estimate.shape or reference.size == 0:
        raise ValueError("Metric images must have the same nonempty shape.")
    for name, array in (("truth", reference), ("reconstruction", estimate)):
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise ValueError(f"{name} must contain real numeric intensities.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains nonfinite values.")
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError(f"{name} intensities must lie in [0, 1].")
    return reference, estimate


def compute_metrics(truth: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    """Return MAE, MSE, PSNR (dB), and SSIM using ``data_range=1``.

    Errors accumulate in float64, including when source images are float32.
    These are image quality metrics, not physical terrain accuracy measures.
    """
    reference, estimate = _metric_images(truth, reconstruction)
    residual = np.subtract(reference, estimate, dtype=np.float64)
    mae = float(np.mean(np.abs(residual), dtype=np.float64))
    mse = float(np.mean(residual * residual, dtype=np.float64))
    psnr = float("inf") if mse == 0.0 else float(-10.0 * np.log10(mse))
    minimum_dimension = min(reference.shape)
    if minimum_dimension < 3:
        ssim = float("nan")
    elif mse == 0.0:
        ssim = 1.0
    else:
        window = min(7, minimum_dimension)
        window -= 1 - window % 2
        ssim = float(
            structural_similarity(
                reference,
                estimate,
                data_range=1.0,
                win_size=window,
                gaussian_weights=False,
                use_sample_covariance=True,
            )
        )
    return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim}


def evaluate_reconstruction(
    truth: np.ndarray,
    reconstruction: np.ndarray,
    *,
    initial_mse: float,
    observations: int,
    total_cost: float,
    coverage: float,
) -> dict[str, float | int]:
    """Build one history record with cumulative, explicitly defined gains.

    ``improvement_per_observation`` is cumulative MSE reduction divided by
    observations so far, not the marginal improvement from the latest reveal.
    Relative reduction and per-observation improvement are zero for their
    respective zero-denominator cases. Negative improvement is retained.
    """
    if not isfinite(float(initial_mse)) or initial_mse < 0:
        raise ValueError("initial_mse must be finite and nonnegative.")
    if isinstance(observations, (bool, np.bool_)) or not isinstance(observations, (int, np.integer)) or observations < 0:
        raise ValueError("observations must be a nonnegative integer.")
    if not isfinite(float(total_cost)) or total_cost < 0:
        raise ValueError("total_cost must be finite and nonnegative.")
    if not isfinite(float(coverage)) or not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must be a finite fraction in [0, 1].")
    metrics = compute_metrics(truth, reconstruction)
    reduction = float(initial_mse - metrics["mse"])
    return {
        "observations": int(observations),
        "total_cost": float(total_cost),
        "coverage": float(coverage),
        **metrics,
        "mse_reduction": reduction,
        "relative_mse_reduction": reduction / initial_mse if initial_mse > 0.0 else 0.0,
        "improvement_per_observation": reduction / observations if observations else 0.0,
    }
