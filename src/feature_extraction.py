"""Interpretable descriptors of the *available* image, not hidden terrain.

All intensity thresholds assume the fixed [0, 1] input scale. The simulator
calls ``extract_features`` exactly once on its initial degraded map. A caller
may explore other images, but those descriptors must not be passed to a
planner unless the corresponding observations have actually been revealed.

Features, in order:
  variance     population variance of intensity inside the region;
  edge_density fraction of pixels whose Sobel magnitude exceeds 0.04;
  gradient     mean Sobel magnitude (Sobel kernels divided by eight);
  texture      RMS high-frequency residual I - Gaussian(I, sigma=1 pixel);
  entropy      Shannon entropy of a 32-bin intensity histogram, in bits;
  contrast     95th minus 5th intensity percentile;
  information  (entropy / 5) * (contrast + edge_density) / 2.

The last feature is a deliberately simple, correlated composite proxy, not
mutual information, calibrated uncertainty, or expected scientific utility.
Texture and shadows do not measure physical slope, elevation, or safety.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage

from .preprocessing import validate_image
from .segmentation import Region

FEATURE_NAMES = ("variance", "edge_density", "gradient", "texture", "entropy", "contrast", "information")
HISTOGRAM_BINS = 32
EDGE_THRESHOLD = 0.04


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    dx = ndimage.sobel(image, axis=1, mode="reflect") / 8.0
    dy = ndimage.sobel(image, axis=0, mode="reflect") / 8.0
    return np.hypot(dx, dy)


def extract_features(initial_map: np.ndarray, regions: Sequence[Region]) -> np.ndarray:
    """Return an N×7 float64 descriptor matrix for the supplied image regions.

    Sobel and Gaussian filters use reflected image borders. Filtering the full
    available map before slicing avoids inventing independent tile boundaries.
    Regions may include partial edge tiles, and a one-pixel tile is valid.
    """
    image = validate_image(initial_map)
    regions = tuple(regions)
    if len({r.id for r in regions if isinstance(r, Region)}) != len(regions):
        raise ValueError("regions must be Region objects with unique IDs")
    for region in regions:
        if region.y0 + region.height > image.shape[0] or region.x0 + region.width > image.shape[1]:
            raise ValueError(f"region {region.id} extends outside the image")
    gradient = _gradient_magnitude(image)
    residual = image - ndimage.gaussian_filter(image, sigma=1.0, mode="reflect")
    features = np.zeros((len(regions), len(FEATURE_NAMES)), dtype=np.float64)
    for index, region in enumerate(regions):
        patch = image[region.slices]
        local_gradient = gradient[region.slices]
        variance = float(np.var(patch, dtype=np.float64))
        edge_density = float(np.mean(local_gradient > EDGE_THRESHOLD))
        mean_gradient = float(np.mean(local_gradient, dtype=np.float64))
        high_frequency = residual[region.slices].astype(np.float64)
        texture = float(np.sqrt(np.mean(high_frequency * high_frequency)))
        # The upper endpoint 1.0 belongs to the last bin, not an extra bin.
        quantized = np.minimum((patch * HISTOGRAM_BINS).astype(np.int32), HISTOGRAM_BINS - 1)
        counts = np.bincount(quantized.ravel(), minlength=HISTOGRAM_BINS)
        probabilities = counts[counts > 0].astype(np.float64) / patch.size
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
        low, high = np.percentile(patch, [5.0, 95.0])
        contrast = float(high - low)
        information = entropy / np.log2(HISTOGRAM_BINS) * (contrast + edge_density) / 2.0
        features[index] = variance, edge_density, mean_gradient, texture, entropy, contrast, information
    return features


def normalise_features(features: np.ndarray) -> np.ndarray:
    """Per-column min-max normalization; constant columns contain only zeros.

    The simulator fits this scale on the initial available map and keeps it
    fixed for the whole run. No ground-truth descriptor enters this operation.
    """
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"features must have shape (N, {len(FEATURE_NAMES)})")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError("features must be a finite real numeric matrix")
    values = values.astype(np.float64, copy=True)
    if not np.all(np.isfinite(values)):
        raise ValueError("features must contain only finite values")
    if len(values) == 0:
        return values
    minimum = values.min(axis=0)
    span = values.max(axis=0) - minimum
    result = np.zeros_like(values)
    np.divide(values - minimum, span, out=result, where=span > 0)
    return np.clip(result, 0.0, 1.0)


def terrain_proxies(image: np.ndarray) -> dict:
    """Explore dark, sharp, textured, and circular image patterns.

    Candidate circles are Canny/Hough detections on an image bounded to 256
    pixels along its longest dimension, with at most twelve peaks. Coordinates
    are mapped back to original pixels. The detector can miss real craters or
    mistake circular texture for craters; Hough scores are not probabilities.
    This optional exploration is never used by the simulator's planner.
    """
    from skimage.feature import canny
    from skimage.transform import hough_circle, hough_circle_peaks, resize

    values = validate_image(image)
    gradient = _gradient_magnitude(values)
    residual = values - ndimage.gaussian_filter(values, sigma=1.0, mode="reflect")
    local_energy = ndimage.uniform_filter(residual * residual, size=7, mode="reflect")
    # The running-sum filter can produce tiny negative round-off near zero.
    roughness = np.sqrt(np.maximum(local_energy, 0.0))
    # An absolute floor ensures a perfectly flat map has no rough-looking mask.
    rough_threshold = max(0.015, float(np.percentile(roughness, 75)))
    circles: list[dict[str, float]] = []
    if min(values.shape) >= 16 and float(np.ptp(values)) > 0.02:
        scale = min(1.0, 256.0 / max(values.shape))
        reduced_shape = tuple(max(1, int(round(axis * scale))) for axis in values.shape)
        reduced = resize(values, reduced_shape, anti_aliasing=True, preserve_range=True)
        edges = canny(reduced, sigma=1.2, low_threshold=0.06, high_threshold=0.15)
        maximum_radius = min(40, min(reduced_shape) // 3)
        radii = np.arange(5, maximum_radius + 1, 3, dtype=int)
        if radii.size and np.any(edges):
            accumulator = hough_circle(edges, radii, normalize=True)
            scores, xs, ys, found_radii = hough_circle_peaks(
                accumulator, radii, min_xdistance=8, min_ydistance=8,
                threshold=0.35, total_num_peaks=12, normalize=True,
            )
            sx, sy = values.shape[1] / reduced_shape[1], values.shape[0] / reduced_shape[0]
            circles = [
                {"x": float(x * sx), "y": float(y * sy),
                 "radius": float(radius * (sx + sy) / 2.0), "score": float(score)}
                for score, x, y, radius in zip(scores, xs, ys, found_radii)
            ]
    return {
        "shadow_mask": values < 0.15,
        "boundary_mask": gradient > EDGE_THRESHOLD,
        "rough_texture_mask": roughness > rough_threshold,
        "gradient_magnitude": gradient,
        "roughness_proxy": roughness,
        "candidate_circles": circles,
        "notes": {
            "shadow_mask": "Intensity below 0.15: a dark-image proxy, not confirmed shadow or depth.",
            "boundary_mask": "Sobel magnitude above 0.04: sharp image boundaries, not terrain slope.",
            "rough_texture_mask": "Local high-pass RMS above max(0.015, image 75th percentile); not physical roughness.",
            "candidate_circles": "Canny/Hough circle candidates, not verified craters; score is accumulator strength.",
        },
    }
