"""Reproducible lunar-like illustration textures; these are NOT NASA data.

This generator is a test fixture, not a model of lunar geology, elevation,
illumination physics, or sensor radiometry. The hidden_detail style explicitly
stress-tests the low-resolution observability limitation of feature ranking.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.ndimage import gaussian_filter

STYLES = ("mixed", "rugged", "hidden_detail")


def generate_terrain(size: int = 512, seed: int = 0, style: str = "mixed") -> np.ndarray:
    """Return deterministic procedural grayscale pixels in [0, 1]."""
    if isinstance(size, bool) or not isinstance(size, Integral) or size < 32:
        raise ValueError("Synthetic image size must be an integer >= 32.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    if style not in STYLES:
        raise ValueError(f"Unknown terrain style {style!r}; choose from {STYLES}.")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    xn, yn = xx / size, yy / size
    texture = np.zeros((size, size), dtype=np.float64)
    for sigma, weight in ((size/16, .055), (size/64, .025), (1.4, .013), (.35, .006)):
        noise = gaussian_filter(rng.normal(size=(size, size)), sigma, mode="reflect")
        texture += weight * noise / max(float(noise.std()), 1e-8)
    roughness = .45 + .9 / (1 + np.exp(-(xn - .48) * 14))
    if style == "rugged":
        roughness += .7 * (np.sin(10 * xn + 7 * yn) ** 2)
    result = .46 + roughness * texture + .035 * xn
    # Elliptical rims, darker bowls, and directional contrast evoke craters.
    crater_count = 32 if style == "rugged" else 20
    for _ in range(crater_count):
        cx, cy = rng.uniform(.03, .97, 2) * size
        radius = rng.uniform(.017, .105) * size
        aspect = rng.uniform(.8, 1.18)
        dx, dy = (xx - cx) / radius, (yy - cy) / (radius * aspect)
        radius_map = np.hypot(dx, dy)
        rim = np.exp(-((radius_map - 1) / .13) ** 2)
        bowl = np.exp(-((radius_map / .82) ** 4))
        illumination = dx / np.maximum(radius_map, .05)
        result += .11 * rim + .16 * rim * illumination - .105 * bowl
        result += .055 * bowl * dx
    if style == "hidden_detail":
        # A smooth, textured-looking left half and a high-frequency right-half
        # checker pattern yield a deliberate example of invisible fine detail.
        right = xn > .55
        result[right] = .48 + .18 * ((xx[right].astype(int) + yy[right].astype(int)) % 2 * 2 - 1)
        result[right] += .005 * texture[right]
    return np.clip(result, 0, 1).astype(np.float32)

