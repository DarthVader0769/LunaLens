"""Fixed-range image validation and an explicit, reproducible sensor surrogate.

The degradation is a Gaussian optical blur, area averaging onto a coarse grid,
and bilinear interpolation back to the reference grid. It is a simulated initial
measurement, not a calibrated LROC sensor model or super-resolution algorithm.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize, resize_local_mean


def validate_image(image: np.ndarray) -> np.ndarray:
    """Return a detached float32 copy of a finite 2-D image in [0, 1].

    Explicit validation prevents silent NaN propagation and ambiguous intensity
    scales. Integer image files must first go through ``load_image``.
    """
    data = np.asarray(image)
    if data.ndim != 2 or not data.size or min(data.shape) < 1:
        raise ValueError("Image must be a nonempty 2-D grayscale array.")
    if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
        raise ValueError("Image values must be real numbers.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Image must contain only finite values.")
    if np.any(data < 0) or np.any(data > 1):
        raise ValueError("Image values must be in [0, 1]; use load_image for files.")
    return np.array(data, dtype=np.float32, order="C", copy=True)


def coarse_shape(shape: tuple[int, int], factor: int) -> tuple[int, int]:
    """Use ceil division so non-divisible shapes retain their entire field."""
    if isinstance(factor, bool) or not isinstance(factor, Integral) or factor < 1:
        raise ValueError("Downsample factor must be a positive integer.")
    return tuple(max(1, (int(n) + int(factor) - 1) // int(factor)) for n in shape)


def degrade_image(
    image: np.ndarray, factor: int = 4, blur_sigma: float = 1.0
) -> np.ndarray:
    """Produce an initially available map with the same shape as the reference.

    ``factor`` is a linear sampling ratio; factor 4 means approximately 1/16
    as many coarse samples. Sigma is expressed in reference-grid pixels.
    Setting factor=1 and sigma=0 is an exact identity (useful for tests).
    No per-image contrast stretch or truth-derived feature is introduced.
    """
    reference = validate_image(image)
    target = coarse_shape(reference.shape, factor)
    if isinstance(blur_sigma, bool) or not isinstance(blur_sigma, Real):
        raise ValueError("Blur sigma must be a finite nonnegative number.")
    if not np.isfinite(blur_sigma) or blur_sigma < 0:
        raise ValueError("Blur sigma must be a finite nonnegative number.")
    blurred = (
        gaussian_filter(reference, sigma=float(blur_sigma), mode="reflect")
        if blur_sigma > 0
        else reference.copy()
    )
    if target == reference.shape:
        return blurred.astype(np.float32, copy=True)
    coarse = resize_local_mean(blurred, target, grid_mode=True, preserve_range=True)
    restored = resize(
        coarse, reference.shape, order=1, mode="reflect", anti_aliasing=False,
        preserve_range=True,
    )
    return np.clip(restored, 0, 1).astype(np.float32)

