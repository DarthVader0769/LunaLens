"""Complete, deterministic square tiling of a two-dimensional lunar image.

Boundary regions retain their real dimensions: no padding is observed, scored,
or counted toward coverage. Coordinates follow NumPy's (row, column) convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return value


@dataclass(frozen=True, slots=True)
class Region:
    """Immutable footprint of one possible observation; ``cost`` is positive."""

    id: int
    row: int
    col: int
    y0: int
    x0: int
    height: int
    width: int
    cost: float = 1.0

    def __post_init__(self) -> None:
        for name in ("id", "row", "col", "y0", "x0"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name, allow_zero=True))
        for name in ("height", "width"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if isinstance(self.cost, (bool, np.bool_)) or not isinstance(self.cost, Real):
            raise ValueError("region cost must be a finite positive number")
        cost = float(self.cost)
        if not np.isfinite(cost) or cost <= 0:
            raise ValueError("region cost must be a finite positive number")
        object.__setattr__(self, "cost", cost)

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y0 + self.height), slice(self.x0, self.x0 + self.width)

    @property
    def area(self) -> int:
        return self.height * self.width


def segment_image(
    shape: Sequence[int], region_size: int, cost_mode: str = "uniform"
) -> tuple[Region, ...]:
    """Tile ``shape`` completely in row-major order, starting with region ID 0.

    ``uniform`` charges one budget unit for every tile, including partial edge
    tiles. ``area`` charges ``tile.area / region_size**2``. Neither cost model
    attempts to represent spacecraft slew, downlink, or orbital constraints.
    """
    try:
        if len(shape) != 2:
            raise ValueError("shape must contain exactly two image dimensions")
        height = _positive_int(shape[0], "image height")
        width = _positive_int(shape[1], "image width")
    except TypeError as exc:
        raise ValueError("shape must contain exactly two image dimensions") from exc
    size = _positive_int(region_size, "region_size")
    if cost_mode not in ("uniform", "area"):
        raise ValueError("cost_mode must be 'uniform' or 'area'")
    regions: list[Region] = []
    for row, y0 in enumerate(range(0, height, size)):
        for col, x0 in enumerate(range(0, width, size)):
            tile_height, tile_width = min(size, height - y0), min(size, width - x0)
            cost = 1.0 if cost_mode == "uniform" else tile_height * tile_width / size**2
            regions.append(Region(len(regions), row, col, y0, x0, tile_height, tile_width, cost))
    return tuple(regions)
