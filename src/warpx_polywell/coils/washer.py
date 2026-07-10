"""
Washer composite — a thick annular winding, discretized into coaxial loops.

A `Washer` takes the same locating arguments as a single `Loop` (axis,
position, current) but replaces the single radius with an inner radius, an outer
radius, and a resolution: it `expand()`s into `resolution` coaxial `Loop`s whose
radii are evenly spaced over [r_inner, r_outer], each carrying the full current.

Because it expands to plain `list[Loop]`, a Washer feeds every field adapter
(magpylib Collection, analytic B, analytic A_ext) with no special-casing — which
is the whole point: unlike a bare magpylib.Collection, the loops retain the
per-loop geometry the analytic vector-potential builder needs.

Units are strict SI: metres and amperes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from warpx_polywell.coils.primitives import Loop, AXES


@dataclass(frozen=True)
class Washer:
    """
    A radially-discretized annular coil.

    Parameters
    ----------
    axis       : 'x' | 'y' | 'z' — symmetry axis (shared by every sub-loop).
    position   : signed centre coordinate along `axis`, in metres.
    current    : signed current carried by each sub-loop, in amperes.
    r_inner    : innermost loop radius, in metres (> 0).
    r_outer    : outermost loop radius, in metres (>= r_inner).
    resolution : number of coaxial loops (>= 1). resolution == 1 degenerates to
                 a single loop at r_inner.
    """
    axis: str
    position: float
    current: float
    r_inner: float
    r_outer: float
    resolution: int

    def __post_init__(self):
        if self.axis not in AXES:
            raise ValueError(f"axis must be one of {AXES}, got {self.axis!r}")
        if self.r_inner <= 0:
            raise ValueError(f"r_inner must be > 0, got {self.r_inner}")
        if self.r_outer < self.r_inner:
            raise ValueError(
                f"r_outer ({self.r_outer}) must be >= r_inner ({self.r_inner})")
        if self.resolution < 1:
            raise ValueError(f"resolution must be >= 1, got {self.resolution}")

    def expand(self) -> list[Loop]:
        radii = np.linspace(self.r_inner, self.r_outer, self.resolution)
        return [Loop(axis=self.axis,
                     position=self.position,
                     radius=float(r),
                     current=self.current)
                for r in radii]
