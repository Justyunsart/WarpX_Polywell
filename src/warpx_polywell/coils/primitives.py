"""
Canonical coil primitives — the single source of truth for coil geometry.

`Loop` is a plain, magpylib-independent description of one circular current
loop. Every downstream consumer (magpylib Collection, analytic B parser
expressions, analytic A_ext parser expressions) is an *adapter* that reads a
`list[Loop]` — so the polywell layout, washer discretization, etc. are each
defined exactly once, as a composite that `.expand()`s into loops.

Units are strict SI: lengths in metres, current in amperes. Positions and
currents are signed.

See docs/design/coil_constructs.md for the full rationale.
"""
from __future__ import annotations

from dataclasses import dataclass

AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Loop:
    """
    One circular current loop, axis-aligned.

    Parameters
    ----------
    axis     : 'x' | 'y' | 'z' — the loop's symmetry axis (its normal).
    position : signed centre coordinate along `axis`, in metres.
    radius   : loop radius, in metres.
    current  : signed current, in amperes.
    """
    axis: str
    position: float
    radius: float
    current: float

    def __post_init__(self):
        if self.axis not in AXES:
            raise ValueError(f"axis must be one of {AXES}, got {self.axis!r}")
        if self.radius <= 0:
            raise ValueError(f"radius must be > 0, got {self.radius}")


@dataclass(frozen=True)
class Polywell:
    """
    Six-coil polywell: one loop on each face of a cube centred at the origin,
    with alternating current sign (the standard cusp configuration).

    Parameters
    ----------
    current  : coil current magnitude, in amperes.
    diameter : coil diameter, in metres.
    offset   : distance from origin to each coil centre, in metres.

    This is now the single source of truth for the layout (formerly the
    hardcoded analytic.POLYWELL_COILS). `expand()` yields, as
    (axis, position sign, current sign):
    (x,-,-) (x,+,+) (y,-,-) (y,+,+) (z,-,-) (z,+,+).
    """
    current: float
    diameter: float
    offset: float

    # (axis, position sign, current sign) — the canonical polywell layout.
    _LAYOUT = (
        ("x", -1, -1), ("x", 1, 1),
        ("y", -1, -1), ("y", 1, 1),
        ("z", -1, -1), ("z", 1, 1),
    )

    def expand(self) -> list[Loop]:
        a = self.diameter / 2
        return [
            Loop(axis=axis,
                 position=pos_sign * self.offset,
                 radius=a,
                 current=i_sign * self.current)
            for axis, pos_sign, i_sign in self._LAYOUT
        ]