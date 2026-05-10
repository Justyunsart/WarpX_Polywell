"""
Domain spec for full-domain and symmetry-reduced (octant) polywell runs.

The user always types `L` and `N` in full-domain physical terms (box spans
[-L, +L] in each direction; N cells per axis covers that range). `symmetry`
is the single knob that decides whether the actual simulated domain is the
full cube or just the +x/+y/+z octant. Everything else — bounds, cell count,
field BCs, particle BCs — is derived here so the input deck never branches
on `symmetry` after `derive_domain`.
"""
from dataclasses import dataclass

import numpy as np


VALID_SYMMETRIES = ("full", "octant")


@dataclass(frozen=True)
class Domain:
    L: float
    N: int
    symmetry: str
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    n_cells: tuple[int, int, int]
    field_bc_lo: tuple[str, str, str]
    field_bc_hi: tuple[str, str, str]
    particle_bc_lo: tuple[str, str, str]
    particle_bc_hi: tuple[str, str, str]


def derive_domain(symmetry: str, L: float, N: int) -> Domain:
    if symmetry == "full":
        return Domain(
            L=L, N=N, symmetry="full",
            lower=(-L, -L, -L),
            upper=(+L, +L, +L),
            n_cells=(N, N, N),
            field_bc_lo=("open", "open", "open"),
            field_bc_hi=("open", "open", "open"),
            particle_bc_lo=("absorbing", "absorbing", "absorbing"),
            particle_bc_hi=("absorbing", "absorbing", "absorbing"),
        )
    if symmetry == "octant":
        if N % 2:
            raise ValueError(f"N must be even for octant symmetry (got {N})")
        half = N // 2
        # PMC, not PEC, is the symmetry-plane BC for a mirror-symmetric source
        # of an axial-vector (magnetic) field. WarpX docs: PMC "models a
        # symmetric surface where charges and currents are symmetric across
        # the boundary"; it sets tangential B = 0 and normal E = 0, both of
        # which the polywell's mirror-symmetric B and self-consistent E
        # satisfy on x=0, y=0, z=0. PEC (tangential E = 0, normal B = 0) is
        # the opposite symmetry class and would zero out the normal B that
        # actually passes through the cusp.
        return Domain(
            L=L, N=N, symmetry="octant",
            lower=(0.0, 0.0, 0.0),
            upper=(+L, +L, +L),
            n_cells=(half, half, half),
            field_bc_lo=("pmc", "pmc", "pmc"),
            field_bc_hi=("open", "open", "open"),
            particle_bc_lo=("reflecting", "reflecting", "reflecting"),
            particle_bc_hi=("absorbing", "absorbing", "absorbing"),
        )
    raise ValueError(
        f"Unknown symmetry mode '{symmetry}' (use one of {VALID_SYMMETRIES})"
    )


def plasma_bounds(domain: Domain, plasma_bounding: float):
    """Plasma loading bounds, clipped to the simulated domain.

    `plasma_bounding` is the user's full-domain fraction (e.g. 0.11 means
    plasma is loaded inside [-0.11 L, +0.11 L] in full mode). In octant mode
    the lower bound is clipped to the symmetry plane at 0.
    """
    L = domain.L
    physical_lo = -plasma_bounding * np.array([L, L, L])
    physical_hi = +plasma_bounding * np.array([L, L, L])
    lo = np.maximum(np.array(domain.lower), physical_lo)
    hi = np.minimum(np.array(domain.upper), physical_hi)
    return lo, hi
