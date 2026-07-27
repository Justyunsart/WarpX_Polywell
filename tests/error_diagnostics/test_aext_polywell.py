"""
Validation test for src/bext/aext.py's six-coil polywell assembly
(compute_A_polywell + curlA) against magpylib as ground truth.

Percentile-based thresholds are used instead of a single max/RMS check:
grid points sitting essentially on a coil's conductor are genuinely
singular (the true field there is enormous), so finite-grid curl error
blows up there regardless of formula correctness. The bulk of the domain
should still agree with magpylib to a fraction of a percent -- only a
small fraction of points immediately adjacent to a conductor are
expected to disagree more.
"""

import numpy as np

from src.warpx_polywell.bext.aext import compute_A_polywell, curlA
from src.warpx_polywell.bext.make_collection import make_polywell_collection


def _build_grid(L, N):
    xs = np.linspace(-L, L, N)
    ys = np.linspace(-L, L, N)
    zs = np.linspace(-L, L, N)
    dx, dy, dz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    mesh = np.moveaxis(np.meshgrid(xs, ys, zs, indexing='ij'), 0, -1)
    return X, Y, Z, mesh, dx, dy, dz


def test_polywell_matches_magpylib():
    L, N = 0.2, 128
    a = b = 0.05
    I = 1e6
    offset = 0.1
    dia = 0.1
    n = 1

    X, Y, Z, mesh, dx, dy, dz = _build_grid(L, N)

    Ax, Ay, Az = compute_A_polywell(X, Y, Z, I, offset, a, b, n)
    B_curlA = curlA(Ax, Ay, Az, dx, dy, dz)
    Bx_c, By_c, Bz_c = B_curlA['x'], B_curlA['y'], B_curlA['z']

    collection = make_polywell_collection(I, dia, offset)
    B_magpy = collection.getB(mesh)
    Bx_m, By_m, Bz_m = np.moveaxis(B_magpy, -1, 0)

    err_mag = np.sqrt((Bx_c - Bx_m) ** 2 + (By_c - By_m) ** 2 + (Bz_c - Bz_m) ** 2)
    B_mag = np.sqrt(Bx_m ** 2 + By_m ** 2 + Bz_m ** 2)
    rel_err = err_mag / (B_mag + 1e-12)

    # interior only -- exclude the boundary layer np.gradient can't center-difference
    sl = np.s_[1:-1, 1:-1, 1:-1]
    rel_err = rel_err[sl]

    median = np.median(rel_err)
    p95 = np.percentile(rel_err, 95)
    p99 = np.percentile(rel_err, 99)
    frac_above_10pct = np.mean(rel_err > 0.10)

    print(f"L: {L}\nN: {N}\ndx: {L / N}")
    print(f"median: {median}\np95: {p95}\np99: {p99}\n>10% error: {frac_above_10pct}")

    assert median < 0.01, f"median rel_err too high: {median*100:.3f}%"
    assert p95 < 0.05, f"95th percentile rel_err too high: {p95*100:.3f}%"
    assert p99 < 0.15, f"99th percentile rel_err too high: {p99*100:.3f}%"
    assert frac_above_10pct < 0.01, (
        f"too many points (>1%) disagree by >10%: {frac_above_10pct*100:.3f}%"
    )

test_polywell_matches_magpylib()