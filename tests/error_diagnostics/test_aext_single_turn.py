"""
Validation tests for src/aext/aext.py's vector potential implementation.

Confirms curl(A) reproduces the known closed-form B-field of a circular
current loop, and that residual error shrinks with grid resolution rather
than plateauing -- the signature that distinguishes a real formula bug
from ordinary finite-difference discretization error.
"""

import numpy as np
from scipy.special import ellipk, ellipe

from src.warpx_polywell.bext.aext import _A_single_n_turn_coil, curlA
from src.warpx_polywell.bext.analytic import B_single_loop, _eval_loop_cartesian

MU0 = 4e-7 * np.pi

def _build_grid(n, extent_xy, z_lo, z_hi):
    xs = np.linspace(-extent_xy, extent_xy, n)
    ys = np.linspace(-extent_xy, extent_xy, n)
    zs = np.linspace(z_lo, z_hi, n)
    dx, dy, dz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    return X, Y, Z, dx, dy, dz, xs, ys, zs


def _cartesian_ground_truth(axis, x, y, z, pos, R, I):
    if axis not in ['x', 'y', 'z']:
        raise ValueError(axis)
    return np.asarray(_eval_loop_cartesian(x, y, z, axis, pos, R, I))


def test_aphi_matches_ground_truth():
    R, I, pos = 0.5, 1e6, 1.0
    n = 81
    rel_tol = 0.02  # separates fixed (~0.1-0.5%) from buggy (~12-14%) formula

    for axis in ('x', 'y', 'z'):
        X, Y, Z, dx, dy, dz, xs, ys, zs = _build_grid(n, extent_xy=1.5, z_lo=-1.0, z_hi=3.0)
        Ax, Ay, Az = _A_single_n_turn_coil(X, Y, Z, axis, pos, I=I, a=R, b=R, n=1)
        B = curlA(Ax, Ay, Az, dx, dy, dz)

        i, j, k = n // 2 + 10, n // 2 + 5, n // 2 + 15  # interior, off-axis, away from wire
        x0, y0, z0 = X[i, j, k], Y[i, j, k], Z[i, j, k]

        B_true = _cartesian_ground_truth(axis, x0, y0, z0, pos, R, I)
        B_calc = np.array([B['x'][i, j, k], B['y'][i, j, k], B['z'][i, j, k]])

        rel_err = np.abs(B_true - B_calc) / np.abs(B_true)
        print(f"[curlA = cartesian ground truth]")
        print(f"B_true: {B_true}")
        print(f"B_calc: {B_calc}")
        print(f"rel err: {rel_err}")
        assert np.all(rel_err < rel_tol), (
            f"axis={axis}: B_true={B_true}, B_calc={B_calc}, rel_err={rel_err}"
        )

def test_aphi_resolution_convergence():
    """Error should shrink as resolution increases. A real formula bug
    plateaus instead -- this is what separates a genuine bug from ordinary
    finite-difference discretization error."""
    R, I, pos = 0.5, 1e6, 1.0
    x0, y0, z0 = 0.375, 0.1875, 1.75

    errors = []
    for n in (41, 81, 161):
        X, Y, Z, dx, dy, dz, xs, ys, zs = _build_grid(n, extent_xy=1.5, z_lo=-1.0, z_hi=3.0)
        i = np.argmin(np.abs(xs - x0))
        j = np.argmin(np.abs(ys - y0))
        k = np.argmin(np.abs(zs - z0))
        xt, yt, zt = X[i, j, k], Y[i, j, k], Z[i, j, k]

        Ax, Ay, Az = _A_single_n_turn_coil(X, Y, Z, 'z', pos, I=I, a=R, b=R, n=1)
        B = curlA(Ax, Ay, Az, dx, dy, dz)

        B_true = _cartesian_ground_truth('z', xt, yt, zt, pos, R, I)
        rel_err = abs(B_true[2] - B['z'][i, j, k]) / abs(B_true[2])
        errors.append(rel_err)

    assert errors[1] < errors[0], f"error did not shrink: {errors}"
    assert errors[2] < errors[1], f"error did not shrink: {errors}"

test_aphi_matches_ground_truth()
test_aphi_resolution_convergence()