"""
Validation tests for src/aext/aext.py's vector potential implementation.

Confirms curl(A) reproduces the known closed-form B-field of a circular
current loop, and that residual error shrinks with grid resolution rather
than plateauing -- the signature that distinguishes a real formula bug
from ordinary finite-difference discretization error.
"""

import numpy as np
from scipy.special import ellipk, ellipe

import src.warpx_polywell.bext.aext as aext
from src.warpx_polywell.bext.aext import _A_single_n_turn_coil, curlA, compute_A_polywell, POLYWELL_COILS
from src.warpx_polywell.bext.analytic import B_single_loop, _eval_loop_cartesian
from src.warpx_polywell.bext.bext import make_polywell_collection
from src.warpx_polywell.bext.make_collection import return_n_turn_coil
import magpylib

import matplotlib.pyplot as plt

aext.POLYWELL_COILS = [('x', +1, +1)] # single coil test

MU0 = 4e-7 * np.pi

def _build_grid(L, N):
    xs = np.linspace(-L, L, N)
    ys = np.linspace(-L, L, N)
    zs = np.linspace(-L, L, N)
    dx, dy, dz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]
    mesh = X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    return X, Y, Z, dx, dy, dz, xs, ys, zs, mesh


def _cartesian_ground_truth(axis, x, y, z, pos, R, I, B_ref):
    a = 0.05
    b = 0.9
    turns = 10
    rs = np.linspace(a, b, turns)
    B = np.zeros_like(B_ref)
    print(B.shape)
    if axis not in ['x', 'y', 'z']:
        raise ValueError(axis)
    for r in rs:
        B += np.asarray(_eval_loop_cartesian(x, y, z, axis, pos, r, I))
    return B


def test_aphi_matches_ground_truth():
    R, I, pos = 0.5, 1e6, 1.0
    a = 0.05
    b = 0.9
    turns = 10
    n = 81
    rel_tol = 0.02  # separates fixed (~0.1-0.5%) from buggy (~12-14%) formula
    L = 1.5

    for axis in ('x', 'y', 'z'):
        X, Y, Z, dx, dy, dz, xs, ys, zs = _build_grid(n, L, L, L)
        Ax, Ay, Az = _A_single_n_turn_coil(X, Y, Z, axis, pos, I=I, a=a, b=b, n=turns)
        B = curlA(Ax, Ay, Az, dx, dy, dz)
        print(B['x'].shape)

        i, j, k = n // 2 + 10, n // 2 + 5, n // 2 + 15  # interior, off-axis, away from wire
        x0, y0, z0 = X[i, j, k], Y[i, j, k], Z[i, j, k]

        B_true = _cartesian_ground_truth(axis, x0, y0, z0, pos, R, I, B['x'])
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
    L = 1.5

    errors = []
    for n in (41, 81, 161):
        X, Y, Z, dx, dy, dz, xs, ys, zs = _build_grid(n, L, L, L)
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

def test_matches_magpylib():
    L, N = 1.5, 128
    a = 0.05
    b = 0.9
    I = 2.25e4
    offset = 0.0
    dia = 0.1
    n = 10

    X, Y, Z, dx, dy, dz, xs, ys, zs, mesh = _build_grid(L, N)

    Ax, Ay, Az = compute_A_polywell(X, Y, Z, I, offset, a, b, n)
    print(Ax)
    B_curlA = curlA(Ax, Ay, Az, dx, dy, dz)
    Bx_c, By_c, Bz_c = B_curlA['x'], B_curlA['y'], B_curlA['z']

    print(Bx_c)

    collection = return_n_turn_coil(I, offset, 0, a, b, n)
    print(collection)
    mesh = np.moveaxis(mesh, 0, -1) # magpylib wants (Nx, Ny, Nz, 3)
    B_magpy = collection.getB(mesh)
    #magpylib.show(collection)
    Bx_m, By_m, Bz_m = np.moveaxis(B_magpy, -1, 0)

    print(X.shape, Bx_c.shape)
    print(Y.shape, By_c.shape)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.streamplot(
        X[:, 32, 32], 
        Y[32, :, 32], 
        Bx_c[:, :, 32].T,
        By_c[:, :, 32].T,
        color='blue',
        density=1.2
        )
    
    plt.show()

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

    print(f"L: {L}\nN: {N}\ndx: {2*L / N}")
    print(f"median: {median}\np95: {p95}\np99: {p99}\n>10% error: {frac_above_10pct}")

    assert median < 0.01, f"median rel_err too high: {median*100:.3f}%"
    assert p95 < 0.05, f"95th percentile rel_err too high: {p95*100:.3f}%"
    assert p99 < 0.15, f"99th percentile rel_err too high: {p99*100:.3f}%"
    assert frac_above_10pct < 0.01, (
        f"too many points (>1%) disagree by >10%: {frac_above_10pct*100:.3f}%"
    )



# test_aphi_matches_ground_truth()
# test_aphi_resolution_convergence()
test_matches_magpylib()