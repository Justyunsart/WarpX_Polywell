"""
Phase 0 verification: differential oracle + golden regression.

Two independent jobs, both printed with PASS/FAIL and the numbers behind them:

  A. DIFFERENTIAL ORACLE  — do the independent field representations agree?
       1. magpylib getB   vs  analytic B_polywell      → exposes any cm/m unit
                                                          mismatch between the two
                                                          representations
       2. curl(analytic A) vs analytic B_polywell      → validates the A_phi
                                                          physics the washer reuses
     This runs against whatever code is currently checked out. In Phase 0 it
     tells us whether the two pipelines agree TODAY (before any refactor).

  B. GOLDEN REGRESSION    — does current code still match the frozen snapshot?
       exact string equality for the 3 parser-expression fixtures, and
       np.allclose for the magpylib B grid. Skipped (with a note) if fixtures
       don't exist yet.

Run:
    python tests/coil_refactor/check.py
Exit code is non-zero if any gate fails.
"""
import json
import pathlib
import sys

import numpy as np

from warpx_polywell.bext.analytic import (
    B_polywell,
    build_bext_expressions,
    build_aext_expressions,
    build_n_turn_aext_expression,
)
from warpx_polywell.bext.make_collection import make_polywell_collection
from warpx_polywell.bext.vector_potential import curl_A

from _config import PARAMS, NTURN, eval_grid, A_polywell

FIX = pathlib.Path(__file__).parent / "fixtures"


def _rel_linf(num, ref, mask=None):
    """Relative L-inf error ||num-ref||_inf / max|ref|, optionally masked."""
    d = np.abs(num - ref)
    r = np.abs(ref)
    if mask is not None:
        d, r = d[mask], r[mask]
    return d.max() / max(r.max(), 1e-30)


def oracle():
    print("=" * 68)
    print("A. DIFFERENTIAL ORACLE  (independent representations agree?)")
    print("=" * 68)
    X, Y, Z, h = eval_grid()
    mesh = np.stack([X, Y, Z], axis=-1)
    I, dia, off = PARAMS["I"], PARAMS["dia"], PARAMS["offset"]

    # --- leg 1: magpylib vs analytic B ---
    B_mag = make_polywell_collection(I, dia, off).getB(mesh)
    Bx_a, By_a, Bz_a = B_polywell(X, Y, Z, I, dia, off)
    e_mag = max(_rel_linf(B_mag[..., 0], Bx_a),
                _rel_linf(B_mag[..., 1], By_a),
                _rel_linf(B_mag[..., 2], Bz_a))
    ok1 = e_mag < 1e-3
    print(f"  1. magpylib getB   vs analytic B_polywell : rel-L_inf = {e_mag:.3e}"
          f"   [{'PASS' if ok1 else 'FAIL'}]  (tol 1e-3)")
    if not ok1:
        print("     -> the two representations DISAGREE. If it's a clean scale"
              " factor, suspect a cm/m unit mismatch (see design note §5 trap 1).")

    # --- leg 2: curl(analytic A) vs analytic B ---
    Ax, Ay, Az = A_polywell(X, Y, Z, I, dia, off)
    Bx_c, By_c, Bz_c = curl_A(Ax, Ay, Az, (h, h, h))
    # np.gradient is only 2nd-order in the interior and 1/rho amplifies noise
    # near the axes, so compare the interior away from rho->0.
    interior = np.zeros_like(X, dtype=bool)
    interior[2:-2, 2:-2, 2:-2] = True
    rho_min = np.sqrt(np.minimum.reduce([X**2 + Y**2, Y**2 + Z**2, X**2 + Z**2]))
    mask = interior & (rho_min > 0.05)
    e_curl = max(_rel_linf(Bx_c, Bx_a, mask),
                 _rel_linf(By_c, By_a, mask),
                 _rel_linf(Bz_c, Bz_a, mask))
    ok2 = e_curl < 5e-2
    print(f"  2. curl(analytic A) vs analytic B_polywell: rel-L_inf = {e_curl:.3e}"
          f"   [{'PASS' if ok2 else 'FAIL'}]  (tol 5e-2, finite-diff on coarse grid)")

    print(f"\n  |B|_max (magpylib) = {np.abs(B_mag).max():.4e} T")
    print(f"  |B|_max (analytic) = {max(np.abs(Bx_a).max(), np.abs(By_a).max(), np.abs(Bz_a).max()):.4e} T")
    return ok1 and ok2


def regression():
    print("\n" + "=" * 68)
    print("B. GOLDEN REGRESSION  (current code == frozen snapshot?)")
    print("=" * 68)
    if not FIX.exists() or not (FIX / "bext_expr.json").exists():
        print("  (no fixtures yet — run snapshot.py first; skipping regression)")
        return True

    ok = True
    checks = [
        ("bext_expr.json", build_bext_expressions(**PARAMS)),
        ("aext_expr.json", build_aext_expressions(**PARAMS)),
        ("nturn_aext_expr.json", build_n_turn_aext_expression(**NTURN)),
    ]
    for fname, current in checks:
        golden = json.loads((FIX / fname).read_text())
        match = golden == current
        ok &= match
        print(f"  string equality {fname:24s}: [{'PASS' if match else 'FAIL'}]")

    X, Y, Z, _ = eval_grid()
    mesh = np.stack([X, Y, Z], axis=-1)
    B_now = make_polywell_collection(PARAMS["I"], PARAMS["dia"], PARAMS["offset"]).getB(mesh)
    B_gold = np.load(FIX / "magpylib_B.npy")
    close = np.allclose(B_now, B_gold, rtol=1e-10, atol=1e-14)
    ok &= close
    print(f"  allclose magpylib_B.npy         : [{'PASS' if close else 'FAIL'}]"
          f"   max|Δ| = {np.abs(B_now - B_gold).max():.2e}")
    return ok


def main():
    a = oracle()
    b = regression()
    print("\n" + "=" * 68)
    verdict = a and b
    print(f"OVERALL: {'PASS' if verdict else 'FAIL'}")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()