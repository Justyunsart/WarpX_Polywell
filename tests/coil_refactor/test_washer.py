"""
Phase 4 gate: the Washer composite.

  A. STRUCTURE  — expand() yields `resolution` coaxial loops at radii
     linspace(r_inner, r_outer, resolution), all sharing axis/position/current;
     degenerate + invalid cases behave.

  B. DIFFERENTIAL ORACLE on a washer config — the three independent field
     representations agree when fed the washer's loops, proving the Washer plugs
     into every adapter (magpylib, analytic B, analytic A) consistently:
       1. to_collection(loops).getB  vs  analytic B_from_loops(loops)
       2. curl(A_from_loops(loops))  vs  analytic B_from_loops(loops)

  C. CONSISTENCY  — a per-face Washer uses the same radii the legacy
     build_n_turn_aext_expression sweeps, and build_aext_from_loops emits one
     self-contained coil entry per sub-loop (valid A_external for WarpX).

Run:
    python tests/coil_refactor/test_washer.py
"""
import sys

import numpy as np

from warpx_polywell.coils import Loop, Washer
from warpx_polywell.bext.make_collection import to_collection
from warpx_polywell.bext.analytic import B_from_loops, build_aext_from_loops
from warpx_polywell.bext.vector_potential import curl_A

from _config import eval_grid, A_from_loops


def _rel_linf(num, ref, mask=None):
    d, r = np.abs(num - ref), np.abs(ref)
    if mask is not None:
        d, r = d[mask], r[mask]
    return d.max() / max(r.max(), 1e-30)


def structure():
    print("A. STRUCTURE")
    ok = True

    w = Washer(axis="z", position=0.435, current=1.0e6,
               r_inner=0.40, r_outer=0.60, resolution=5)
    loops = w.expand()

    count_ok = len(loops) == 5
    ok &= count_ok
    print(f"  expand() count == resolution        : [{'PASS' if count_ok else 'FAIL'}]")

    radii = [lp.radius for lp in loops]
    radii_ok = np.allclose(radii, np.linspace(0.40, 0.60, 5))
    ok &= radii_ok
    print(f"  radii == linspace(inner,outer,res)  : [{'PASS' if radii_ok else 'FAIL'}]")

    shared_ok = all(lp.axis == "z" and lp.position == 0.435 and lp.current == 1.0e6
                    for lp in loops)
    ok &= shared_ok
    print(f"  axis/position/current shared        : [{'PASS' if shared_ok else 'FAIL'}]")

    degen_ok = len(Washer("x", 0.0, 1.0, 0.5, 0.5, 1).expand()) == 1
    ok &= degen_ok
    print(f"  resolution=1 -> single loop         : [{'PASS' if degen_ok else 'FAIL'}]")

    bad = 0
    for args in [("z", 0.0, 1.0, -0.1, 0.5, 3),   # r_inner <= 0
                 ("z", 0.0, 1.0, 0.6, 0.5, 3),    # r_outer < r_inner
                 ("z", 0.0, 1.0, 0.4, 0.5, 0),    # resolution < 1
                 ("w", 0.0, 1.0, 0.4, 0.5, 3)]:   # bad axis
        try:
            Washer(*args)
        except ValueError:
            bad += 1
    val_ok = bad == 4
    ok &= val_ok
    print(f"  rejects invalid parameters          : [{'PASS' if val_ok else 'FAIL'}]")
    return ok


def oracle():
    print("\nB. DIFFERENTIAL ORACLE (washer config)")
    ok = True
    X, Y, Z, h = eval_grid()
    mesh = np.stack([X, Y, Z], axis=-1)

    # A single z-face washer (radii safely > grid extent, so no on-wire points).
    loops = Washer(axis="z", position=0.435, current=1.0e6,
                   r_inner=0.40, r_outer=0.60, resolution=6).expand()

    B_mag = to_collection(loops).getB(mesh)
    Bx_a, By_a, Bz_a = B_from_loops(X, Y, Z, loops)
    e1 = max(_rel_linf(B_mag[..., 0], Bx_a),
             _rel_linf(B_mag[..., 1], By_a),
             _rel_linf(B_mag[..., 2], Bz_a))
    ok1 = e1 < 1e-3
    ok &= ok1
    print(f"  1. magpylib getB vs analytic B      : rel-L_inf = {e1:.3e}  [{'PASS' if ok1 else 'FAIL'}]")

    Ax, Ay, Az = A_from_loops(X, Y, Z, loops)
    Bx_c, By_c, Bz_c = curl_A(Ax, Ay, Az, (h, h, h))
    interior = np.zeros_like(X, dtype=bool)
    interior[2:-2, 2:-2, 2:-2] = True
    rho = np.sqrt(X**2 + Y**2)          # z-axis washer -> rho about z
    mask = interior & (rho > 0.05)
    e2 = max(_rel_linf(Bx_c, Bx_a, mask),
             _rel_linf(By_c, By_a, mask),
             _rel_linf(Bz_c, Bz_a, mask))
    ok2 = e2 < 5e-2
    ok &= ok2
    print(f"  2. curl(A) vs analytic B            : rel-L_inf = {e2:.3e}  [{'PASS' if ok2 else 'FAIL'}]")
    return ok


def consistency():
    print("\nC. CONSISTENCY (feeds analytic A adapter)")
    ok = True
    # Per-face washer radii match the legacy n-turn linspace(a, b, n).
    a, b, n = 0.40, 0.60, 5
    loops = Washer("z", 0.435, 1e6, a, b, n).expand()
    A = build_aext_from_loops(loops)
    keys_ok = list(A.keys()) == [f"coil_{i+1}" for i in range(n)]
    ok &= keys_ok
    print(f"  build_aext_from_loops -> coil_1..n  : [{'PASS' if keys_ok else 'FAIL'}]")

    entry_ok = all(set(v) == {"Ax_external_function", "Ay_external_function",
                              "Az_external_function", "A_time_external_function"}
                   for v in A.values())
    ok &= entry_ok
    print(f"  each coil is a full A_external entry: [{'PASS' if entry_ok else 'FAIL'}]")
    return ok


def main():
    ok = structure() & oracle() & consistency()
    print(f"\nPhase 4 gate (Washer): {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
