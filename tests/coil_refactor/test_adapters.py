"""
Phase 2 gate: the adapters reproduce prior behavior from a list[Loop].

  1. to_collection(Polywell.expand()).getB  ==  golden magpylib_B  (allclose)
  2. to_collection    ==  make_polywell_collection (same physical field, since the
     old constructor is still present in Phase 2) — machine precision.

String-equality of the refactored analytic builders is covered by re-running
check.py (its regression block). Run both:

    python tests/coil_refactor/check.py
    python tests/coil_refactor/test_adapters.py
"""
import pathlib
import sys

import numpy as np

from warpx_polywell.coils import Polywell
from warpx_polywell.bext.make_collection import to_collection, make_polywell_collection

from _config import PARAMS, eval_grid

FIX = pathlib.Path(__file__).parent / "fixtures"


def main():
    ok = True
    I, dia, off = PARAMS["I"], PARAMS["dia"], PARAMS["offset"]
    X, Y, Z, _ = eval_grid()
    mesh = np.stack([X, Y, Z], axis=-1)

    loops = Polywell(I, dia, off).expand()
    B_new = to_collection(loops).getB(mesh)

    # 1. vs frozen golden array
    B_gold = np.load(FIX / "magpylib_B.npy")
    d1 = np.abs(B_new - B_gold).max()
    ok1 = np.allclose(B_new, B_gold, rtol=1e-9, atol=1e-12)
    ok &= ok1
    print(f"  to_collection.getB vs golden magpylib_B : max|Δ|={d1:.2e}  [{'PASS' if ok1 else 'FAIL'}]")

    # 2. vs the still-present old constructor
    B_old = make_polywell_collection(I, dia, off).getB(mesh)
    d2 = np.abs(B_new - B_old).max()
    ok2 = np.allclose(B_new, B_old, rtol=1e-9, atol=1e-12)
    ok &= ok2
    print(f"  to_collection vs make_polywell_collection: max|Δ|={d2:.2e}  [{'PASS' if ok2 else 'FAIL'}]")

    print(f"\nPhase 2 gate (adapters): {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
