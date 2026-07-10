"""
Phase 0 golden-master snapshot.

Runs the CURRENT coil code and freezes its outputs to fixtures/ so the refactor
can be verified with `assert new == old`:

  fixtures/bext_expr.json        build_bext_expressions(**PARAMS)          (exact strings)
  fixtures/aext_expr.json        build_aext_expressions(**PARAMS)          (exact strings)
  fixtures/nturn_aext_expr.json  build_n_turn_aext_expression(**NTURN)     (exact strings)
  fixtures/magpylib_B.npy        make_polywell_collection(...).getB(grid)  (numeric)

Run once, against the pre-refactor code, then commit the fixtures:
    python tests/coil_refactor/snapshot.py
"""
import json
import pathlib

import numpy as np

from warpx_polywell.bext.analytic import (
    build_bext_expressions,
    build_aext_expressions,
    build_n_turn_aext_expression,
)
from warpx_polywell.bext.make_collection import make_polywell_collection

from _config import PARAMS, NTURN, eval_grid

FIX = pathlib.Path(__file__).parent / "fixtures"


def main():
    FIX.mkdir(exist_ok=True)

    # --- exact parser strings (tolerance-free regression pins) ---
    (FIX / "bext_expr.json").write_text(
        json.dumps(build_bext_expressions(**PARAMS), indent=2, sort_keys=True))
    (FIX / "aext_expr.json").write_text(
        json.dumps(build_aext_expressions(**PARAMS), indent=2, sort_keys=True))
    (FIX / "nturn_aext_expr.json").write_text(
        json.dumps(build_n_turn_aext_expression(**NTURN), indent=2, sort_keys=True))

    # --- magpylib B on the fixed grid (numeric pin for to_collection) ---
    X, Y, Z, _ = eval_grid()
    mesh = np.stack([X, Y, Z], axis=-1)
    coll = make_polywell_collection(PARAMS["I"], PARAMS["dia"], PARAMS["offset"])
    B = coll.getB(mesh)                       # (N,N,N,3), Tesla
    np.save(FIX / "magpylib_B.npy", B)

    print(f"wrote fixtures to {FIX}/")
    print(f"  bext_expr.json        ({len((FIX/'bext_expr.json').read_text())} bytes)")
    print(f"  aext_expr.json        ({len((FIX/'aext_expr.json').read_text())} bytes)")
    print(f"  nturn_aext_expr.json  ({len((FIX/'nturn_aext_expr.json').read_text())} bytes)")
    print(f"  magpylib_B.npy        shape={B.shape}, |B|_max={np.abs(B).max():.4e} T")


if __name__ == "__main__":
    main()