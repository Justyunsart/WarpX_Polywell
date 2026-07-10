"""
Phase 1 gate: the Loop primitive + Polywell composite reproduce the layout that
was previously hardcoded as analytic.POLYWELL_COILS.

Deriving (axis, position sign, current sign) back out of Polywell.expand() and
asserting it equals POLYWELL_COILS proves the new single-source-of-truth encodes
exactly the same six coils — no behavior change, just a new representation.

Run:
    python tests/coil_refactor/test_primitives.py
"""
import sys

from warpx_polywell.coils import Loop, Polywell

# The canonical polywell layout (axis, position sign, current sign). This was
# formerly analytic.POLYWELL_COILS; it now lives only in Polywell, so the test
# pins Polywell against this literal spec.
EXPECTED_LAYOUT = [
    ("x", -1, -1), ("x", 1, 1),
    ("y", -1, -1), ("y", 1, 1),
    ("z", -1, -1), ("z", 1, 1),
]


def main():
    ok = True

    # Use unit-ish params so signs are trivially recoverable:
    #   position sign = sign(position) since offset=1
    #   current  sign = sign(current)  since I=1
    loops = Polywell(current=1.0, diameter=2.0, offset=1.0).expand()

    derived = [(lp.axis,
                int(round(lp.position)),
                int(round(lp.current)))
               for lp in loops]
    match = derived == EXPECTED_LAYOUT
    ok &= match
    print(f"  Polywell.expand() == canonical layout: [{'PASS' if match else 'FAIL'}]")
    if not match:
        print(f"    derived:  {derived}")
        print(f"    expected: {EXPECTED_LAYOUT}")

    # radius = diameter/2 for every loop
    radii_ok = all(abs(lp.radius - 1.0) < 1e-15 for lp in loops)
    ok &= radii_ok
    print(f"  radius == diameter/2                : [{'PASS' if radii_ok else 'FAIL'}]")

    # A physical config maps through correctly (spot check one coil).
    pw = Polywell(current=1.0e6, diameter=1.0, offset=0.435)
    lp0 = pw.expand()[0]                      # ('x', -1, -1) face
    spot_ok = (lp0.axis == "x" and abs(lp0.position + 0.435) < 1e-15
               and abs(lp0.radius - 0.5) < 1e-15 and abs(lp0.current + 1.0e6) < 1e-9)
    ok &= spot_ok
    print(f"  physical config maps correctly      : [{'PASS' if spot_ok else 'FAIL'}]")

    # Loop validates its axis.
    try:
        Loop(axis="w", position=0.0, radius=1.0, current=1.0)
        val_ok = False
    except ValueError:
        val_ok = True
    ok &= val_ok
    print(f"  Loop rejects bad axis               : [{'PASS' if val_ok else 'FAIL'}]")

    print(f"\nPhase 1 gate: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()