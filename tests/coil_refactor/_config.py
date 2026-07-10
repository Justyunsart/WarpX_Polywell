"""
Shared configuration for the coil-refactor Phase 0 baseline.

Single source of truth for the fixed parameter sets and evaluation grid used by
both the golden-master snapshot (`snapshot.py`) and the differential oracle /
regression check (`check.py`). Keeping params here means the snapshot and the
comparison can never drift apart.

All lengths are SI (metres), currents in amperes — matching the target
convention of the refactor.
"""
import numpy as np
from scipy.special import ellipk, ellipe

MU0 = 4e-7 * np.pi

# ----------------------------------------------------------------------
# Fixed physical parameters (a real polywell geometry: radius 0.5 m, coils
# offset 0.435 m from origin — matches the vector_potential.py demo).
# ----------------------------------------------------------------------
PARAMS = dict(I=1.0e6, dia=1.0, offset=0.435)

# n-turn / washer sweep parameters (inner->outer radius, resolution)
NTURN = dict(I=1.0e6, offset=0.435, a=0.40, b=0.60, n=5)


def eval_grid():
    """
    Uniform 3D grid used for every numeric comparison.

    Bounded to |x|,|y|,|z| <= 0.30 m so no sample point ever lands on a coil
    wire (coil radius is 0.5 m at offset 0.435 m), which would be singular.
    Returns X, Y, Z (each shape (N,N,N)) and the isotropic spacing h.
    """
    N = 21
    lo, hi = -0.30, 0.30
    x = np.linspace(lo, hi, N)
    h = x[1] - x[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return X, Y, Z, h


# ----------------------------------------------------------------------
# NumPy reference for the analytic *vector potential* (A_phi) of a single loop.
#
# This mirrors the formula emitted as a parser string by
# analytic.build_aext_expressions, evaluated in pure NumPy so we can curl it and
# check ∇×A == B independently of WarpX. It is the physics the washer will reuse,
# so pinning it now protects the highest-risk path.
#
#   A_phi = (mu0 I / pi) * sqrt(a/rho) * [(1 - k^2/2) K(k) - E(k)] / k
#   with k^2 = 4 a rho / ((rho+a)^2 + zeta^2)
# ----------------------------------------------------------------------

# Polywell coil layout, identical to analytic.POLYWELL_COILS
# (axis, position sign, current sign)
POLYWELL_COILS = [
    ("x", -1, -1), ("x", 1, 1),
    ("y", -1, -1), ("y", 1, 1),
    ("z", -1, -1), ("z", 1, 1),
]


def _aphi_single(rho, zeta, a, I):
    """Scalar azimuthal vector potential A_phi of one loop (radius a, current I)."""
    rho = np.asarray(rho, dtype=float)
    rho_safe = np.where(rho < 1e-30, 1e-30, rho)
    b2 = (rho_safe + a) ** 2 + zeta ** 2
    k2 = np.clip(4 * a * rho_safe / b2, 0.0, 1 - 1e-12)
    k = np.sqrt(k2)
    K = ellipk(k2)   # scipy takes m = k^2
    E = ellipe(k2)
    aphi = (MU0 * I / np.pi) * np.sqrt(a / rho_safe) * ((1 - 0.5 * k2) * K - E) / k
    return np.where(rho < 1e-12, 0.0, aphi)


def _loop_A_cartesian(X, Y, Z, axis, pos, a, I):
    """One loop's A in Cartesian components, matching analytic._coil_aext_term."""
    if axis == "z":
        rho = np.sqrt(X ** 2 + Y ** 2)
        zeta = Z - pos
        aphi = _aphi_single(rho, zeta, a, I)
        r = np.where(rho < 1e-30, 1e-30, rho)
        return -aphi * Y / r, aphi * X / r, np.zeros_like(X)
    if axis == "x":
        rho = np.sqrt(Z ** 2 + Y ** 2)
        zeta = X - pos
        aphi = _aphi_single(rho, zeta, a, I)
        r = np.where(rho < 1e-30, 1e-30, rho)
        return np.zeros_like(X), -aphi * Z / r, aphi * Y / r
    # axis == "y"
    rho = np.sqrt(Z ** 2 + X ** 2)
    zeta = Y - pos
    aphi = _aphi_single(rho, zeta, a, I)
    r = np.where(rho < 1e-30, 1e-30, rho)
    return aphi * Z / r, np.zeros_like(X), -aphi * X / r


def A_from_loops(X, Y, Z, loops):
    """Total vector potential of an arbitrary list[Loop] (NumPy reference)."""
    Ax = np.zeros_like(X, dtype=float)
    Ay = np.zeros_like(X, dtype=float)
    Az = np.zeros_like(X, dtype=float)
    for lp in loops:
        ax, ay, az = _loop_A_cartesian(X, Y, Z, lp.axis, lp.position, lp.radius, lp.current)
        Ax += ax; Ay += ay; Az += az
    return Ax, Ay, Az


def A_polywell(X, Y, Z, I, dia, offset):
    """Total vector potential of the 6-coil polywell (NumPy reference)."""
    a = dia / 2
    Ax = np.zeros_like(X, dtype=float)
    Ay = np.zeros_like(X, dtype=float)
    Az = np.zeros_like(X, dtype=float)
    for axis, ps, isg in POLYWELL_COILS:
        ax, ay, az = _loop_A_cartesian(X, Y, Z, axis, ps * offset, a, isg * I)
        Ax += ax; Ay += ay; Az += az
    return Ax, Ay, Az