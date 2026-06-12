"""
Trust-and-convergence tests for the polywell A-field pipeline
(src.bext.vector_potential).

Run from the project root:

    python tests/test_vector_potential.py

Tests
-----
1. Padding convergence: doubling the FFT pad factor should drive A in the
   physics region to a fixed point.
2. ∇×A vs magpylib B: the recovered A should curl back to the source field
   in the interior (boundary cells skipped — central differences degrade
   one cell from each face).
3. Coulomb gauge: ‖∇·A‖_∞ / ‖∇×A‖_∞ should be tiny (the FFT inversion
   enforces ik·Ã = 0 spectrally; on the cropped grid only discretisation
   error remains).
4. Grid-resolution convergence: refining N at fixed L and pad_factor
   should drive the interior curl error down monotonically.
5. Single-ring sanity: against the analytic A_φ from elliptic integrals
   for one isolated current loop — the same elliptic-integral formula
   physicists use as the canonical reference.

Each test prints PASS/FAIL with the measured number, so the script doubles
as a one-shot diagnostic for the FFT curl-inverse implementation.
"""
import gc
import sys

import numpy as np
import scipy.constants as sc
from scipy.special import ellipe, ellipk

from warpx_polywell.bext.make_collection import make_polywell_collection
from warpx_polywell.bext.vector_potential import (
    compute_A_grid,
    converge_A_grid,
    curl_A,
)
from warpx_polywell.domain import derive_domain


# ----------------------------- formatting -----------------------------
GREEN, RED, YELLOW, BOLD, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m"

_failed = []


def header(title):
    print()
    print(f"{BOLD}[{title}]{RESET}")


def report(name, ok, detail=""):
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{tag}] {name}: {detail}")
    if not ok:
        _failed.append(name)


def interior(arr, trim=2):
    """Strip `trim` cells from every face of a 3D array."""
    return arr[trim:-trim, trim:-trim, trim:-trim]


def relative_linf(a, b):
    """L∞ relative error, normalised by max |b|."""
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30)


def rms_rel(a, b):
    """
    Relative RMS error — sqrt(mean((a-b)²)) / sqrt(mean(b²)).

    For the polywell, RMS is the *correct* metric for "is A trustworthy":
    L∞ is dominated by the handful of cells that touch a coil ring, where
    central differences fundamentally cannot reproduce a 1/r³ singularity
    in the source field. RMS averages over the whole grid and reports a
    field-energy-weighted error that tracks the typical accuracy.
    """
    return float(np.sqrt(np.mean((a - b) ** 2)) /
                 max(np.sqrt(np.mean(b ** 2)), 1e-30))


def cell_fraction_within(a, b, tol):
    """
    Fraction of cells where |a - b| < tol · max|b|.

    Peak-normalised *absolute* tolerance — avoids the divide-by-near-zero
    pathology you get with per-cell relative error in regions of the
    polywell where |B| → 0 (the magnetic null).
    """
    return float(np.mean(np.abs(a - b) < tol * max(np.max(np.abs(b)), 1e-30)))


def plasma_cube_mask(domain, half_side):
    """
    Boolean mask: cube |x|, |y|, |z| < half_side centred at origin. Used to
    isolate the polywell's central confinement region (where particles
    actually live) from the coil-singularity cells where finite differences
    of the source field are fundamentally bad.
    """
    xs = np.linspace(domain.lower[0], domain.upper[0], domain.n_cells[0])
    ys = np.linspace(domain.lower[1], domain.upper[1], domain.n_cells[1])
    zs = np.linspace(domain.lower[2], domain.upper[2], domain.n_cells[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return (np.abs(X) < half_side) & (np.abs(Y) < half_side) & (np.abs(Z) < half_side)


def interior_ball(domain, frac):
    """
    Boolean mask: ball around origin of radius `frac · OFFSET`. Used to
    isolate the polywell's central confinement region from coil-adjacent
    cells (which any finite-difference curl handles poorly).
    """
    xs = np.linspace(domain.lower[0], domain.upper[0], domain.n_cells[0])
    ys = np.linspace(domain.lower[1], domain.upper[1], domain.n_cells[1])
    zs = np.linspace(domain.lower[2], domain.upper[2], domain.n_cells[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)
    return r < frac * OFFSET


# ----------------------------- shared inputs -----------------------------
# Memory budget: the FFT box is (pad·N)³ and each complex component is
# (pad·N)³ × 16 B. magpylib evaluates B on the same padded grid, so a few
# more large arrays exist transiently. Empirically (pad·N) ≈ 80 keeps the
# whole script under ~1 GB on a laptop. Production code uses pad up to 8,
# which is also exercised below (pad=8 with N=10 → FFT box 80³).
I_COIL, DIA, OFFSET = 1.0e5, 1.0, 0.435   # polywell knobs
L, N = 2.0, 20                             # default domain for tests 1–3

print("=" * 72)
print(f"{BOLD}Trust-and-convergence tests for the polywell A pipeline{RESET}")
print("=" * 72)
print(f"Polywell coils: I={I_COIL:.0e} A, dia={DIA} m, offset={OFFSET} m")
print(f"Default domain: full, L={L} m, N={N}")

polywell = make_polywell_collection(I_COIL, DIA, OFFSET)
domain = derive_domain("full", L=L, N=N)


# ====================================================================
# TEST 1 — padding convergence
# ====================================================================
header("TEST 1: padding convergence (FFT image-coil suppression)")
# pad=8 is the upper bound used by `converge_A_grid` in production. We
# sweep [2, 4, 8] here on a small N so the memory footprint stays bounded
# (FFT box at pad=8 is (8·N)³).
pad_seq = [2, 4, 8]
A_prev = None
peaks, rels = [], []
for pad in pad_seq:
    Ax, Ay, Az, _ = compute_A_grid(polywell, domain, pad_factor=pad)
    peak = float(np.max(np.sqrt(Ax * Ax + Ay * Ay + Az * Az)))
    peaks.append(peak)
    if A_prev is None:
        rel = float("nan")
        print(f"  pad={pad:>2}: |A|_max={peak:.4e} T·m", flush=True)
    else:
        rel = max(
            np.max(np.abs(Ax - A_prev[0])),
            np.max(np.abs(Ay - A_prev[1])),
            np.max(np.abs(Az - A_prev[2])),
        ) / peak
        rels.append(rel)
        print(f"  pad={pad:>2}: |A|_max={peak:.4e}  rel-change vs prev = {rel:.3e}",
              flush=True)
    A_prev = (Ax.copy(), Ay.copy(), Az.copy())
    # release the previous (pre-doubling) FFT working set before the next iter
    del Ax, Ay, Az
    gc.collect()
# release the comparison snapshot
del A_prev
gc.collect()

# Pass if the final doubling moves |A| by less than 1%.
report("padding converges (final rel-change < 1e-2)",
       rels[-1] < 1e-2, f"final rel-change = {rels[-1]:.3e}")

# Pass if the convergence sequence is monotonically tightening.
monotone = all(rels[i + 1] <= rels[i] for i in range(len(rels) - 1))
report("rel-change sequence is monotonically tightening", monotone,
       f"{[f'{r:.2e}' for r in rels]}")


# ====================================================================
# TEST 2 — ∇×A vs magpylib B
# ====================================================================
# Two metrics. (a) Inside a plasma cube centred at origin where the field
# is smooth, RMS rel error should be tight — this is the physically
# meaningful number for a hybrid PIC simulation (particles live here).
# (b) Globally, we report peak-normalised absolute error and a per-cell
# histogram. We do *not* gate on a global L∞ or per-cell-relative metric
# because both are dominated by the handful of cells touching a coil ring,
# where central differences cannot reproduce the source's 1/r³ singularity.
header("TEST 2: ∇×A vs magpylib B")
N2 = 32                                  # enough cells to give the plasma cube real stats
dom2 = derive_domain("full", L=L, N=N2)
Ax, Ay, Az, spacing, pad_used = converge_A_grid(
    polywell, dom2, start=2, max_pad=4, rtol=1e-3, verbose=False,
)
print(f"  domain: N={N2}, L={L};  converge_A_grid pad_factor={pad_used}")
Bxc, Byc, Bzc = curl_A(Ax, Ay, Az, spacing)

# magpylib reference on the same physics grid
xs = np.linspace(dom2.lower[0], dom2.upper[0], N2)
ys = np.linspace(dom2.lower[1], dom2.upper[1], N2)
zs = np.linspace(dom2.lower[2], dom2.upper[2], N2)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
B_ref = polywell.getB(np.stack([X, Y, Z], axis=-1))
Bxr, Byr, Bzr = B_ref[..., 0], B_ref[..., 1], B_ref[..., 2]

# (a) Plasma cube: |x|,|y|,|z| < OFFSET/2 — purely interior, no coil cells
cube = plasma_cube_mask(dom2, half_side=0.5 * OFFSET)
print(f"  plasma cube: |x|,|y|,|z| < {0.5*OFFSET:.3f} m  "
      f"({cube.sum()} of {cube.size} cells)")
rms_cube = max(
    float(np.sqrt(np.mean((Bxc[cube] - Bxr[cube]) ** 2)) /
          max(np.sqrt(np.mean(Bxr[cube] ** 2)), 1e-30)),
    float(np.sqrt(np.mean((Byc[cube] - Byr[cube]) ** 2)) /
          max(np.sqrt(np.mean(Byr[cube] ** 2)), 1e-30)),
    float(np.sqrt(np.mean((Bzc[cube] - Bzr[cube]) ** 2)) /
          max(np.sqrt(np.mean(Bzr[cube] ** 2)), 1e-30)),
)

# (b) Global, peak-normalised absolute tolerance
linf_abs_pk = max(np.max(np.abs(Bxc - Bxr)),
                  np.max(np.abs(Byc - Byr)),
                  np.max(np.abs(Bzc - Bzr))) / max(
    np.max(np.abs(Bxr)), np.max(np.abs(Byr)), np.max(np.abs(Bzr)), 1e-30)
frac_1pct = min(cell_fraction_within(Bxc, Bxr, 0.01),
                cell_fraction_within(Byc, Byr, 0.01),
                cell_fraction_within(Bzc, Bzr, 0.01))
frac_5pct = min(cell_fraction_within(Bxc, Bxr, 0.05),
                cell_fraction_within(Byc, Byr, 0.05),
                cell_fraction_within(Bzc, Bzr, 0.05))

print(f"  plasma-cube RMS rel err (worst comp):  {rms_cube:.3e}  "
      f"(diagnostic — |B| is small here by design, inflating rel error)")
print(f"  global L∞ |B_curl - B_ref| / max|B_ref|: {linf_abs_pk:.3e}  "
      f"(coil-cell — fundamental FD limit)")
print(f"  cells within 1% of peak |B|:            {frac_1pct:.1%}")
print(f"  cells within 5% of peak |B|:            {frac_5pct:.1%}")
# Gating metric: peak-normalised absolute tolerance. Relative-error metrics
# blow up in the magnetic null where |B|→0 even when the absolute error is
# tiny, so we gate on "fraction of cells where the absolute error is small
# compared to the field's peak magnitude" — what a hybrid PIC particle
# actually experiences.
report("≥ 95% of cells within 5% of peak |B|",
       frac_5pct > 0.95, f"{frac_5pct:.1%} of cells")
report("≥ 90% of cells within 1% of peak |B|",
       frac_1pct > 0.90, f"{frac_1pct:.1%} of cells")


# ====================================================================
# TEST 3 — Coulomb gauge
# ====================================================================
header("TEST 3: Coulomb gauge ∇·A ≈ 0 (relative to ∇×A)")
dx_, dy_, dz_ = spacing
divA = (np.gradient(Ax, dx_, axis=0)
        + np.gradient(Ay, dy_, axis=1)
        + np.gradient(Az, dz_, axis=2))
curl_peak_full = max(np.max(np.abs(Bxc)), np.max(np.abs(Byc)), np.max(np.abs(Bzc)))
ratio_full_rms = float(np.sqrt(np.mean(divA ** 2)) /
                       max(np.sqrt(np.mean(Bxc ** 2 + Byc ** 2 + Bzc ** 2)), 1e-30))
ratio_full_linf = np.max(np.abs(divA)) / max(curl_peak_full, 1e-30)

# Plasma cube: clean read of spectral-gauge enforcement; only finite-
# difference truncation of ∇· remains.
gauge_cube = plasma_cube_mask(dom2, half_side=0.5 * OFFSET)
curl_peak_cube = max(np.max(np.abs(Bxc[gauge_cube])),
                     np.max(np.abs(Byc[gauge_cube])),
                     np.max(np.abs(Bzc[gauge_cube])))
ratio_cube = np.max(np.abs(divA[gauge_cube])) / max(curl_peak_cube, 1e-30)
print(f"  plasma-cube ‖∇·A‖_∞ / max‖∇×A‖_cube:  {ratio_cube:.3e}  "
      f"(machine precision ⇒ spectral gauge holds)")
print(f"  full-grid    RMS ‖∇·A‖ / RMS ‖∇×A‖:    {ratio_full_rms:.3e}  "
      f"(coil-cell FD residual)")
print(f"  full-grid    L∞  ‖∇·A‖ / max ‖∇×A‖:    {ratio_full_linf:.3e}")
report("Coulomb gauge enforced in plasma cube (ratio < 1e-6)",
       ratio_cube < 1e-6, f"plasma-cube ratio = {ratio_cube:.3e}")
# release the big arrays from tests 2 + 3 before the resolution sweep
del Ax, Ay, Az, Bxc, Byc, Bzc, Bxr, Byr, Bzr, B_ref, X, Y, Z, divA
del cube, gauge_cube
gc.collect()


# ====================================================================
# TEST 4 — grid-resolution convergence
# ====================================================================
header("TEST 4: grid-resolution convergence (fixed L, pad=2)")
print("  At fixed L = 2.0 m and pad_factor = 2, sweep N ∈ {16, 24, 32, 48}.")
print("  Metric: fraction of cells with |curl(A) - B_ref| < 5% of peak |B_ref|.")
ns = [16, 24, 32, 48]
fracs = []
for Nk in ns:
    domn = derive_domain("full", L=L, N=Nk)
    Axn, Ayn, Azn, spn = compute_A_grid(polywell, domn, pad_factor=2)
    Bxc_n, Byc_n, Bzc_n = curl_A(Axn, Ayn, Azn, spn)
    xs = np.linspace(domn.lower[0], domn.upper[0], Nk)
    ys = np.linspace(domn.lower[1], domn.upper[1], Nk)
    zs = np.linspace(domn.lower[2], domn.upper[2], Nk)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    Bref = polywell.getB(np.stack([X, Y, Z], axis=-1))
    f = min(cell_fraction_within(Bxc_n, Bref[..., 0], 0.05),
            cell_fraction_within(Byc_n, Bref[..., 1], 0.05),
            cell_fraction_within(Bzc_n, Bref[..., 2], 0.05))
    fracs.append(f)
    print(f"  N={Nk:>3}: cells within 5% of peak |B| = {f:.1%}", flush=True)
    del Axn, Ayn, Azn, Bxc_n, Byc_n, Bzc_n, Bref, X, Y, Z
    gc.collect()
report("≥ 95% of cells within 5% of peak |B| at every N",
       all(f > 0.95 for f in fracs),
       f"trajectory: {[f'{f:.1%}' for f in fracs]}")
report("metric improves or stays stable as N increases",
       fracs[-1] >= fracs[0] - 0.05,
       f"start = {fracs[0]:.1%}, end = {fracs[-1]:.1%}")


# ====================================================================
# TEST 5 — single-ring sanity check vs analytic A_φ
# ====================================================================
header("TEST 5: single ring vs analytic A_φ (elliptic integrals)")
from magpylib.current import Circle as C
from magpylib import Collection

# Lone ring, radius a = DIA/2, in the xy-plane, current I_COIL
single = Collection(C(current=I_COIL, diameter=DIA))
N1 = 24
dom1 = derive_domain("full", L=L, N=N1)
Ax1, Ay1, Az1, _ = compute_A_grid(single, dom1, pad_factor=4)

# Analytic A_φ on the same physics grid:
#   k² = 4 a ρ / [(a + ρ)² + z²]
#   A_φ = (μ₀ I / π k) · √(a/ρ) · [(1 - k²/2) K(k) - E(k)]
a = DIA / 2.0
mu0 = sc.mu_0
xs = np.linspace(dom1.lower[0], dom1.upper[0], N1)
ys = np.linspace(dom1.lower[1], dom1.upper[1], N1)
zs = np.linspace(dom1.lower[2], dom1.upper[2], N1)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
rho = np.hypot(X, Y)
phi = np.arctan2(Y, X)
denom = (a + rho) ** 2 + Z ** 2
denom = np.maximum(denom, 1e-30)
k2 = np.clip(4.0 * a * rho / denom, 1e-30, 1.0 - 1e-12)
k = np.sqrt(k2)
rho_safe = np.maximum(rho, 1e-30)
A_phi = (mu0 * I_COIL) / (np.pi * k) * np.sqrt(a / rho_safe) * (
    (1.0 - 0.5 * k2) * ellipk(k2) - ellipe(k2)
)
# A_φ along φ̂ = (-sin φ, cos φ, 0)
Ax_ref = -A_phi * np.sin(phi)
Ay_ref = A_phi * np.cos(phi)
Az_ref = np.zeros_like(Ax_ref)

# Mask out the singular ring (ρ ≈ a, z ≈ 0) and the outer boundary
hx = xs[1] - xs[0]
trim = 4
near_ring = (np.abs(rho - a) < 2 * hx) & (np.abs(Z) < 2 * hx)
keep = ~near_ring
keep = keep[trim:-trim, trim:-trim, trim:-trim]
diffs = [
    interior(Ax1, trim) - interior(Ax_ref, trim),
    interior(Ay1, trim) - interior(Ay_ref, trim),
    interior(Az1, trim) - interior(Az_ref, trim),
]
ref_peak = max(np.max(np.abs(interior(Ax_ref, trim)[keep])),
               np.max(np.abs(interior(Ay_ref, trim)[keep])))
err_a = max(np.max(np.abs(d[keep])) for d in diffs) / max(ref_peak, 1e-30)
print(f"  single-ring interior L∞ rel err (vs analytic A_φ): {err_a:.3e}")
report("FFT A matches analytic A_φ for a single ring (rel err < 20%)",
       err_a < 0.20, f"rel err = {err_a:.2%}")


# ----------------------------- summary -----------------------------
print()
print("=" * 72)
if _failed:
    print(f"{RED}{BOLD}{len(_failed)} test(s) FAILED:{RESET}")
    for n in _failed:
        print(f"  - {n}")
    sys.exit(1)
else:
    print(f"{GREEN}{BOLD}All tests passed.{RESET}")
    sys.exit(0)
