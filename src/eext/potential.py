"""
Electrostatic scalar potential φ from 6 charged rings in a polywell arrangement.

Closed form for a single ring of radius a, total charge Q, in its local
cylindrical frame (ring lies in the z = 0 plane, centered at the origin):

    k²    = 4 a ρ / ((a + ρ)² + z²)
    φ(ρ,z) = Q · K(k) / (2 π² ε₀ √((a + ρ)² + z²))

where K is the complete elliptic integral of the first kind (scipy convention:
`ellipk(m)` with m = k²). On-axis (ρ → 0) this collapses to the familiar
Q/(4πε₀ √(a² + z²)) form.

The polywell layout (positions + orientations) is reused from
make_polywell_collection so the cube symmetry stays consistent with the B-field
setup. Each ring contributes +Q (we use the collection only for coil placement,
matching the existing eext.py convention).

Output:
- φ on the WarpX grid (V).
- Optionally E = -∇φ via central differences (np.gradient).

Usage:
    from src.eext.potential import compute_phi_grid, compute_E_from_phi
    phi, dx = compute_phi_grid(Q=1e-9, dia=0.75, offset=1.1, domain=domain)
    Ex, Ey, Ez = compute_E_from_phi(phi, dx)
"""
import numpy as np
import scipy.constants as sc
from scipy.special import ellipk

from src.bext.make_collection import make_polywell_collection


# ----------------------------------------------------------------------
# Closed-form φ for one ring (vectorized over grid points)
# ----------------------------------------------------------------------

def phi_ring_local(rho, z, a, Q):
    """
    Scalar potential of a single charged ring (radius `a`, total charge `Q`)
    evaluated at points (rho, z) in the ring's local cylindrical frame.

    Vectorized: rho and z can be arrays of any matching shape.
    """
    eps0 = sc.epsilon_0
    denom_sq = (a + rho) ** 2 + z ** 2
    # numerical floor to keep things finite right on the ring (rho≈a, z≈0)
    denom_sq = np.maximum(denom_sq, 1e-30)
    k2 = 4.0 * a * rho / denom_sq
    # scipy.special.ellipk takes the parameter m = k², and diverges at m = 1
    k2 = np.clip(k2, 0.0, 1.0 - 1e-12)
    return Q * ellipk(k2) / (2.0 * np.pi ** 2 * eps0 * np.sqrt(denom_sq))


# ----------------------------------------------------------------------
# Polywell-wide φ on the WarpX grid
# ----------------------------------------------------------------------

def compute_phi_grid(Q, dia, offset, domain):
    """
    Superpose φ from 6 polywell rings on the WarpX grid.

    Parameters
    ----------
    Q : float    — per-ring total charge (Coulombs)
    dia : float  — ring diameter (m)
    offset : float — ring-center distance from origin (m)
    domain : src.domain.Domain — simulated grid spec

    Returns
    -------
    phi : ndarray of shape domain.n_cells, in Volts
    spacing : tuple (dx, dy, dz) in meters
    """
    a = dia / 2.0
    nx, ny, nz = domain.n_cells
    _x = np.linspace(domain.lower[0], domain.upper[0], nx)
    _y = np.linspace(domain.lower[1], domain.upper[1], ny)
    _z = np.linspace(domain.lower[2], domain.upper[2], nz)
    spacing = (_x[1] - _x[0], _y[1] - _y[0], _z[1] - _z[0])

    X, Y, Z = np.meshgrid(_x, _y, _z, indexing="ij")
    pts_lab = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)   # (N, 3)

    # Reuse the polywell layout for coil positions + orientations only.
    # The charge magnitude carried in the collection is ignored; we apply +Q
    # uniformly to all 6 rings (matches the convention in src/eext/eext.py).
    collection = make_polywell_collection(Q, dia, offset)

    phi_flat = np.zeros(pts_lab.shape[0], dtype=np.float64)
    for c in collection:
        # transform all lab-frame points into this coil's local frame
        pts_local = c.orientation.inv().apply(pts_lab - np.asarray(c.position))
        rho = np.hypot(pts_local[:, 0], pts_local[:, 1])
        zl = pts_local[:, 2]
        phi_flat += phi_ring_local(rho, zl, a, Q)

    return phi_flat.reshape(nx, ny, nz), spacing


def compute_E_from_phi(phi, spacing):
    """E = -∇φ via second-order central differences."""
    gx, gy, gz = np.gradient(phi, spacing[0], spacing[1], spacing[2])
    return -gx, -gy, -gz


# ----------------------------------------------------------------------
# Standalone sanity-check / demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick demo: compute φ and E = -∇φ for the default biasing rings, print
    on-axis sanity values, and plot φ + |E| on the z = 0 mid-plane.
    """
    import matplotlib.pyplot as plt
    from src.domain import derive_domain

    domain = derive_domain("full", L=2.0, N=64)
    Q, dia, offset = 1e-9, 0.75, 1.1

    phi, dx = compute_phi_grid(Q=Q, dia=dia, offset=offset, domain=domain)
    Ex, Ey, Ez = compute_E_from_phi(phi, dx)

    print(f"[demo] phi  max={phi.max():.3e} V, min={phi.min():.3e} V")
    Emag_all = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    print(f"[demo] |E| max = {Emag_all.max():.3e} V/m")

    mid = domain.n_cells[2] // 2
    extent = [domain.lower[0], domain.upper[0],
              domain.lower[1], domain.upper[1]]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    im1 = a1.imshow(phi[:, :, mid].T, origin="lower", cmap="RdBu_r", extent=extent)
    fig.colorbar(im1, ax=a1, label="φ (V)")
    a1.set_xlabel("x (m)"); a1.set_ylabel("y (m)")
    a1.set_title("polywell φ at z = 0")
    a1.set_aspect("equal")

    im2 = a2.imshow(Emag_all[:, :, mid].T, origin="lower", cmap="viridis", extent=extent)
    fig.colorbar(im2, ax=a2, label="|E| (V/m)")
    a2.set_xlabel("x (m)"); a2.set_ylabel("y (m)")
    a2.set_title("polywell |E| = |-∇φ| at z = 0")
    a2.set_aspect("equal")

    plt.tight_layout()
    plt.show()
