"""
Vector potential A for the polywell B-field via FFT curl-inverse.

The Coulomb-gauge (∇·A = 0) inversion of ∇×A = B is a one-liner in Fourier space:

    Ã(k) = i (k × B̃(k)) / |k|²

with Ã(0) = 0 (DC mode is gauge freedom; the A → 0 at infinity convention sets it to 0).

The catch is that np.fft.fftn assumes periodic boundary conditions, which for a
localized coil means the box is replicated as an infinite lattice. To suppress
image-coil contamination we evaluate B on a zero-padded grid (so the coil is
buried deep in a much larger box of near-empty space), invert the curl in Fourier
space, then crop back to the physics region.

`compute_A_grid` does one pad / FFT / crop pass for a fixed pad_factor.
`converge_A_grid` doubles pad_factor until A in the physics region stops moving.

Usage:
    from warpx_polywell.bext.make_collection import make_polywell_collection
    from warpx_polywell.bext.vector_potential import converge_A_grid
    from warpx_polywell.domain import derive_domain

    domain = derive_domain("full", L=2.0, N=64)
    coll   = make_polywell_collection(I=1e6, dia=1.0, d=0.435)
    Ax, Ay, Az, dx, pad = converge_A_grid(coll, domain, start=2, max_pad=8, verbose=True)

Returns A in T·m (= V·s/m) on a grid matching domain.n_cells.
"""
import numpy as np


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _padded_grid(domain, pad_factor):
    """
    Build a uniform grid that is `pad_factor`× the physics domain in each axis.
    Returns the meshgrid (Nx,Ny,Nz,3), the slice that recovers the physics
    region, and the per-axis spacing.

    Cell spacing is taken from the physics grid so that, after the inverse FFT,
    the cropped subarray lies exactly on the WarpX cells.
    """
    nx, ny, nz = domain.n_cells
    lo = np.asarray(domain.lower, dtype=float)
    hi = np.asarray(domain.upper, dtype=float)

    dx = (hi[0] - lo[0]) / max(nx - 1, 1)
    dy = (hi[1] - lo[1]) / max(ny - 1, 1)
    dz = (hi[2] - lo[2]) / max(nz - 1, 1)

    # extra cells per side
    ex = int(round((pad_factor - 1) * nx / 2))
    ey = int(round((pad_factor - 1) * ny / 2))
    ez = int(round((pad_factor - 1) * nz / 2))

    Nx, Ny, Nz = nx + 2 * ex, ny + 2 * ey, nz + 2 * ez

    x = lo[0] + (np.arange(Nx) - ex) * dx
    y = lo[1] + (np.arange(Ny) - ey) * dy
    z = lo[2] + (np.arange(Nz) - ez) * dz

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    mesh = np.stack([X, Y, Z], axis=-1)               # magpylib wants (..., 3)

    crop = (slice(ex, ex + nx), slice(ey, ey + ny), slice(ez, ez + nz))
    return mesh, crop, (dx, dy, dz)


def _coulomb_gauge_A_from_B(B, spacing):
    """
    Coulomb-gauge curl-inverse via FFT.

    Parameters
    ----------
    B : (Nx, Ny, Nz, 3) array
        Magnetic field on a uniform grid (Tesla).
    spacing : (dx, dy, dz)
        Per-axis cell size in meters.

    Returns
    -------
    A : (Nx, Ny, Nz, 3) array
        Vector potential in T·m.
    """
    Nx, Ny, Nz, _ = B.shape
    dx, dy, dz = spacing

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0                                  # avoid div/0; Ak[0,0,0] zeroed below

    Bk = np.fft.fftn(B, axes=(0, 1, 2))

    Akx = 1j * (KY * Bk[..., 2] - KZ * Bk[..., 1]) / K2
    Aky = 1j * (KZ * Bk[..., 0] - KX * Bk[..., 2]) / K2
    Akz = 1j * (KX * Bk[..., 1] - KY * Bk[..., 0]) / K2

    # DC mode is pure gauge — pick A_DC = 0 (corresponds to A → 0 at infinity).
    Akx[0, 0, 0] = 0
    Aky[0, 0, 0] = 0
    Akz[0, 0, 0] = 0

    Ax = np.fft.ifftn(Akx, axes=(0, 1, 2)).real
    Ay = np.fft.ifftn(Aky, axes=(0, 1, 2)).real
    Az = np.fft.ifftn(Akz, axes=(0, 1, 2)).real

    return np.stack([Ax, Ay, Az], axis=-1)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def compute_A_grid(collection, domain, pad_factor=2):
    """
    One-shot: pad → magpylib B → FFT curl-inverse → crop.

    Parameters
    ----------
    collection : magpylib.Collection
        Source coils (see src.bext.make_collection.make_polywell_collection).
    domain : src.domain.Domain
        Physics domain (defines crop region + spacing).
    pad_factor : float
        Box-extension factor in each axis. 2 = 8× the volume; 3 = 27×.

    Returns
    -------
    Ax, Ay, Az : ndarrays of shape `domain.n_cells`, in T·m.
    spacing    : tuple (dx, dy, dz) in meters.
    """
    mesh, crop, dx = _padded_grid(domain, pad_factor)
    B = collection.getB(mesh)                          # (Nx, Ny, Nz, 3) in T
    A_pad = _coulomb_gauge_A_from_B(B, dx)
    A = A_pad[crop[0], crop[1], crop[2], :]
    return A[..., 0], A[..., 1], A[..., 2], dx


def converge_A_grid(collection, domain, *,
                    start=2, max_pad=8, rtol=1e-3, verbose=False):
    """
    Double the padding until A in the physics region stops moving by more than
    `rtol` in L∞ norm relative to its current peak magnitude.

    Pad-factor sequence: start, 2·start, 4·start, ... capped at max_pad.

    Returns
    -------
    Ax, Ay, Az : converged A on the physics grid.
    spacing    : (dx, dy, dz) in meters.
    pad        : final pad_factor used.
    """
    pad = start
    Ax, Ay, Az, dx = compute_A_grid(collection, domain, pad_factor=pad)
    peak0 = max(np.max(np.abs(Ax)), np.max(np.abs(Ay)), np.max(np.abs(Az)))
    if verbose:
        print(f"[converge_A] pad={pad}  |A|_max={peak0:.4e} T·m")

    while pad < max_pad:
        pad_next = pad * 2
        Ax2, Ay2, Az2, dx2 = compute_A_grid(collection, domain, pad_factor=pad_next)
        peak = max(np.max(np.abs(Ax2)), np.max(np.abs(Ay2)), np.max(np.abs(Az2)), 1e-30)
        diff = max(np.max(np.abs(Ax2 - Ax)),
                   np.max(np.abs(Ay2 - Ay)),
                   np.max(np.abs(Az2 - Az)))
        rel = diff / peak
        if verbose:
            print(f"[converge_A] pad={pad_next}  |A|_max={peak:.4e}  rel-change={rel:.3e}")
        Ax, Ay, Az, dx = Ax2, Ay2, Az2, dx2
        pad = pad_next
        if rel < rtol:
            break
    else:
        if verbose:
            print(f"[converge_A] reached max_pad={max_pad} without hitting rtol={rtol}")
    return Ax, Ay, Az, dx, pad


def curl_A(Ax, Ay, Az, spacing):
    """
    B = ∇×A via second-order central differences.

    Parameters
    ----------
    Ax, Ay, Az : ndarrays of shape (Nx, Ny, Nz) — vector potential components.
    spacing    : (dx, dy, dz) — per-axis cell sizes.

    Returns
    -------
    Bx, By, Bz : ndarrays of shape (Nx, Ny, Nz).
    """
    dx, dy, dz = spacing
    dAx_dy, dAx_dz = np.gradient(Ax, dy, dz, axis=(1, 2))
    dAy_dx, dAy_dz = np.gradient(Ay, dx, dz, axis=(0, 2))
    dAz_dx, dAz_dy = np.gradient(Az, dx, dy, axis=(0, 1))
    Bx = dAz_dy - dAy_dz
    By = dAx_dz - dAz_dx
    Bz = dAy_dx - dAx_dy
    return Bx, By, Bz


def check_curl(Ax, Ay, Az, spacing, collection, domain):
    """
    Sanity check: compute ∇×A on the physics grid via central differences and
    compare to the magpylib B at the same points. Returns relative L∞ error.
    """
    Bx_curl, By_curl, Bz_curl = curl_A(Ax, Ay, Az, spacing)

    # magpylib B on the same grid
    nx, ny, nz = domain.n_cells
    _x = np.linspace(domain.lower[0], domain.upper[0], nx)
    _y = np.linspace(domain.lower[1], domain.upper[1], ny)
    _z = np.linspace(domain.lower[2], domain.upper[2], nz)
    X, Y, Z = np.meshgrid(_x, _y, _z, indexing="ij")
    B_ref = collection.getB(np.stack([X, Y, Z], axis=-1))
    Bx, By, Bz = B_ref[..., 0], B_ref[..., 1], B_ref[..., 2]

    num = max(np.max(np.abs(Bx_curl - Bx)),
              np.max(np.abs(By_curl - By)),
              np.max(np.abs(Bz_curl - Bz)))
    den = max(np.max(np.abs(Bx)), np.max(np.abs(By)), np.max(np.abs(Bz)), 1e-30)
    return num / den


# ----------------------------------------------------------------------
# Standalone sanity-check / demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick demo: compute A for the default polywell, run the convergence sweep,
    verify ∇×A ≈ B, and plot |A| on the z = 0 mid-plane.
    """
    import matplotlib.pyplot as plt
    from warpx_polywell.bext.make_collection import make_polywell_collection
    from warpx_polywell.domain import derive_domain

    domain = derive_domain("full", L=2.0, N=48)
    coll = make_polywell_collection(1e6, dia=1.0, d=0.435)

    Ax, Ay, Az, dx, pad = converge_A_grid(
        coll, domain, start=2, max_pad=8, rtol=1e-3, verbose=True,
    )
    err = check_curl(Ax, Ay, Az, dx, coll, domain)
    print(f"[demo] converged pad_factor={pad}, curl(A) vs B relative error = {err:.3e}")

    mid = domain.n_cells[2] // 2
    Amag = np.sqrt(Ax**2 + Ay**2 + Az**2)[:, :, mid]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        Amag.T, origin="lower", cmap="viridis",
        extent=[domain.lower[0], domain.upper[0],
                domain.lower[1], domain.upper[1]],
    )
    fig.colorbar(im, ax=ax, label="|A| (T·m)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"polywell |A| at z=0  (pad_factor={pad})")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
