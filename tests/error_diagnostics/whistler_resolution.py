import scipy.constants as sc
import numpy as np
import src.warpx_polywell.bext.aext as aext 
import matplotlib.pyplot as plt

aext.POLYWELL_COILS = [('x', 1, 1)]

def _build_grid(n, L):
    xs = np.linspace(-L, L, n)
    ys = np.linspace(-L, L, n)
    zs = np.linspace(-L, L, n)
    dx, dy, dz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    return X, Y, Z, dx, dy, dz, xs, ys, zs

N = 64
L = 1.5
I = 2.25e4
offset = 0
R = 0.5
B_coil = (sc.mu_0 * I) / R*2
w_ci = sc.e * B_coil / sc.m_p
dt = 1.0 / (100.0 * w_ci)
substeps = 100

density_to_substeps = {
    1e17: 100,
    1e18: 32,
    1e19: 32,
    1e20: 32,
    1e21: 32
}

X, Y, Z, dx, dy, dz, xs, ys, zs = _build_grid(N, L)

Ax, Ay, Az = aext.compute_A_polywell(X, Y, Z, I, offset, R, R, N)

B_curlA = aext.curlA(Ax, Ay, Az, dx, dy, dz)

B_mag = np.sqrt(B_curlA['x']**2 + B_curlA['y']**2 + B_curlA['z']**2)

plt.streamplot(
    X[:, 32, :].T, 
    Z[:, 32, :].T, 
    B_curlA['x'][:, 32, :].T, 
    B_curlA['z'][:, 32, :].T,
    color=B_mag[:, 32, :].T)
plt.show()

def ion_skin_depth(n):  # n in m^-3
    w_pe = np.sqrt(n * sc.e**2 / (sc.epsilon_0 * sc.m_p))
    return sc.c / w_pe

dx = 3.0 / 64

for n in [1e21, 1e20, 1e19, 1e18, 1e17]:
    d_i = ion_skin_depth(n)
    print(f"n={n:.0e}  d_i={d_i:.3e} m")
    print(f"dx / d_e = {dx / d_i}")

def whistler_metric(Bx, By, Bz, n, dx, dt, substeps, B_coil=B_coil):
    """
    Bx,By,Bz : curl-derived B components (3D arrays, same grid as coil_3d.py)
    n        : plasma density (m^-3) for this run
    dx       : uniform grid spacing (m)
    dt       : field dt (pre-substep)
    substeps : substep count used for this density
    """
    # sl = np.s_[1:-1, 1:-1, 1:-1]
    # Bx = Bx[sl]
    # By = By[sl]
    # Bz = Bz[sl]
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)          # local |B|, includes curl discretization error
    w_ci = sc.e * Bmag / sc.m_p                     # local ion cyclotron freq

    w_pi = np.sqrt(n * sc.e**2 / (sc.epsilon_0 * sc.m_p))
    d_i = sc.c / w_pi                                # ion skin depth for this density

    k_ny = np.pi / dx                                # grid-Nyquist wavenumber
    w_whistler = k_ny**2 * d_i**2 * w_ci             # Hall-term whistler at grid scale

    metric = w_whistler * dt / substeps              # phase advance per field substep
    return metric, Bmag

# Histogram: where does metric sit relative to stability threshold (~0.1-0.2)?
def plot_metric_histogram(ax, metric, label=""):
    # plt.figure(figsize=(6,4))
    ax.hist(metric.flatten(), bins=100, log=True)   # log-y: blowup points are rare, need to see the tail
    ax.axvline(0.1, color='orange', ls='--', label='marginal (0.1)')
    ax.axvline(0.5, color='red', ls='--', label='likely unstable (0.5)')
    ax.set_xlabel("whistler metric")
    ax.set_ylabel("grid point count (log)")
    ax.set_title(f"Whistler metric distribution {label}")
    ax.legend()

# plot_metric_histogram(metric)

fig, axs = plt.subplots(5, 1, figsize=(6, 4))
fig.tight_layout()

densities = [1e21, 1e20, 1e19, 1e18, 1e17]

frac_above_thresh1 = []
frac_above_thresh5 = []

for i, n in enumerate(densities):
    
    d_i = ion_skin_depth(n)
    # print(f"n={n:.0e}  d_i={d_i:.3e} m")
    # print(f"dx / d_e = {dx / d_i}")
    metric, Bmag = whistler_metric(B_curlA['x'], B_curlA['y'], B_curlA['z'], n=n, dx=dx, dt=dt, substeps=density_to_substeps[n])
    print(f"ERRORS {n}")
    print(f"n={n:.0e}  max_metric={metric.max():.3f}  at idx={np.unravel_index(np.argmax(metric), metric.shape)}")
    above_1 = metric > 0.1
    above_5 = metric > 0.5
    frac_above_1 = np.mean(above_1)
    frac_above_5 = np.mean(above_5)
    frac_above_thresh1.append(frac_above_1)
    frac_above_thresh5.append(frac_above_5)
    plot_metric_histogram(axs[i], metric, label=f"Density: {n}")

plt.show()

plt.plot(np.log10(densities), frac_above_thresh1, label="0.1")
plt.plot(np.log10(densities), frac_above_thresh5, label="0.5")
for i, n in enumerate(densities):
    plt.axvline(np.log10(n), linestyle='dashed')
    plt.axhline(frac_above_thresh1[i], linestyle='dashed')
    plt.text(s=f"N={n:.2e}, {frac_above_thresh1[i]:.2f}", x = np.log10(n), y = frac_above_thresh1[i])
    plt.axhline(frac_above_thresh5[i], linestyle='dashed')
    plt.text(s=f"N={n:.2e},{frac_above_thresh5[i]:.2f}", x = np.log10(n), y = frac_above_thresh5[i] + 0.1)
plt.legend()
plt.title("Fraction of points that underresolve whistler wave at nyquist")

plt.show()

# Spatial clustering check: where do high-metric cells sit relative to the coil?
# Coil is a ring of radius R_coil in the y-z plane at x=0 (single loop, axis along x)

def coil_proximity_analysis(ax, metric, X, Y, Z, R_coil, threshold=0.1):
    """
    X,Y,Z    : grid coordinate arrays (same shape as metric)
    R_coil   : coil loop radius (0.5 m for your dia=1.0)
    threshold: metric cutoff to flag as 'at risk'
    """
    # approximate distance from each grid point to the coil conductor (a ring, not a point)
    rho = np.sqrt(Y**2 + Z**2)              # radial distance from x-axis
    dist_to_ring = np.sqrt((rho - R_coil)**2 + X**2)   # distance to nearest point on the ring

    farther_than_ring = dist_to_ring > 2 * dx
    flagged = (metric > threshold) & (farther_than_ring)

    print(f"threshold={threshold}")
    print(f"  flagged points: {flagged.sum()} / {metric.size} ({100*flagged.mean():.1f}%)")
    print(f"  median dist_to_ring (flagged):   {np.median(dist_to_ring[flagged]):.4f} m")
    print(f"  median dist_to_ring (unflagged): {np.median(dist_to_ring[~flagged]):.4f} m")
    print(f"  dx for reference: {dx:.4f} m")

    # scatter: metric vs distance to conductor -- if coil-proximal, expect metric to fall off with distance
    ax.scatter(dist_to_ring.flatten(), metric.flatten(), s=1, alpha=0.1)
    ax.axhline(threshold, color='red', ls='--')
    ax.set_xlabel("distance to coil conductor (m)")
    ax.set_ylabel("whistler metric")
    ax.set_yscale('log')
    ax.set_title("Metric vs. distance from coil")

    return dist_to_ring, flagged

# usage per density:
fig, axs = plt.subplots(5, 1, figsize=(6, 4))
for i, n in enumerate(densities):
    metric, Bmag = whistler_metric(B_curlA['x'], B_curlA['y'], B_curlA['z'], n=n, dx=dx, dt=dt, substeps=density_to_substeps[n])
    dist_to_ring, flagged = coil_proximity_analysis(axs[i], metric, X, Y, Z, R_coil=R, threshold=0.1)

plt.show()