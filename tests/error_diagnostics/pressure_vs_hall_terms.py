import numpy as np
import scipy.constants as sc
import openpmd_viewer as OPMD
import matplotlib.pyplot as plt

from coil_3d import SingleCoil3DConfig
cfg = SingleCoil3DConfig()
static_params = {
    "$n_{stream}$": cfg.n_stream,
    "$v_{drift} (m/s)$": cfg.v_drift,
    "$T_{i}(eV)$": cfg.T_i_eV,
    "$T_{e}(eV)$": cfg.T_e_eV,
    "I(A)": cfg.I,
    "R(m)": cfg.R_coil,
}

# Restructured to keep full vector components through the pipeline --
# needed for true |J x B| (angle-dependent), and lets you inspect
# per-direction contributions (e.g. is the coil-proximal spike dominated
# by J_z x B_y, or something more isotropic?) rather than just magnitudes.

def pressure_vs_hall_ratio_vec(n_field, Te_eV, B_vec, J_vec, dx):
    """
    n_field : density array (m^-3)
    Te_eV   : electron temp (eV), scalar per your simplified isothermal closure
    B_vec   : (Bx, By, Bz) arrays
    J_vec   : (Jx, Jy, Jz) arrays  -- J_ion + j_displacement per component
    dx      : grid spacing (m)
    """
    Bx, By, Bz = B_vec
    Jx, Jy, Jz = J_vec

    Te_J = Te_eV * sc.e
    Pe = cfg.n_stream * Te_J * (n_field / cfg.n_stream)**(5/3)
    grad_Pe = np.gradient(Pe, dx)
    grad_Pe_mag = np.sqrt(sum(g**2 for g in grad_Pe))
    pressure_term = grad_Pe_mag / (n_field * sc.e)

    # true cross product J x B, componentwise -- not |J|*|B|
    Cx = Jy*Bz - Jz*By
    Cy = Jz*Bx - Jx*Bz
    Cz = Jx*By - Jy*Bx

    hall_x = Cx / (n_field * sc.e)
    hall_y = Cy / (n_field * sc.e)
    hall_z = Cz / (n_field * sc.e)
    hall_mag = np.sqrt(hall_x**2 + hall_y**2 + hall_z**2)

    ratio = pressure_term / (hall_mag + 1e-12)

    return {
        "ratio": ratio,
        "pressure_term": pressure_term,
        "hall_mag": hall_mag,
        "hall_components": (hall_x, hall_y, hall_z),
        "J_vec": (Jx, Jy, Jz),
        "B_vec": (Bx, By, Bz),
    }

ratios = []
pressure_terms = []
hall_terms = []

save_path = 'diags'
series_f   = OPMD.OpenPMDTimeSeries(save_path + '/field_diag')
iterations = series_f.iterations[1:]

results = []
iterations_used = []

for k, it in enumerate(iterations):
    Bx, info = series_f.get_field('B', coord='x', iteration=it)
    By, _    = series_f.get_field('B', coord='y', iteration=it)
    Bz, _    = series_f.get_field('B', coord='z', iteration=it)

    Js = []
    for coord in ['x', 'y', 'z']:
        Jion = series_f.get_field('j', coord=coord, iteration=it)
        Je   = series_f.get_field('j_displacement', coord=coord, iteration=it)
        Js.append(Jion[0] + Je[0])

    rho, _ = series_f.get_field('rho', iteration=it)   # confirm this key exists in your diag
    n_field = (np.abs(rho) + 1e-12) / sc.e

    res = pressure_vs_hall_ratio_vec(
        n_field, cfg.T_e_eV, (Bx, By, Bz), tuple(Js), cfg.dx
    )
    results.append(res)

r = results[-1]  # e.g. inspect the timestep near the spike you saw

plt.plot(iterations[3:], [np.max(p['pressure_term']) for p in results[3:]])

for r in results:
    # Locate pressure-term peak independently -- don't assume it coincides with Hall peak
    pressure_peak_idx = np.unravel_index(np.argmax(r["pressure_term"]), r["pressure_term"].shape)

    px, py, pz = pressure_peak_idx
    print(f"pressure peak value: {r['pressure_term'][pressure_peak_idx]:.3e}")
    print(f"pressure peak idx: {pressure_peak_idx}  (grid shape: {r['pressure_term'].shape})")

    # compare against where hall peaks, and against local n/Te to sanity check
    hall_peak_idx = np.unravel_index(np.argmax(r["hall_mag"]), r["hall_mag"].shape)
    print(f"hall peak idx:     {hall_peak_idx}")

    # distance between the two peak locations, in grid cells
    dist_cells = np.sqrt(sum((a-b)**2 for a,b in zip(pressure_peak_idx, hall_peak_idx)))
    print(f"separation between pressure-peak and hall-peak: {dist_cells:.1f} cells "
        f"({dist_cells*cfg.dx:.4f} m)")

    # what's the ratio AT the pressure peak (not the hall peak)?
    print(f"ratio at pressure peak: {r['ratio'][pressure_peak_idx]:.3e}")
    print(f"hall_mag at pressure peak: {r['hall_mag'][pressure_peak_idx]:.3e}")

def plot_peak_slices(res):
    pressure_term = res["pressure_term"]
    hall_mag = res["hall_mag"]

    p_idx = np.unravel_index(np.argmax(pressure_term), pressure_term.shape)
    h_idx = np.unravel_index(np.argmax(hall_mag), hall_mag.shape)

    # slice through the hall-peak's x-index (roughly the coil plane if hall peaks there)
    x_slice = h_idx[0]

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    im0 = axs[0].imshow(pressure_term[x_slice].T, origin='lower', cmap='viridis')
    axs[0].set_title(f"Pressure term (x-slice={x_slice})")
    axs[0].plot(p_idx[1], p_idx[2], 'r*', markersize=15, label='pressure peak (global)')
    if p_idx[0] == x_slice:
        axs[0].legend()
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(hall_mag[x_slice].T, origin='lower', cmap='inferno')
    axs[1].set_title(f"Hall magnitude (x-slice={x_slice})")
    axs[1].plot(h_idx[1], h_idx[2], 'c*', markersize=15, label='hall peak')
    axs[1].legend()
    plt.colorbar(im1, ax=axs[1])

    plt.suptitle(f"Peak separation: {np.sqrt(sum((a-b)**2 for a,b in zip(p_idx,h_idx))):.1f} cells")
    plt.tight_layout()
    plt.show()

    return p_idx, h_idx

# usage:
p_idx, h_idx = plot_peak_slices(r)

# second pass, centered on pressure peak's plane
x_slice2 = p_idx[0]
plt.imshow(r["pressure_term"][x_slice2].T, origin='lower', cmap='viridis')
plt.plot(p_idx[1], p_idx[2], 'r*', markersize=15)
plt.title(f"Pressure term at its own peak plane (x-slice={x_slice2})")
plt.colorbar()
plt.show()