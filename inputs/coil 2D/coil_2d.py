# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 2D ion-kinetic stream vs. transverse magnetic dipole (Hybrid-PIC)
#
# 2D Cartesian (x, z) Hybrid-PIC run in WarpX. Quasineutral by construction: only ion macroparticles are pushed; electrons are a massless adiabatic fluid with fixed `T_e`. The stream is injected from the `x_hi` face with drift velocity `-x`. The transverse dipole is two infinite line currents along the out-of-plane (y) axis at `z = ± d/2`, anti-parallel; in hybrid mode the external field is supplied as a **vector potential** `A_y(x,z)` (Wb/m) — WarpX reconstructs `B = ∇×A` internally. Field boundaries are `neumann` on all four faces (E is algebraic in hybrid — no PML needed); particle BCs are `absorbing`. A non-perturbing flux-counter callback records ion loss across the coil-to-coil midplane segment (`x=0, |z|<d/2`).
#
# Upstream ram pressure `ρ v²` is matched to `B² / 2μ₀` at the intended standoff (Chapman–Ferraro). Diagnostics dump full EM fields, ion charge density, and ion phase-space so post-processing can build `β_th = 2μ₀(n_i k_B T_i + n_e k_B T_e)/B²` (with `T_e` fixed by the solver, `T_i` measured) and `β_dyn = 2μ₀ ρ v² / B²` per cell; the `β_dyn = 1` contour is the empirical standoff surface.

# %%
# %load_ext wurlitzer

# %%
import os
import numpy as np
import scipy.constants as sc
from pywarpx import picmi, warpx

# %% [markdown]
# ## Physical parameters and Chapman–Ferraro standoff
#
# `T_e_eV` is now a **fluid input** to the hybrid solver (`hybrid_pic_model.elec_temp`), not a kinetic species temperature. `T_i_eV` is the spread on the ion injection distribution. `r_CF` comes from `ρ v² = B²/2μ₀` with the 2D line-dipole far-field `|B| ≈ μ₀ I d / (2π r²)`.

# %%
# Stream (upstream)
n_stream = 1.0e17          # m^-3
v_drift  = 5.0e5           # m/s, magnitude (flow toward -x)
T_e_eV   = 10.0            # electron FLUID temperature (hybrid input)
T_i_eV   = 10.0            # ion thermal spread on injection

# Line-dipole geometry: two infinite wires along y at z = +/- d/2.
I_line   = 1e5           # A per wire (anti-parallel)
d_sep    = 0.5            # m

# Derived
m_i   = sc.m_p
rho   = n_stream * m_i                   # ion mass density dominates
P_ram = rho * v_drift**2

# r_CF: P_ram = B^2/(2 mu0) with |B| = mu0 I d / (2 pi r^2)
r_CF = (sc.mu_0 * I_line**2 * d_sep**2 / (8.0 * np.pi**2 * P_ram)) ** 0.25

# Ion-inertial / gyrofrequency scales (hybrid cell + timestep targets)
B_ref       = sc.mu_0 * I_line * d_sep / (2.0 * np.pi * r_CF**2)
d_i         = sc.c / np.sqrt(n_stream * sc.e**2 / (sc.epsilon_0 * m_i))
Omega_ci    = sc.e * B_ref / m_i

# large resistive "conductor" islands at the two wires, ~zero elsewhere
eta_bg   = 1.0e-7        # your background value
eta_coil = 1.0e3         # high-resistivity coil interior (tune)
w        = 7.8e-2        # mask width ~ one cell, matches your eps

print(f"P_ram       = {P_ram:.3e} Pa")
print(f"r_CF        = {r_CF*100:.2f} cm  (predicted standoff)")
print(f"B at r_CF   = {B_ref*1e4:.2f} G")
print(f"d_i (n_inf) = {d_i*100:.2f} cm  (ion inertial length upstream)")
print(f"Omega_ci    = {Omega_ci:.3e} rad/s  -> 1/Omega_ci = {1/Omega_ci:.3e} s")

# %% [markdown]
# ## External vector potential (line-current pair)
#
# For `+I` along `+y` at `(x=0, z=+d/2)` and `-I` along `+y` at `(x=0, z=-d/2)`:
#
# $$A_y(x,z) = \frac{\mu_0 I}{4\pi}\,\ln\!\frac{x^2 + (z+d/2)^2}{x^2 + (z-d/2)^2},\quad A_x = A_z = 0$$
#
# so `B = ∇×A` gives `B_x = ∂A_y/∂z`, `B_z = -∂A_y/∂x` — same field as the anti-parallel line-current pair. `ε²` is added to each denominator to regularize the singularity at the wires; setting it to roughly half a cell (`~dx/2`) avoids a giant unresolved spike at the wire location that the grid couldn't see anyway. Since `A_x = A_z = 0` and `A_y` has no `y` dependence, `∇·A = 0` exactly, so we can disable WarpX's divA cleaner.

# %%
K   = sc.mu_0 * I_line / (4.0 * np.pi)   # note: 4π for A, vs 2π for B
dh  = d_sep / 2.0
# Smooth over one full cell (dx = 2*Lx/Nx = 5/64 ≈ 0.078 m). This caps the peak |B|
# in the wire cell at ~μ₀·I/(2π·ε) ≈ 0.13 T (with I_line=5e4) — comfortable for the
# B-substep CFL. Since ε << d_sep/2, the far-field dipole structure at r_CF is preserved.
eps = 7.8e-2   # m  (~ one cell at 64² resolution)

Ay_expr = (
    f"{K} * log( (x*x + (z+{dh})*(z+{dh}) + {eps*eps})"
    f"        / (x*x + (z-{dh})*(z-{dh}) + {eps*eps}) )"
)

# NOTE: WarpX's `plasma_resistivity(rho,J)` parser exposes only `rho` and `J` —
# it has no spatial coordinates, so resistivity cannot be made a function of
# (x, z). The high-resistivity "coil islands" are therefore emulated below by a
# post-deposition callback (`damp_current_at_coils`) that suppresses the plasma
# current in the wire cells each step. `plasma_resistivity` itself is left as the
# uniform background value `eta_bg`.

A_external = {
    "dipole": {
        "Ax_external_function":     "0",
        "Ay_external_function":     Ay_expr,
        "Az_external_function":     "0",
        "A_time_external_function": "1",   # static field
    }
}

# %% [markdown]
# ## Grid, boundaries, hybrid solver
#
# Field BCs are `neumann` (zero-gradient on `B` at the outer faces) — PML doesn't apply because `E` is algebraic from generalized Ohm's law. Particle BCs are `absorbing`. Cell size is targeted at the upstream ion inertial length; collocated grid + linear particle shape are recommended for hybrid.
#
# Three numerical-stability levers for hybrid-PIC are all engaged here:
# - `n_floor` clips the `1/n` term in Ohm's law in the rarefied cavity behind the standoff.
# - `holmstrom_vacuum_region=True` additionally suppresses the Hall and pressure terms when density falls to the floor — without this the cavity stays pathological even with a floor.
# - `plasma_hyper_resistivity` adds an η_H ∇²J term that damps grid-scale whistlers (whose phase speed scales as k², so they pin dt at the Nyquist wavenumber). Tune downward; if structures look over-smoothed, halve it.
# - A small `plasma_resistivity` damps long-wavelength modes — think of it as artificial viscosity. `0` is the most aggressive choice and tends to ring.
#
# `warpx_max_grid_size=16` on a 64² domain gives 16 grids (4×4) — enough decomposition to keep MPI ranks / OpenMP tiles fed.

# %%
Lx = 2.5     # m (half-extent) — sized to ~1.4 × r_CF
Lz = 2.5     # m
Nx = 64      # dx ≈ 7.8 cm — resolves r_CF with ~22 cells, d_i with ~29, ρ_i,drift with ~21
Nz = 64      # ρ_i,thermal ≈ 1.8 cells (under-resolved — fine for standoff, refine for sheath)

grid = picmi.Cartesian2DGrid(
    number_of_cells=[Nx, Nz],
    lower_bound=[-Lx, -Lz],
    upper_bound=[+Lx, +Lz],
    lower_boundary_conditions=["neumann", "neumann"],
    upper_boundary_conditions=["neumann", "neumann"],
    lower_boundary_conditions_particles=["absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
    warpx_max_grid_size=16,
)

solver = picmi.HybridPICSolver(
    grid=grid,
    Te=T_e_eV,
    n0=n_stream,
    gamma=5.0/3.0,
    n_floor=0.05 * n_stream,                 # 5% of upstream — caps 1/n amplification in the deepening cavity
    plasma_resistivity=eta_bg,               # uniform; coil islands emulated via callback
    plasma_hyper_resistivity=3.0e-3,         # Ohm·m^3; overdamps grid-Nyquist whistlers at peak |B|~0.13T
    holmstrom_vacuum_region=True,            # suppress Hall/pressure terms in the cavity
    substeps=100,                           # dt_sub ≈ 3e-11 s; clears whistler CFL at peak |B| with ~3x margin
    A_external=A_external,
    do_external_diva_cleaning=False,         # A is analytically div-free
)

# %% [markdown]
# ## Timestep — explicit `const_dt` for hybrid
#
# Without `warpx.const_dt`, WarpX falls back to the speed-of-light CFL (`dx/c`) — correct for EM solvers, but lethal for hybrid: each hybrid step is *more* expensive than an EM step (the `substeps` loop multiplies B-field work), so the speedup only materializes when the **outer** `dt` is ion-scale.
#
# Pick the smaller of:
# - ion-cyclotron resolution: `1 / (50 · Ω_ci)` at the standoff-field strength
# - half-cell crossing at the electron-fluid thermal speed (electron thermal speed bounds whistler dispersion on the grid)

# %%
MAX_STEPS = 1000

dx       = (2.0 * Lx) / Nx
ve_th    = np.sqrt(T_e_eV * sc.eV / sc.m_e)   # electron-fluid thermal speed

dt_cyclo = 1.0 / (50.0 * Omega_ci)            # ~1/50 of ion gyroperiod at r_CF
dt_cross = 0.5 * dx / ve_th                   # half-cell at electron thermal speed
const_dt = dt_cyclo

warpx.const_dt = const_dt

t_sim     = const_dt * MAX_STEPS
transits  = t_sim * v_drift / (2.0 * Lx)

print(f"dx          = {dx*1e2:.2f} cm")
print(f"dt (cyclo)  = {dt_cyclo:.3e} s")
#print(f"dt (cross)  = {dt_cross:.3e} s")
print(f"const_dt    = {const_dt:.3e} s   <- min(cyclo, cross)")
print(f"sim time    = {t_sim*1e6:.2f} us  ({MAX_STEPS} steps)")
print(f"transits    = {transits:.2f}     (need >~3-5 for steady-state standoff)")
if r_CF > Lx:
    print(f"WARNING: r_CF = {r_CF*100:.1f} cm exceeds Lx = {Lx*100:.1f} cm — standoff is OUTSIDE the domain")
else:
    print(f"r_CF / Lx   = {r_CF/Lx:.2f}     (standoff inside domain)")

# %% [markdown]
# ## Ion populations: background fill + ongoing flux injection
#
# Hybrid-PIC operates on `1/n` in Ohm's law and substeps the B-field at a rate that goes as `B / sqrt(μ₀·n·m_i)` (whistler/Alfvén CFL). With an *empty* box on step 0 the local density is just `n_floor` everywhere, the Alfvén speed near the wires (|B| ~ few T) reaches ~10¹⁰ m/s, and the B-substep CFL becomes impossibly tight — step 1 stalls in `HybridPICEvolveFields` even before particles arrive. Flux injection alone can't seed the bulk fast enough (one cell per ~6 steps at `v_drift`).
#
# Fix: pre-fill the domain with the upstream plasma drifting at `-v_drift`, and keep the `x_hi` flux injector running to maintain inflow as particles get scraped on the downstream face. The same proton kind is used for both so the hybrid solver sees a single ion density.

# %%
vi_rms = np.sqrt(T_i_eV * sc.eV / m_i)
flux   = n_stream * v_drift  # ions / m^2 / s

background_dist = picmi.UniformDistribution(
    density=n_stream,
    rms_velocity=[vi_rms, vi_rms, vi_rms],
    directed_velocity=[-v_drift, 0.0, 0.0],
    fill_in=True,
)
background_i = picmi.Species(
    particle_type="proton", name="background_i",
    initial_distribution=background_dist,
)

stream_i_dist = picmi.UniformFluxDistribution(
    flux=flux,
    flux_normal_axis="x",
    surface_flux_position=+Lx,
    flux_direction=-1,
    gaussian_flux_momentum_distribution=False,
    rms_velocity=[vi_rms, vi_rms, vi_rms],
    directed_velocity=[-v_drift, 0.0, 0.0],
)
stream_i = picmi.Species(
    particle_type="proton", name="stream_i",
    initial_distribution=stream_i_dist,
)

layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=4, grid=grid)

# %% [markdown]
# ## Diagnostics
#
# Full EM fields + ion charge density at a fixed cadence, plus ion phase-space dumps for post-processing. Particle loss across the coil-to-coil midplane segment is measured non-perturbatively by the flux-counter callback below (no boundary-scraping diagnostic).

# %%
PERIOD = 10

field_diag = picmi.FieldDiagnostic(
    name="diag",
    grid=grid,
    period=PERIOD,
    data_list=["Ex", "Ey", "Ez", "Bx", "By", "Bz",
               "rho_background_i", "rho_stream_i"],
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)
part_diag = picmi.ParticleDiagnostic(
    name="diag",
    period=PERIOD,
    species=[background_i, stream_i],
    data_list=["x", "z", "ux", "uy", "uz", "weighting"],
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)

# %% [markdown]
# ## Build + step
#
# Linear particle shape is recommended for hybrid. `sim.step()` is left commented as a checkpoint; uncomment when ready to run.

# %%
sim = picmi.Simulation(
    solver=solver,
    max_steps=MAX_STEPS,
    verbose=True,
    particle_shape="linear",
    warpx_grid_type="collocated",   # recommended for hybrid
)
sim.add_species(background_i, layout=layout)
sim.add_species(stream_i, layout=layout)
sim.add_diagnostic(field_diag)
sim.add_diagnostic(part_diag)

# %% [markdown]
# ## Resistive coil islands via a current-damping callback
#
# `plasma_resistivity(rho,J)` cannot depend on position, so we emulate the two
# high-η coil interiors by multiplying the freshly-deposited plasma current by a
# spatial factor that → 0 at `(x=0, z=±d/2)` and → 1 in the bulk. With contrast
# `eta_coil/eta_bg`, the wire cells carry essentially no current — a stand-in for
# a conductor island. The mask is built once from the (collocated) current mesh
# and cached. Installed `afterdeposition` so it runs before the hybrid E-solve
# consumes the current each step.

# %%
from pywarpx.callbacks import installafterdeposition

_coil_damp_mask = {}

def damp_current_at_coils():
    Direction = sim.extension.libwarpx_so.Direction
    contrast = eta_coil / eta_bg
    for idir in (0, 1, 2):
        mf = sim.fields.get("current_fp", dir=Direction(idir), level=0)
        factor = _coil_damp_mask.get(idir)
        if factor is None:
            xs = mf.mesh("x")
            zs = mf.mesh("z")
            X = xs[:, None]
            Z = zs[None, :]
            g = (np.exp(-(X * X + (Z - dh) ** 2) / (w * w))
                 + np.exp(-(X * X + (Z + dh) ** 2) / (w * w)))
            factor = 1.0 / (1.0 + contrast * g)
            _coil_damp_mask[idir] = factor
        arr = mf[:, :]
        fac = factor.reshape(arr.shape[:2] + (1,) * (arr.ndim - 2))
        mf[:, :] = arr * fac

installafterdeposition(damp_current_at_coils)

# %% [markdown]
# ## Non-perturbing midplane flux counter
#
# Instead of an absorbing boundary, we measure ion loss across the line segment
# joining the two coils (`x=0, |z| < d/2`) with a read-only `afterstep` callback.
# It only *reads* the live particle arrays, so the run is bit-for-bit identical to
# one without it. Crossing rate is the standard "flux through a plane" estimator:
# for particles momentarily inside a one-cell-wide slab `|x| < L/2` straddling the
# plane, `rate = Σ w·|v_x| / L`, split by sign of `v_x`. A particle drifts
# `v·dt ≈ 0.01 m ≪ L ≈ 0.078 m` per step, so it sits in the slab for ~8 steps and
# is never skipped; integrating `rate·dt` recovers ~`w` per crossing. `rate_minus`
# (v_x<0, downstream toward the midplane) is the leak that penetrates the standoff;
# `rate_plus` is the return flux. Output: `diags/segment_flux.npz`.

# %%
from pywarpx.callbacks import installafterstep

_seg_L = 2.0 * Lx / Nx                 # slab width ≈ one cell (dx)
_seg_t, _seg_minus, _seg_plus = [], [], []
_seg_n = [0]                           # mutable step counter (closure-friendly)
_seg_state = {}                        # lazily-cached containers + ParIter

def _seg_gather(container, parit):
    """Concatenate per-tile (x, z, ux, w) SoA arrays for one species (level 0).
    Reads share WarpX memory (copy=False) — purely a measurement, no writes."""
    ix = container.get_real_comp_index("x")
    iz = container.get_real_comp_index("z")
    iu = container.get_real_comp_index("ux")
    iw = container.get_real_comp_index("w")
    xs, zs, us, ws = [], [], [], []
    for pti in parit(container, 0):
        soa = pti.soa()
        xs.append(np.array(soa.get_real_data(ix), copy=False))
        zs.append(np.array(soa.get_real_data(iz), copy=False))
        us.append(np.array(soa.get_real_data(iu), copy=False))
        ws.append(np.array(soa.get_real_data(iw), copy=False))
    if not xs:
        e = np.empty(0)
        return e, e, e, e
    return (np.concatenate(xs), np.concatenate(zs),
            np.concatenate(us), np.concatenate(ws))

def count_segment_flux():
    if not _seg_state:
        # `sim.particles.get(name)` -> pyAMReX WarpXParticleContainer (the
        # non-deprecated path; the old ParticleContainerWrapper is broken against
        # this pyAMReX build). Cache the containers + the ParIter type once.
        _seg_state["parit"] = sim.extension.libwarpx_so.WarpXParIter
        _seg_state["pc"] = {s: sim.particles.get(s)
                            for s in ("background_i", "stream_i")}
    parit = _seg_state["parit"]
    rminus = rplus = 0.0
    for container in _seg_state["pc"].values():
        x, z, vx, wt = _seg_gather(container, parit)   # vx = u ≈ v_x (non-rel)
        if x.size == 0:
            continue
        m = (np.abs(x) < 0.5 * _seg_L) & (np.abs(z) < dh)
        if m.any():
            vm = vx[m]; wm = wt[m]
            neg = vm < 0
            rminus += float(np.sum(wm[neg]  * -vm[neg])) / _seg_L
            rplus  += float(np.sum(wm[~neg] *  vm[~neg])) / _seg_L
    _seg_t.append(_seg_n[0] * const_dt)
    _seg_minus.append(rminus)
    _seg_plus.append(rplus)
    _seg_n[0] += 1

installafterstep(count_segment_flux)

sim.step()

# Reduce across ranks (slab particles may live on any rank) and write once on root.
try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _seg_minus = _comm.allreduce(np.asarray(_seg_minus), op=MPI.SUM)
    _seg_plus = _comm.allreduce(np.asarray(_seg_plus), op=MPI.SUM)
    _is_root = _comm.Get_rank() == 0
except Exception:
    _is_root = True
if _is_root:
    os.makedirs("diags", exist_ok=True)
    np.savez(os.path.join("diags", "segment_flux.npz"),
             t_s=np.asarray(_seg_t, dtype=float),
             rate_minus=np.asarray(_seg_minus, dtype=float),
             rate_plus=np.asarray(_seg_plus, dtype=float),
             dh=dh, slab_L=_seg_L)
    print(f"[segment_flux] wrote diags/segment_flux.npz "
          f"({len(_seg_t)} steps, peak rate_minus={np.max(_seg_minus):.3e})")

# %% [markdown]
# ## Post-processing sketch — β_dyn = 1 vs. Chapman–Ferraro
#
# After `sim.step()` completes:
#
# 1. Open `diags/diag/openpmd_%T.h5` with `openpmd-api`.
# 2. For each iteration, bin the ion particle dump onto the field grid to build `n_i(x,z)`, `⟨v_i⟩(x,z)`, and `T_i(x,z) = m_i⟨(v - ⟨v⟩)²⟩/k_B` per cell. `n_e = n_i` is implicit in hybrid; `T_e` is the fixed solver input.
# 3. Compute `B² = Bx² + By² + Bz²` from the grid.
# 4. Compute the dimensionless ratios and overlay the empirical and analytic standoffs.

# %%
# Sketch (uncomment + adapt once a run exists):
#
# from openpmd_api import Series, Access
# import matplotlib.pyplot as plt
#
# series = Series("diags/diag/openpmd_%T.h5", Access.read_only)
# it = series.iterations[max(series.iterations)]
#
# Bx = it.meshes["B"]["x"].load_chunk()
# By = it.meshes["B"]["y"].load_chunk()
# Bz = it.meshes["B"]["z"].load_chunk()
# series.flush()
# B2 = Bx**2 + By**2 + Bz**2
#
# # n_i, <v_i>, T_i from ion particle dump (numpy.histogram2d on (x,z),
# # weighted by w, w*ux/..., w*v^2) -> grids n_grid, v2_grid, T_i_grid.
# # rho_grid = n_grid * m_i  (ions dominate mass density in hybrid).
#
# beta_dyn = 2.0 * sc.mu_0 * rho_grid * v2_grid / B2
# beta_th  = 2.0 * sc.mu_0 * n_grid * sc.k * (T_i_grid + T_e_eV * sc.eV / sc.k) / B2
#
# fig, ax = plt.subplots(figsize=(6, 6))
# X, Z = np.meshgrid(np.linspace(-Lx, Lx, Nx), np.linspace(-Lz, Lz, Nz), indexing="ij")
# ax.contour(X, Z, beta_dyn, levels=[1.0], colors="k")            # empirical standoff
# theta = np.linspace(0, 2*np.pi, 200)
# ax.plot(r_CF*np.cos(theta), r_CF*np.sin(theta), "r--", label="r_CF")  # analytic
# ax.set_aspect("equal"); ax.legend()
