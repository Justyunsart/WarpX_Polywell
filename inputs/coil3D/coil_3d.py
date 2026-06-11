"""
Stability issues, trying a smaller timestep to determine if it remains the point of plasma-coil contact


"""

import numpy as np
import scipy.constants as sc
from pywarpx import picmi, warpx
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

C    = sc.c
E_C  = sc.e
M_P  = sc.m_p
M_E  = sc.m_e
EV   = sc.eV
MU0  = 4 * np.pi * 1e-7
EPS0 = sc.epsilon_0
MAX_STEPS = 1000

@dataclass
class SingleCoil3DConfig:

    # Stream params
    n_stream:   float = 1.0e17     # m^-3
    v_drift:    float = 5.0e5      # m/s, magnitude (flow toward -x)
    T_e_eV:     float = 10.0       # electron fluid temperature
    T_i_eV:     float = 10.0       # ion thermal spread on injection

    # Coil params (coil sits at x=0, axis along x)
    I:          float = 1e5        # A
    dia:        float = 0.5        # m

    # Grid
    # Increasing L softens coil B-field anomaly (eps = dx = L / N)
    L:          float = 2.5        # m (half-extent, all axes)
    N:          int   = 64         # cells per axis

    # Numerical stability (mirrors coil_2d.py)
    eta_bg:     float = 1.0e-7     # background resistivity (Ohm·m)
    eta_coil:   float = 1.0e3      # coil island resistivity (Ohm·m)
    eta_H:      float = 3.0e-3     # hyper-resistivity (Ohm·m^3)
    n_floor_frac: float = 0.05     # n_floor as fraction of n_stream
    substeps:   int   = 100        # B-field substeps

    # Derived — not set by user
    R_coil:     float = field(init=False)   # coil radius (m)
    dx:         float = field(init=False)   # cell width (m)
    eps:        float = field(init=False)   # wire regularization width (~dx)
    m_i:        float = field(init=False)
    rho:        float = field(init=False)   # upstream mass density
    P_ram:      float = field(init=False)   # ram pressure
    B_ref:      float = field(init=False)   # |B| at r_CF
    r_CF:       float = field(init=False)   # Chapman-Ferraro standoff (m)
    Omega_ci:   float = field(init=False)   # ion cyclotron freq at r_CF
    d_i:        float = field(init=False)   # ion inertial length
    dt:         float = field(init=False)   # outer timestep
    n_floor:    float = field(init=False)   # absolute density floor

    def __post_init__(self):
        MU0 = sc.mu_0

        self.R_coil = self.dia / 2
        self.dx     = 2 * self.L / self.N
        self.eps    = self.dx                        # one cell, matches coil_2d convention

        self.m_i    = sc.m_p
        self.rho    = self.n_stream * self.m_i
        self.P_ram  = self.rho * self.v_drift**2

        # On-axis B for a circular loop: B_x(r) = mu0*I*R^2 / (2*(R^2+r^2)^(3/2))
        # Solve P_ram = B^2 / (2*mu0) numerically for r_CF
        # sqrt(P_ram * 2 * mu0) = B
        # sqrt(P_ram * 2 * MU0) = MU0 * I * R^2 / 2 * (R^2 +)
        self.r_CF = np.sqrt((MU0 * self.I * self.R_coil**2 / (2 * np.sqrt(2 * MU0 * self.P_ram)))**(2/3) - self.R_coil**2)
        if self.r_CF < 0 or not np.isreal(self.r_CF):
            raise ValueError(f"r_CF is imaginary — P_ram too large for coil to stand off the flow. Increase I or reduce n_stream/v_drift.")

        self.B_ref    = MU0 * self.I * self.R_coil**2 / (2 * (self.R_coil**2 + self.r_CF**2)**1.5)
        self.Omega_ci = sc.e * self.B_ref / self.m_i
        self.d_i      = sc.c / np.sqrt(self.n_stream * sc.e**2 / (sc.epsilon_0 * self.m_i))
        self.n_floor  = self.n_floor_frac * self.n_stream

        # Timestep: min of 1/50 cyclotron period and 0.4*dx/v_e_th
        v_e_th    = np.sqrt(self.T_e_eV * sc.eV / sc.m_e)
        dt_cyclo  = 1.0 / (50.0 * self.Omega_ci)
        dt_efluid = 0.4 * self.dx / v_e_th
        self.dt   = min(dt_cyclo, dt_efluid)

        t_sim     = self.dt * MAX_STEPS
        transits  = t_sim * self.v_drift / (2.0 * self.L)

        print(f"P_ram       = {self.P_ram:.3e} Pa")
        print(f"r_CF        = {self.r_CF*100:.2f} cm  (predicted standoff)")
        print(f"B at r_CF   = {self.B_ref*1e4:.2f} G")
        print(f"d_i (n_inf) = {self.d_i*100:.2f} cm  (ion inertial length upstream)")
        print(f"Omega_ci    = {self.Omega_ci:.3e} rad/s  -> 1/Omega_ci = {1/self.Omega_ci:.3e} s")
        print(f"dx          = {self.dx*1e2:.2f} cm")
        print(f"dt (cyclo)  = {dt_cyclo:.3e} s")
        #print(f"dt (cross)  = {dt_cross:.3e} s")
        print(f"const_dt    = {self.dt:.3e} s   <- min(cyclo, cross)")
        print(f"sim time    = {t_sim*1e6:.2f} us  ({MAX_STEPS} steps)")
        print(f"transits    = {transits:.2f}     (need >~3-5 for steady-state standoff)")
        if self.r_CF > self.L:
            print(f"WARNING: r_CF = {self.r_CF*100:.1f} cm exceeds Lx = {self.L*100:.1f} cm — standoff is OUTSIDE the domain")
        else:
            print(f"r_CF / Lx   = {self.r_CF/self.L:.2f}     (standoff inside domain)")

        v_A_ref = self.B_ref / np.sqrt(sc.mu_0 * self.n_floor * self.m_i)
        v_A_peak = 0.25 / np.sqrt(sc.mu_0 * self.n_floor * self.m_i)
        print(v_A_ref, (v_A_ref * self.dt / 100) / self.dx, (v_A_peak * self.dt / 100) / self.dx)

        eta_H_max = sc.mu_0 * self.dx**2 / self.dt
        print(f"eta_H_max = {eta_H_max:.3e}")

        ratio = self.eta_H / (self.eta_bg * self.dx**2)
        print(f"eta_H / (eta_bg * dx²) = {ratio:.3e}")

        eta_bg_balanced = self.eta_H / self.dx**2
        print(f"balanced eta_bg = {eta_bg_balanced:.3e}")

cfg = SingleCoil3DConfig()

grid = picmi.Cartesian3DGrid(
    number_of_cells=[cfg.N, cfg.N, cfg.N],
    lower_bound=[-cfg.L]*3,
    upper_bound=[cfg.L]*3,
    lower_boundary_conditions=["neumann"]*3,
    upper_boundary_conditions=["neumann"]*3,
    lower_boundary_conditions_particles=["absorbing"]*3,
    upper_boundary_conditions_particles=["absorbing"]*3,
    warpx_max_grid_size=32,
)

import src.bext.analytic as analytic

analytic.POLYWELL_COILS = [('x', 1, 1)]
A_external = analytic.build_aext_expressions(I=cfg.I, dia=cfg.dia, offset=0.0, eps=cfg.eps)
print(A_external)

solver = picmi.HybridPICSolver(
    grid=grid,
    Te=cfg.T_e_eV,
    n0=cfg.n_stream,
    gamma=5.0/3.0,
    n_floor=cfg.n_floor,                 # 10% of upstream — caps 1/n amplification in the deepening cavity
    plasma_resistivity=cfg.eta_bg,               # uniform; coil islands emulated via callback
    plasma_hyper_resistivity=cfg.eta_H,         # Ohm·m^3; overdamps grid-Nyquist whistlers at peak |B|~0.13T
    holmstrom_vacuum_region=True,            # suppress Hall/pressure terms in the cavity
    substeps=cfg.substeps,                           # dt_sub ≈ 3e-11 s; clears whistler CFL at peak |B| with ~3x margin
    A_external=A_external,
    do_external_diva_cleaning=False,         # A is analytically div-free
)

vi_rms = np.sqrt(cfg.T_i_eV * sc.eV / cfg.m_i)
flux   = cfg.n_stream * cfg.v_drift  # ions / m^2 / s

background_dist = picmi.UniformDistribution(
    density=cfg.n_stream,
    rms_velocity=[vi_rms, vi_rms, vi_rms],
    directed_velocity=[-cfg.v_drift, 0.0, 0.0],
    fill_in=True,
)
background_i = picmi.Species(
    particle_type="proton", name="background_i",
    initial_distribution=background_dist,
)

stream_i_dist = picmi.UniformFluxDistribution(
    flux=flux,
    flux_normal_axis="x",
    surface_flux_position=cfg.L,
    flux_direction=-1,
    gaussian_flux_momentum_distribution=False,
    rms_velocity=[vi_rms, vi_rms, vi_rms],
    directed_velocity=[-cfg.v_drift, 0.0, 0.0],
)
stream_i = picmi.Species(
    particle_type="proton", name="stream_i",
    initial_distribution=stream_i_dist,
    warpx_save_particles_at_xlo = True,
    warpx_save_particles_at_xhi = True,
    warpx_save_particles_at_ylo = True,
    warpx_save_particles_at_yhi = True,
    warpx_save_particles_at_zlo = True,
    warpx_save_particles_at_zhi = True
)

layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=4, grid=grid)

PERIOD = 10

field_diag = picmi.FieldDiagnostic(
    name="diag",
    grid=grid,
    period=PERIOD,
    data_list=["Ex", "Ey", "Ez", "Bx", "By", "Bz",
                "rho_stream_i", "rho_background_i"],
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)
part_diag = picmi.ParticleDiagnostic(
    name="diag",
    period=PERIOD,
    species=[stream_i],
    data_list=["x", "y", "z", "ux", "uy", "uz", "weighting"],
    warpx_format="openpmd",
    warpx_openpmd_backend="h5",
)

# Boundary scraping diagnostics
scraping_diag = picmi.ParticleBoundaryScrapingDiagnostic(
    name="boundary_scraping",
    period=PERIOD,
    species=[stream_i],
    data_list=["x", "y", "z", "ux", "uy", "uz", "weighting"],
    warpx_format='openpmd',
    warpx_openpmd_backend='h5',
)

field_energy_reduced_diag = picmi.ReducedDiagnostic(
    diag_type="FieldEnergy",
    name="field_energy",
    period=1
)

particle_energy_reduced_diag = picmi.ReducedDiagnostic(
    diag_type="ParticleEnergy",
    name="particle_energy",
    period=1
)

# CHANGED: Added filtering to smooth currents with a binomial filter 2-pass
warpx.const_dt = cfg.dt
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
sim.add_diagnostic(scraping_diag)
sim.add_diagnostic(field_energy_reduced_diag)
sim.add_diagnostic(particle_energy_reduced_diag)

#from pywarpx.callbacks import installbeforeesolve
from pywarpx.callbacks import installafterdeposition, installbeforeEsolve

_coil_damp_mask = {}

def damp_current_at_coils():
    try:
        Direction = sim.extension.libwarpx_so.Direction
        contrast = cfg.eta_coil / cfg.eta_bg
        for idir in (0, 1, 2):
            mf = sim.fields.get("current_fp", dir=Direction(idir), level=0)
            factor = _coil_damp_mask.get(idir)
            if factor is None:
                xs = mf.mesh("x")
                ys = mf.mesh("y")
                zs = mf.mesh("z")
                X = xs[:, None, None]
                Y = ys[None, :, None]
                Z = zs[None, None, :]
                d2 = X**2 + (np.sqrt(Y**2 + Z**2) - cfg.R_coil)**2
                g = np.exp(-d2 / cfg.dx**2)
                factor = 1.0 / (1.0 + contrast * g)
                _coil_damp_mask[idir] = factor
                print(f"[coil J] dir {idir}: factor.min = {factor.min():.3e} (cached)")
            arr = mf[...]
            fac = factor.reshape(arr.shape[:3] + (1,) * (arr.ndim - 3))
            mf[...] = arr * fac
    except Exception as e:
        print(f"[damp_current_at_coils] ERROR: {e}")
        raise

installafterdeposition(damp_current_at_coils)

from pywarpx.callbacks import installafterstep

_slab_L   = cfg.dx
_flux_t, _flux_minus, _flux_plus = [], [], []
_flux_n   = [0]
_flux_pc  = {}   # lazily cached

def count_coil_flux():
    if not _flux_pc:
        _flux_pc["pc"] = sim.particles.get("stream_i")
    pc = _flux_pc["pc"]

    rminus = rplus = 0.0
    for pti in pc.iterator(level=0):
        x  = np.array(pti['x'],  copy=False)
        y  = np.array(pti['y'],  copy=False)
        z  = np.array(pti['z'],  copy=False)
        vx = np.array(pti['ux'], copy=False)
        wt = np.array(pti['w'],  copy=False)

        r    = np.sqrt(y**2 + z**2)
        mask = (np.abs(x) < 0.5 * _slab_L) & (r < cfg.R_coil)
        if mask.any():
            vm = vx[mask]; wm = wt[mask]
            neg = vm < 0
            # _slab_L = dx = 7.8cm
            rminus += float(np.sum(wm[neg]  * -vm[neg]))  / _slab_L
            rplus  += float(np.sum(wm[~neg] *  vm[~neg])) / _slab_L

    _flux_t.append(_flux_n[0] * cfg.dt)
    _flux_minus.append(rminus)
    _flux_plus.append(rplus)
    _flux_n[0] += 1

installafterstep(count_coil_flux)

def save_cusp_losses():
    np.savez("diags/cusp_flux.npz",
             times=np.asarray(_flux_t, dtype=float),
             flux_minus=np.asarray(_flux_minus, dtype=float),
             flux_plus=np.asarray(_flux_plus, dtype=float),
             )
    
from pywarpx.callbacks import installafterdiagnostics

installafterdiagnostics(save_cusp_losses)

import os

os.chdir('/Users/hehe/Documents/plasma_research/warpx/WarpX_Polywell/inputs/coil3D')
os.makedirs("diags", exist_ok=True)

sim.step()