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
    eta_H:      float = 3.0e-4     # hyper-resistivity (Ohm·m^3)
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

        print(f"[gyroradius]")
        v_ion = np.sqrt(2 * self.T_i_eV * sc.eV / sc.m_p)
        B_coil = (MU0 * self.I) / self.dia
        w_ci = E_C * B_coil / sc.m_p
        print("[vion] ", v_ion)
        print("[B_coil] ", B_coil)
        print("[w_ci] ", w_ci)
        r_i = v_ion / w_ci 
        print(f"[gyroradius] {r_i*100:.3f} cm")
        w_pi = np.sqrt(self.p_density * E_C**2 / (EPS0 * sc.m_p))
        inertial_length = (sc.c / w_pi)
        print(inertial_length*100)
        exit()

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

import warpx_polywell.bext.analytic as analytic

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

# TODO: Increasing n_macroparticles_per_cell could decrease noise, going to try bumping to 5, though this is more computationally expensive
layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=5, grid=grid)

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
    warpx_grid_type="collocated",   # recommended for hybrid,
    # NOTE: Filtering may not work, but interesting to note it is a filter as a function of wavenumber
    # NOTE: This did not work at 2, it may be sufficient to have one pass
    # NOTE: Tried: npass_each_dim = [2, 2, 2], broke around 426
    # NOTE: Tried: [3, 3, 3], boke around 485
    # NOTE: Tried [1, 1, 1], broke around 468
    # WarpX implements a strided filter, allowing for optimal high-pass filtering
    # They use techniques from Vay et al 2011 which show that strided filtering (n-stride filter -> smooth with neighbors a distance n away from current point)
    # doing a combination of 1-, 2-, 3-, 4-stride n-pass filtering was beneficial in paralelization and speed.
    # doi:10.1016/J.Jcp.2011.04.003
    # Becomes a suppression of the signal at integer multiples of the Nyquist wavelength
    # Wide-band, low-pass filter
    # The gain is a function of wavenumber, $g = \alpha + (1-\alpha)\cos(kx)$
    # The binomial filter uses alpha = 0.5
    # Ideally we have total attenuation ~ 1 for physics we care about, and ~ 0 for unphysical waves
    # Due to the formula, k->0 implies g = 1, while k -> 2dx -> g = 0.5 - 0.5 = 0
    # Then a compensation is added, to bring back low wavenumber waves, while keeping the damping on the high wavenumbers
    warpx_use_filter=True # Bilinear filtering smooths the charge and currents on the mesh, after depositing them from the macro-particles
)
warpx.filter_npass_each_dir = [3, 3, 3]
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
import time

os.chdir('./inputs/coil3D')
new_run_dir = f"runs_{time.time()}"
os.makedirs(new_run_dir, exist_ok=True)
os.chdir(new_run_dir)

sim.step()