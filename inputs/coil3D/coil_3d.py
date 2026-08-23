"""
Stability issues, trying a smaller timestep to determine if it remains the point of plasma-coil contact


"""

import numpy as np
import scipy.constants as sc
from pywarpx import picmi, warpx
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import warpx_polywell.bext.aext as aext

C    = sc.c
E_C  = sc.e
M_P  = sc.m_p
M_E  = sc.m_e
EV   = sc.eV
MU0  = 4 * np.pi * 1e-7
EPS0 = sc.epsilon_0
# NOTE: Benchmarking
MAX_STEPS = 100
TOTAL_TIME = 20e-6
NUM_DIAGS = 100

@dataclass
class SingleCoil3DConfig:

    # Stream params
    n_stream:   float = 1e17     # m^-3
    v_drift:    float = 1e5      # m/s, magnitude (flow toward -x)
    T_e_eV:     float = 1000.0       # electron fluid temperature
    T_i_eV:     float = 1000.0       # ion thermal spread on injection

    # Coil params (coil sits at x=0, axis along x)
    I:          float = 2.25e4       # A, reference current when using disk
    dia:        float = 1.0        # m

    # Grid
    # Increasing L softens coil B-field anomaly (eps = dx = L / N)
    L:          float = 1.5        # m (half-extent, all axes)
    N:          int   = 64          # cells per axis

    # Numerical stability (mirrors coil_2d.py)
    eta_bg:     float = 1.0e-7     # background resistivity (Ohm·m)
    eta_coil:   float = 1.0e3      # coil island resistivity (Ohm·m)
    eta_H:      float = 9.0e-4     # hyper-resistivity (Ohm·m^3)
    n_floor_frac: float = 0.05     # n_floor as fraction of n_stream
    substeps:   int   = 100         # B-field substeps

    # disk vs ring configuration
    disk:       bool  = True       # whether to use a ring or a disk 
    r1:         float = 0.1         # inner radius of disk
    r2:         float = 0.6         # outer radius of disk
    n_turns:    int   = 10          # number of turns between this r1 and r2
    scale_down: bool  = False

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
    v_drift_scale: float = field(init=False) 

    def __post_init__(self):

        # If we use a disk, the current will be scaled, and this will be stored within the config file
        if self.disk:
            aext.POLYWELL_COILS = ['x',1,1]
            xs = np.linspace(-self.L, self.L, self.N)
            dx = xs[1] - xs[0]
            X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
            self_scale = 10
            disk_Bx, _, _, scale = aext.get_B_disk(X, Y, Z, self.I / self_scale, self.r1, self.r2, self.n_turns, 0.5, dx, dx, dx, self.scale_down)
            ring_Bx, _, _ = aext.get_B_ring(X, Y, Z, self.I, 0.5, dx, dx, dx)
            self.I /= self_scale
            print(f"I used: {self.I} A")
            slice_x = np.s_[:, self.N//2, self.N//2]
            print(f"Disk vs Ring @ center:\nRing @ center: {np.max(ring_Bx[slice_x])}\nDisk @ center: {np.max(disk_Bx[slice_x])}")

        self.v_drift_scale = 1e5 / self.v_drift

        MU0 = sc.mu_0

        self.R_coil = self.dia / 2
        self.dx     = 2 * self.L / self.N
        self.eps    = self.dx                        # one cell, matches coil_2d convention

        self.m_i    = sc.m_p
        self.rho    = self.n_stream * self.m_i
        self.P_ram  = self.rho * self.v_drift**2
        Ti_J = self.T_i_eV * sc.eV
        Te_J = self.T_e_eV * sc.eV
        self.P_th = self.n_stream * (Ti_J + Te_J)

        self.d_i      = sc.c / np.sqrt(self.n_stream * sc.e**2 / (sc.epsilon_0 * self.m_i))
        self.n_floor  = self.n_floor_frac * self.n_stream

        v_ion = np.sqrt(2 * self.T_i_eV * sc.eV / sc.m_p)
        B_coil = (MU0 * self.I) / self.dia if not self.disk else np.max(disk_Bx[:, self.N // 2, self.N // 2])
        w_ci = E_C * B_coil / sc.m_p
        conservative_dt = 1.0 / (50.0 * w_ci)
        print(f"[vion] {v_ion:.3e} m/s")
        print(f"[B_coil] {B_coil:.3e} T")
        print(f"[w_ci] {w_ci:.3e} rad/s")
        print(f"[conservative dt] {conservative_dt:3e}")
        print(f"[conservative b_dt] {conservative_dt / self.substeps:.3e}")
        self.dt = conservative_dt
        r_i = self.v_drift / w_ci 
        r_ci_thermal = v_ion / w_ci
        print(f"[gyroradius min, ram] {r_i*100:.3f} cm")
        print(f"[gyroradius min, th] {r_ci_thermal*100:.3f} cm")
        w_pi = np.sqrt(self.n_stream * E_C**2 / (EPS0 * sc.m_p))
        print(f"[ion frequency] {w_pi:.3e}")
        inertial_length = (sc.c / w_pi)
        print(f"[ion inertial length]: {inertial_length*100:.2e} cm, {self.d_i*100:.2e} cm")

        Ti_J = self.T_i_eV * sc.eV
        Te_J = self.T_e_eV * sc.eV
        print("BETA THERMAL AT COIL CENTER: ")
        p_plasma = self.n_stream * (Ti_J + Te_J)
        p_B = B_coil**2 / (2 * MU0)
        print(f'[beta thermal] {p_plasma / p_B:.3e}')

        print("BETA DYNAMIC AT COIL CENTER:")
        print(f"[beta dynamic] {self.P_ram / p_B:.3e}")

        print(f"[true beta] {(p_plasma + self.P_ram) / p_B:.3e}")

        print("possible eta_H")
        alpha = 0.003
        num = alpha * inertial_length**2 * w_ci * (sc.m_e / sc.m_p) * self.dx**2
        print(f"{num:.2f}")

if __name__ == "__main__":

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
    if cfg.disk:
        A_external = analytic.build_n_turn_aext_expression(I=cfg.I, a=cfg.r1, b=cfg.r2, n=cfg.n_turns, offset=0.0)

    else:
        A_external = analytic.build_aext_expressions(I=cfg.I, dia=cfg.dia, offset=0.0)

    solver = picmi.HybridPICSolver(
        grid=grid,
        Te=cfg.T_e_eV,
        n0=cfg.n_stream,
        gamma=5.0/3.0,
        n_floor=cfg.n_floor,                 # 5% of upstream — caps 1/n amplification in the deepening cavity
        plasma_resistivity=cfg.eta_bg,               # uniform; coil islands emulated via callback
        plasma_hyper_resistivity=cfg.eta_H,         # Ohm·m^3; overdamps grid-Nyquist whistlers at peak |B|~0.13T
        holmstrom_vacuum_region=True,            # suppress Hall/pressure terms in the cavity
        substeps=cfg.substeps,                           # dt_sub ≈ 3e-11 s; clears whistler CFL at peak |B| with ~3x margin
        A_external=A_external,
        do_external_diva_cleaning=False,         # A is analytically div-free,
    )

    vi_rms = np.sqrt(cfg.T_i_eV * sc.eV / cfg.m_i)
    flux   = cfg.n_stream * cfg.v_drift  # ions / m^2 / s

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

    # This ensures we always run for 20us regardless of the standoff
    # This compensates for slowly drift velocities as well
    MAX_STEPS = int((TOTAL_TIME / cfg.dt) * cfg.v_drift_scale)
    print(f"[max steps] {MAX_STEPS}")
    PERIOD = MAX_STEPS // NUM_DIAGS
    print(f"[diag period] {PERIOD}")
    t_sim = MAX_STEPS * cfg.dt
    print(f"[t sim] {t_sim * 1e6:.3e} us")

    print("[time needed]")
    transits = t_sim * cfg.v_drift / (2 * cfg.L)
    print(f"[transits] {transits}")

    field_diag = picmi.FieldDiagnostic(
        name="field_diag",
        grid=grid,
        period=PERIOD,
        data_list=["E", "B",
                    'rho_stream_i', "rho", 
                    "J", 
                    "J_displacement",
                    "T_stream_i", 
                    "part_per_cell", 
                    "divE", "divB"], # added divs for quality control/verification no funky stuff
        warpx_format="openpmd",
        warpx_openpmd_backend="h5",
    )
    part_diag = picmi.ParticleDiagnostic(
        name="part_diag",
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
        warpx_use_filter=True,
    )
    sim.add_species(stream_i, layout=layout)
    sim.add_diagnostic(field_diag)
    sim.add_diagnostic(part_diag)
    sim.add_diagnostic(scraping_diag)
    sim.add_diagnostic(field_energy_reduced_diag)
    sim.add_diagnostic(particle_energy_reduced_diag)

    # #from pywarpx.callbacks import installbeforeesolve
    from pywarpx.callbacks import installafterdeposition, installbeforeEsolve

    _coil_damp_mask = {}

    def damp_current_at_coils():
        try:
            Direction = sim.extension.libwarpx_so.Direction
            contrast = cfg.eta_coil / cfg.eta_bg
            for idir in (0, 1, 2):
                mf = sim.fields.get("current_fp", dir=Direction(idir), level=0)
                factor = _coil_damp_mask.get(idir)
                assert False, "This is running, and catching appropriately"
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

        TOTAL_RS = 5

        rs = np.linspace(cfg.R_coil / TOTAL_RS, cfg.R_coil, TOTAL_RS)

        rminus, rplus = [0]*TOTAL_RS, [0]*TOTAL_RS
        for pti in pc.iterator(level=0):
            x  = np.array(pti['x'],  copy=False)
            y  = np.array(pti['y'],  copy=False)
            z  = np.array(pti['z'],  copy=False)
            vx = np.array(pti['ux'], copy=False)
            wt = np.array(pti['w'],  copy=False)

            # radial distance 
            R    = np.sqrt(y**2 + z**2)

            # The disk will have a smaller inner radius, and hence, particles are more likely to be incident 
            # upon the disk, then pass through it, so we provide a reference x slightly in front to capture 
            # incidence at different radii
            if cfg.disk:
                ref_x = cfg.dx 
            else:
                ref_x = 0

            for k, r in enumerate(rs):
                # axial distance within dx, radial distance within current radius
                mask = (np.abs(x - ref_x) < 0.5 * _slab_L) & (R < r)
                if mask.any():
                    vm = vx[mask]; wm = wt[mask]
                    neg = vm < 0
                    # _slab_L = dx
                    rminus[k] += float(np.sum(wm[neg]  * -vm[neg]))  / _slab_L
                    rplus[k]  += float(np.sum(wm[~neg] *  vm[~neg])) / _slab_L

        _flux_t.append(_flux_n[0] * cfg.dt)
        _flux_minus.append(rminus)
        _flux_plus.append(rplus)
        _flux_n[0] += 1

    installafterstep(count_coil_flux)

    from pywarpx.callbacks import installafterdiagnostics

    def save_cusp_losses():
        np.savez("diags/cusp_flux.npz",
                times=np.asarray(_flux_t, dtype=float),
                flux_minus=np.asarray(_flux_minus, dtype=float),
                flux_plus=np.asarray(_flux_plus, dtype=float),
                )

    installafterdiagnostics(save_cusp_losses)

    # Group output under OUTPUT_DIR/coil_3d/run_* and register the run. run_session
    # chdir's into the run dir (so diagnostics + the cusp-flux callback land inside),
    # marks it completed on clean exit, and removes it on failure.
    from warpx_polywell.db.runs import run_session
    with run_session(__file__, {}):
        sim.step()