"""
This class handles specifically the Hybrid case.

It requires an instantiation of a PolywellHybridConfig or a json formatted file that you can import.

There already exists one config in configs/polywell_hybrid_config.py

HYBRID_CONFIG 
- This is a single turn HybridPICSolver config

One may wish to run with a flat n-turn disk, in this case they may provide the following arguments
    --nturns: int
    --r1: float
    --r2: float

Note that the bounds are quite small, this is for testing stability in a quicker manner, and this 
implementation could benefit from a larger scale system. 
"""

"""
Solutions for current state of the script:

1) Current damping about coils via resistivity
- This reduces numerical artifacts that arise when particles pass through the coils, and acts as a spacial resistivity formula taking into account coil geometry (thanks Yoon)
- It might be worthwhile to construct a global dynamic resistivity formula that allows for dispersion/advection standoffs to come about more naturally

2) Hyper resistivity (Ohm * m^2)
- I initially had this at 1e-11, but bumping this up to 1e-8 provided smoother B dispersion, no random spikes when plasma pressure influences the field
- Plasma resistivity is kept at 3e-8 (Ohm * m)
"""

import os
import shutil

# Pywarpx utils, plus callbacks which can be useful for diagnostic (live saving of fields for example)
from pywarpx import picmi, callbacks, warpx

from warpx_polywell.db.runs import RunsDB, new_run_dir

from configs.polywell_hybrid_config import PolywellHybridConfig, HYBRID_CONFIG
from warpx_polywell.domain import plasma_bounds
from warpx_polywell.spawn import make_layout

import numpy as np
import scipy.constants as sc

import argparse
from pathlib import Path

from warpx_polywell.bext.analytic import build_aext_expressions, build_n_turn_aext_expression

from dataclasses import asdict
import json

parser = argparse.ArgumentParser(
    description="Polywell simulation"
)

parser.add_argument(
    "--cfg", default=None,
    help="Provide a .json file path to use for loading the config"
)

parser.add_argument(
    "--nturns", 
    type=int,
    help="Optional --- specify n-turns for the flat disc coils, defaults to 1"
)

parser.add_argument(
    "--r1", 
    type=float,
    help="Optional only if --nturns not specified --- specify inner radius of disc coils, "
)

parser.add_argument(
    "--r2", 
    type=float,
    help="Optional only if --nturns not specified --- specify outer radius of disc coils, "
)

args = parser.parse_args()
cfg = args.cfg
n_turns = args.nturns
r1 = args.r1
r2 = args.r2

C    = sc.c
E_C  = sc.e
M_P  = sc.m_p
M_E  = sc.m_e
EV   = sc.eV
MU0  = 4 * np.pi * 1e-7
EPS0 = sc.epsilon_0

# When not loading a config from a .json file
if not isinstance(cfg, str):

    cfg = HYBRID_CONFIG

    if n_turns:
        assert r1, "Need to specify inner radius with --r1"
        assert r2, "Need to specify outer radius with --r2"
        cfg.n_turns = n_turns
        cfg.r1 = r1
        cfg.r2 = r2
        print("*********************************")
        print(f"Using n-turn hybrid config with n = {cfg.n_turns}, r1 = {cfg.r1}, r2 = {cfg.r2}")
        print("*********************************")

    cfg.compute_b()

class PollywellSixCoilHybrid: 
    def __init__(self, cfg: PolywellHybridConfig | str):
        """
        cfg: PolywellHybridConfig | Path to json-formatted polywell config
        """
        self.store_config_params(cfg)
        self.sim = picmi.Simulation(
            verbose=True, 
            warpx_grid_type = "collocated",
            warpx_use_filter = True
            )
        warpx.filter_npass_each_dir = [2, 2, 2]
        self.get_plasma_quantities()
        self.get_sim_length()
        exit()
        self.set_grid()
        self.set_b_field()
        self.set_solver() 
        self.add_species()
        self.add_diagnostics()
        self.define_run_management()
        self.sim.initialize_inputs()
        self.sim.initialize_warpx()
        self.add_injection_callback()
        self.zero_coil_current()

    def save_cfg_to_json(self, cfg: PolywellHybridConfig, filepath: str) -> None:
        """Converts a dataclass instance to JSON and saves it to a file."""
        # Convert dataclass to a standard dictionary
        data_dict = asdict(cfg)
        # these are instantiated using initial arguments, hence not needed for init of PolywellConfig
        not_required_for_init = ['domain', 'Ti_J', 'Te_J', 'dx', 'B_coil']
        for not_required_key in not_required_for_init:
            data_dict.pop(not_required_key)
        # remove n_turns, r1, r2 if n_turns == 1
        if data_dict['n_turns'] == 1:
            print("REMOVING N_TURNS R1 AND R2 FROM CONFIG")
            data_dict.pop('n_turns')
            data_dict.pop('r1')
            data_dict.pop('r2')
        filename = 'config_used.json'
        filepath = Path(filepath, filename)
        # Save the dictionary as a pretty-printed JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=4)
        print(f"Successfully saved data to {filepath}")

    def load_cfg_from_json(self, filepath: str):
        """Loads a JSON file and unpacks it back into the specified dataclass."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
            
        # Unpack the dictionary keys as keyword arguments (**kwargs) into the class
        return PolywellHybridConfig(**data_dict)

    def store_config_params(self, cfg):
        if isinstance(cfg, str):
            try:
                print("[config loading] LOADING CFG FROM PATH")
                cfg = self.load_cfg_from_json(cfg)
            except Exception as e:
                print(f"[config loading] ERROR::{e}")
                raise ValueError()
        # save config
        self.cfg = cfg
        # Hybrid vs EM
        self.use_potentials = True
        # octant/full
        self.domain = cfg.domain
        # density/count
        self.particle_mode = cfg.particle_mode
        # if density mode
        self.p_density = cfg.p_density
        # if count mode
        self.n_test_particles_per_cell = cfg.n_test_particles_per_cell
        # % volume of inner sphere initial particles occupy
        # NOTE --- Spawning at high-beta leads to instability, a concern to think about
        # NOTE --- hence spawns closer to center of core where beta is lower
        if cfg.plasma_bounding * cfg.L > cfg.b_offset:
            print(f"[WARNING] Plasma bounding spawns particles outside of reactor, re-scaling")
            # this ensures we spawn within the system, with a scaling factor to spawn closer to center
            # if no scaling, spawn at b_offset
            new_plasma_bounding = 0.5 * (cfg.b_offset / cfg.L)
            cfg.plasma_bounding = new_plasma_bounding
            print(f"[new plasma bounding value] {cfg.plasma_bounding}, spawning particles in central radius: {new_plasma_bounding * cfg.L}")
        self.plasma_bounding = cfg.plasma_bounding
        
        self.M = cfg.mass * M_P

        self.Ti_eV = cfg.Ti_eV
        self.Ti_J  = cfg.Ti_eV * EV

        # B field params
        self.b_method = cfg.b_method
        self.b_dia = cfg.b_dia
        self.b_offset = cfg.b_offset 
        self.I = cfg.I
        self.B_coil = cfg.B_coil

        # For n-turn coils
        self.n_turns = cfg.n_turns
        self.r1 = cfg.r1 
        self.r2 = cfg.r2

        print(f"\n{'='*60}")
        if cfg.particle_mode == "density":
            print(f"  density = {cfg.p_density:.1e} m^-3")
        elif cfg.particle_mode == "count":
            print(f"  n_test_particles_per_cell = {cfg.n_test_particles_per_cell}")
        print(f"{'='*60}\n")

        # Electron/E-field parameters
        self.Te_eV = cfg.Te_eV
        self.Te_J  = cfg.Te_eV * EV
        
        # Cube side length
        self.L   = cfg.L        
        # Cells per axis
        self.N   = cfg.N
        # Cell size in (m)
        self.dx  = cfg.dx
        # Number of particles per cell
        self.n_per_cell_each_dim = cfg.n_per_cell_each_dim

        print(f"[grid] Values for symmetry: {self.cfg.symmetry}")
        if self.cfg.symmetry == "full":
            print(f"[grid]  {cfg.N}^3 = {cfg.N**3:,} cells,  dx = {cfg.dx*1000:.3f} mm")
        else:
            print(f"[grid]  {cfg.N/2}^3 = {(cfg.N/2)**3:,} cells,  dx = {cfg.dx*1000:.0f} mm")
        
    def get_plasma_quantities(self):

        print("\n[get_plasma_quantities] **********************************")
        print(f"Working with a B-Coil Center: {self.B_coil}")

        # Thermal speed, need dt < dx / v_ion
        # v_most_probably = v_rms / sqrt(2)
        self.v_ion     = np.sqrt(self.Ti_J / self.M)          # thermal speed, m/s, kinetic energy formula

        # Ion plasma frequency
        self.w_pi  = np.sqrt(self.p_density * E_C**2 / (EPS0 * self.M))  # ion plasma freq, rad/s
        
        # Cyclotron angular frequency (rad/s) and period (s)
        self.w_ci  = E_C * self.B_coil / self.M      
        self.T_ci      = 2 * np.pi / self.w_ci   

        # Ion skin depth, inertial length
        self.l_i = (C / self.w_pi)
        print(f"[plasma] ion skin depth ~ {self.l_i*1000:.3f} mm")
        assert self.dx < self.l_i, f"Ion skin depth is not resolved"

        #ion gyroradius
        self.ri = self.v_ion / self.w_ci
        print(f"[plasma] gyroradius ~ {self.ri*1000:.1f} mm ({self.ri / self.dx:.1f} cells per gyroradius)")
        assert self.dx < self.ri, f"dx > gyroradius, {self.dx} > {self.ri}, will not resolve ion cyclotron scale"
        
        # When using the EM solver, these need to be resolved
        # Electron thermal speed (most probable)
        self.v_e = np.sqrt(self.Te_J / M_E)
        # Electron cyclotron frequency (rad/s)
        self.w_ce = E_C * self.B_coil / M_E   
        # Electron plasma frequency... TBD in future iterations
        # calculate plasma density based on electron plasma frequency
        self.w_pe = np.sqrt( ( self.p_density * E_C**2 ) / ( M_E * EPS0 ) )
        self.ratio_pe_ce = self.w_pe / self.w_ce
        assert self.ratio_pe_ce > 1, "Plasma is highly magnetized at the boundary, no diamagnetic effects will occur"
        # debye length
        self.debye = np.sqrt(EPS0 * self.Ti_J / (self.p_density * E_C**2))
        # one mirror bounce time, s, approximation of linear trajectory
        self.t_bounce  = self.b_offset / self.v_ion                         
        # Alfven speed, speed at which magnetic tension waves propagate through a magnetized plasma
        # (m/s)
        # upper limit as null zone would make this approach 0
        self.v_A      = self.B_coil / np.sqrt(MU0 * self.p_density * (M_E + self.M))  # Alfven speed
        
        # Diagnostic print: characterize the plasma regime
        print(f"[plasma]  v_ion    = {self.v_ion/1e3:.2f} km/s, {self.v_ion / C:.3f}% speed of light")
        print(f"[plasma]  v_Alfven = {self.v_A/1e3:.2f} km/s ")
        print(f"[plasma]  omega_pi = {self.w_pi:.3e} rad/s  (not used in hybrid dt)")
        print(f"[plasma]  omega_ci = {self.w_ci:.3e} rad/s,  T_ci = {self.T_ci*1e9:.3f} ns")
        print(f"[plasma]  omega_ce = {self.w_ce:.3e} rad/s")
        print(f"[plasma]  omega_pe/omega_ce = {self.ratio_pe_ce:.3f}  ")
        print(f"[plasma]  t_bounce = {self.t_bounce*1e9:.1f} ns")

    def get_sim_length(self):

        print("\n[get_sim_length] **********************************")
        self.dt_Alfven  = 0.5 * self.dx / self.v_A      # Alfven CFL: MHD wave can't cross a cell per step
        self.dt_cyclo   = self.T_ci / 100                # cyclotron: 100 steps per gyration minimum
        self.dt_thermal = self.dx / self.v_ion
        print(self.dt_Alfven, self.dt_cyclo, self.dt_thermal)
        self.dt_whistler = 0.5 * self.dx / self.v_e
        self.dt = self.dt_cyclo
        # need substeps to account for dt_cyclo / (2*substeps) < dt_Alfven
        # substeps >= dt_cyclo / dt_Alfven / 2
        substeps_min = int(np.ceil(self.dt / (2 * self.dt_Alfven)))
        substeps_min = int(np.ceil(max(self.dt / self.dt_Alfven, self.dt / self.dt_whistler)))
        self.substeps = max(substeps_min, 20)
        print(f"[substeps] derived substeps = {self.substeps} = max(derived, 20)"
                f"(dt_Bfield = {self.dt/(self.substeps):.3e} s, "
                f"dt_Alfven = {self.dt_Alfven:.3e} s, "
                f"dt_whistler = {self.dt_whistler:.3e} s)")

        assert self.dt / (self.substeps) < self.dt_whistler, "Whistler CFL violated in substeps"

        self.binding = "dt_cyclo and dt_Alfven"
        
        # print(f"\n[timestep]  dt_ion_CFL    = {self.dt_icfl:.3e} s")
        print(f"[timestep]  dt_Alfven_CFL = {self.dt_Alfven:.3e} s")
        print(f"[timestep]  dt_cyclotron  = {self.dt_cyclo:.3e} s")
        print(f"[timestep]  --> const_dt  = {self.dt:.3e} s  (binding: {self.binding})")

        steps_10bounce = int(10 * self.t_bounce / self.dt)
        steps_10ci     = int(10 * self.T_ci     / self.dt)
        self.max_steps      = max(steps_10bounce, steps_10ci)

        print(f"\n[steps]  steps per bounce    = {int(self.t_bounce/self.dt):,}")
        print(f"[steps]  steps per T_ci      = {int(self.T_ci/self.dt):,}")
        print(f"[steps]  steps_10bounce      = {steps_10bounce:,}")
        print(f"[steps]  steps_10ci          = {steps_10ci:,}")
        print(f"[steps]  max_steps (binding) = {self.max_steps:,}  "
            f"({'bounce' if steps_10bounce > steps_10ci else 'cyclotron'})")
        
        print(f"[total runtime] {(self.dt * self.max_steps) / 1e-9:.6f} ns")
            
        # UPDATE SIMULATION PARAMETERS
        self.sim.time_step_size = self.dt
        self.sim.max_steps = self.max_steps
        
    def set_grid(self):
        # Computer started yelling at me but neumann is equivalent to pmc by picmi docs
        # https://warpx.readthedocs.io/en/latest/usage/parameters.html#boundary.field_lo-hi
        
        # NOTE, hybrid is most stable in pmc/neumann constraints
        # Hence, PMC/Neumann is used for field boundaries
        # This necessitates a large enough L
        # TODO - look into embedded boundaries around system that might better capture boundary conditions
        grid = picmi.Cartesian3DGrid(
            number_of_cells=list(self.domain.n_cells),
            lower_bound=list(self.domain.lower),
            upper_bound=list(self.domain.upper),
            # if using pmc and pmc is allowed, okay, else use neumann (pmc <=> neumann)
            lower_boundary_conditions=list(self.domain.field_bc_lo),
            upper_boundary_conditions=list(self.domain.field_bc_hi),
            lower_boundary_conditions_particles=list(self.domain.particle_bc_lo),
            upper_boundary_conditions_particles=list(self.domain.particle_bc_hi),
            warpx_max_grid_size=self.N // 2,
        )
        print()
        print("[set_grid] **********************************")
        print("[grid] Full/Octant: ", self.domain.symmetry)
        print("[grid] lower_bound domain: ", grid.lower_bound)
        print("[grid] upper bound domain: ", grid.upper_bound)
        print("[grid] lower boundary conditions fields: ", grid.lower_boundary_conditions)
        print("[grid] upper boundary conditions fields: ", grid.upper_boundary_conditions)
        print("[grid] lower boundary conditions particles: ", grid.lower_boundary_conditions_particles)
        print("[grid] upper boundary conditions particles: ", grid.upper_boundary_conditions_particles)
        self.grid = grid
        
    def set_b_field(self):
        print("[set_b_field] **********************************")
        print("[b-field] Initializing analytic solution to vector potential")
        # We need it to be an Vector Potential Field (A External)
        if self.r1 == self.r2:
            print("[b-field] Using single turn ring coils")
            A_external = build_aext_expressions(self.I, self.b_dia, self.b_offset)
        else:
            print(f"[b-field] Using {self.n_turns}-turn discs")
            A_external = build_n_turn_aext_expression(self.I, self.b_offset, self.r1, self.r1, self.n_turns)
        self.A_external = A_external

    def set_embedded_coils(self):
        print('[stl] Adding coils from stl file')
        coil_structure = {
            'r1': self.r1,
            'r2': self.r2,
            'n_turns': self.n_turns
        }
        path = make_coil_stl(coil_structure, self.b_offset, self.L, self.N, full=self.domain.symmetry == "full")
        coil_ebs = picmi.EmbeddedBoundary(
            stl_file=path,
            # stl_scale=1.0,
            # stl_center=[0, 0, 0],
            # stl_reverse_normal=False,
        )
        #self.sim.warpx_embedded_boundary = coil_ebs
        print('[stl] Coils added as embedded boundaries from STL file')
        print('[stl] Boundary conditions default to PEC for fields and absorbing for particles')
        return coil_ebs

    def set_solver(self):

        """
        Notes:

        1) Lower Te_eV compared to Ti_eV
            - Allows electrons to heat up overtime as the system stabilizes
            - Could allow for stabilizing effect on the overall simulation
            - Let heating occur self-consistently through current dissipation/compression
            - Electron pressure in Ohm's scales with Te
                Lower Te = less initial pressure-driven current
        
        2) Gamma in Isothermal/Adiabatic 
            - Gamma is used in calculation of electron pressure
            - n0 * Te0 * (n_e / n0)**(gamma)
            - Controls intensity of pressure changes from electron density
            - Gamma = 1 -> isothermal limit
                - Electron pressure less sensitive to electron density
            - Gamma = 5/3 -> adiabatic limit
                - Electron pressure more sensitive to electron density
                - Compression produces more pressure (density increases)

        3) Holmstrom vacuum 
            - Allows for lower n_floor
            - Uses a magnetic diffusion equation when density drops below threshold
                - threshold = n_floor
            - Vacuum regions as infinite resistivity
                - Faraday's law -> magnetic diffusion equation
            - This solves the diffusion equation in low charge density regions
                - if n < n_floor → use magnetic diffusion (vacuum treatment)
                - if n > n_floor → use full Ohm's law (plasma treatment)

        4) Hyperresistivity
            - diffusion operator acting on current density — similar to how viscosity acts on velocity in fluid mechanics.
            - Discretizing Faraday's law has no numerical diffusivity (poor approximation)
            - Smoothing is needed, used in E-field computation
            - Maron (2008) introduces hypers of different orders for a MHD solver
            - Here, hr tacked onto -hr*div2(J) term 
            - Higher order derivatives of Laplacian diffuse high freqs, preserve low frewqs
            - Can increase max stable time step
            - Discretized with standard second order finite difference stencils
            - hyper-resistivity is present in all regions including vacuum
        """

        # solve for vacuum resistivity
        # t_diff = mu0 * L**2 / eta
        # eta = mu0 * L**2 / t_diff
        self.eta_plasma = 3e-8
        n_floor = 0.1 * self.p_density
        print("[set_solver] **********************************")
        solver = picmi.HybridPICSolver(
            grid=self.grid,
            Te= self.Te_eV,                      # electron temperature, eV
            n0=self.p_density,                  # reference density, m^-3
            gamma=1,                            # isothermal limit = 1, adiabatic limit = 5/3
            plasma_resistivity=self.eta_plasma,            # ~collisionless — increase to ~1e-6 if unstable
            plasma_hyper_resistivity=1e-8,    # damping of the current term's second order derivative
            n_floor=n_floor,             # density floor for numerical stability,
            # NEED SUBSTEPS
            substeps=self.substeps,
            # use_rkf45 = True,
            # substep_rtol = 1e-6,
            # substep_atol = 1e-10,
            # substep_safety = 0.5,
            # substep_max_growth = 2.0,
            # max_substep_attempts = 2000, 
            A_external = self.A_external,
            holmstrom_vacuum_region=True,
            do_external_diva_cleaning=False,
            warpx_verbose=True
        )
        self.sim.particle_shape = 1             # recommended by docs for hybrid, "linear"
        self.sim.current_deposition_algo = "direct"
        print(f"[solver] hybrid solver set with:")
        print(f"[solver] substeps: {self.substeps}")
        print(f"[solver] Te_eV = {self.Te_eV}")
        print(f"[solver] n0 (reference density for quasi-neutrality ne ~ np = {self.p_density})")
        print(f"[solver] NOTE: n_floor is density floor for numerical stability, higher values may be unphysical, this may need to decrease.")
        self.solver = solver
        self.sim.solver = solver

    def add_species(self):
        # TODO: inititalize_self_field may be of importance here in add_species
        # initialize_self_field (bool, optional) – 
        #   Whether the initial space-charge fields of this species is calculated and added to the simulation
        """
        As of now there is no electric particle inclusion, even in the test case. 

        In the hybrid PIC we don't have to worry about this either.
        """
        print("[add_species] **********************************")
        print("[species] Adding species to simulation")
        self.species_added = []

        # ALLOWS FOR COUNT VS DENSITY INPUTS
        # picmi.GriddedLayout
        layout = make_layout(
            mode=self.particle_mode, 
            grid=self.grid,
            n_macroparticle_per_cell=self.n_per_cell_each_dim,
            n_test_particles_per_cell=self.n_test_particles_per_cell
        )

        self.layout = layout

        print(f"[species] Made species layout using {self.particle_mode}")

        plasma_bounds_lo, plasma_bounds_hi = plasma_bounds(self.domain, self.plasma_bounding)
        vi_rms = np.sqrt(self.Ti_J / self.M)
        plasma_radius = self.plasma_bounding * self.L
        density_expr = f"if(x*x+y*y+z*z<{plasma_radius * plasma_radius}, {self.p_density}, 0.)"
        print(f"[species] Using density expression for analytic distribution: {density_expr}")

        # Ions: isotropic Maxwellian at Ti (cooler; will be accelerated by potential well)
        ion_dist = picmi.AnalyticDistribution(
            density_expression=density_expr,
            lower_bound=plasma_bounds_lo,
            upper_bound=plasma_bounds_hi,
            fill_in=True,
            rms_velocity=[vi_rms, vi_rms, vi_rms],
        )

        plasma_i = picmi.Species(
                particle_type='proton',
                name='plasma_i',
                initial_distribution=ion_dist,
                warpx_save_particles_at_eb = True if hasattr(self.sim, 'warpx_embedded_boundary') else False,
                warpx_save_particles_at_xlo = True,
                warpx_save_particles_at_xhi = True,
                warpx_save_particles_at_ylo = True,
                warpx_save_particles_at_yhi = True,
                warpx_save_particles_at_zlo = True,
                warpx_save_particles_at_zhi = True
            )

        self.plasma_i = plasma_i
        self.sim.add_species(
            plasma_i,
            layout=layout,
        )

        self.species_added.append(plasma_i)

        print(f"[species] Added initial plasma to the simulation")

    def zero_coil_current(self):
        """
        Magnetic diffusivity is directly related to resistivity

        diff_m = eta / mu0


        """

        print("ADDING COIL CURRENT-DAMPING CALLBACK")

        # 3D translation of coil_2d.py's `damp_current_at_coils`: emulate
        # high-resistivity "conductor island" coils by smoothly damping the
        # deposited plasma current toward zero at each coil ring. See
        # docs/simulation/hybrid_resistivity.md for why plasma_resistivity
        # cannot be made position-dependent.
        r_coil = self.b_dia / 2
        w = self.dx  # Gaussian island width ~ one cell

        # eta_plasma MUST track the solver's plasma_resistivity (see set_solver);
        # the wire cells are suppressed by the eta_coil/eta_plasma contrast.
        eta_coil = 1e3
        contrast = eta_coil / self.eta_plasma

        coil_positions = [('x', -self.b_offset), ('x', self.b_offset),
                          ('y', -self.b_offset), ('y', self.b_offset),
                          ('z', -self.b_offset), ('z', self.b_offset)]

        def ring_distance2(X, Y, Z, axis, pos):
            # squared distance from each grid point to the coil ring wire
            # (radius r_coil, centered at `pos` along `axis`)
            if axis == 'x':
                axial = X - pos
                radial = np.sqrt(Y**2 + Z**2)
            elif axis == 'y':
                axial = Y - pos
                radial = np.sqrt(X**2 + Z**2)
            elif axis == 'z':
                axial = Z - pos
                radial = np.sqrt(X**2 + Y**2)
            return axial**2 + (radial - r_coil)**2

        def build_factor(mf):
            # Sum of ring Gaussians -> smooth damping factor.
            # At a coil g~1 => factor~eta_plasma/eta_coil~0; in bulk g~0 => factor~1.
            xs = mf.mesh("x")
            ys = mf.mesh("y")
            zs = mf.mesh("z")
            X = xs[:, None, None]
            Y = ys[None, :, None]
            Z = zs[None, None, :]
            g = np.zeros((xs.size, ys.size, zs.size))
            for axis, pos in coil_positions:
                g = g + np.exp(-ring_distance2(X, Y, Z, axis, pos) / (w * w))
            return 1.0 / (1.0 + contrast * g)

        # Per-direction cache: the mask is grid-static, so build it once.
        # Each component's mesh carries its own centering (ghost cells differ
        # per axis), hence a separate factor per direction.
        _factor_cache = {}

        def _damp_J():
            try:
                Direction = self.sim.extension.libwarpx_so.Direction
                for idir in (0, 1, 2):
                    mf = self.sim.fields.get("current_fp", dir=Direction(idir), level=0)
                    factor = _factor_cache.get(idir)
                    if factor is None:
                        factor = build_factor(mf)
                        _factor_cache[idir] = factor
                        print(f"[coil J] dir {idir}: factor.min = {factor.min():.3e} "
                              f"(cached)")
                    arr = mf[...]
                    fac = factor.reshape(arr.shape[:3] + (1,) * (arr.ndim - 3))
                    mf[...] = arr * fac
            except Exception as e:
                print(f"[_damp_J] ERROR: {e}")
                raise

        # The hybrid solver does NOT trigger the 'afterdeposition' hook (unlike
        # the EM loop the 2D coil deck assumed); 'beforeEsolve' is the per-step
        # hook that fires after the ion current is deposited but before
        # HybridPICEvolveFields consumes current_fp.
        callbacks.installcallback('beforeEsolve', _damp_J)

        assert callbacks.isinstalled('beforeEsolve', _damp_J), "NEVER INSTALLED CALLBACK"

    def add_diagnostics(self):
        # NOTE: Unused and diagnostics are set for every step
        self.diag_period = max(1, self.max_steps // 200)

        # Field diagnostic: B-field components + electron fluid quantities.
        # 'rho' is the ion charge density (from particle deposition).
        # 'Ez', 'Ex', 'Ey' show the self-consistent ambipolar E-field — this is
        # Watch for E pointing inward (toward x=y=z=0) — that's the ambipolar
        # field trying to confine ions.
        field_diag = picmi.FieldDiagnostic(
            name="field_diag",
            grid=self.grid,
            period=self.diag_period,
            data_list=["Bx", "By", "Bz",
                    "Ex", "Ey", "Ez",
                    "Bx_fp_external", "By_fp_external", "Bz_fp_external",
                    "Ex_fp_external", "Ey_fp_external", "Ez_fp_external",
                    "Jx", "Jy", "Jz", "part_per_cell", "rho",
                    "eb_covered"],
            warpx_format='openpmd',
            warpx_openpmd_backend='h5',
        )

        field_diag.diag_type = "Full"
        field_diag.fields_to_plot = ['Bx', 'By', "Bz"]

        # Particle diagnostic: ion positions and momenta.
        # Same format as Stage 1, used by analyze_stage2.py.
        # We output fewer fields than Stage 1 to keep file sizes down at high max_steps.
        part_diag = picmi.ParticleDiagnostic(
            name="part_diag",
            period=self.diag_period,
            species=[self.plasma_i],
            data_list=["x", "y", "z", "ux", "uy", "uz", "weighting"],
            warpx_format='openpmd',
            warpx_openpmd_backend='h5',
        )

        # Boundary scraping diagnostics
        scraping_diag = picmi.ParticleBoundaryScrapingDiagnostic(
            name="boundary_scraping",
            period=10,
            species=[self.plasma_i],
            data_list=["x", "y", "z", "ux", "uy", "uz", "weighting"],
            warpx_format='openpmd',
            warpx_openpmd_backend='h5',
        )

        # Reduced diagnostic: particle count every step.
        # This is the primary output for confinement time measurement.
        # The decay curve N(t) -> exponential fit -> tau_sim.
        # We compare tau_sim across densities to extract the scaling exponent.
        # Output every step (cheap — just a scalar sum) for maximum time resolution.
        particle_num_reduced_diag = picmi.ReducedDiagnostic(
            diag_type="ParticleNumber",
            name="particle_count",
            period=1,
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

        self.sim.add_diagnostic(field_diag)
        self.sim.add_diagnostic(part_diag)
        self.sim.add_diagnostic(scraping_diag)
        self.sim.add_diagnostic(field_energy_reduced_diag)
        self.sim.add_diagnostic(particle_energy_reduced_diag)

    def cusp_loss_count(self, pc):

        injection_sphere_count = 0.0
        injection_radius = self.plasma_bounding * self.L

        for pti in pc.iterator(level=0):
            x  = np.array(pti['x'], copy=False)
            y  = np.array(pti['y'], copy=False)
            z  = np.array(pti['z'], copy=False)

            # Tried doing this as a loop outside but doesn't work, for some reason works here
            # if we want that blast to endure, we can resort to keeping the initial density at every timestep
            in_sphere = (x**2 + y**2 + z**2 < injection_radius**2)
            cur_injection_sphere_count = np.sum(in_sphere)
            injection_sphere_count += cur_injection_sphere_count

        print("MADE IT TO INJECTION SPHERE COUNT: ", injection_sphere_count)

        total = 0.0
        per_face = []
        
        faces = [
            (0, +1), (0, -1),
            (1, +1), (1, -1),
            (2, +1), (2, -1),
        ]
        
        for ax, sign in faces:
            # get the coordinates for this slab slice
            ax1, ax2 = (ax+1)%3, (ax+2)%3
            # slab of width about sign * b_offset
            face_center = sign * self.b_offset
            loss = 0.0
            
            for pti in pc.iterator(level=0):
                x  = np.array(pti['x'], copy=False)
                y  = np.array(pti['y'], copy=False)
                z  = np.array(pti['z'], copy=False)
                ux = np.array(pti['ux'], copy=False)
                uy = np.array(pti['uy'], copy=False)
                uz = np.array(pti['uz'], copy=False)
                
                coords = [x, y, z]
                vels   = [ux, uy, uz]   
                
                # width of slab on axis, dx / 2 since it's width is dx
                in_slab  = np.abs(coords[ax] - face_center) < self.dx / 2
                # slice on perpendicular axes
                # basically 
                in_face  = (np.abs(coords[ax1]) < self.b_dia / 2) & \
                        (np.abs(coords[ax2]) < self.b_dia / 2)
                in_face = (coords[ax1]**2 + coords[ax2]**2 < (self.b_dia / 2)**2 )
                # is it leaving?
                outgoing = sign * vels[ax] > 0
                # ensure we only capture particles that will be leaving the slab (no counting twice)
                # if they're normal velocity and position imply they will leave after this time step
                # calculate x value after time step using current normal velocity
                next_pos = coords[ax] + vels[ax] * self.dt
                # scale by sign so a single comparison operator works
                # |next_pos| > |slab_end|
                will_exit = sign * next_pos > sign * (face_center + sign * self.dx / 2)
                mask = in_slab & in_face & outgoing & will_exit
                loss += np.sum(mask)
            
            per_face.append(loss)
            total += loss

        print("MADE IT TO CUSP LOSS: ", total)
        
        return int(round(total)), per_face, injection_sphere_count

    def add_injection_callback(self):
        # TODO: Make injection outweigh loss, that initial burst effect should remain as a continuous effect, have N_inject increase up to max loss and remain as such. 
        # TODO: Make injection an increasing trend
        inject_radius = self.plasma_bounding * self.L
        v_rms = np.sqrt(self.Ti_J / self.M)
        particle_container = self.sim.particles.get("plasma_i")
        _loss_log = {
            "times": [], 
            "per_face": [], 
            "from_injection_volume": [],
            "peak_cusp_loss": 0
            }  # closure state
        df = particle_container.to_df(local=True)
        weight = df['w'].iloc[0]
        N_t0 = len(df['w'])
        print(f"INITIAL PARTICLE COUNT: {N_t0:3e}")
        def inject_particles():
            total_loss, per_face, injection_sphere_count = self.cusp_loss_count(particle_container)
            # Have to divide by 6 since we count these six times, it's a waste but random bug occurred when doing it "properly"
            injection_sphere_count = injection_sphere_count // 6
            print("INJECTION SPHERE COUNT: ", injection_sphere_count)
            injection_sphere_loss = max(N_t0 - injection_sphere_count, 0)
            _loss_log['from_injection_volume'].append(injection_sphere_loss)
            # Retain that density lost to the burst, 
            N_inject = int(max(_loss_log['peak_cusp_loss'], total_loss, injection_sphere_loss))
            # Keep track of losses through cusps
            _loss_log['peak_cusp_loss'] = max(_loss_log['peak_cusp_loss'], total_loss)

            print(f"[injection callback] Total face-cusp loss: {total_loss}\n"
                  f"[injection callback] Loss from initial spawn sphere: {injection_sphere_loss}\n"
                  f"[injection callback] Injection count: {N_inject}\n"
                  )

            _loss_log['times'].append(self.sim.extension.warpx.gett_new(0))
            _loss_log['per_face'].append(per_face)
            # simply return if nothing to add back in
            if N_inject == 0:
                return
            w = np.ones(N_inject) * weight
            print("CREATED WEIGHT ARRAY")
            # uniform sphere sampling
            r = inject_radius * np.cbrt(np.random.uniform(0, 1, N_inject))
            theta = np.arccos(np.random.uniform(-1, 1, N_inject))
            phi = np.random.uniform(0, 2 * np.pi, N_inject)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            # This is passed to picmi 
            ux = np.random.normal(0, v_rms, N_inject)
            uy = np.random.normal(0, v_rms, N_inject)
            uz = np.random.normal(0, v_rms, N_inject)

            particle_container.add_particles(
                x=x, y=y, z=z,
                ux=ux, uy=uy, uz=uz,
                w=w,
                unique_particles=False,
            )

        def save_cusp_losses():
            np.savez("diags/cusp_flux.npz",
                    times=np.array(_loss_log["times"]),
                    losses=np.array(_loss_log["per_face"]),
                    injected=_loss_log['peak_cusp_loss'],
                    from_injection_volume=_loss_log['from_injection_volume'],
                    face_labels=["x_hi","x_lo","y_hi","y_lo","z_hi","z_lo"])

        callbacks.installcallback('afterstep',inject_particles)
        callbacks.installcallback('afterdiagnostics', save_cusp_losses)

        print("[injection callback] ADDED CALLBACK TO INJECT PARTICLES")

    def define_run_management(self):

        script_name = "polwell_hybrid.py"
        run_dir = new_run_dir()
        try:
            shutil.copy2(__file__, run_dir / script_name)
        except Exception:
            pass

        # save config as json file for easily referencing it
        self.save_cfg_to_json(self.cfg, run_dir)
        run_params = {
            # simulation control
            "max_steps":            self.max_steps,
            "const_dt":             self.dt,
            # plasma
            "Ti_eV":                self.Ti_eV,
            "Te_eV":                self.Te_eV,
            "p_density":            self.p_density,
            "plasma_bounding":      self.plasma_bounding,
            # B-field
            #"B_coil":               B_coil,
            "b_method":             self.b_method,
            "coil_current":         self.I,
            "b_dia":                self.b_dia,
            "b_offset":             self.b_offset,
            # grid
            "grid_L":               self.L,
            "grid_N":               self.N,
            "particles_per_cell":   self.n_per_cell_each_dim,
            "solver_type":          type(self.solver).__name__,
            "solver_method":        getattr(self.solver, "method", None),
            # diagnostics
            "diag_period":           self.diag_period,
            "diag_path":             str(run_dir),
        }

        self.run_dir = run_dir
        self.run_params = run_params
        os.chdir(self.run_dir)
        self.sim.write_input_file("inputs_test")

    def run(self):
        db = RunsDB()
        _prev = os.getcwd()
        try:
            with db.run_context(self.run_dir, self.run_params) as run_id:
                print(f"\n[run] id={run_id}  dir={self.run_dir}")
                print(f"[run] density={self.p_density:.1e}"
                    f"dt={self.dt:.2e}s  steps={self.max_steps:,}")
                print(f"[run] simulated time = {self.max_steps*self.dt*1e9:.1f} ns  "
                    f"(~{self.max_steps*self.dt/self.t_bounce:.1f} bounces, "
                    f"~{self.max_steps*self.dt/self.T_ci:.0f} cyclotron periods)")
                os.chdir(self.run_dir)
                try:
                    self.sim.step()
                finally:
                    os.chdir(_prev)
        finally:
            db.close()

# python3.10 -u -m inputs.polywell_sim --test >> ./output/hybrid_run_logs/"run_log_$(date +%s).txt" 2>&1

polywell = PollywellSixCoilHybrid(cfg)
polywell.run()