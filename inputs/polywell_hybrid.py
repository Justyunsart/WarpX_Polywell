"""
This class handles specifically the Hybrid case and is currently only tested in the full cube symmetry.

It requires an instantiation of a PolywellHybridConfig or a json formatted file that you can import.

There already exists two configs in configs/polywell_hybrid_config.py

HYBRID_CONFIG 
- This is a single turn HybridPICSolver config

HYBRID_10_TURN_CONFIG
- This is a 10-turn config with
    - r1 = 0.02
    - r2 = 0.09

Note that the bounds are quite small, this is for testing stability in a quicker manner, and this 
implementation could benefit from a larger scale system. 
"""

"""
Ran into issues:

1) malloc errors (Maybe a mac issue?)
2) Non-finite E-field

Main issue:

There is a clear particle effect on the magnetic field, and it's intense

This rapid shift in field structure creates massive instability in the solver

Particles are reshaping the B-field at that core and the solver can't keep up

Main solution:

Spawn location at/near coil boundary provides immediate extreme conditions (high-beta regime is highly unstable)

Hence spawning them closer to the center seems to allows the solver to smooths things out a lot better

Tried:

1) reduce timesteps
effect: allows running longer, but crash happens at same B-field chaos
- Helped confirm it's happening at a specific point

2) Increase substeps
effect: Doesn't seem to change, but haven't tried 1000+, which would be infeasible unless I run all night
- If it even works...

3) Increase n_floor:
effect: allows to run longer, but becomes less realistic as there definitely are vacuums...

4) RKF45 adapative error correction
- I didn't see much change using this

5) Plasma bounding decrease
- This helps the sim run longer, though it doesn't begin in that high beta regime...

Possible solutions to look into:
- Some kind of smoothing for charge density
- Some kind of smoothing for particle clumping
- 


Deeper center spawning helps immensely
"""

import os
import shutil

# Pywarpx utils, plus callbacks which can be useful for diagnostic (live saving of fields for example)
from pywarpx import picmi, warpx, particles, fields, amrex
from pywarpx.callbacks import installcallback

# Custom code for loading B-field, E-field, and saving them in a database
from src.bext.bext import setup_bext, make_bext_file
from src.eext.eext import fill_eext_file # should run AFTER B-field init.
from src.eext.methods import EMethods # enum registry for available methods
from src.db.runs import RunsDB, new_run_dir

from configs.polywell_hybrid_config import PolywellHybridConfig, HYBRID_CONFIG
from src.domain import plasma_bounds
from src.spawn import make_layout

import numpy as np
import scipy.constants as sc

import argparse
from pathlib import Path

from src.bext.analytic import build_aext_expressions, build_n_turn_aext_expression

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
simulation = picmi.Simulation(verbose=True)

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

class PollywellSixCoilHybrid: 
    def __init__(self, cfg: PolywellHybridConfig | str, sim=None, test=True):
        """
        cfg: PolywellHybridConfig | Path to json-formatted polywell config
        """
        self.sim = sim or picmi.Simulation(verbose=True)
        self.store_config_params(cfg)
        self.get_plasma_quantities()
        self.get_sim_length()
        self.set_grid()
        self.set_b_field()
        self.set_solver() 
        self.add_species()
        self.add_diagnostics()
        self.define_run_management()
        simulation.initialize_inputs()
        simulation.initialize_warpx()

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
            new_plasma_bounding = 5/10 * (cfg.b_offset / cfg.L)
            cfg.plasma_bounding = new_plasma_bounding
            print(f"[new plasma bounding value] {cfg.plasma_bounding}")
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

        # Plasma resistivity - used to dampen the mode excitation
        # Needed for HybridPICSolver
        self.eta = 1e-7
        
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

        self.dt_Alfven  = 0.5 * self.dx / self.v_A      # Alfven CFL: MHD wave can't cross a cell per step
        self.dt_cyclo   = self.T_ci / 100                # cyclotron: 100 steps per gyration minimum

    def get_sim_length(self):

        print("\n[get_sim_length] **********************************")
        self.dt = self.dt_cyclo
        
        # need substeps to account for dt_cyclo / (2*substeps) < dt_Alfven
        # substeps >= dt_cyclo / dt_Alfven / 2
        substeps_min = int(np.ceil(self.dt / (2 * self.dt_Alfven)))
        self.substeps = max(substeps_min, 32)
        print(f"[substeps] derived substeps = {self.substeps} = max(derived, 32)"
                f"(dt_Bfield = {self.dt/(2*self.substeps):.3e} s, "
                f"dt_Alfven = {self.dt_Alfven:.3e} s)")

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

    def set_solver(self):

        print("[set_solver] **********************************")
        solver = picmi.HybridPICSolver(
            grid=self.grid,
            Te=self.Te_eV,                      # electron temperature, eV
            n0=self.p_density,                  # reference density, m^-3
            gamma=1,                            # isothermal electrons = 1, adiabatic = 5/3
            plasma_resistivity=3e-8,            # ~collisionless — increase to ~1e-6 if unstable
            n_floor=0.5 * self.p_density,             # density floor for numerical stability,
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
        self.sim.warpx_grid_type = "collocated" # recommended by docs for Hybrid
        self.sim.particle_shape = 1             # recommended by docs for hybrid
        self.sim.current_deposition_algo = "direct"
        print(f"[solver] hybrid solver set with:")
        print(f"[solver] substeps: {self.substeps}")
        print(f"[solver] Te_eV = {self.Te_eV}")
        print(f"[solver] n0 (reference density for quasi-neutrality ne ~ np = {self.p_density})")
        print(f"[solver] NOTE: n_floor is density floor for numerical stability, higher values may be unphysical, this may need to decrease.")
        self.solver = solver
        self.sim.solver = solver

    def add_species(self):
        """
        As of now there is no electric particle inclusion, even in the test case. 

        In the hybrid PIC we don't have to worry about this either.
        """
        print("[add_species] **********************************")
        print("[species] Adding species to simulation")
        species_added = []

        # ALLOWS FOR COUNT VS DENSITY INPUTS
        # picmi.GriddedLayout
        layout = make_layout(
            mode=self.particle_mode, 
            grid=self.grid,
            n_macroparticle_per_cell=self.n_per_cell_each_dim,
            n_test_particles_per_cell=self.n_test_particles_per_cell
        )

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
            )

        self.plasma_i = plasma_i
        self.sim.add_species(
            plasma_i,
            layout=layout,
        )

        species_added.append("plasma_i")

        print(f"[species] Added {species_added} to the simulation")

    def add_diagnostics(self):
        # NOTE: Unused and diagnostics are set for every step
        diag_period = max(1, self.max_steps // 400)

        # NOTE, diag period = 1 for smoother trajectory mapping, but makes diag file sizes much bigger
        self.diag_period = 1

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
                    "Jx", "Jy", "Jz", "part_per_cell", "rho"],
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

        # Reduced diagnostic: particle count every step.
        # This is the primary output for confinement time measurement.
        # The decay curve N(t) -> exponential fit -> tau_sim.
        # We compare tau_sim across densities to extract the scaling exponent.
        # Output every step (cheap — just a scalar sum) for maximum time resolution.
        # NOTE - this stopped working for some reason, but we can get particle counts in paraview using weights
        reduced_diag = picmi.ReducedDiagnostic(
            diag_type="ParticleNumber",
            name="particle_count",
            period=1,
        )

        self.field_diag = field_diag 
        self.part_diag = part_diag 
        self.reduced_diag = reduced_diag

        self.sim.add_diagnostic(field_diag)
        self.sim.add_diagnostic(part_diag)
        self.sim.add_diagnostic(reduced_diag)

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

polywell = PollywellSixCoilHybrid(cfg, sim=simulation)
polywell.run()