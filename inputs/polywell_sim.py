"""
This is a class version that allows for the test phase (Yoon's code) and more dynamic
customization of the code.

Organized in a way that provides modulation and separation of concerns.
"""

import os
import shutil

# Pywarpx utils, plus callbacks which can be useful for diagnostic (live saving of fields for example)
from pywarpx import picmi, warpx, particles, fields
from pywarpx.callbacks import installcallback

# Custom code for loading B-field, E-field, and saving them in a database
from src.bext.bext import setup_bext, make_bext_file
from src.eext.eext import fill_eext_file # should run AFTER B-field init.
from src.eext.methods import EMethods # enum registry for available methods
from src.db.runs import RunsDB, new_run_dir

from configs.polywell_config import PolywellConfig, TEST_CONFIG, HYBRID_CONFIG
from src.domain import plasma_bounds
from src.spawn import make_layout

import numpy as np
import scipy.constants as sc

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Polywell simulation"
)

parser.add_argument(
    "--test", "-t", action="store_true",
    help="Whether to run initial test simulation by Yoon."
)

args = parser.parse_args()
test = args.test

# print(f"DENSITY: {p_density}")
print(f"TESTING MODE: {test}")

C    = sc.c
E_C  = sc.e
M_P  = sc.m_p
M_E  = sc.m_e
EV   = sc.eV
MU0  = 4 * np.pi * 1e-7
EPS0 = sc.epsilon_0
simulation = picmi.Simulation(verbose=True)

if test:
    cfg = TEST_CONFIG
else:
    cfg = HYBRID_CONFIG

class PollywellSixCoil: 
    def __init__(self, cfg: PolywellConfig, sim=None, test=True):
        self.sim = sim or picmi.Simulation(verbose=True)
        self.store_config_params(cfg)
        self.get_plasma_quantities()
        self.get_sim_length()
        self.set_grid()
        self.set_b_field()
        if self.test:
            self.set_e_field()
        self.set_solver() 
        self.add_species()
        self.add_diagnostics()
        self.define_run_management()

    def store_config_params(self, cfg):
        # save config
        self.cfg = cfg
        # denote testing or not
        self.test = test
        # Hybrid vs EM
        self.solver_type = cfg.solver_type
        # octant/full
        self.domain = cfg.domain
        # density/count
        self.particle_mode = cfg.particle_mode
        # if density mode
        self.p_density = cfg.p_density
        # if count mode
        self.n_test_particles_per_cell = cfg.n_test_particles_per_cell
        # % volume of inner sphere initial particles occupy
        self.plasma_bounding=cfg.plasma_bounding

        self.Ti_eV = cfg.Ti_eV
        self.Ti_J  = cfg.Ti_eV * EV

        # Number of substeps used to update B, as of right now it is hardcoded. Maybe a mathematical expression to calculate this? Alfven waves?
        # This is only relevant for HybridPIC
        self.substeps = 40

        # B field params
        self.b_method = cfg.b_method
        self.b_dia = cfg.b_dia
        self.b_offset = cfg.b_offset 
        self.I = cfg.I
        self.B_coil = cfg.B_coil
        # beta = n * kT * 2 * mu0 / B^2  (using ion temperature as proxy for total pressure)
        self.beta    = cfg.p_density * cfg.Ti_J * 2 * MU0 / cfg.B_coil**2

        print(f"\n{'='*60}")
        if cfg.particle_mode == "density":
            print(f"  density = {cfg.p_density:.1e} m^-3")
        elif cfg.particle_mode == "count":
            print(f"  n_test_particles_per_cell = {cfg.n_test_particles_per_cell}")
        print(f"  beta    = {cfg.beta:.3e}  {'(HIGH-BETA REGIME)' if (np.isclose([cfg.beta], [1.0], 0.1) or cfg.beta > 1.0) else '(low-beta)'}")
        print(f"{'='*60}\n")

        # Electron/E-field parameters
        self.Te_eV = cfg.Te_eV
        self.Te_J  = cfg.Te_eV * EV

        # E-field params
        self.e_method = cfg.e_method
        self.Q = cfg.Q 
        self.e_dia = cfg.e_dia 
        self.e_offset = cfg.e_offset
        
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
        ion_gyroradius = np.sqrt(2*cfg.Ti_J*M_P)/(E_C*cfg.B_coil)
        print(f"[grid]  Ion gyroradius ~ {ion_gyroradius*1000:.1f} mm  "
            f"({(ion_gyroradius)/cfg.dx:.1f} cells per gyroradius)")

        assert cfg.dx < ion_gyroradius, f"FAILED DUE TO DX{cfg.dx} > ION GYRORADIUS {ion_gyroradius}"
        
        # Plasma resistivity - used to dampen the mode excitation
        # Needed for HybridPICSolver
        self.eta = 1e-7
        
    def get_plasma_quantities(self):

        # Thermal speed, need dt < dx / v_ion
        self.v_ion     = np.sqrt(2 * self.Ti_J / M_P)          # thermal speed, m/s, kinetic energy formula

        # Ion plasma frequency from charge displacement, relevant for EM Solver
        # Account for resolving many plasma oscillations (scale by 1/num oscillation)
        self.w_pi  = np.sqrt(self.p_density * E_C**2 / (EPS0 * M_P))  # ion plasma freq, rad/s
        
        # Cyclotron angular frequency (rad/s) and period (s)
        self.w_ci  = E_C * self.B_coil / M_P         
        self.T_ci      = 2 * np.pi / self.w_ci      
        
        # Electron cyclotron frequency (rad/s)
        self.w_ce = E_C * self.B_coil / M_E   
        
        # Electron plasma frequency... TBD in future iterations
        # calculate plasma density based on electron plasma frequency
        self.w_pe = np.sqrt( ( self.p_density * E_C**2 ) / ( M_E * EPS0 ) )

        self.ratio_pe_ce = self.w_pe / self.w_ce

        # Ion skin depth
        self.l_i = C / self.w_pi

        # one mirror bounce time, s, approximation of linear trajectory
        self.t_bounce  = self.L / self.v_ion                         
        # Alfven speed, speed at which magnetic tension waves propagate through a magnetized plasma
        # (m/s)
        self.v_A      = self.B_coil / np.sqrt(MU0 * self.p_density * M_P)  # Alfven speed
        
        # Diagnostic print: characterize the plasma regime
        print(f"\n[plasma]  v_ion    = {self.v_ion/1e6:.2f} km/s")
        print(f"[plasma]  v_Alfven = {self.v_A/1e6:.2f} km/s  "
            f"({'Alfven > ion: Alfven CFL will bind' if self.v_A > self.v_ion else 'ion > Alfven: ion CFL will bind'})")
        print(f"[plasma]  omega_pi = {self.w_pi:.3e} rad/s  (not used in hybrid dt)")
        print(f"[plasma]  omega_ci = {self.w_ci:.3e} rad/s,  T_ci = {self.T_ci*1e9:.3f} ns")
        print(f"[plasma]  omega_ce = {self.w_ce:.3e} rad/s")
        print(f"[plasma]  omega_pe/omega_ce = {self.ratio_pe_ce:.3f}  "
            f"({'WARNING: hybrid validity questionable (need << 1)' if self.ratio_pe_ce > 0.5 else 'OK for hybrid'})")
        print(f"[plasma]  t_bounce = {self.t_bounce*1e9:.1f} ns")

        self.dt_ifreq  = 0.05 / self.w_pi             # ion plasma freq (20 steps/period), The Collective Motion Limit, relevant for non quasistatic (None HybridPIC)
        self.dt_icfl  = 0.5 * self.dx / self.v_ion    # ion CFL: particle can't cross a cell per step
        self.dt_Acfl  = 0.5 * self.dx / self.v_A      # Alfven CFL: MHD wave can't cross a cell per step
        self.dt_ici   = self.T_ci / 10                # cyclotron: 10 steps per gyration minimum

    def get_sim_length(self):

        # omega_pi constraint intentionally excluded — not applicable to hybrid PIC
        self.dt = min(self.dt_icfl, self.dt_Acfl, self.dt_ici)

        self.binding  = ("ion CFL"     if self.dt == self.dt_icfl else
            "Alfvén CFL"  if self.dt == self.dt_Acfl  else
            "ion cyclotron")
        
        print(f"\n[timestep]  dt_ion_CFL    = {self.dt_icfl:.3e} s")
        print(f"[timestep]  dt_Alfven_CFL = {self.dt_Acfl:.3e} s")
        print(f"[timestep]  dt_cyclotron  = {self.dt_ici:.3e} s")
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
            
        if self.test:
            self.dt = 1e-9
            self.max_steps = 1000
            print("********** OVER RIDING ABOVE BINDING FOR TEST MODE *********")
            print(f"[test mode] Using dt = {self.dt}"
                  f"[test mode] Max steps = {self.max_steps}"
                  )
            print("********** OVER RIDING ABOVE BINDING FOR TEST MODE *********")
            
        # UPDATE SIMULATION PARAMETERS
        self.sim.time_step_size = self.dt
        self.sim.max_steps = self.max_steps

        exit()
        
    def set_grid(self):
        print(f"[INITIATING GRID WITH DOMAIN]")
        print(f"SYMMETRY:{self.domain.symmetry}")
        print(f"LOWER:{self.domain.lower}")
        print(f"UPPER:{self.domain.upper}")
        print(f"FIELD BC LO: {self.domain.field_bc_lo}")
        print(f"FIELD BC HI: {self.domain.field_bc_hi}")
        print(f"PARTICLE BC LO: {self.domain.particle_bc_lo}")
        print(f"PARTICLE BC HI: {self.domain.particle_bc_hi}")
        # Computer started yelling at me but neumann is equivalent to pmc by picmi docs
        # https://warpx.readthedocs.io/en/latest/usage/parameters.html#boundary.field_lo-hi
        pmc_needed = self.domain.field_bc_lo[0] == "pmc"
        pmc_okay = "pmc" in picmi.BC_map
        if self.solver_type == "EM":
            grid = picmi.Cartesian3DGrid(
                number_of_cells=list(self.domain.n_cells),
                lower_bound=list(self.domain.lower),
                upper_bound=list(self.domain.upper),
                # if using pmc and pmc is allowed, okay, else use neumann
                lower_boundary_conditions=list(self.domain.field_bc_lo) if (not pmc_needed or pmc_okay) else ('neumann', 'neumann', 'neumann'),
                upper_boundary_conditions=list(self.domain.field_bc_hi),
                lower_boundary_conditions_particles=list(self.domain.particle_bc_lo),
                upper_boundary_conditions_particles=list(self.domain.particle_bc_hi),
                warpx_max_grid_size=32,
            )
            print(f"FINISHED SETTING UP EM SOLVER GRID")
        elif self.solver_type == "hybrid":
            grid = picmi.Cartesian3DGrid(
                number_of_cells=[self.N]*3,
                lower_bound=[-self.L]*3,
                upper_bound=[self.L]*3,
                # Field boundary conditions:
                # I'm unsure how to set these, neumann works, open doesn't seem to work
                # Otherwise it throws a fit
                lower_boundary_conditions=["neumann"]*3,
                upper_boundary_conditions=["neumann"]*3,
                lower_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
                upper_boundary_conditions_particles=["absorbing",  "absorbing",  "absorbing" ],
                warpx_max_grid_size=self.N,   # single box (small grid, no need to decompose)                                                    
            )
        print(grid.lower_boundary_conditions)
        self.grid = grid
        
    def set_b_field(self):
        print(f"CREATING B FIELD WITH SOLVER {self.solver_type} AND METHOD {self.b_method}")
        ext_path = setup_bext(
            method='file',
            particles=particles,
            warpx_module=warpx,
            I=self.I,
            dia=self.b_dia,
            offset=self.b_offset,
            domain=self.domain,
            solver=self.solver_type
        )
        # hybrid needs grid fields, and not fields on particles
        if self.solver_type == "hybrid":
            warpx.B_ext_grid_init_style = "read_from_file"
            warpx.read_fields_from_path = ext_path
        print(ext_path)
        self.ext_path = ext_path
        # Field is div free, and for numerical stability, this is turned off
        warpx.do_initial_div_cleaning = 0

    def set_e_field(self):
        ## Configure E-field (file-based only for now)
        # note: "file" B-field and E-field share the same .h5 file.
        # "analytic" B-field means E needs its own file, OR its own parse setup.
        # Updates file path to read from after appending the E-field to it
        if self.e_method is not None:
            method = EMethods[self.e_method].value[0]
            if self.b_method == "file":
                # B already made the .h5 file — append E-field data to it
                particles.E_ext_particle_init_style = "read_from_file"
                self.ext_path = fill_eext_file(self.ext_path, method,
                                        self.e_dia, self.e_offset,
                                        self.Q, self.domain)
                particles.read_fields_from_path = self.ext_path
            else:
                # Analytic B doesn't make an .h5. E-field needs its own file.
                particles.E_ext_particle_init_style = "read_from_file"
                dummy_ext = make_bext_file(0, self.b_dia, self.b_offset, self.L, self.N)  # zero-current B file as scaffold
                self.ext_path = fill_eext_file(dummy_ext, method,
                                        self.e_dia, self.e_offset,
                                        self.Q, self.domain)
                particles.read_fields_from_path = self.ext_path

    def set_solver(self):

        print(f"SOLVER TYPE BEING SET AS: {self.solver_type}")

        if self.solver_type == "EM":
            solver = picmi.ElectromagneticSolver(
                grid=self.grid,
                method="Yee",   # options: "Yee", "CKC", "psatd"
                cfl=0.99,
            )
            print("FINISHED SETTING UP EM SOLVER")

        elif self.solver_type == "hybrid":
            # TODO TODO ONLY DOES HYBRID AT THE MOMENT
            solver = picmi.HybridPICSolver(
                grid=self.grid,
                Te=self.Te_eV,                      # electron temperature, eV
                n0=self.p_density,                  # reference density, m^-3
                gamma=1,                            # isothermal electrons
                plasma_resistivity=1e-7,            # ~collisionless — increase to ~1e-6 if unstable
                n_floor=1e-4 * self.p_density,      # density floor for numerical stability,
                # NEED SUBSTEPS
                substeps=self.substeps,
                warpx_verbose=True
            )
            self.sim.warpx_grid_type = "collocated" # recommended by docs for Hybrid
            self.sim.particle_shape = 1             # recommended by docs for hybrid
            self.sim.current_deposition_algo = "direct"
        self.solver = solver
        self.sim.solver = solver

    def add_species(self):
        """
        As of now there is no electric particle inclusion, even in the test case. 

        In the hybrid PIC we don't have to worry about this either.
        """

        # ALLOWS FOR COUNT VS DENSITY INPUTS
        # picmi.GriddedLayout
        layout = make_layout(
            mode=self.particle_mode, 
            grid=self.grid,
            n_macroparticle_per_cell=self.n_per_cell_each_dim,
            n_test_particles_per_cell=self.n_test_particles_per_cell
        )

        plasma_bounds_lo, plasma_bounds_hi = plasma_bounds(self.domain, self.plasma_bounding)
        ve_rms = np.sqrt(self.Te_J / M_E)
        vi_rms = np.sqrt(self.Ti_J / M_P)
        plasma_radius = self.plasma_bounding * self.L
        density_expr = f"if(x*x+y*y+z*z<{plasma_radius * plasma_radius}, {self.p_density}, 0.)"

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

        if self.test:

            # NOT USED
            # contains definition for electrons, but are not added to simulation
            # Electrons: isotropic Maxwellian at Te, no net drift
            electron_dist = picmi.AnalyticDistribution(
                density_expression=density_expr,
                lower_bound=plasma_bounds_lo,
                upper_bound=plasma_bounds_hi,
                fill_in=True,
                rms_velocity=[ve_rms, ve_rms, ve_rms],
            )

            # create the plasma species with their respective distributions
            plasma_e = picmi.Species(
                particle_type='electron', 
                name='plasma_e', 
                initial_distribution=electron_dist,
            )

            plasma_e.do_not_deposit = 1 # test particles
            plasma_i.do_not_deposit = 1 # test particles

        # In test, particles don't influence field
        # Ideally they would deposit charge and influence the field
        self.plasma_i = plasma_i
        self.sim.add_species(
            plasma_i,
            layout=layout,
        )

    def add_diagnostics(self):
        diag_period = max(1, self.max_steps // 100)

        if self.test:
            diag_period = 10

        self.diag_period = diag_period

        # Field diagnostic: B-field components + electron fluid quantities.
        # 'rho' is the ion charge density (from particle deposition).
        # 'Ez', 'Ex', 'Ey' show the self-consistent ambipolar E-field — this is
        # Watch for E pointing inward (toward x=y=z=0) — that's the ambipolar
        # field trying to confine ions.
        field_diag = picmi.FieldDiagnostic(
            name="field_diag",
            grid=self.grid,
            period=diag_period,
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
            period=diag_period,
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
        # TODO TODO TODO Can possibly do this to quantify a single side of each cube?
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
        if self.test:
            script_name = "polywell_input_test.py"
        else:
            script_name = "polwell_hybrid_input.py"
        run_dir = new_run_dir()
        try:
            shutil.copy2(__file__, run_dir / script_name)
        except Exception:
            pass

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
                print(f"[run] density={self.p_density:.1e}  beta={self.beta:.2e}  "
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

polywell = PollywellSixCoil(cfg, sim=simulation, test=test)
polywell.run()