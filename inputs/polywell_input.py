import os
import shutil
from pywarpx import picmi, warpx, particles

# PICMI's BC_map has no entry for WarpX's native "pmc" boundary; register a
# passthrough so octant-symmetry planes (tangential B = 0, normal E = 0) work.
picmi.BC_map["pmc"] = "pmc"
from src.bext.bext import setup_bext, make_bext_file
from src.eext.eext import fill_eext_file # should run AFTER B-field init.
from src.eext.methods import EMethods # enum registry for available methods
from src.db.runs import RunsDB, new_run_dir
from src.domain import derive_domain, plasma_bounds, VALID_SYMMETRIES
from src.spawn import make_layout, VALID_PARTICLE_MODES
import numpy as np
import scipy.constants as sc

constants = picmi.constants
#######################
# === USER INPUTS === #
#######################
# SIM #
max_steps = 1000
warpx.const_dt = 1e-9
p_density = 1e18

    # B-Field # (polywell)
b_method = "file" # "analytic" (parser expressions, no grid file) or "file" (magpylib -> HDF5 grid)
I = 1e6 # current of coil, Amperes
b_dia = 1 # diameter of coil, m
b_offset = 0.435 # dist. of coil center from origin, m

    # E-Field #
    # note: if e_method is set to None, all other E-field parameters will be ignored
e_method = None # either None (no E-field) or a name of an entry in EMethods (yes E-field)
Q = 1e-9 # Coulombs
e_dia = 0.75 # m
e_offset = 1.1 # m

    # SOLVER #
    # When True, use the Hybrid-PIC solver. Hybrid is the only WarpX solver
    # that natively consumes external vector potentials, so this is also the
    # only situation where the potentials-based field pipelines are useful;
    # `use_potentials` is therefore derived from this toggle (see below).
use_hybrid = False

    # POTENTIALS (derived) #
    # The B file is built from A via Coulomb-gauge FFT curl-inverse
    # (src.bext.vector_potential) and the E file is built from φ via the
    # closed-form ring scalar potential (src.eext.potential). The generated
    # HDF5 cache files carry a "_potentials" tag so they never collide with
    # the magpylib-direct / analytic-E pipeline outputs.
use_potentials = use_hybrid

    # Plasma temperatures #
Te = 50e3 * sc.eV  # electron temperature: 50 keV (typical polywell target)
#Ti = 1e3  * sc.eV  # ion temperature: 1 keV (ions gain energy from potential well)
Ti = 1e4  * sc.eV

# GRID #
L = 2 # full-domain half-extent (box spans [-L, +L] in each direction)
N = 80 # full-domain cell count per axis (must be divisible by MPI rank count)
number_per_cell_each_dim = [10, 10, 10] # macroparticles per cell (density mode only)

# SYMMETRY #
symmetry = "octant" # "full" or "octant". Octant simulates [0, +L]^3 with pmc+reflecting
                  # on the inner faces; requires N even, and N/2 divisible by ranks.

# PARTICLE SPAWNING #
particle_mode = "count"   # "density": N_cells * ppc particles via GriddedLayout
                            # "count":   n_test_particles_per_cell randomly per cell
                            #            (WarpX's AnalyticDistribution doesn't support
                            #            a global total with PseudoRandomLayout)
n_test_particles_per_cell = 10   # used only when particle_mode == "count"

# SCALE FACTORS #
plasma_bounding = 0.25 # plasma sphere radius as fraction of full-domain half-extent L

## Validate toggles and derive the simulated-domain spec.
## Everything downstream (grid, plasma bounds, field caches, runs DB) reads
## from `domain`/`layout` rather than touching the toggles directly.
assert symmetry in VALID_SYMMETRIES, f"Unknown symmetry mode '{symmetry}'"
assert particle_mode in VALID_PARTICLE_MODES, f"Unknown particle_mode '{particle_mode}'"
domain = derive_domain(symmetry, L, N)

#################
# === GRIDS === # (and solver)
#################
grid = picmi.Cartesian3DGrid(
    number_of_cells=list(domain.n_cells),
    lower_bound=list(domain.lower),
    upper_bound=list(domain.upper),
    lower_boundary_conditions=list(domain.field_bc_lo),
    upper_boundary_conditions=list(domain.field_bc_hi),
    lower_boundary_conditions_particles=list(domain.particle_bc_lo),
    upper_boundary_conditions_particles=list(domain.particle_bc_hi),
    warpx_max_grid_size=32,
)

# note: The Hybrid-PIC solver does not support external E-fields applied to
# particles; it consumes the external B-field through a vector potential A.
# The `use_hybrid` toggle above flips both the solver class and the field
# pipelines (via `use_potentials`) in lockstep so the cache files match.
if use_hybrid:
    solver = picmi.HybridPICSolver(
        grid=grid,
        Te=1.0,
        n0=p_density,
        gamma=1,
        plasma_resistivity=0,
        n_floor=(p_density * plasma_bounding),
    )
else:
    solver = picmi.ElectromagneticSolver(
        grid=grid,
        method="Yee",   # options: "Yee", "CKC", "psatd"
        cfl=0.99,
    )

###################
# === SPECIES === #
###################
# Thermal rms velocities from temperature (non-relativistic approximation)
# For 50 keV electrons ve_rms ~ 1.3e8 m/s (mildly relativistic but WarpX handles this)
ve_rms = np.sqrt(Te / sc.m_e)
vi_rms = np.sqrt(Ti / sc.m_p)

plasma_bounds_lo, plasma_bounds_hi = plasma_bounds(domain, plasma_bounding)

# Spatial profile: uniformly sampled sphere of radius `plasma_radius` centred at
# the origin. AnalyticDistribution still samples within the bounding cube
# (plasma_bounds_lo/hi); the step-function density makes particles outside the
# sphere carry zero weight, so the realised macroparticle cloud is spherical.
# In octant symmetry the cube is already clipped to [0,+R]^3, which gives the
# +x/+y/+z octant of the same sphere.
plasma_radius = plasma_bounding * L
density_expr = f"if(x*x+y*y+z*z<{plasma_radius * plasma_radius}, {p_density}, 0.)"

# Electrons: isotropic Maxwellian at Te, no net drift
electron_distribution = picmi.AnalyticDistribution(
    density_expression=density_expr,
    lower_bound=plasma_bounds_lo,
    upper_bound=plasma_bounds_hi,
    fill_in=True,
    rms_velocity=[ve_rms, ve_rms, ve_rms],
)
# Ions: isotropic Maxwellian at Ti (cooler; will be accelerated by potential well)
ion_distribution = picmi.AnalyticDistribution(
    density_expression=density_expr,
    lower_bound=plasma_bounds_lo,
    upper_bound=plasma_bounds_hi,
    fill_in=True,
    rms_velocity=[vi_rms, vi_rms, vi_rms],
)

# create the plasma species with their respective distributions
plasma_e = picmi.Species(
    particle_type='electron', name='plasma_e', initial_distribution=electron_distribution,
)
plasma_i = picmi.Species(
    particle_type='proton', name='plasma_i', initial_distribution=ion_distribution,
)

plasma_e.do_not_deposit = 1
plasma_i.do_not_deposit = 1 # test particles

##################
# === FIELDS === #
##################
## set external B-field initialization mode
    # errors were happening without turning initial div cleaning off when doing an EM solver with open bc
warpx.do_initial_div_cleaning = 0 # not needed, since both solutions are already div. free

## Configure B-field via the modular dispatcher
ext_path = setup_bext(
    method=b_method,
    particles=particles,
    warpx_module=warpx,
    I=I, dia=b_dia, offset=b_offset,
    domain=domain,                     # only used by "file" method; ignored by "analytic"
    solver="hybrid" if use_hybrid else None,
    use_potentials=use_potentials,
)

## Configure E-field (file-based only for now)
    # note: "file" B-field and E-field share the same .h5 file.
    # "analytic" B-field means E needs its own file, OR its own parse setup.
if e_method is not None:
    # `method` is ignored when use_potentials=True (φ → -∇φ pipeline takes over).
    method = EMethods[e_method].value[0] if not use_potentials else None
    if b_method == "file":
        # B already made the .h5 file — append E-field data to it
        particles.E_ext_particle_init_style = "read_from_file"
        ext_path = fill_eext_file(ext_path, method,
                                  e_dia, e_offset,
                                  Q, domain,
                                  use_potentials=use_potentials)
        particles.read_fields_from_path = ext_path
    else:
        # Analytic B doesn't make an .h5. E-field needs its own file.
        particles.E_ext_particle_init_style = "read_from_file"
        dummy_ext = make_bext_file(0, b_dia, b_offset, domain,
                                   use_potentials=use_potentials)  # zero-current scaffold
        ext_path = fill_eext_file(dummy_ext, method,
                                  e_dia, e_offset,
                                  Q, domain,
                                  use_potentials=use_potentials)
        particles.read_fields_from_path = ext_path

###############
# === SIM === #
###############
# create the simulation object
sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=True,

)

layout = make_layout(
    particle_mode,
    grid=grid,
    n_macroparticle_per_cell=number_per_cell_each_dim,
    n_test_particles_per_cell=n_test_particles_per_cell,
)

# add species to the simulation
"""sim.add_species(plasma_e, layout=layout)"""
sim.add_species(plasma_i, layout=layout)

# create, add tallies to the simulation
# Field + particle diagnostics share `name="diag1"` so openPMD writes them
# into a single series — each iteration file carries both `meshes/` (fields)
# and `particles/` (species) groups, instead of producing two parallel
# diags/field_diag/ and diags/part_diag/ folders.
field_diag = picmi.FieldDiagnostic(
    name="diag",
    grid=grid,
    period=10,
    # WarpX's FullDiagnostics::isFieldOutputType does NOT accept *_fp_external
    # field names — those are internal MultiFab labels, not output types. To
    # stream-trace the external fields in ParaView, open the pre-generated
    # openPMD file at `output/bext/B_ext_…h5` as a separate source — it
    # already contains B/x,y,z and E/x,y,z as proper vector records.
    # In the current test-particle regime (do_not_deposit=1 on both species)
    # no PIC self-field is generated, so the `B` written here is effectively
    # equal to the external B anyway.
    data_list=["Bx", "By", "Bz",
               "Jx", "Jy", "Jz", "part_per_cell"],
    warpx_format='openpmd',  # Options: 'openpmd', 'plotfile', 'checkpoint'
    warpx_openpmd_backend='h5',  # For openPMD: 'h5', 'bp', or 'json'
)

part_diag = picmi.ParticleDiagnostic(
    name="diag1",
    period=10,
    species=[plasma_i],
    data_list=["x", "y", "z", "ux", "uy", "uz", "weighting"],
    warpx_format='openpmd',  # Options: 'openpmd', 'plotfile', 'checkpoint'
    warpx_openpmd_backend='h5'  # For openPMD: 'h5', 'bp', or 'json'
)
#warpx.sort_intervals = -1


sim.add_diagnostic(field_diag)
sim.add_diagnostic(part_diag)


#######################
# === RUN + LOGGING ===
#######################
# Allocate a per-run directory, snapshot this input script for reproducibility,
# and register the run in the SQLite runs database before stepping. Diagnostics
# are redirected into the run directory by chdir'ing before sim.step().
run_dir = new_run_dir()
try:
    shutil.copy2(__file__, run_dir / "polywell_input.py")
except Exception:
    pass

run_params = {
    # simulation control
    "max_steps":         max_steps,
    "const_dt":          warpx.const_dt,
    # plasma
    "p_density":         p_density,
    "Te_eV":             float(Te / sc.eV),
    "Ti_eV":             float(Ti / sc.eV),
    "plasma_bounding":   plasma_bounding,
    # B-field
    "b_method":          b_method,
    "coil_current":      I,
    "b_dia":             b_dia,
    "b_offset":          b_offset,
    # E-field
    "e_method":          str(e_method),
    "e_charge":          Q,
    "e_dia":             e_dia,
    "e_offset":          e_offset,
    # grid
    "grid_L":            L,
    "grid_N":            N,
    "symmetry":          symmetry,
    "particles_per_cell": number_per_cell_each_dim,
    "particle_mode":     particle_mode,
    "n_test_particles_per_cell": n_test_particles_per_cell,
    # solver
    "solver_type":       type(solver).__name__,
    "solver_method":     getattr(solver, "method", None),
    "cfl":               getattr(solver, "cfl", None),
    "use_hybrid":        int(bool(use_hybrid)),
    # diagnostics
    "diag_period":       field_diag.period,
    "diag_path":         str(run_dir),
}

db = RunsDB()
_prev_cwd = os.getcwd()
try:
    with db.run_context(run_dir, run_params) as run_id:
        print(f"[runs.db] registered run id={run_id} at {run_dir}")
        os.chdir(run_dir)
        try:
            # Populate PICMI inputs (idempotent — sim.step() will skip the
            # re-init via its self.inputs_initialized guard), then extend the
            # WarpX-side fields_to_plot with names PICMI's parser dropped.
            sim.step()
        finally:
            os.chdir(_prev_cwd)
finally:
    db.close()