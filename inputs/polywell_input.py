from pywarpx import picmi, warpx
from src.bext.bext import make_bext_file
from src.eext.eext import fill_eext_file # should run AFTER B-field init.
from src.eext.methods import EMethods # enum registry for available methods
import numpy as np

constants = picmi.constants
#######################
# === USER INPUTS === #
#######################
# SIM #
max_steps = 10000
warpx.const_dt = 1e-12
p_density = 1e18

    # B-Field # (polywell)
I = 1e6 # current of coil, Amperes
b_dia = 1 # diameter of coil, m
b_offset = 1.1 # dist. of coil center from origin, m

    # E-Field #
    # note: if e_method is set to None, all other E-field parameters will be ignored
e_method = "FW" # either None (no E-field) or a name of an entry in EMethods (yes E-field)
Q = 1e-9 # Coulombs
e_dia = 0.75 # m
e_offset = 1.1 # m

# GRID #
L = 3 # length of the simulation grid
N = 72 # resolution of each grid axis. My system runs with 8 cores, so this must be divisible by 8
number_per_cell_each_dim = [10, 10, 10] # number of macroparticles in each cell

# SCALE FACTORS #
plasma_bounding = 0.11 # the % of the grid length plasma is allowed to start in

## From user inputs, derive stuff
lx = ly = lz = L
nx = ny = nz = N

#################
# === GRIDS === # (and solver)
#################
grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[-lx, -ly, -lz],
    upper_bound=[lx, ly, lz],
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    warpx_max_grid_size=32,
)

#solver = picmi.ElectrostaticSolver(grid=grid)
solver = picmi.ElectromagneticSolver(
    grid=grid,
    method="Yee",   # options: "Yee", "CKC", "psatd"
    cfl=0.99,
)
# note: The Hybrid-PIC solver does not support external E-fields (needs external vector potentials instead)
"""solver = picmi.HybridPICSolver(
    grid=grid,
    Te=1.0,
    n0=p_density,
    gamma=1,
    plasma_resistivity=0,
    n_floor=(p_density*plasma_bounding),
)"""

###################
# === SPECIES === #
###################
# first create the plasma distribution
plasma_distribution = picmi.UniformDistribution(
    density=p_density,
    lower_bound=(np.array([-lx, -ly, -lz]) * plasma_bounding),
    upper_bound=(np.array([lx, ly, lz]) * plasma_bounding),
    fill_in = True,
    directed_velocity=[0.0, 0.0, 0.9 * picmi.constants.c]
)
# you can now create the plasma species (plasma electrons, ions)
plasma_e = picmi.Species(
    particle_type='electron', name='plasma_e', initial_distribution=plasma_distribution,
)
plasma_i = picmi.Species(
    particle_type='proton', name='plasma_i', initial_distribution=plasma_distribution,
)

##################
# === FIELDS === #
##################
## set external B-field initialization mode
    # note: if both B and E fields are set to "read_from_file", it expects both to be in the same file.
warpx.B_ext_field_init_style = "read_from_file"
warpx.E_ext_field_init_style = "read_from_file"
    # errors were happening without turning initial div cleaning off when doing an EM solver with open bc
warpx.do_initial_div_cleaning = 0 # not needed, since magpylib's solution is already div. free

## dynamically create the .h5 grid input based on sim params
ext_path = make_bext_file(I, b_dia, b_offset, L, N) # make and fill B-field info.
if e_method is not None: # update file with E-field information (if a method is selected)
    method = EMethods[e_method].value[0]
    ext_path = fill_eext_file(ext_path, method,
                              e_dia, e_offset,
                              Q, L, N)


## tell WarpX to read the file
warpx.read_fields_from_path = ext_path # tell the program to read the .h5 file

###############
# === SIM === #
###############
# create the simulation object
sim = picmi.Simulation(
    solver=solver,
    max_steps=max_steps,
    verbose=True,

)

# add species to the simulation
sim.add_species(
    plasma_e,
    layout=picmi.GriddedLayout(
        grid=grid, n_macroparticle_per_cell=number_per_cell_each_dim,
    ),
)
sim.add_species(
    plasma_i,
    layout=picmi.GriddedLayout(
        grid=grid, n_macroparticle_per_cell=number_per_cell_each_dim,
    ),
)

# create, add tallies to the simulation
field_diag = picmi.FieldDiagnostic(
    name="field_diag",
    grid=grid,
    period=100,
    data_list=["Bx", "By", "Bz", "Jx", "Jy", "Jz", "part_per_cell"],
    warpx_format='openpmd',  # Options: 'openpmd', 'plotfile', 'checkpoint'
    warpx_openpmd_backend='h5',  # For openPMD: 'h5', 'bp', or 'json'
)

part_diag = picmi.ParticleDiagnostic(
    name="part_diag",
    period=100,
    species=[plasma_e, plasma_i],
    data_list=["x", "y", "z", "ux", "uy", "uz", "weighting"],
    warpx_format='openpmd',  # Options: 'openpmd', 'plotfile', 'checkpoint'
    warpx_openpmd_backend='h5'  # For openPMD: 'h5', 'bp', or 'json'
)

sim.add_diagnostic(field_diag)
sim.add_diagnostic(part_diag)


sim.step()