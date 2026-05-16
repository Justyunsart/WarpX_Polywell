"""
External B-field module for WarpX polywell simulations.

Supports two pipelines, selectable via setup_bext():
  - "file"     : Pre-compute B on a grid with magpylib, write to openPMD HDF5,
                  and tell WarpX to read_from_file. (original behavior)
  - "analytic" : Generate parser expression strings from the exact elliptic-integral
                  solution, and tell WarpX to parse_B_ext_particle_function. No grid file needed.
"""
from src.bext.make_collection import make_polywell_collection
from src.bext.analytic import build_bext_expressions
from src.domain import Domain
import h5py
import numpy as np
from datetime import datetime
from src.utils.storage import get_backend


# ================================================================
# Public entry point — pick your pipeline here
# ================================================================

def setup_bext(method, particles, warpx_module=None, *,
               I, dia, offset, domain: Domain = None, solver = None):
    """
    Configure WarpX's external B-field for particles.

    Parameters
    ----------
    method : str
        "file"     — magpylib grid → openPMD HDF5 → read_from_file
        "analytic" — elliptic-integral parser expressions → parse_B_ext_particle_function
    particles : pywarpx.particles module
        The `particles` object from pywarpx (used to set init style and paths).
    warpx_module : pywarpx.warpx module, optional
        The `warpx` object from pywarpx (needed for my_constants in analytic mode).
    I       : float  — coil current (A)
    dia     : float  — coil diameter (m)
    offset  : float  — coil center distance from origin (m)
    domain  : Domain — simulated-domain spec. Required for "file" method.

    Returns
    -------
    ext_path : str or None
        For "file" mode, returns the path to the generated .h5 file.
        For "analytic" mode, returns None (no file involved).
    """
    method = method.lower()

    # This resolves first to ensure the solver dictates where the field is applied
    # Important since HybridPICSolver does not allow fields applied to particles
    if solver == "hybrid":
        if domain is None:
            raise ValueError("'file' method requires L and N grid parameters.")
        ext_path = make_bext_file(I, dia, offset, domain)
        return ext_path

    elif method == "file":
        if domain is None:
            raise ValueError("'file' method requires a domain parameter.")
        particles.B_ext_particle_init_style = "read_from_file"
        ext_path = make_bext_file(I, dia, offset, domain)
        particles.read_fields_from_path = ext_path
        return ext_path

    elif method == "analytic":
        particles.B_ext_particle_init_style = "parse_B_ext_particle_function"
        exprs = build_bext_expressions(I, dia, offset)
        # WarpX's ParmParse keys literally include the argument signature:
        # `particles.Bx_external_particle_function(x,y,z,t)`. That's not a
        # legal Python attribute name, so we have to use setattr rather than
        # dot syntax — pywarpx's Bucket stores the key verbatim, which makes
        # it findable on the C++ side.
        setattr(particles, "Bx_external_particle_function(x,y,z,t)", exprs['Bx'])
        setattr(particles, "By_external_particle_function(x,y,z,t)", exprs['By'])
        setattr(particles, "Bz_external_particle_function(x,y,z,t)", exprs['Bz'])
        print(f"[setup_bext] Analytic B-field configured (6 coils, I={I} A, dia={dia} m, offset={offset} m)")
        print(f"[setup_bext] Expression lengths: Bx={len(exprs['Bx'])}, By={len(exprs['By'])}, Bz={len(exprs['Bz'])} chars")
        return None

    else:
        raise ValueError(f"Unknown B-field method '{method}'. Use 'file' or 'analytic'.")


# ================================================================
# File-based pipeline (original implementation, unchanged)
# ================================================================

def _make_empty_ext_h5(filename)->None:
    """
    Creates an empty .h5 file with the given filename.
    The file will be formatted with the expected group and dataset
    structure of an external field input file.

    :param filename: (str or Pathlike) abs path to the .h5 file to be made
    """
    with h5py.File(filename, "w") as f:
        # Add root-level openPMD attributes
        f.attrs['openPMD'] = '1.1.0'
        f.attrs['openPMDextension'] = np.uint32(0)
        f.attrs['basePath'] = '/data/%T/'
        f.attrs['meshesPath'] = 'meshes/'
        f.attrs['particlesPath'] = 'particles/'
        f.attrs['iterationEncoding'] = 'fileBased'
        f.attrs['iterationFormat'] = '/data/%T/'
        f.attrs['software'] = 'custom'
        f.attrs['softwareVersion'] = '1.0'
        f.attrs['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S %z')

        # root group has the 'data' subgroup
        data = f.create_group('data')
        one = data.create_group('1')
        meshes = one.create_group('meshes')

        # Add iteration attributes
        one.attrs['time'] = 0.0
        one.attrs['dt'] = 0.0
        one.attrs['timeUnitSI'] = 1.0

        # 'data/1/meshes' has the B and E groups
        B = meshes.create_group('B')
        E = meshes.create_group('E')

        # B and E groups each have x, y, z datasets
        for field_group, field_name in [(B, 'B'), (E, 'E')]:
            field_group.attrs['geometry'] = 'cartesian'
            field_group.attrs['gridSpacing'] = np.array([1.0, 1.0, 1.0])
            field_group.attrs['gridGlobalOffset'] = np.array([0.0, 0.0, 0.0])
            field_group.attrs['gridUnitSI'] = 1.0
            field_group.attrs['dataOrder'] = 'C'
            field_group.attrs['axisLabels'] = np.array(['x', 'y', 'z'], dtype='S')
            field_group.attrs['unitDimension'] = np.array(
                [0.0, 1.0, 1.0, -2.0, 0.0, 0.0, -1.0]) if field_name == 'B' else np.array(
                [1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0])
            field_group.attrs['timeOffset'] = 0.0

def _fill_h5_file(filepath, Bx, By, Bz, grid_spacing, grid_offset):
    with h5py.File(filepath, "r+") as f:
        meshes = f["data/1/meshes"]

        # Update grid attributes with actual values
        for field_name in ['B', 'E']:
            field_group = meshes[field_name]
            field_group.attrs['gridSpacing'] = np.array(grid_spacing)
            field_group.attrs['gridGlobalOffset'] = np.array(grid_offset)

        # Write B-field data
        for component, data in [('x', Bx), ('y', By), ('z', Bz)]:
            dataset_path = f"data/1/meshes/B/{component}"

            # Create dataset with actual data
            if dataset_path in f:
                del f[dataset_path]

            dset = f.create_dataset(dataset_path, data=data, dtype='f8')

            # Add required openPMD attributes for each dataset
            dset.attrs['unitSI'] = 1.0  # Tesla
            dset.attrs['position'] = np.array([0.5, 0.5, 0.5])  # Cell-centered
            dset.attrs['shape'] = np.array(data.shape)

        # Set E-field to zero (if WarpX requires it)
        for component in ['x', 'y', 'z']:
            dataset_path = f"data/1/meshes/E/{component}"
            if dataset_path in f:
                del f[dataset_path]

            dset = f.create_dataset(dataset_path, data=np.zeros_like(Bx), dtype='f8')
            dset.attrs['unitSI'] = 1.0  # V/m
            dset.attrs['position'] = np.array([0.5, 0.5, 0.5])
            dset.attrs['shape'] = np.array(Bx.shape)

def get_bext_file_name(I, dia, offset, domain: Domain):
    """
    Returns the name of the external B-field's .h5 file.

    The user-facing L and N (full-domain spec) plus the symmetry tag are
    embedded so that octant and full-domain caches never collide and the
    filename stays recognisable to a user reading from the cache dir.
    """
    return (
        f"B_ext_I-{I}A_D-{dia}m_Off-{offset}m_"
        f"L-{domain.L}m_N-{domain.N}_sym-{domain.symmetry}.h5"
    )

def make_bext_file(I, dia, offset, domain: Domain):
    """
    Checks whether external B-field file exists, or if it should calculate and
    create a new .h5 file for the given configuration.

    I, dia, and offset are parameters for the coils.
    domain encodes the simulated grid (bounds, cell count, symmetry tag).
    """
    backend = get_backend(subdir="bext")

    # get the name of the requested file
    file_name = get_bext_file_name(I, dia, offset, domain)

    # return file if it exists. If not, continue with the pipeline for making the file
    if backend.exists(file_name):
        print("File exists, returning")
        # For DriveBackend, download to get a usable local path; LocalBackend resolve() is already local.
        if hasattr(backend, "download"):
            return backend.download(file_name)
        return backend.resolve(file_name)
    else:
        print("File does not exist. Continuing with .h5 generation")

    # local staging path where h5py will write the file
    file_path = backend.resolve(file_name)

    ###############
    # CREATE DATA #
    ###############
    # create the magpylib.Collections object to calculate the B-field with
    collection = make_polywell_collection(I, dia, offset)

    # next, create the grid of points for the simulated domain
    _x = np.linspace(domain.lower[0], domain.upper[0], domain.n_cells[0])
    _y = np.linspace(domain.lower[1], domain.upper[1], domain.n_cells[1])
    _z = np.linspace(domain.lower[2], domain.upper[2], domain.n_cells[2])
    grid_spacing = [_x[1] - _x[0], _y[1] - _y[0], _z[1] - _z[0]]
    _mesh = np.meshgrid(_x, _y, _z, indexing='ij') # (3, Nx, Ny, Nz)
    mesh = np.moveaxis(_mesh, 0, -1) # magpylib's getB() expects (Nx, Ny, Nz, 3)

    # calculate the B-field at these grid points
    B = collection.getB(mesh)
    Bx, By, Bz = np.moveaxis(B, -1, 0) # each array of shape (Nx, Ny, Nz)

    #################
    # POPULATE FILE #
    #################
    # create the empty .h5 file
    _make_empty_ext_h5(file_path)
    # populate the .h5 file
    _fill_h5_file(file_path, Bx, By, Bz,
                  grid_spacing=grid_spacing,
                  grid_offset=tuple(domain.lower))

    # route the finished file through the storage backend
    return backend.save(file_path, file_name)