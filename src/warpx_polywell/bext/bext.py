"""
External B-field module for WarpX polywell simulations.

Supports two pipelines, selectable via setup_bext():
  - "file"     : Pre-compute B on a grid with magpylib, write to openPMD HDF5,
                  and tell WarpX to read_from_file. (original behavior)
  - "analytic" : Generate parser expression strings from the exact elliptic-integral
                  solution, and tell WarpX to parse_B_ext_particle_function. No grid file needed.
"""
from warpx_polywell.bext.make_collection import make_polywell_collection
from warpx_polywell.bext.analytic import build_bext_expressions
from warpx_polywell.domain import Domain
import h5py
import numpy as np
from datetime import datetime
from warpx_polywell.utils.storage import get_backend


# ================================================================
# Public entry point — pick your pipeline here
# ================================================================

def setup_bext(method, particles, warpx_module=None, *,
               I, dia, offset, domain: Domain = None, solver = None,
               use_potentials: bool = False):
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
    use_potentials : bool
        If True, the file pipeline computes A via FFT curl-inverse on a padded
        grid (src.bext.vector_potential) and derives B = ∇×A instead of taking
        B directly from magpylib. The generated filename carries a
        "_potentials" tag so cached files from the two pipelines never collide.
        Ignored for the "analytic" method.

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
        if not use_potentials:
            # Hybrid PIC consumes the external B-field through a vector
            # potential A, not B directly. Forcing use_potentials here keeps
            # the cache filename ("_potentials" tag) consistent with what
            # WarpX is actually going to read.
            print("[setup_bext] solver='hybrid' implies use_potentials=True; flipping.")
            use_potentials = True
        ext_path = make_bext_file(I, dia, offset, domain, use_potentials=use_potentials)
        _wire_hybrid_external_A(ext_path)
        return ext_path

    elif method == "file":
        if domain is None:
            raise ValueError("'file' method requires a domain parameter.")
        particles.B_ext_particle_init_style = "read_from_file"
        ext_path = make_bext_file(I, dia, offset, domain, use_potentials=use_potentials)
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

        # 'data/1/meshes' has the B, E and A groups (A is only populated when
        # the file is generated via the use_potentials pipeline; in the
        # default magpylib-direct pipeline the A datasets stay absent.)
        B = meshes.create_group('B')
        E = meshes.create_group('E')
        A = meshes.create_group('A')

        # B, E and A groups each have x, y, z datasets
        # unitDimension exponents follow openPMD's [L, M, T, I, θ, N, J]
        # convention. The B and E entries below preserve the previously
        # written values for cache compatibility; the A entry is the
        # openPMD-standard exponent vector for Wb/m (= T·m).
        for field_group, field_name in [(B, 'B'), (E, 'E'), (A, 'A')]:
            field_group.attrs['geometry'] = 'cartesian'
            field_group.attrs['gridSpacing'] = np.array([1.0, 1.0, 1.0])
            field_group.attrs['gridGlobalOffset'] = np.array([0.0, 0.0, 0.0])
            field_group.attrs['gridUnitSI'] = 1.0
            field_group.attrs['dataOrder'] = 'C'
            field_group.attrs['axisLabels'] = np.array(['x', 'y', 'z'], dtype='S')
            if field_name == 'B':
                field_group.attrs['unitDimension'] = np.array(
                    [0.0, 1.0, 1.0, -2.0, 0.0, 0.0, -1.0])
            elif field_name == 'E':
                field_group.attrs['unitDimension'] = np.array(
                    [1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0])
            else:  # 'A' — vector potential, Wb/m = kg·m·s⁻²·A⁻¹
                field_group.attrs['unitDimension'] = np.array(
                    [1.0, 1.0, -2.0, -1.0, 0.0, 0.0, 0.0])
            field_group.attrs['timeOffset'] = 0.0

def _fill_h5_file(filepath, Bx, By, Bz, grid_spacing, grid_offset,
                  Ax=None, Ay=None, Az=None):
    """
    Populate the B and E meshes (and optionally A) of an external-fields .h5
    file that was previously initialised by _make_empty_ext_h5.

    When Ax/Ay/Az are provided, they are written to the `A` mesh in T·m
    (Wb/m). WarpX's hybrid-PIC solver reads A from this mesh via
    external_vector_potential.<name>.read_from_file + .path.
    """
    with h5py.File(filepath, "r+") as f:
        meshes = f["data/1/meshes"]

        # Update grid attributes with actual values (always B and E; A only
        # when the A mesh is going to be populated below).
        grid_groups = ['B', 'E']
        if Ax is not None:
            grid_groups.append('A')
        for field_name in grid_groups:
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

        # Optional A-field data (vector potential, units Wb/m = T·m).
        if Ax is not None:
            for component, data in [('x', Ax), ('y', Ay), ('z', Az)]:
                dataset_path = f"data/1/meshes/A/{component}"
                if dataset_path in f:
                    del f[dataset_path]
                dset = f.create_dataset(dataset_path, data=data, dtype='f8')
                dset.attrs['unitSI'] = 1.0  # Wb/m
                dset.attrs['position'] = np.array([0.5, 0.5, 0.5])
                dset.attrs['shape'] = np.array(data.shape)

def get_bext_file_name(I, dia, offset, domain: Domain, use_potentials: bool = False):
    """
    Returns the name of the external B-field's .h5 file.

    The user-facing L and N (full-domain spec) plus the symmetry tag are
    embedded so that octant and full-domain caches never collide and the
    filename stays recognisable to a user reading from the cache dir.

    When `use_potentials` is True the name carries a "_potentials" tag so the
    A → ∇×A pipeline and the magpylib-direct pipeline never share a cache
    entry, even at otherwise identical parameters.
    """
    tag = "_potentials" if use_potentials else ""
    return (
        f"B_ext{tag}_I-{I}A_D-{dia}m_Off-{offset}m_"
        f"L-{domain.L}m_N-{domain.N}_sym-{domain.symmetry}.h5"
    )

def _compute_b_via_potentials(I, dia, offset, domain):
    """
    Compute A and B on the WarpX grid via the potentials pipeline.

    Solves for the vector potential A on a zero-padded grid using the
    Coulomb-gauge FFT curl-inverse of the magpylib B (src.bext.vector_potential),
    crops to the physics region, then derives B = ∇×A via central differences.

    Imports are local so importing src.bext.bext stays cheap when the
    potentials path isn't used.

    Returns
    -------
    Bx, By, Bz : derived from ∇×A (Tesla)
    Ax, Ay, Az : the vector potential itself (T·m = Wb/m)
    spacing    : list [dx, dy, dz] (m)
    """
    from warpx_polywell.bext.vector_potential import converge_A_grid, curl_A

    collection = make_polywell_collection(I, dia, offset)
    Ax, Ay, Az, spacing, pad = converge_A_grid(
        collection, domain, start=2, max_pad=8, rtol=1e-3, verbose=True,
    )
    print(f"[bext-potentials] A converged at pad_factor={pad}")
    Bx, By, Bz = curl_A(Ax, Ay, Az, spacing)
    return Bx, By, Bz, Ax, Ay, Az, list(spacing)


def make_bext_file(I, dia, offset, domain: Domain, use_potentials: bool = False):
    """
    Checks whether external B-field file exists, or if it should calculate and
    create a new .h5 file for the given configuration.

    I, dia, and offset are parameters for the coils.
    domain encodes the simulated grid (bounds, cell count, symmetry tag).

    When `use_potentials` is True, B is derived from A = (FFT curl-inverse of
    magpylib B) rather than taken directly from magpylib. The filename gets a
    "_potentials" tag.
    """
    backend = get_backend(subdir="bext")

    # get the name of the requested file
    file_name = get_bext_file_name(I, dia, offset, domain, use_potentials=use_potentials)

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
    Ax = Ay = Az = None
    if use_potentials:
        # A → ∇×A pipeline (Coulomb-gauge FFT curl-inverse on a padded grid).
        # Both A and B are kept: A feeds WarpX's external_vector_potential
        # (hybrid solver), B is written to the same file for downstream
        # diagnostics / non-hybrid use.
        Bx, By, Bz, Ax, Ay, Az, grid_spacing = _compute_b_via_potentials(
            I, dia, offset, domain,
        )
    else:
        # Direct magpylib B on the physics grid.
        collection = make_polywell_collection(I, dia, offset)
        _x = np.linspace(domain.lower[0], domain.upper[0], domain.n_cells[0])
        _y = np.linspace(domain.lower[1], domain.upper[1], domain.n_cells[1])
        _z = np.linspace(domain.lower[2], domain.upper[2], domain.n_cells[2])
        grid_spacing = [_x[1] - _x[0], _y[1] - _y[0], _z[1] - _z[0]]
        _mesh = np.meshgrid(_x, _y, _z, indexing='ij')   # (3, Nx, Ny, Nz)
        mesh = np.moveaxis(_mesh, 0, -1)                  # magpylib wants (Nx, Ny, Nz, 3)
        B = collection.getB(mesh)
        Bx, By, Bz = np.moveaxis(B, -1, 0)

    #################
    # POPULATE FILE #
    #################
    # create the empty .h5 file
    _make_empty_ext_h5(file_path)
    # populate the .h5 file (A datasets are written only when use_potentials)
    _fill_h5_file(file_path, Bx, By, Bz,
                  grid_spacing=grid_spacing,
                  grid_offset=tuple(domain.lower),
                  Ax=Ax, Ay=Ay, Az=Az)

    # route the finished file through the storage backend
    return backend.save(file_path, file_name)


# ================================================================
# Hybrid-PIC external vector potential wiring
# ================================================================

# Name used for the external_vector_potential field entry. The same string
# appears in three ParmParse keys so we factor it out.
_HYBRID_A_FIELD_NAME = "polywell"


def _wire_hybrid_external_A(ext_path):
    """
    Wire the openPMD file at `ext_path` into WarpX's hybrid-PIC external
    vector potential machinery. Equivalent to writing the following block in
    a WarpX inputs file:

        hybrid_pic_model.add_external_fields = 1
        external_vector_potential.fields = polywell
        external_vector_potential.do_diva_cleaning = 0
        external_vector_potential.polywell.read_from_file = 1
        external_vector_potential.polywell.path = <ext_path>
        external_vector_potential.polywell.A_time_external_grid_function(t) = 1

    The A is treated as static (`A_time_external_grid_function(t) = 1`) and
    divergence cleaning is disabled because the FFT curl-inverse already
    enforces Coulomb gauge (∇·A = 0) by construction.
    """
    from pywarpx import hybridpicmodel, external_vector_potential

    hybridpicmodel.add_external_fields = 1
    external_vector_potential.fields = _HYBRID_A_FIELD_NAME
    external_vector_potential.do_diva_cleaning = 0
    # Per-field sub-keys: pywarpx Buckets accept arbitrary string keys, so
    # the verbatim ParmParse "polywell.read_from_file" style works via
    # setattr. Match WarpX's expected format exactly — including the
    # (t) suffix on the time function.
    setattr(external_vector_potential, f"{_HYBRID_A_FIELD_NAME}.read_from_file", 1)
    setattr(external_vector_potential, f"{_HYBRID_A_FIELD_NAME}.path", str(ext_path))
    setattr(external_vector_potential,
            f"{_HYBRID_A_FIELD_NAME}.A_time_external_grid_function(t)", "1")
    print(f"[setup_bext] hybrid external A wired: "
          f"external_vector_potential.{_HYBRID_A_FIELD_NAME}.path={ext_path}")