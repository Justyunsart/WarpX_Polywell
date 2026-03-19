"""
To utilize warpx's ability to set external fields with a grid file,
we want to create an .h5 file for the B-field grid.
"""
from src.bext.make_collection import make_polywell_collection
import h5py
import numpy as np
from datetime import datetime
from src.utils.paths import BEXT_DIR #where to place the external B-field .h5 files

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

def get_bext_file_name(I, dia, offset, L, N):
    """
    Returns the name of the external B-field's .h5 file.
    """
    return f"B_ext_I-{I}A_D-{dia}m_Off-{offset}m_L-{L}m_N-{N}.h5"

def make_bext_file(I, dia, offset, L:int, N:int):
    """
    Checks whether external B-field file exists, or if it should calculate and
    create a new .h5 file for the given configuration.

    I, dia, and offset are parameters for the coils.
    L and N are parameters for the grid. (L = length in each axis, N = resolution in each axis)
    """
    # get the name of the requested file
    file_name = get_bext_file_name(I, dia, offset, L, N)
    file_path = BEXT_DIR / file_name # the full path of where the file should be/will be

    # return file if it exists. If not, continue with the pipeline for making the file
    if file_path.is_file():
        print("File exists, returning")
        return file_path
    else:
        print("File does not exist. Continuing with .h5 generation")

    ###############
    # CREATE DATA #
    ###############
    # create the magpylib.Collections object to calculate the B-field with
    collection = make_polywell_collection(I, dia, offset)

    # next, create the grid of points for the grid
    _x = np.linspace(-L, L, N)
    _y = np.linspace(-L, L, N)
    _z = np.linspace(-L, L, N)
    interval = _x[1] - _x[0]
    grid_spacing = [interval, interval, interval]
    _mesh = np.meshgrid(_x, _y, _z, indexing='ij') # (3, N, N, N)
    mesh = np.moveaxis(_mesh, 0, -1) # magpylib's getB() expects (N, N, N, 3)

    # calculate the B-field at these grid points
    B = collection.getB(mesh)
    Bx, By, Bz = np.moveaxis(B, -1, 0) # each array of shape (N, N, N)

    #################
    # POPULATE FILE #
    #################
    # create the empty .h5 file
    _make_empty_ext_h5(file_path)
    # populate the .h5 file
    _fill_h5_file(file_path, Bx, By, Bz,
                  grid_spacing=grid_spacing,
                  grid_offset=(-L, -L, -L))

    return file_path # returns the full path to the file