"""
Functions to fill the external fields file with the E-field computed with parameters from the polywell input.
"""
from src.bext.make_collection import make_polywell_collection # This is a convenient container for getting rotations
from src.domain import Domain
from magpylib.current import Circle
from src.utils.cyl import toCyl, toCart
from src.utils.storage import get_backend
import numpy as np
import pathlib
from typing import Callable
import h5py

def get_e_field_data(method:Callable, dia, offset, Q, domain: Domain):
    """
    Inputs define the grid parameters. Returns a corresponding grid with the same shape
    that represents the E-field values for those grid points.
    """
    print(
        f"[get_e_field_data] Starting E-field computation: dia={dia}m, offset={offset}m, "
        f"Q={Q}C, domain={domain.symmetry} L={domain.L}m N={domain.N}"
    )

    # Used BEFORE each call to the analytic method
    # BEFORE: orient_point() -> to_cyl() -> method
    def orient_point(c: Circle, point):
        """
        Puts the point (cartesian) in a circle's local space (centered at the origin, aligned with horizontal plane)
        - Only works with cartesian coordinates!
        """
        point = np.array(point, dtype=np.float64)
        # Reset rotation to identity
        rotation = c.orientation
        # print(f"coil rotation: {rotation.as_euler('xyz', degrees=True)}")
        # subtract the coil's position from the rotated point to make it centered at the origin.
        p = np.array(point - np.array(c.position, dtype=np.float64), dtype=np.float64)
        # after subtracting, the rotation then can be applied. This makes the point rotate about the coil center.
        inv_rotation = rotation.inv()

        rotated_point = inv_rotation.apply(p)

        # print(f"started with {point}, ended with {rotated_point}")
        return rotated_point

    ###############
    # CREATE DATA #
    ###############
    # create the magpylib.Collections object to calculate the B-field with
    collection = make_polywell_collection(Q, dia, offset)
    print(f"[get_e_field_data] Polywell collection created with {len(list(collection))} coils")

    # next, create the grid of points for the simulated domain
    nx, ny, nz = domain.n_cells
    _x = np.linspace(domain.lower[0], domain.upper[0], nx)
    _y = np.linspace(domain.lower[1], domain.upper[1], ny)
    _z = np.linspace(domain.lower[2], domain.upper[2], nz)
    grid_spacing = [_x[1] - _x[0], _y[1] - _y[0], _z[1] - _z[0]]
    X, Y, Z = np.meshgrid(_x, _y, _z, indexing='ij')  # (Nx, Ny, Nz)
    # Three SEPARATE arrays. Do not write `Ex = Ey = Ez = np.zeros(...)`:
    # that makes all three names alias the same buffer, so every +=
    # accumulates into a single array and Ex, Ey, Ez end up identical.
    Ex = np.zeros((nx, ny, nz))
    Ey = np.zeros((nx, ny, nz))
    Ez = np.zeros((nx, ny, nz))
    print(
        f"[get_e_field_data] Grid created: {nx}x{ny}x{nz} = {nx*ny*nz} points, "
        f"spacing=({grid_spacing[0]:.4f},{grid_spacing[1]:.4f},{grid_spacing[2]:.4f})m"
    )

    # constants for all function calls
    a = dia/2

    total_points = nx * ny * nz
    progress_interval = max(1, total_points // 10)

    # because the analytic methods aren't vectorized, we need to iterate over all points and accumulate.
    # from all coils' contributions.
    for idx, (i, j, k) in enumerate(np.ndindex(nx, ny, nz)):
        if idx % progress_interval == 0:
            print(f"[get_e_field_data] Computing E-field: {idx}/{total_points} points ({100*idx//total_points}%) done")

        input_cartesian = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])

        # loop over all coils to accumulate
        for c in collection:
            ## prepare input (cylindrical, transformed to meet function assumptions)
            rot_cartesian = orient_point(c=c, point=input_cartesian)
            input_cylindrical = toCyl(rot_cartesian)

            ## get output in the coil's LOCAL cylindrical frame
            phi_local = input_cylindrical[1]
            E_r, E_z_local = method(r=input_cylindrical[0], z=input_cylindrical[2],
                                    a=a, Q=Q)

            ## Build the field vector in the coil's LOCAL cartesian frame.
            ## The radial unit vector at this point is (cos phi, sin phi, 0)
            ## expressed in local cartesian, so E_r decomposes as below.
            E_local = np.array([E_r * np.cos(phi_local),
                                E_r * np.sin(phi_local),
                                E_z_local])

            ## Rotate the field vector back to the LAB frame using the
            ## coil's forward orientation. Without this step, coils with
            ## non-identity orientation contribute in the wrong basis and
            ## the total field loses the expected octahedral symmetry of
            ## the 6-coil polywell.
            E_lab = c.orientation.apply(E_local)

            Ex[i, j, k] += E_lab[0]
            Ey[i, j, k] += E_lab[1]
            Ez[i, j, k] += E_lab[2]

    print(f"[get_e_field_data] E-field computation complete. "
          f"Peak magnitude: {np.max(np.sqrt(Ex**2 + Ey**2 + Ez**2)):.4e} V/m")
    return Ex, Ey, Ez, grid_spacing


def _fill_efield_datasets(filepath, Ex, Ey, Ez, grid_spacing, grid_offset):
    """
    Open existing .h5 file for external initial field grid information
    and update the E-field datasets, including openPMD-expected properties.
    """
    print(f"[_fill_efield_datasets] Opening file for writing: {filepath}")
    with h5py.File(filepath, "r+") as f:
        meshes = f["data/1/meshes"]

        # Update E group grid attributes with actual values
        E_group = meshes['E']
        E_group.attrs['gridSpacing'] = np.array(grid_spacing)
        E_group.attrs['gridGlobalOffset'] = np.array(grid_offset)
        print(f"[_fill_efield_datasets] Set gridSpacing={grid_spacing}, gridGlobalOffset={grid_offset}")

        # Write E-field data
        for component, data in [('x', Ex), ('y', Ey), ('z', Ez)]:
            dataset_path = f"data/1/meshes/E/{component}"

            if dataset_path in f:
                print(f"[_fill_efield_datasets] Deleting existing dataset: {dataset_path}")
                del f[dataset_path]

            dset = f.create_dataset(dataset_path, data=data, dtype='f8')
            dset.attrs['unitSI'] = 1.0  # V/m
            dset.attrs['position'] = np.array([0.5, 0.5, 0.5])  # cell-centered
            dset.attrs['shape'] = np.array(data.shape)
            print(f"[_fill_efield_datasets] Wrote E{component} dataset with shape={data.shape}, "
                  f"max={np.max(np.abs(data)):.4e} V/m")

    print(f"[_fill_efield_datasets] Finished writing E-field datasets to {filepath}")


def fill_eext_file(filepath, method:Callable, dia, offset, Q, domain: Domain):
    """
    This function runs after the B-field external file functions are done.
    - Those functions make the .h5 file and also provides the filepath that this function needs

    filepath: the path to the .h5 file
    method: callable method for the analytic E-field
    dia: E-field ring diameter (m)
    Q: the current of the E-field rings (Coulombs)
    domain: simulated-domain spec; the upstream B file already encodes the
            symmetry tag in its stem, so this function's filename addition
            does not need its own.
    """
    print(
        f"[fill_eext_file] Starting E-field external file generation: dia={dia}m, "
        f"offset={offset}m, Q={Q}C, domain={domain.symmetry} L={domain.L}m N={domain.N}"
    )

    backend = get_backend(subdir="bext")

    ## To even determine if the methods should be run, check if a file with the same name exists
        # the E-field parameters to append the filepath name with
    filepath_addition = (
        f"_E_ext_Q-{Q}_D-{dia}m_offset-{offset}m_C_L-{domain.L}m_N-{domain.N}.h5"
    )
        # derive the final file name from the input filepath stem
    filepath = pathlib.Path(filepath) # ensure pathlib can work with the input filepath
    new_name = filepath.stem + filepath_addition
    print(f"[fill_eext_file] Target output name: {new_name}")

    if backend.exists(new_name):
        print(f"[fill_eext_file] File already exists, skipping computation and returning.")
        if hasattr(backend, "download"):
            return backend.download(new_name)
        return backend.resolve(new_name)
    else:
        print(f"[fill_eext_file] File does not exist. Continuing with .h5 generation")

    ## First, get the data
    print(f"[fill_eext_file] Computing E-field data using method: {method.__name__}")
    Ex, Ey, Ez, grid_spacing = get_e_field_data(method, dia, offset, Q, domain)

    ## Next, fill the .h5 file with the data
        # note: the fill function used below uses a 'with open()' block, so
        # the file is guaranteed to be closed after this function call finishes.
    grid_offset = tuple(domain.lower)
    print(f"[fill_eext_file] Writing E-field data to file with grid_offset={grid_offset}")
    _fill_efield_datasets(filepath, Ex, Ey, Ez, grid_spacing, grid_offset)

    ## After all .h5 writing operations are done, route the renamed file through the backend.
    print(f"[fill_eext_file] Saving as '{new_name}' via storage backend")
    filepath.rename(filepath.parent / new_name)  # rename locally first
    result = backend.save(filepath.parent / new_name, new_name)
    print(f"[fill_eext_file] Done. Output: {result}")

    return result
