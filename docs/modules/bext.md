# `src/bext` — External B-field Module

Provides the external magnetic field for WarpX polywell simulations. Supports
two modes selectable at runtime: a **file-based** pipeline (pre-computed grid in
an openPMD HDF5 file) and an **analytic** pipeline (exact elliptic-integral
expressions evaluated per-particle). See
[External Particle Field Modes](external_particle_fields.md) for the full
conceptual background, physics, and user guide.

---

## Files

| File | Purpose |
|---|---|
| `src/bext/bext.py` | `setup_bext()` dispatcher + file-based HDF5 creation |
| `src/bext/analytic.py` | Analytic elliptic-integral kernel (NumPy + AMReX parser expressions) |
| `src/bext/make_collection.py` | Builds the magpylib coil geometry (file mode only) |

---

## `make_collection.py`

### `make_polywell_collection(a, dia, d)`

Constructs a `magpylib.Collection` of 6 circular current loops arranged in a polywell
formation (one coil on each face of a cube centered at the origin).

```
Parameters:
    a    : float — current in Amperes
    dia  : float — coil diameter in metres
    d    : float — distance from origin to coil center in metres

Returns:
    magpylib.Collection
```

**Coil placement and orientation:**

| Coil | Position | Current direction | Rotation |
|---|---|---|---|
| s1 | `(-d, 0, 0)` | `+a` | 90° around Y |
| s2 | `(+d, 0, 0)` | `-a` | 90° around Y |
| s3 | `(0, -d, 0)` | `-a` | -90° around X |
| s4 | `(0, +d, 0)` | `+a` | -90° around X |
| s5 | `(0, 0, -d)` | `+a` | 90° around Z |
| s6 | `(0, 0, +d)` | `-a` | 90° around Z |

Alternating currents on opposing faces create the characteristic magnetic well structure.

### `make_helmholtz_collection(a, dia, d)` (test utility)

Creates a two-coil Helmholtz pair for testing/validation. Not used in the production pipeline.

---

## `bext.py`

### `setup_bext(method, particles, warpx_module, *, I, dia, offset, domain)`

**Modular dispatcher.** Configures WarpX's external B-field for either file-based
or analytic mode. This is the only function the input deck needs to call.

```
Parameters:
    method       : str — "file" or "analytic"
    particles    : pywarpx.particles module
    warpx_module : pywarpx.warpx module (optional)
    I, dia, offset : coil parameters
    domain       : src.domain.Domain — simulated-domain spec
                   (required for "file" mode; ignored by "analytic")

Returns:
    str or None — .h5 path (file mode) or None (analytic mode)
```

The `domain` parameter carries the simulated bounds, cell count, and the symmetry tag.
The grid sampled in file mode is `linspace(domain.lower[i], domain.upper[i], domain.n_cells[i])`
per axis — automatically `(N, N, N)` in full mode and `(N/2, N/2, N/2)` in octant mode.

See [External Particle Field Modes](external_particle_fields.md) for detailed
behavior of each mode, and [domain module](domain.md) for the `Domain` dataclass.

### `get_bext_file_name(I, dia, offset, domain)` → `str`

Returns the expected filename for a given set of parameters. Used to check for cached files.

```python
# Example
from src.domain import derive_domain
d = derive_domain("full", 2, 72)
name = get_bext_file_name(1e6, 1.0, 1.1, d)
# → "B_ext_I-1000000.0A_D-1m_Off-1.1m_L-2m_N-72_sym-full.h5"
```

The `_sym-…` token is essential because the sampled grid differs between modes at the same `L`/`N`.

### `make_bext_file(I, dia, offset, domain)` → `Path`

**Main public function.** Checks whether a cached file already exists; if not, computes
and writes a new one.

```
Parameters:
    I      : float          — coil current (A)
    dia    : float          — coil diameter (m)
    offset : float          — coil center distance from origin (m)
    domain : src.domain.Domain — simulated-domain spec

Returns:
    pathlib.Path — absolute path to the .h5 file
```

**Internal pipeline when file does not exist:**

1. Call `make_polywell_collection(I, dia, offset)` to build the magpylib coil set
2. Create a 3D meshgrid: `linspace(domain.lower[i], domain.upper[i], domain.n_cells[i])` per axis (indexing `'ij'`)
3. Reshape mesh to `(Nx, Ny, Nz, 3)` as expected by magpylib's `getB()`
4. Call `collection.getB(mesh)` — vectorised field evaluation over all grid points
5. Decompose result into `Bx, By, Bz` arrays
6. Call `_make_empty_ext_h5(file_path)` to create the skeleton HDF5
7. Call `_fill_h5_file(...)` with `grid_offset = domain.lower` to write the B-field data (E-field set to zeros as placeholder)

### `_make_empty_ext_h5(filename)` (private)

Creates a skeleton HDF5 file with all required openPMD metadata groups and attributes,
but no actual field data yet.

**openPMD attributes set on root:**

| Attribute | Value |
|---|---|
| `openPMD` | `"1.1.0"` |
| `basePath` | `"/data/%T/"` |
| `meshesPath` | `"meshes/"` |
| `iterationEncoding` | `"fileBased"` |

**Per-field-group attributes (`B` and `E`):**

| Attribute | Value |
|---|---|
| `geometry` | `"cartesian"` |
| `gridSpacing` | `[1.0, 1.0, 1.0]` (placeholder, overwritten later) |
| `gridGlobalOffset` | `[0.0, 0.0, 0.0]` (placeholder, overwritten later) |
| `unitDimension` | B: `[0, 1, 1, -2, 0, 0, -1]` (Tesla); E: `[1, 1, -3, -1, 0, 0, 0]` (V/m) |

### `_fill_h5_file(filepath, Bx, By, Bz, grid_spacing, grid_offset)` (private)

Writes actual B-field data and zeroed E-field placeholders into an existing skeleton file.

- Updates `gridSpacing` and `gridGlobalOffset` on both B and E groups
- Creates `data/1/meshes/B/{x,y,z}` datasets from `Bx, By, Bz`
- Creates `data/1/meshes/E/{x,y,z}` datasets as `zeros_like(Bx)`
- Each dataset gets `unitSI=1.0` and `position=[0.5, 0.5, 0.5]` (cell-centered)

---

## `analytic.py`

Provides the analytic B-field pipeline: exact elliptic-integral evaluation with
no grid files. Contains both a NumPy interface (for testing/plotting) and an
AMReX parser expression builder (for WarpX runtime).

### `build_bext_expressions(I, dia, offset)` → `dict`

Builds three AMReX parser expression strings for the full 6-coil polywell field.
Each expression uses native `comp_ellint_1(k)` / `comp_ellint_2(k)` functions
and local-variable semicolon syntax. Returns `{'Bx': ..., 'By': ..., 'Bz': ...}`.

### `B_polywell(X, Y, Z, I, dia, offset)` → `(Bx, By, Bz)`

NumPy evaluation of the same 6-coil field. Used for testing and validation — not
called by WarpX at runtime.

### `B_single_loop(rho, zeta, a, I)` → `(B_rho, B_zeta)`

Core kernel — single current loop field in cylindrical coordinates via
`scipy.special.ellipk` / `ellipe`.

For the full API reference and physics background, see
[External Particle Field Modes](external_particle_fields.md).

---

## Data Flow Summary

```
make_bext_file(I, dia, offset, domain)
        │
        ├─ [file exists?] ──yes──→ return cached path
        │
        └─ no
           │
           ├─ make_polywell_collection(I, dia, offset)
           │        → magpylib.Collection (6 Circle coils)
           │
           ├─ np.meshgrid over domain.lower..upper with domain.n_cells points
           │        → mesh shape (Nx, Ny, Nz, 3)
           │
           ├─ collection.getB(mesh)
           │        → B shape (Nx, Ny, Nz, 3)
           │        → Bx, By, Bz each (Nx, Ny, Nz)
           │
           ├─ _make_empty_ext_h5(path)
           │        → skeleton HDF5 with openPMD metadata
           │
           └─ _fill_h5_file(path, Bx, By, Bz, spacing, domain.lower)
                    → B datasets written, E zeroed
                    → return path
```
