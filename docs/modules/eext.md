# `src/eext` — External E-field Module

Computes the electric field from a set of charged rings (same geometry as the magnetic coils)
and writes the result into an existing B-field HDF5 file, overwriting the zeroed E-field
placeholder that `bext.py` left behind.

---

## Files

| File | Purpose |
|---|---|
| `src/eext/eext.py` | Grid computation and HDF5 population |
| `src/eext/methods.py` | Analytic E-field integrands + `EMethods` enum |

---

## `methods.py`

### `fw_e(r, z, a, Q, resolution=500, **kwargs)` → `(E_r, E_z)`

Numerical integration of the electric field from a uniformly charged ring using
a Riemann sum over the angle `θ ∈ [0, 2π]`.

```
Parameters:
    r          : float — radial distance from ring axis (m)
    z          : float — axial distance from ring plane (m)
    a          : float — ring radius (m)
    Q          : float — total charge on ring (C)
    resolution : int   — number of angle quadrature points (default 500)

Returns:
    (E_r, E_z) : tuple of floats — radial and axial E-field components (V/m)
```

**Physics:**
The linear charge density is `λ = Q / (2πa)`. The field components are:

```
E_r = (1/4πε₀) · λ · a · ∫₀²π (r - a·cos θ) / D³ dθ
E_z = (1/4πε₀) · λ · a · z · ∫₀²π 1 / D³ dθ

where D = sqrt(r² + a² - 2ar·cos θ + z²)
```

---

### `bob_e(r, z, a, Q, resolution=100, **kwargs)` → `(E_r, E_z)`

Alternative normalised integration from the same physical model, integrating over
`θ ∈ [0, π]` with Coulomb's constant `k = 8.99e9 N·m²/C²`.

```
Parameters:
    r          : float — radial distance (m), clamped to 1e-10 if near zero
    z          : float — axial distance (m)
    a          : float — ring radius (m)
    Q          : float — total charge (C)
    resolution : int   — number of angle quadrature points (default 100)

Returns:
    (E_r, E_z) : tuple of floats
```

**Note on singularity handling:** If `|r| < 1e-10`, `r` is clamped to `1e-10` to avoid
division by zero in the `frho` integrand.

---

### `EMethods` (Enum)

Registry of available methods for use in `polywell_input.py`.

```python
class EMethods(Enum):
    FW  = (fw_e,)
    BOB = (bob_e,)
```

Usage in input file:
```python
e_method = "FW"                         # string name
method = EMethods[e_method].value[0]    # callable function
```

---

## `eext.py`

### `fill_eext_file(filepath, method, dia, offset, Q, L, N)` → `Path`

**Main public function.** Computes the E-field grid and writes it into the HDF5 file
that `make_bext_file` produced. Renames the file to include E-field parameters.

```
Parameters:
    filepath : str or Path — path to existing B-field .h5 file
    method   : Callable    — one of the functions from methods.py
    dia      : float       — ring diameter (m)
    offset   : float       — ring center offset from origin (m)
    Q        : float       — total charge per ring (C)
    L        : int         — grid half-length (m)
    N        : int         — grid resolution per axis

Returns:
    pathlib.Path — path to the updated (and renamed) .h5 file
```

**Caching:** Before computing anything, constructs the expected output filename
and returns immediately if it already exists:
```
{original_stem}_E_ext_Q-{Q}_D-{dia}m_offset-{offset}m_C_L-{L}m_N-{N}.h5
```

**Pipeline when file does not exist:**
1. Call `get_e_field_data(method, ...)` to compute `Ex, Ey, Ez`
2. Call `_fill_efield_datasets(filepath, ...)` to write into the HDF5
3. Rename the file to append E-field parameters to the name

---

### `get_e_field_data(method, dia, offset, Q, L, N)` → `(Ex, Ey, Ez, grid_spacing)`

Computes the full 3D E-field grid by iterating over every grid point and accumulating
contributions from all 6 charged rings.

```
Parameters:
    method : Callable — analytic method (fw_e or bob_e)
    dia    : float    — ring diameter (m)
    offset : float    — ring center distance from origin (m)
    Q      : float    — total charge per ring (C)
    L      : int      — grid half-length (m)
    N      : int      — grid resolution per axis

Returns:
    Ex, Ey, Ez   : ndarray of shape (N, N, N) — E-field components (V/m)
    grid_spacing : list[float] — [dx, dy, dz]
```

**Per-point computation (inner loop):**

For each grid point `(i,j,k)` and each coil `c` in the collection:
1. `orient_point(c, point)` — transforms the Cartesian point into the coil's local frame
   (subtract coil position, apply inverse rotation)
2. `toCyl(rotated_point)` — convert to cylindrical `(r, φ, z)`
3. `method(r, z, a, Q)` — get `(E_r, E_z)` in cylindrical
4. `toCart(E_r, φ, E_z)` — convert back to Cartesian `(Ex, Ey, Ez)`
5. Accumulate into output arrays

> **Performance note:** This loop is O(N³ × 6) and is not vectorised.
> For `N=72`, that is ~2.2M method calls. For large `N`, this is the primary bottleneck.
> Progress is printed every 10%.

---

### `_fill_efield_datasets(filepath, Ex, Ey, Ez, grid_spacing, grid_offset)` (private)

Opens the existing HDF5 file in read-write mode and replaces the zeroed E-field datasets
with the computed values.

- Updates `E` group attributes: `gridSpacing`, `gridGlobalOffset`
- Deletes any existing `data/1/meshes/E/{x,y,z}` datasets
- Creates new float64 datasets with `unitSI=1.0` and `position=[0.5, 0.5, 0.5]`

---

## Coordinate Transform Pipeline

The analytic methods assume the ring is centered at the origin in the xy-plane.
Each real coil is at a different position and orientation, so each point must be
transformed into each coil's local frame before calling the method.

```
World Cartesian point
        │
        ▼
orient_point(coil, point)
  - subtract coil.position
  - apply coil.orientation.inv().apply(...)
        │
        ▼
Local Cartesian (ring centered at origin, axis = z)
        │
        ▼
toCyl(local_cartesian)
  → (r, φ, z) in cylindrical
        │
        ▼
method(r, z, a, Q)
  → (E_r, E_z)  [φ component = 0 by symmetry]
        │
        ▼
toCart(E_r, φ, E_z)
  → (ΔEx, ΔEy, ΔEz) contribution from this coil
        │
        ▼
Accumulate into Ex[i,j,k], Ey[i,j,k], Ez[i,j,k]
```

---

## Data Flow Summary

```
fill_eext_file(filepath, method, dia, offset, Q, L, N)
        │
        ├─ [renamed file exists?] ──yes──→ return cached path
        │
        └─ no
           │
           ├─ get_e_field_data(method, dia, offset, Q, L, N)
           │   │
           │   ├─ make_polywell_collection(Q, dia, offset)  [6 coils]
           │   │
           │   └─ for each (i,j,k) in N×N×N, for each coil:
           │       orient_point → toCyl → method → toCart → accumulate
           │
           ├─ _fill_efield_datasets(filepath, Ex, Ey, Ez, ...)
           │       → overwrites E datasets in existing .h5
           │
           └─ filepath.rename(new_filepath)
                   → encodes E-field params in filename
                   → return new_filepath
```
