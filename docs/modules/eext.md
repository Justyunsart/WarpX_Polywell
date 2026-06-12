# `src/warpx_polywell/eext` — External E-field Module

Computes the electric field from a set of charged rings (same geometry as the magnetic coils)
and writes the result into an existing B-field HDF5 file, overwriting the zeroed E-field
placeholder that `bext.py` left behind.

---

## Files

| File | Purpose |
|---|---|
| `src/warpx_polywell/eext/eext.py` | Grid computation and HDF5 population |
| `src/warpx_polywell/eext/methods.py` | Analytic E-field integrands + `EMethods` enum |
| `src/warpx_polywell/eext/potential.py` | Closed-form ring scalar potential φ (elliptic integrals); fully vectorised — typically ~100× faster than the per-point `methods.py` loop for the same physics |

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

### `fill_eext_file(filepath, method, dia, offset, Q, domain, use_potentials=False)` → `Path`

**Main public function.** Computes the E-field grid and writes it into the HDF5 file
that `make_bext_file` produced. Renames the file to include E-field parameters.

```
Parameters:
    filepath       : str or Path       — path to existing B-field .h5 file
    method         : Callable          — one of the functions from methods.py.
                                          Ignored when use_potentials=True
                                          (the φ → -∇φ pipeline takes over).
    dia            : float             — ring diameter (m)
    offset         : float             — ring center offset from origin (m)
    Q              : float             — total charge per ring (C)
    domain         : warpx_polywell.domain.Domain — simulated-domain spec
    use_potentials : bool              — if True, build E by summing the
                                          closed-form ring potential φ from
                                          warpx_polywell.eext.potential.compute_phi_grid
                                          over 6 polywell rings and taking
                                          E = -∇φ. The appended filename
                                          segment carries a `_potentials` tag.

Returns:
    pathlib.Path — path to the updated (and renamed) .h5 file
```

**Caching:** Before computing anything, constructs the expected output filename
and returns immediately if it already exists. The appended segment depends on
`use_potentials`:

```
# use_potentials=False (default)
{original_stem}_E_ext_Q-{Q}_D-{dia}m_offset-{offset}m_C_L-{domain.L}m_N-{domain.N}.h5

# use_potentials=True
{original_stem}_E_ext_potentials_Q-{Q}_D-{dia}m_offset-{offset}m_C_L-{domain.L}m_N-{domain.N}.h5
```

The E filename does not carry its own `_sym-…` token because the
`{original_stem}` (from the B file) already encodes the symmetry. The
`_potentials` tag prevents collision between the analytic-E and
`φ → -∇φ` caches.

**Pipeline when file does not exist:**
1. If `use_potentials=False`: call `get_e_field_data(method, ..., domain)` —
   per-point loop over `methods.py::fw_e`/`bob_e`. Else: call
   `_get_e_field_data_via_potentials(dia, offset, Q, domain)` — vectorised
   `compute_phi_grid` followed by `compute_E_from_phi`.
2. Call `_fill_efield_datasets(filepath, ..., grid_offset=domain.lower)` to write into the HDF5
3. Rename the file to append E-field parameters to the name

---

### `get_e_field_data(method, dia, offset, Q, domain)` → `(Ex, Ey, Ez, grid_spacing)`

Computes the 3D E-field grid by iterating over every grid point in the simulated domain and accumulating contributions from all 6 charged rings.

```
Parameters:
    method : Callable          — analytic method (fw_e or bob_e)
    dia    : float             — ring diameter (m)
    offset : float             — ring center distance from origin (m)
    Q      : float             — total charge per ring (C)
    domain : warpx_polywell.domain.Domain — simulated-domain spec

Returns:
    Ex, Ey, Ez   : ndarray of shape domain.n_cells — E-field components (V/m)
    grid_spacing : list[float] — [dx, dy, dz]
```

In octant mode this does (N/2)³ work instead of N³ — 8× faster naturally, with no special code path.

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
fill_eext_file(filepath, method, dia, offset, Q, domain)
        │
        ├─ [renamed file exists?] ──yes──→ return cached path
        │
        └─ no
           │
           ├─ get_e_field_data(method, dia, offset, Q, domain)
           │   │
           │   ├─ make_polywell_collection(Q, dia, offset)  [6 coils]
           │   │
           │   └─ for each (i,j,k) in domain.n_cells, for each coil:
           │       orient_point → toCyl → method → toCart → accumulate
           │
           ├─ _fill_efield_datasets(filepath, Ex, Ey, Ez, ..., grid_offset=domain.lower)
           │       → overwrites E datasets in existing .h5
           │
           └─ filepath.rename(new_filepath)
                   → encodes E-field params in filename
                   → return new_filepath
```

---

## `potential.py`

Closed-form **scalar potential φ** from 6 polywell-arranged charged rings,
fully vectorised over the WarpX grid. When `fill_eext_file(...,
use_potentials=True)` is invoked, this module replaces the per-point
`methods.py` loop entirely.

### Physics

For a single charged ring of radius `a` carrying total charge `Q`, in its
local cylindrical frame:

```
k² = 4 a ρ / [(a + ρ)² + z²]
φ(ρ, z) = Q · K(k) / (2 π² ε₀ √((a + ρ)² + z²))
```

where `K(k)` is the complete elliptic integral of the first kind
(`scipy.special.ellipk(m)` with `m = k²`). On-axis (`ρ → 0`) this collapses
to the familiar `Q / (4πε₀ √(a² + z²))`.

### `phi_ring_local(rho, z, a, Q)` → ndarray

Vectorised single-ring φ in the ring's local cylindrical coordinates.
`rho` and `z` can be arrays of any matching shape.

### `compute_phi_grid(Q, dia, offset, domain)` → `(phi, (dx, dy, dz))`

Superposes φ from 6 polywell-placed rings on the WarpX grid:

1. Builds the magpylib `Collection` via `make_polywell_collection(Q, dia, offset)`
   — used only for the ring positions and orientations (same convention as
   the existing `get_e_field_data` in `eext.py`; the per-coil charge magnitude
   carried by the Collection is ignored and `+Q` is applied uniformly).
2. For each ring, transforms all grid points into the ring's local frame
   via `c.orientation.inv().apply(pts_lab - c.position)`.
3. Accumulates `phi_ring_local(ρ, z, a, Q)` into a flat array.
4. Reshapes back to `(nx, ny, nz)` and returns alongside the grid spacing.

The vectorisation is per-ring rather than per-point: each ring iteration
evaluates φ at *all* grid points at once via NumPy + `scipy.special.ellipk`.

### `compute_E_from_phi(phi, spacing)` → `(Ex, Ey, Ez)`

`E = -∇φ` via second-order central differences (`np.gradient`).

### Why bother

The existing `get_e_field_data` loops over grid points and over coils in
Python — `O(N³ × 6)` `fw_e`/`bob_e` calls. For `N = 64`, that's ~1.6M
function calls and dominates the field-cache build time. The potentials
pipeline expresses the same physics (sum of 6 oriented-ring fields) as
vectorised numpy and is typically ~100× faster at `N = 64` — and the
speedup grows with N.

---
