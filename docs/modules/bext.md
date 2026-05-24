# `src/bext` — External B-field Module

Provides the external magnetic field for WarpX polywell simulations. Supports
three modes selectable at runtime: a **file-based** pipeline (pre-computed B
grid in an openPMD HDF5 file), an **analytic** pipeline (exact
elliptic-integral expressions evaluated per-particle), and a **potentials**
overlay on the file pipeline (build the same HDF5 from A via Coulomb-gauge
FFT curl-inverse instead of taking B directly from magpylib — required for
the Hybrid-PIC solver, which consumes A rather than B). See
[External Particle Field Modes](external_particle_fields.md) for the full
conceptual background, physics, and user guide.

---

## Files

| File | Purpose |
|---|---|
| `src/bext/bext.py` | `setup_bext()` dispatcher + file-based HDF5 creation + Hybrid-PIC `external_vector_potential` wiring |
| `src/bext/analytic.py` | Analytic elliptic-integral kernel (NumPy + AMReX parser expressions) |
| `src/bext/vector_potential.py` | A via Coulomb-gauge FFT curl-inverse on a zero-padded grid (potentials overlay; required for Hybrid-PIC) |
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

### `setup_bext(method, particles, warpx_module, *, I, dia, offset, domain, solver=None, use_potentials=False)`

**Modular dispatcher.** Configures WarpX's external B-field for either file-based
or analytic mode. This is the only function the input deck needs to call.

```
Parameters:
    method         : str — "file" or "analytic"
    particles      : pywarpx.particles module
    warpx_module   : pywarpx.warpx module (optional)
    I, dia, offset : coil parameters
    domain         : src.domain.Domain — simulated-domain spec
                     (required for "file" mode; ignored by "analytic")
    solver         : "hybrid" or None — when "hybrid", forces use_potentials=True
                     and wires `external_vector_potential.*` ParmParse keys
                     so WarpX reads A from the generated openPMD file
    use_potentials : bool — if True, the file pipeline computes A via
                     Coulomb-gauge FFT curl-inverse (src.bext.vector_potential)
                     and derives B = ∇×A instead of taking B directly from
                     magpylib. Generated filename carries a `_potentials` tag
                     so caches from the two pipelines never collide. Ignored
                     by the "analytic" method.

Returns:
    str or None — .h5 path (file mode) or None (analytic mode)
```

The `domain` parameter carries the simulated bounds, cell count, and the symmetry tag.
The grid sampled in file mode is `linspace(domain.lower[i], domain.upper[i], domain.n_cells[i])`
per axis — automatically `(N, N, N)` in full mode and `(N/2, N/2, N/2)` in octant mode.

When `solver="hybrid"`, the dispatcher also calls `_wire_hybrid_external_A(ext_path)`
(see below) so WarpX's hybrid solver reads A from the file via
`external_vector_potential.polywell.{read_from_file, path}`.

See [External Particle Field Modes](external_particle_fields.md) for detailed
behavior of each mode, and [domain module](domain.md) for the `Domain` dataclass.

### `get_bext_file_name(I, dia, offset, domain, use_potentials=False)` → `str`

Returns the expected filename for a given set of parameters. Used to check for cached files.

```python
# Example — default (magpylib-direct B)
from src.domain import derive_domain
d = derive_domain("full", 2, 72)
name = get_bext_file_name(1e6, 1.0, 1.1, d)
# → "B_ext_I-1000000.0A_D-1m_Off-1.1m_L-2m_N-72_sym-full.h5"

# Same parameters but built via the potentials pipeline:
name = get_bext_file_name(1e6, 1.0, 1.1, d, use_potentials=True)
# → "B_ext_potentials_I-1000000.0A_D-1m_Off-1.1m_L-2m_N-72_sym-full.h5"
```

The `_sym-…` token is essential because the sampled grid differs between
modes at the same `L`/`N`. The `_potentials` tag (when present) prevents
collision between magpylib-direct and FFT-curl-inverse caches.

### `make_bext_file(I, dia, offset, domain, use_potentials=False)` → `Path`

**Main public function.** Checks whether a cached file already exists; if not, computes
and writes a new one.

```
Parameters:
    I              : float            — coil current (A)
    dia            : float            — coil diameter (m)
    offset         : float            — coil center distance from origin (m)
    domain         : src.domain.Domain — simulated-domain spec
    use_potentials : bool             — if True, build B via FFT curl-inverse
                                        of A (see _compute_b_via_potentials)
                                        instead of taking B directly from
                                        magpylib; filename gets `_potentials` tag

Returns:
    pathlib.Path — absolute path to the .h5 file
```

**Default pipeline (`use_potentials=False`), when file does not exist:**

1. Call `make_polywell_collection(I, dia, offset)` to build the magpylib coil set
2. Create a 3D meshgrid: `linspace(domain.lower[i], domain.upper[i], domain.n_cells[i])` per axis (indexing `'ij'`)
3. Reshape mesh to `(Nx, Ny, Nz, 3)` as expected by magpylib's `getB()`
4. Call `collection.getB(mesh)` — vectorised field evaluation over all grid points
5. Decompose result into `Bx, By, Bz` arrays
6. Call `_make_empty_ext_h5(file_path)` to create the skeleton HDF5
7. Call `_fill_h5_file(...)` with `grid_offset = domain.lower` to write the B-field data (E-field set to zeros as placeholder; A mesh left empty)

**Potentials pipeline (`use_potentials=True`):**

1. Call `_compute_b_via_potentials(I, dia, offset, domain)`, which itself calls
   `converge_A_grid(...)` from `vector_potential.py` to solve for A on the
   physics grid, then derives `Bx, By, Bz` via `curl_A` (central differences)
2. Call `_make_empty_ext_h5(file_path)` to create the skeleton HDF5
3. Call `_fill_h5_file(..., Ax=Ax, Ay=Ay, Az=Az)` — writes B *and* A meshes

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

**Per-field-group attributes (`B`, `E`, and `A`):**

| Attribute | Value |
|---|---|
| `geometry` | `"cartesian"` |
| `gridSpacing` | `[1.0, 1.0, 1.0]` (placeholder, overwritten later) |
| `gridGlobalOffset` | `[0.0, 0.0, 0.0]` (placeholder, overwritten later) |
| `unitDimension` | B: `[0, 1, 1, -2, 0, 0, -1]` (Tesla); E: `[1, 1, -3, -1, 0, 0, 0]` (V/m); A: `[1, 1, -2, -1, 0, 0, 0]` (Wb/m) |

The `A` mesh is always created (kept empty by default) so all generated files
share a uniform schema. It's populated only when `_fill_h5_file` is called with
`Ax/Ay/Az` arrays — i.e. when `make_bext_file` runs the potentials pipeline.

### `_fill_h5_file(filepath, Bx, By, Bz, grid_spacing, grid_offset, Ax=None, Ay=None, Az=None)` (private)

Writes actual B-field data and zeroed E-field placeholders into an existing skeleton file.

- Updates `gridSpacing` and `gridGlobalOffset` on B and E groups (and A
  when `Ax` is provided)
- Creates `data/1/meshes/B/{x,y,z}` datasets from `Bx, By, Bz`
- Creates `data/1/meshes/E/{x,y,z}` datasets as `zeros_like(Bx)`
- When `Ax/Ay/Az` are provided, creates `data/1/meshes/A/{x,y,z}` from them
- Each dataset gets `unitSI=1.0` and `position=[0.5, 0.5, 0.5]` (cell-centered)

### `_compute_b_via_potentials(I, dia, offset, domain)` (private)

Returns `(Bx, By, Bz, Ax, Ay, Az, [dx, dy, dz])` for the potentials pipeline.
Calls `converge_A_grid(...)` from `vector_potential.py` (which sweeps
`pad_factor 2 → 4 → 8` until A in the physics region is stable), then
derives `B = ∇×A` via `curl_A`. Both A and B are returned so
`make_bext_file` can write both meshes.

### `_wire_hybrid_external_A(ext_path)` (private)

Drives `pywarpx.hybridpicmodel` and `pywarpx.external_vector_potential`
Buckets so WarpX's hybrid solver reads A from the openPMD file at startup.
Equivalent to writing this ParmParse block:

```
hybrid_pic_model.add_external_fields = 1
external_vector_potential.fields = polywell
external_vector_potential.do_diva_cleaning = 0
external_vector_potential.polywell.read_from_file = 1
external_vector_potential.polywell.path = <ext_path>
external_vector_potential.polywell.A_time_external_grid_function(t) = 1
```

- `A_time_external_grid_function(t) = 1` makes the field static.
- `do_diva_cleaning = 0` because the FFT curl-inverse already enforces
  ∇·A = 0 (Coulomb gauge) by construction; no extra cleaning needed.
- The per-field sub-keys (`polywell.read_from_file`, `polywell.path`,
  `polywell.A_time_external_grid_function(t)`) aren't legal Python
  attribute names, so they're set via `setattr(bucket, "key", value)` —
  the same idiom `setup_bext` uses for
  `particles.Bx_external_particle_function(x,y,z,t)`.

Called automatically from `setup_bext` when `solver="hybrid"`.

---

## `vector_potential.py`

Recovers the vector potential A on the WarpX grid from any
magpylib-evaluable B via a **Coulomb-gauge FFT curl-inverse on a
zero-padded box**. Used by `bext.py::_compute_b_via_potentials` (and
therefore by `make_bext_file` whenever `use_potentials=True`).

### Why

magpylib exposes `getB / getH / getJ / getM` but no `getA`. The cleanest
way to get A for an arbitrary coil arrangement is to evaluate B with
magpylib and invert the curl in Coulomb gauge. In Fourier space,

```
∇·A = 0  ⇔  k·Ã = 0
∇×A = B  ⇒  Ã(k) = i (k × B̃) / |k|²
```

with `Ã(0) = 0` chosen by gauge (corresponds to `A → 0 at infinity`).

The catch is that `np.fft.fftn` enforces periodic boundary conditions, so
an isolated coil in a finite box gets replicated into an infinite lattice
of image coils whose A-contributions leak into the interior. The cure is
zero-padding: evaluate B on a box ≥ `pad_factor`× larger than the physics
region, invert in Fourier space, then crop back. A is converged in `pad_factor`
when image-coil contamination has decayed below the chosen tolerance.

### `compute_A_grid(collection, domain, pad_factor=2)` → `(Ax, Ay, Az, (dx, dy, dz))`

One-shot pad → magpylib `getB` → FFT curl-inverse → crop. Returns A
components on a grid matching `domain.n_cells`, in **T·m (= Wb/m)**.

### `converge_A_grid(collection, domain, *, start=2, max_pad=8, rtol=1e-3, verbose=False)` → `(Ax, Ay, Az, spacing, pad)`

Doubles `pad_factor` (`start → 2·start → 4·start → …`) until A in the
physics region stops moving by more than `rtol` in L∞ norm relative to
its current peak. Caps at `max_pad`. Returns the converged A plus the
final `pad_factor` used.

`max_pad=8` is the production default — see Test 1 in
[`tests/test_vector_potential.py`](../../tests/test_vector_potential.py)
for the convergence trajectory and why it's enough.

### `curl_A(Ax, Ay, Az, spacing)` → `(Bx, By, Bz)`

Second-order central differences `B = ∇×A`. Reusable helper — used by
both `_compute_b_via_potentials` (to derive B for the openPMD file) and
`check_curl` (for verification).

### `check_curl(Ax, Ay, Az, spacing, collection, domain)` → `float`

Sanity check: compute `∇×A` on the physics grid and compare to magpylib's
direct B at the same points. Returns relative L∞ error. Note that this
metric is **always large at coil-adjacent cells** (central differences of
a 1/r³ source can't reproduce the singularity); use the test suite for a
properly-masked accuracy report.

### Trust and convergence

The pipeline is exercised by 5 tests in
[`tests/test_vector_potential.py`](../../tests/test_vector_potential.py),
all of which pass:

1. **Padding convergence** — `|A|_max` rel-change at pad=4→8 is < 0.1%.
2. **∇×A vs magpylib B** — 99.1% of cells within 5% of peak |B|, 97% within 1%.
3. **Coulomb gauge** — `‖∇·A‖_∞ / max‖∇×A‖_∞ = 9×10⁻¹⁶` in the plasma cube
   (machine precision; the spectral `ik·Ã = 0` enforcement survives IFFT and cropping intact).
4. **Resolution convergence** — cells-within-5%-of-peak grows monotonically
   97.3 → 98.8 → 99.1 → 99.5 % as N goes 16 → 48.
5. **Single isolated ring vs analytic A_φ** — 14% off-ring rel error against
   the closed-form `(μ₀ I / π k) √(a/ρ) [(1 - k²/2) K(k) - E(k)]`.

Run with `PYTHONPATH=$(pwd) python tests/test_vector_potential.py`.

> **Don't gate on per-cell relative error for the polywell.** The
> polywell has a magnetic null at origin by design (|B| → 0 there), so
> any per-cell `|a - b| / |b|` blows up to ∞ in the null even when the
> absolute error is tiny. The test suite uses **peak-normalised absolute
> tolerance** (fraction of cells where `|error| < 5% of peak |B|`) and
> RMS rel error — both of which are well-behaved.

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
