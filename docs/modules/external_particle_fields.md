# External Particle Field Modes

This document describes how WarpX applies external electromagnetic fields to
particles in the polywell simulation, covering both the **file-based** and
**analytic** pipelines. It includes the underlying physics, the developer API,
and a practical guide for choosing and configuring each mode.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Physics Background](#2-physics-background)
3. [Mode 1 — File-Based (`read_from_file`)](#3-mode-1--file-based-read_from_file)
4. [Mode 2 — Analytic (`parse_B_ext_particle_function`)](#4-mode-2--analytic-parse_b_ext_particle_function)
4.5. [Mode 3 — Potentials (Hybrid-PIC, `external_vector_potential`)](#45-mode-3--potentials-hybrid-pic-external_vector_potential)
5. [API Reference](#5-api-reference)
6. [User Guide — Choosing a Mode](#6-user-guide--choosing-a-mode)
7. [Conventions and Pitfalls](#7-conventions-and-pitfalls)

---

## 1. Overview

WarpX supports three styles for applying an external B-field. The first two
apply B directly to particles each step; the third (Hybrid-PIC) feeds the
**vector potential A** into the field solver, which curls it internally to
produce the B used by Ohm's law.

| Style | WarpX parameter / namespace | How it works |
|---|---|---|
| **File-based** | `particles.B_ext_particle_init_style = "read_from_file"` | Pre-compute B on a 3D grid, store in an openPMD HDF5 file. WarpX loads the grid at startup and interpolates to each particle position every step. |
| **Analytic** | `particles.B_ext_particle_init_style = "parse_B_ext_particle_function"` | Provide a math expression string for each of Bx, By, Bz as a function of `(x, y, z)`. WarpX evaluates the expression exactly at each particle position every step. No file, no interpolation. |
| **Potentials** (Hybrid-PIC only) | `external_vector_potential.<name>.read_from_file = 1` + `…path = <openPMD .h5>` | Pre-compute A on a 3D grid via Coulomb-gauge FFT curl-inverse of the magpylib B, store as a Wb/m `A` mesh in an openPMD file. WarpX's hybrid solver reads A and computes `B = ∇×A` internally each step. |

The `setup_bext()` dispatcher in `src/bext/bext.py` handles all wiring
automatically. For the potentials path it forces `use_potentials=True`
whenever `solver="hybrid"` is passed (so the cache filename and the
ParmParse keys stay consistent) and calls `_wire_hybrid_external_A` to
populate the `hybrid_pic_model` and `external_vector_potential` Buckets.

---

## 2. Physics Background

### 2.1 Single Current Loop — Elliptic Integral Solution

The magnetic field from a single circular current loop of radius *a* carrying
current *I* is expressed exactly in cylindrical coordinates (ρ, ζ) centered on
the coil axis, where ρ is the radial distance from the axis and ζ is the axial
distance from the coil center.

Define the intermediate quantities:

```
α² = (ρ − a)² + ζ²
β² = (ρ + a)² + ζ²
k² = 4aρ / β²          (the elliptic parameter m = k²)
```

Then the field components are:

```
C = μ₀I / (2π)

B_ζ = C / √β² · [ K(m) + (a² − ρ² − ζ²) / α² · E(m) ]

B_ρ = C · ζ / (ρ√β²) · [ −K(m) + (a² + ρ² + ζ²) / α² · E(m) ]
```

where K(m) and E(m) are the complete elliptic integrals of the first and second
kind, evaluated at the parameter m = k².

This is exact (no approximation) and valid everywhere except on the wire itself
(ρ = a, ζ = 0). The on-axis case (ρ → 0) is handled by setting B_ρ = 0 when
ρ is below a small threshold.

### 2.2 Coordinate Remapping — One Kernel for All Axes

A coil aligned with the Z-axis naturally uses ρ² = x² + y² and ζ = z − z₀. For
coils on the X or Y axis, we remap which lab coordinates feed into ρ and ζ:

| Coil axis | ρ² | ζ | cos θ | sin θ |
|---|---|---|---|---|
| Z | x² + y² | z − pos | x/ρ | y/ρ |
| X | z² + y² | x − pos | z/ρ | y/ρ |
| Y | z² + x² | y − pos | z/ρ | x/ρ |

After computing B_ρ and B_ζ (the "axial" component) in the coil's local
cylindrical frame, we project back to Cartesian:

| Coil axis | Bx | By | Bz |
|---|---|---|---|
| Z | B_ρ cos θ | B_ρ sin θ | B_ζ |
| X | B_ζ | B_ρ sin θ | B_ρ cos θ |
| Y | B_ρ sin θ | B_ζ | B_ρ cos θ |

This means the exact same kernel function handles all 6 coils — only the
coordinate mapping changes.

### 2.3 Polywell Superposition

The total field is a linear superposition of all 6 coils:

```
B_total(x,y,z) = Σᵢ Bᵢ(x,y,z)
```

Each coil has an axis, signed position, and signed current defined in the
`POLYWELL_COILS` constant:

| Coil | Axis | Position | Current |
|---|---|---|---|
| s1 | X | −offset | +I |
| s2 | X | +offset | −I |
| s3 | Y | −offset | −I |
| s4 | Y | +offset | +I |
| s5 | Z | −offset | +I |
| s6 | Z | +offset | +I |

The alternating current signs create magnetic cusps — the characteristic
polywell field topology.

---

## 3. Mode 1 — File-Based (`read_from_file`)

### How It Works

1. `make_polywell_collection()` builds a `magpylib.Collection` of 6 circular
   current loops with the specified geometry.
2. A 3D Cartesian meshgrid is created: `linspace(−L, L, N)` along each axis.
3. `collection.getB(mesh)` evaluates the field at every grid point.
4. The resulting Bx, By, Bz arrays (each of shape N×N×N) are written into an
   openPMD HDF5 file at `data/1/meshes/B/{x,y,z}`.
5. WarpX loads this file at startup and interpolates B to each particle's
   position at every time step.

### Trade-offs

**Advantages:**

- Startup is straightforward — field pre-computation is a one-time cost
  (results are cached by filename).
- The file can be inspected with `openPMD-viewer` or `h5dump` for debugging.
- E-field data can share the same HDF5 file.

**Disadvantages:**

- Interpolation error: the grid is finite, so B at particle positions is
  interpolated (typically linear or quadratic). Accuracy depends on grid
  resolution N.
- Memory: an N³ grid with 3 components requires 3 × N³ × 8 bytes. For N = 72,
  that's ~9 MB; for N = 256, it's ~400 MB.
- Particles that leave the grid domain see B = 0 (or an extrapolated value),
  which may be unphysical.

### Data Flow

```
User parameters (I, dia, offset, L, N)
    │
    ▼
make_polywell_collection(I, dia, offset)
    │ → magpylib.Collection
    ▼
meshgrid(−L..L, N) + collection.getB()
    │ → Bx, By, Bz arrays (N, N, N)
    ▼
Write to openPMD HDF5
    │ → B_ext_*.h5
    ▼
particles.B_ext_particle_init_style = "read_from_file"
particles.read_fields_from_path = <path to .h5>
    │
    ▼
WarpX loads grid, interpolates to particles each step
```

---

## 4. Mode 2 — Analytic (`parse_B_ext_particle_function`)

### How It Works

1. `build_bext_expressions(I, dia, offset)` constructs three expression strings
   (one each for Bx, By, Bz) that encode the exact elliptic-integral field from
   all 6 coils.
2. These strings are assigned to WarpX's parser function slots:
   - `particles.Bx_external_particle_function = exprs['Bx']`
   - `particles.By_external_particle_function = exprs['By']`
   - `particles.Bz_external_particle_function = exprs['Bz']`
3. At runtime, AMReX's built-in parser evaluates each expression at every
   particle's (x, y, z) position, every time step.

There is no grid, no file, no interpolation. The field is computed exactly from
the elliptic integral formulas at each particle's precise location.

### Expression Structure

Each expression uses AMReX's **local variable syntax** — semicolon-separated
assignments where the final expression (after the last semicolon) is the
returned value:

```
var1=expr1; var2=expr2; ...; final_value
```

For the 6-coil polywell, the expression is structured as:

```
┌─────────────────────────────────────────────────────────┐
│  Coil 1 variables (13 assignments):                     │
│    r2_1=z*z+y*y;  r_1=sqrt(r2_1+1e-30);               │
│    z_1=x-(-1.1e+00);  ct_1=z/(r_1+1e-30);             │
│    st_1=y/(r_1+1e-30);                                  │
│    a2_1=(r_1-0.5)**2+z_1**2;                            │
│    b2_1=(r_1+0.5)**2+z_1**2;                            │
│    k_1=sqrt(min(4*0.5*r_1/b2_1, 0.9999999));           │
│    K_1=comp_ellint_1(k_1);                              │
│    E_1=comp_ellint_2(k_1);                              │
│    sb_1=sqrt(b2_1);                                     │
│    Bax_1=...;  Br_1=if(r2_1>1e-24, ..., 0.0);         │
│                                                         │
│  Coil 2 variables ... (same pattern, suffix _2)         │
│  ...                                                    │
│  Coil 6 variables ... (suffix _6)                       │
│                                                         │
│  Final sum:                                             │
│    1e+06*(Bax_1) + -1e+06*(Bax_2) + ...                │
└─────────────────────────────────────────────────────────┘
```

Each coil has uniquely suffixed variable names (`_1` through `_6`) because all
local variables share a flat namespace within one expression. The current (with
sign) is applied as a numeric prefactor in the final summation.

### AMReX Built-in Functions Used

| Function | Description |
|---|---|
| `comp_ellint_1(k)` | Complete elliptic integral of the first kind, K(k) |
| `comp_ellint_2(k)` | Complete elliptic integral of the second kind, E(k) |
| `sqrt(x)` | Square root |
| `min(a, b)` | Minimum of two values |
| `if(cond, t, f)` | Conditional: returns t if cond > 0, else f |

### Trade-offs

**Advantages:**

- Exact: no interpolation error, no grid resolution limitations.
- No files: nothing to generate, cache, or manage on disk.
- Unbounded domain: works correctly at any (x, y, z) — no grid boundary issues.
- Compact: ~3.6 KB per component (total ~11 KB for all three).

**Disadvantages:**

- Per-particle cost: the expression is evaluated for every particle at every
  time step. For very large particle counts, this may be slower than a grid
  lookup with interpolation.
- Six elliptic integral evaluations per particle per component per step (one per
  coil). Each elliptic integral involves an internal iterative computation.
- The E-field still requires its own file (the analytic pipeline only covers B).
  A zero-current scaffold HDF5 is created for E-field data when using analytic B.

### Data Flow

```
User parameters (I, dia, offset)
    │
    ▼
build_bext_expressions(I, dia, offset)
    │ → {'Bx': "r2_1=...; ...", 'By': "...", 'Bz': "..."}
    ▼
particles.B_ext_particle_init_style = "parse_B_ext_particle_function"
particles.Bx_external_particle_function = exprs['Bx']
particles.By_external_particle_function = exprs['By']
particles.Bz_external_particle_function = exprs['Bz']
    │
    ▼
WarpX evaluates expression at each particle's (x,y,z) every step
```

---

## 4.5. Mode 3 — Potentials (Hybrid-PIC, `external_vector_potential`)

The Hybrid-PIC solver in WarpX consumes the external magnetic field through
a **vector potential A** rather than B directly. To run the polywell with
hybrid PIC we therefore need an openPMD file that contains an `A` mesh in
Wb/m, and we need the `external_vector_potential.<name>.read_from_file/path`
ParmParse keys pointing WarpX at it.

This mode is enabled by setting `use_hybrid = True` (or, equivalently,
`use_potentials = True` plus `solver = "hybrid"`) in
`inputs/polywell_input.py`. It's an **overlay on the file pipeline** — the
same openPMD-file scaffolding from Mode 1 is reused, but the data path is
"A first, then B = ∇×A" instead of "B straight from magpylib."

### How It Works

magpylib's API exposes `getB / getH / getJ / getM` but no `getA`. The
cleanest way to recover A for an arbitrary coil arrangement is to invert
the curl in Coulomb gauge, which in Fourier space is a one-liner:

```
∇·A = 0           ⇔   k · Ã(k) = 0          (Coulomb gauge)
∇ × A = B         ⇒   Ã(k) = i (k × B̃(k)) / |k|²
```

with `Ã(0) = 0` chosen by gauge (corresponds to `A → 0 at infinity`).

`np.fft.fftn` enforces periodic boundary conditions on the FFT box, so an
isolated coil in a finite domain gets repeated into an infinite lattice of
image coils. The cure is zero-padding: evaluate B on a box `pad_factor`×
larger than the physics region (with `B = 0` in the pad region), invert in
Fourier space, then crop back. Doubling `pad_factor` until A in the
physics region stops moving gives a clean convergence test.

`src/bext/vector_potential.py` implements all of this:

| Function | Purpose |
|---|---|
| `compute_A_grid(collection, domain, pad_factor)` | One pad → magpylib `getB` → FFT curl-inverse → crop pass. |
| `converge_A_grid(...)` | Doubles `pad_factor` (`2 → 4 → 8`) until A in the physics region is stable to `rtol`. Caps at `max_pad`. |
| `curl_A(Ax, Ay, Az, spacing)` | `B = ∇×A` via central differences. Used both to derive B for the openPMD file and to verify against magpylib. |
| `check_curl(...)` | Sanity-check helper that returns the relative L∞ error between `curl_A(A)` and magpylib's direct B. |

### WarpX wiring (driven by `_wire_hybrid_external_A`)

When `setup_bext` is called with `solver="hybrid"` it auto-flips
`use_potentials = True` (logging a notice if it had been False) and calls
`_wire_hybrid_external_A(ext_path)`, which drives the
`pywarpx.hybridpicmodel` and `pywarpx.external_vector_potential` Buckets
to set:

```
hybrid_pic_model.add_external_fields = 1
external_vector_potential.fields = polywell
external_vector_potential.do_diva_cleaning = 0
external_vector_potential.polywell.read_from_file = 1
external_vector_potential.polywell.path = <ext_path>
external_vector_potential.polywell.A_time_external_grid_function(t) = 1
```

- `A_time_external_grid_function(t) = 1` keeps the field static.
- `do_diva_cleaning = 0` because the FFT curl-inverse already enforces
  `∇·A = 0` by construction — no extra cleaning needed.

### Trade-offs

| Property | Notes |
|---|---|
| **Required for Hybrid-PIC?** | Yes. The hybrid solver does not accept external `B_ext_particle_*` keys; A from `external_vector_potential` is the only path. |
| **Per-step cost** | Same as Mode 1 — WarpX interpolates A once at startup; `B = ∇×A` is recomputed internally each step (cheap compared to the particle push). |
| **Generation cost** | Higher than Mode 1: the FFT box is `(pad_factor·N)³`, so memory and FFT time scale with `pad³`. Production cap is `pad_factor = 8`. |
| **Coulomb gauge** | Enforced spectrally by construction; verified at machine precision (~10⁻¹⁵) in the plasma cube by [`tests/test_vector_potential.py`](../../tests/test_vector_potential.py). |
| **B reconstruction accuracy** | 99.1% of cells within 5% of peak \|B\| at N=32, monotonically improving with N. Coil-adjacent cells have unbounded L∞ error (1/r³ singularity — fundamental FD limit, not pipeline defect). |
| **Cache filename** | Carries `_potentials` tag so caches from this mode never collide with magpylib-direct ones, even at identical parameters. |

### E-field counterpart

When `use_potentials` is True, `fill_eext_file` similarly switches to a
**closed-form scalar potential φ** pipeline (`src/eext/potential.py`):
`compute_phi_grid(Q, dia, offset, domain)` superposes
`φ = Q·K(k) / (2π²ε₀√((a+ρ)² + z²))` from 6 polywell-arranged rings,
vectorised over the whole grid, then `compute_E_from_phi(phi, dx)`
returns `E = -∇φ`. Same physics as the per-point `methods.py::fw_e/bob_e`
loop, ~100× faster at N = 64.

### Data Flow

```
b_method = "file" + use_hybrid = True  (i.e., use_potentials = True, solver = "hybrid")
                            │
                            ▼
setup_bext("file", ..., solver="hybrid", use_potentials=True)
                            │
                            ▼
make_bext_file(I, dia, offset, domain, use_potentials=True)
                            │
                            ▼
_compute_b_via_potentials(I, dia, offset, domain)
            │
            ├─ make_polywell_collection(I, dia, offset)
            │
            ├─ converge_A_grid(collection, domain)
            │     │  pad=2 → 4 → 8 (until rel-change < rtol)
            │     │
            │     └─ per pad:
            │         _padded_grid → magpylib.getB → _coulomb_gauge_A_from_B → crop
            │             → Ax, Ay, Az on physics grid
            │
            └─ curl_A(Ax, Ay, Az, spacing)
                  → Bx, By, Bz on physics grid (∇×A central differences)
                            │
                            ▼
_make_empty_ext_h5(file_path)    [creates B, E, A mesh groups]
_fill_h5_file(..., Ax=Ax, Ay=Ay, Az=Az)
            → openPMD .h5 with populated B and A meshes
                            │
                            ▼
_wire_hybrid_external_A(ext_path)
            → ParmParse keys set on pywarpx Buckets
                            │
                            ▼
WarpX hybrid solver reads A from ext_path, curls it internally each step
```

---

## 5. API Reference

### `setup_bext(method, particles, warpx_module, *, I, dia, offset, domain)`

The single entry point for configuring external B-fields. Located in
`src/bext/bext.py`.

```
Parameters:
    method       : str — "file" or "analytic"
    particles    : pywarpx.particles module
    warpx_module : pywarpx.warpx module (optional, reserved for future use)
    I            : float          — coil current (A)
    dia          : float          — coil diameter (m)
    offset       : float          — coil center distance from origin (m)
    domain       : src.domain.Domain — simulated-domain spec; required for
                   "file" mode (carries bounds/n_cells/symmetry), ignored by
                   "analytic" mode (closed-form expressions are coordinate-agnostic).

Returns:
    str or None — path to the .h5 file (file mode) or None (analytic mode)
```

The `domain` argument lets file mode automatically sample on `(N, N, N)` (full) or `(N/2, N/2, N/2)` (octant) without any conditional logic in the caller. See [domain module](domain.md) for the dataclass.

**What it does internally:**

| Method | Actions performed |
|---|---|
| `"file"` | Sets `B_ext_particle_init_style = "read_from_file"`, calls `make_bext_file()` to generate or retrieve the cached HDF5 file, and sets `read_fields_from_path`. |
| `"analytic"` | Sets `B_ext_particle_init_style = "parse_B_ext_particle_function"`, calls `build_bext_expressions()` to generate the expression strings, and assigns them to `Bx/By/Bz_external_particle_function`. |

### `build_bext_expressions(I, dia, offset)` → `dict`

Located in `src/bext/analytic.py`. Builds the three AMReX parser expression
strings for a 6-coil polywell.

```
Parameters:
    I      : float — coil current (A)
    dia    : float — coil diameter (m)
    offset : float — coil center distance from origin (m)

Returns:
    dict with keys 'Bx', 'By', 'Bz'
    Each value is a parser-ready string using AMReX local variable syntax.
```

### `B_polywell(X, Y, Z, I, dia, offset)` → `(Bx, By, Bz)`

Located in `src/bext/analytic.py`. NumPy evaluation of the same field (for
testing, plotting, and validation — not used by WarpX at runtime).

```
Parameters:
    X, Y, Z : array_like — observation coordinates
    I       : float — coil current (A)
    dia     : float — coil diameter (m)
    offset  : float — coil center distance from origin (m)

Returns:
    (Bx, By, Bz) : numpy arrays — Cartesian field components
```

### `B_single_loop(rho, zeta, a, I)` → `(B_rho, B_zeta)`

Located in `src/bext/analytic.py`. Core kernel — computes the cylindrical field
components from a single current loop using scipy's `ellipk(m)` and `ellipe(m)`.

```
Parameters:
    rho  : array_like — radial distance from coil axis
    zeta : array_like — axial distance from coil center
    a    : float — coil radius (m)
    I    : float — current (A)

Returns:
    (B_rho, B_zeta) : numpy arrays
```

### Internal Helpers (Private)

| Function | Purpose |
|---|---|
| `_coil_var_defs(tag, axis, pos, a)` | Generates 13 local-variable assignment strings for one coil (geometry, elliptic integrals, cylindrical B components) |
| `_coil_cartesian_term(tag, axis, component, I_val)` | Returns the expression fragment projecting one coil's cylindrical field into a Cartesian component, with current prefactor |
| `_eval_loop_cartesian(X, Y, Z, axis, pos, a, I)` | NumPy evaluation of one coil's Cartesian field (used by `B_polywell`) |

---

## 6. User Guide — Choosing a Mode

### Switching Modes

In `inputs/polywell_input.py`, change one line:

```python
b_method = "analytic"   # exact per-particle evaluation, no grid file
# or
b_method = "file"       # pre-computed grid, interpolated to particles
```

Everything else is handled automatically by `setup_bext()`.

### When to Use Each Mode

| Scenario | Recommended mode | Reason |
|---|---|---|
| Development and testing | `"analytic"` | No file generation step, instant iteration, exact fields |
| Very high particle counts (> 10⁸) | `"file"` | Grid interpolation may be faster per-particle than evaluating 6 elliptic integrals |
| Need to inspect the field visually before running | `"file"` | The HDF5 file can be loaded in openPMD-viewer or ParaView |
| Particles near or beyond the grid boundary | `"analytic"` | No grid edge artifacts — valid everywhere |
| Convergence studies (varying grid resolution) | `"analytic"` | Eliminates interpolation as a source of error |
| Comparing analytic vs grid fields | Both | Run with each mode and diff the particle trajectories |

### E-Field Interaction

The E-field pipeline is currently file-based only. When using analytic B:

- A "scaffold" HDF5 file is created with zero B-field data so the E-field
  pipeline has a file to write into.
- The E-field data is appended to this scaffold file.
- WarpX reads E from the file while evaluating B from the parser expressions.

This means both `read_from_file` (for E) and `parse_B_ext_particle_function` (for B) are
active simultaneously — WarpX handles this correctly because the two field types
(B and E) are configured through separate parameter families
(`B_ext_particle_init_style` vs `E_ext_particle_init_style`).

### Diagnostic Output

Both modes produce the same diagnostic output. The field diagnostics
(`Bx_fp_external`, `By_fp_external`, `Bz_fp_external`) record whatever external
field WarpX applied to particles, regardless of whether it came from a file or
an expression.

---

## 7. Conventions and Pitfalls

### Elliptic Integral Convention

This is the single most important convention to be aware of when working with
the analytic expressions:

| Library | Function | Argument | Notation |
|---|---|---|---|
| scipy | `ellipk(m)` | parameter m = k² | K(m) |
| scipy | `ellipe(m)` | parameter m = k² | E(m) |
| AMReX | `comp_ellint_1(k)` | modulus k = √m | K(k) |
| AMReX | `comp_ellint_2(k)` | modulus k = √m | E(k) |

The code handles this by computing `k = sqrt(4aρ/β²)` for the AMReX expressions
(passing the modulus), while the NumPy kernel computes `m = 4aρ/β²` and passes
it to `ellipk(m)` (the parameter). Both produce the same result.

Mixing these up silently produces wrong field values with no error — be careful
when modifying the expressions.

### On-Axis Singularity

At ρ = 0 (on the coil axis), the radial field B_ρ is physically zero but the
formula has a 1/ρ factor that diverges numerically. Both the NumPy kernel and
the parser expressions handle this:

- NumPy: `np.where(abs(rho) < 1e-12, 0.0, B_rho)`
- Parser: `if(r2 > 1e-24, <B_r expression>, 0.0)`

Additionally, `sqrt(rho2 + 1e-30)` prevents division by zero in intermediate
calculations.

### Variable Naming in Parser Expressions

All 6 coils' local variables share a flat namespace within one expression. Each
coil uses a numeric suffix (`_1` through `_6`) to avoid name collisions:

```
r2_1, r_1, z_1, ct_1, st_1, a2_1, b2_1, k_1, K_1, E_1, sb_1, Bax_1, Br_1
r2_2, r_2, z_2, ct_2, st_2, a2_2, b2_2, k_2, K_2, E_2, sb_2, Bax_2, Br_2
...
```

If you add more coils, use the next available suffix. Do not reuse variable
names — AMReX's parser treats them as a flat list of sequential assignments.

### Expression Size

With native AMReX elliptic integral functions, each component expression is
approximately 3.6 KB (78 local variable assignments + final summation). An
earlier implementation using Abramowitz & Stegun polynomial approximations
produced ~28 KB expressions. The native version is both smaller and exact.

### Validation

The analytic expressions have been cross-validated against the NumPy kernel at
1000+ random points in the simulation domain. The maximum relative error is
~6.5 × 10⁻¹⁵ (machine precision), confirming that the parser expressions and
the NumPy kernel compute the same field.

The validation notebook at `tests/coil_field_analysis/Coil_Field_Analysis.ipynb`
provides additional verification including on-axis closed-form comparisons,
Helmholtz pair tests, divergence checks, and symmetry validation.
