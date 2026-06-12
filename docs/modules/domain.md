# `src/warpx_polywell/domain.py` — Simulated-Domain Spec & Symmetry Reduction

A single frozen dataclass collapses every difference between full-domain and octant-symmetry runs into one object. Downstream code (grid construction, field-file generation, particle loading) reads from `Domain` and never branches on the `symmetry` toggle directly.

---

## `Domain` (dataclass, frozen)

| Field | Type | Meaning |
|---|---|---|
| `L` | `float` | User-facing full-domain half-extent (carried through unchanged for filenames and DB) |
| `N` | `int` | User-facing full-domain cell count (carried through unchanged) |
| `symmetry` | `str` | `"full"` or `"octant"` |
| `lower` | `tuple[float, float, float]` | Simulated-domain lower bound |
| `upper` | `tuple[float, float, float]` | Simulated-domain upper bound |
| `n_cells` | `tuple[int, int, int]` | Cells per axis in the simulated domain |
| `field_bc_lo` / `field_bc_hi` | `tuple[str, str, str]` | PICMI field BCs per axis |
| `particle_bc_lo` / `particle_bc_hi` | `tuple[str, str, str]` | PICMI particle BCs per axis |

The dataclass is frozen so callers can pass it around without worrying about mutation.

---

## `derive_domain(symmetry, L, N)` → `Domain`

Single entry point for translating the user's toggles into a `Domain`. Validates the inputs and returns one of two configurations:

### `symmetry = "full"`

| Field | Value |
|---|---|
| `lower` / `upper` | `(-L, -L, -L)` / `(+L, +L, +L)` |
| `n_cells` | `(N, N, N)` |
| `field_bc_lo` / `field_bc_hi` | `("open",) × 3` / `("open",) × 3` |
| `particle_bc_lo` / `particle_bc_hi` | `("absorbing",) × 3` / `("absorbing",) × 3` |

### `symmetry = "octant"`

Requires `N` even (`N % 2 == 0`).

| Field | Value |
|---|---|
| `lower` / `upper` | `(0, 0, 0)` / `(+L, +L, +L)` |
| `n_cells` | `(N/2, N/2, N/2)` |
| `field_bc_lo` / `field_bc_hi` | `("pmc",) × 3` / `("open",) × 3` |
| `particle_bc_lo` / `particle_bc_hi` | `("reflecting",) × 3` / `("absorbing",) × 3` |

### Why PMC, not PEC, on the inner faces

A polywell B-field is an axial (pseudo)vector sourced by mirror-symmetric loop currents. Under reflection through a cardinal plane the tangential components of B vanish and the normal component is unconstrained — exactly the boundary conditions WarpX's **PMC** ("Perfect Magnetic Conductor, equivalent to Neumann; models a symmetric surface where charges and currents are symmetric across the boundary") enforces. PEC (tangential E = 0, normal B = 0) is the opposite symmetry class and would zero out the cusp field that physically passes through the symmetry plane.

The particle BC `reflecting` produces the mirror-image current density required by PMC's "currents symmetric across the boundary" condition.

---

## `plasma_bounds(domain, plasma_bounding)` → `(lo, hi)`

Returns numpy arrays of the plasma loading bounds, clipped to the simulated domain. The user specifies `plasma_bounding` as a fraction of the full-domain extent (e.g. `0.11` means the plasma loads inside `[-0.11 L, +0.11 L]` in full mode). In octant mode the lower bound is clipped to `0` so loading stays inside the simulated region.

---

## `VALID_SYMMETRIES`

Tuple `("full", "octant")`. The input deck asserts `symmetry in VALID_SYMMETRIES` immediately after the user-params block, so an invalid value fails loudly before any PICMI work begins.

---

## Validation recipe

Before relying on octant reduction for production sweeps, run one short full-domain reference at the same `L`/`N`/physics params and compare fields and bulk plasma quantities on the shared octant. A noticeable mismatch is almost always a sign that the field BC (PMC vs PEC vs open) is wrong for the physics being studied — not a bug in `Domain` itself.
