# Input Parameters Reference

All user-facing parameters live in `inputs/polywell_input.py` under the `=== USER INPUTS ===` block.

---

## Simulation Control

| Parameter | Default | Type | Description |
|---|---|---|---|
| `max_steps` | `10000` | `int` | Total number of PIC time steps to advance |
| `warpx.const_dt` | `1e-12` | `float` (s) | Fixed time step size in seconds |
| `p_density` | `1e18` | `float` (m⁻³) | Number density of each plasma species |

---

## B-Field (Polywell Coils)

These parameters define the six circular current loops that create the polywell magnetic well.

| Parameter | Default | Type | Description |
|---|---|---|---|
| `I` | `1e6` | `float` (A) | Current through each coil in Amperes |
| `b_dia` | `1` | `float` (m) | Diameter of each circular coil |
| `b_offset` | `1.1` | `float` (m) | Distance from origin to each coil center |

The six coils are placed at ±x, ±y, ±z at distance `b_offset`, with alternating current
directions. See [make_collection.py](../modules/bext.md#make_polywell_collection) for geometry.

---

## E-Field (Charged Rings)

| Parameter | Default | Type | Description |
|---|---|---|---|
| `e_method` | `"FW"` | `str` or `None` | E-field method name. Set to `None` to disable E-field entirely |
| `Q` | `1e-9` | `float` (C) | Total charge on each ring in Coulombs |
| `e_dia` | `0.75` | `float` (m) | Diameter of each charged ring |
| `e_offset` | `1.1` | `float` (m) | Distance from origin to each ring center |

### Available E-field methods (`EMethods` enum)

| Name string | Function | Notes |
|---|---|---|
| `"FW"` | `fw_e()` | Numerical integration (500 angular quadrature points by default) |
| `"BOB"` | `bob_e()` | Alternative normalised integration (100 points by default) |
| `None` | — | No E-field; `warpx.E_ext_field_init_style` still reads from file but the file contains zeros |

---

## Grid

| Parameter | Default | Type | Description |
|---|---|---|---|
| `L` | `3` | `int` (m) | Half-length of the simulation domain in each axis. Full domain is `[-L, L]³` |
| `N` | `72` | `int` | Number of grid cells per axis. Must be divisible by the MPI rank count |
| `number_per_cell_each_dim` | `[10, 10, 10]` | `list[int]` | Macro-particles per cell per dimension for each species |

> Total macro-particles per species = `N³ × (10 × 10 × 10) × plasma_bounding³` (approximately)

---

## Scale Factors

| Parameter | Default | Type | Description |
|---|---|---|---|
| `plasma_bounding` | `0.11` | `float` (fraction) | Plasma initialises within `±(L × plasma_bounding)` in each axis. 0.11 means the plasma occupies ≈11% of the domain half-length |

---

## Derived Values (set automatically)

These are derived from user inputs and should not need manual editing:

```python
lx = ly = lz = L          # domain half-extent (m)
nx = ny = nz = N          # grid resolution per axis
```

---

## Solver Options

The solver is configured in the `=== GRIDS ===` section:

```python
# Currently active:
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.99)

# Available alternatives (commented out):
# solver = picmi.ElectrostaticSolver(grid=grid)
# solver = picmi.HybridPICSolver(...)
```

| Solver | Notes |
|---|---|
| `ElectromagneticSolver` (Yee) | Full Maxwell; supports external E-field from file |
| `ElectrostaticSolver` | Poisson-based; simpler but no EM waves |
| `HybridPICSolver` | Does **not** support external E-fields from file; requires external vector potentials instead |

---

## Diagnostics

Diagnostics are written every 100 steps to the working directory in openPMD/HDF5 format.

**Field diagnostic** (`field_diag`):
- Fields: `Bx`, `By`, `Bz`, `Jx`, `Jy`, `Jz`, `part_per_cell`

**Particle diagnostic** (`part_diag`):
- Species: `plasma_e`, `plasma_i`
- Data: `x`, `y`, `z`, `ux`, `uy`, `uz`, `weighting`

Both use `warpx_format='openpmd'` with `warpx_openpmd_backend='h5'`.
