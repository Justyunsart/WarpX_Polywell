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
| `b_method` | `"analytic"` | `str` | B-field pipeline: `"analytic"` (exact per-particle expressions) or `"file"` (pre-computed HDF5 grid) |
| `I` | `1e6` | `float` (A) | Current through each coil in Amperes |
| `b_dia` | `1` | `float` (m) | Diameter of each circular coil |
| `b_offset` | `1.1` | `float` (m) | Distance from origin to each coil center |

The six coils are placed at ±x, ±y, ±z at distance `b_offset`, with alternating current
directions. See [make_collection.py](../modules/bext.md#make_polywell_collection) for geometry.

For a detailed comparison of the two modes, see
[External Particle Field Modes](../modules/external_particle_fields.md).

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

`L` and `N` are always written in **full-domain physical terms** regardless of `symmetry` (see below). When `symmetry = "octant"`, the simulated region is `[0, +L]^3` with `N/2` cells per axis, but the user types `L` and `N` as if the run were full-domain.

| Parameter | Default | Type | Description |
|---|---|---|---|
| `L` | `2` | `int` (m) | Half-extent of the full simulation domain. Full mode spans `[-L, +L]³`; octant mode spans `[0, +L]³` |
| `N` | `72` | `int` | Full-domain cell count per axis. Must be divisible by the MPI rank count; in octant mode the simulated cells per axis are `N/2`, which must also be rank-divisible |
| `number_per_cell_each_dim` | `[10, 10, 10]` | `list[int]` | Macroparticles per cell per dimension (density mode only) |

> Total macroparticles per species (density mode) ≈ `(N or N/2)³ × ppc³ × plasma_bounding³`.

---

## Symmetry

The polywell has cubic symmetry. Setting `symmetry = "octant"` reduces the simulated region to one octant (`[0, +L]³`), using PMC (Perfect Magnetic Conductor) on the three inner faces and `reflecting` for particles. Field/particle BCs are derived automatically by `derive_domain(symmetry, L, N)` (`src/domain.py`).

| Parameter | Default | Type | Description |
|---|---|---|---|
| `symmetry` | `"full"` | `str` | `"full"` (no reduction) or `"octant"` (1/8 of the cubic domain). Recorded in the runs DB. |

> **Validation step.** Before relying on octant for production sweeps, run one short full-domain reference at the same `L`/`N`/physics params and confirm the fields and bulk plasma quantities agree on the shared octant.

> **Why PMC.** A polywell B-field is an axial vector sourced by mirror-symmetric loop currents. On a symmetry plane, normal B is unconstrained and tangential B vanishes — exactly what PMC enforces. PEC (tangential E = 0, normal B = 0) is the opposite symmetry class and would zero out the cusp field.

---

## Particle Spawning

Two modes selectable via `particle_mode`. Both spawn particles from the same `UniformDistribution(density=p_density, …)`, so the represented physical density is identical — only the sampling differs.

| Parameter | Default | Type | Description |
|---|---|---|---|
| `particle_mode` | `"density"` | `str` | `"density"` (GriddedLayout, ppc per cell) or `"count"` (PseudoRandomLayout, exact global count) |
| `n_test_particles` | `10000` | `int` | Total macroparticles globally when `particle_mode = "count"`; ignored otherwise |

> **Weight semantics.** In count mode, PICMI sets `weight = density × plasma_volume / n_test_particles`. A density-mode run and a count-mode run at the same `p_density` represent the same plasma — count mode is a Monte Carlo undersampling for orbit/tracer studies.

> Both species already have `do_not_deposit = 1`, so "count" mode is functionally a tracer ensemble: each macroparticle moves under the prescribed external fields without backreacting.

---

## Scale Factors

| Parameter | Default | Type | Description |
|---|---|---|---|
| `plasma_bounding` | `0.11` | `float` (fraction) | Plasma initialises within `±(L × plasma_bounding)` (full mode) or `[0, +L × plasma_bounding]` (octant mode, clipped automatically) per axis |

---

## Solver / Potentials Toggle

A single user toggle, `use_hybrid`, picks the solver class **and** flips
the B/E field pipelines into "potentials mode" in lockstep. Hybrid-PIC is
the only WarpX solver that natively consumes an external vector potential
A (rather than B directly), so the two settings have to move together.

| Parameter | Default | Type | Description |
|---|---|---|---|
| `use_hybrid` | `False` | `bool` | When True: solver is `HybridPICSolver`, the B file is built via Coulomb-gauge FFT curl-inverse for A (`src.bext.vector_potential`) and the openPMD `A` mesh is wired into `external_vector_potential.polywell.{read_from_file, path}`. When False: solver is `ElectromagneticSolver(Yee)`, B comes straight from magpylib, E comes from the analytic ring integrand. |
| `use_potentials` | derived from `use_hybrid` | `bool` | Don't set this by hand. It exists so `setup_bext` / `fill_eext_file` can be driven without solver coupling, but in `polywell_input.py` it's just `use_potentials = use_hybrid`. |

> **Cache safety.** With `use_hybrid=True` the generated `.h5` filenames
> carry a `_potentials` tag (e.g. `B_ext_potentials_…h5`,
> `…_E_ext_potentials_…h5`) so caches from the two pipelines never
> collide, even at otherwise identical parameters.

> **DB column.** `use_hybrid` is a filterable column in `runs.db`. List
> hybrid-mode runs with
> `python -m src.db.runs list use_hybrid=true`.

---

## Solver Options

The solver is selected in the `=== GRIDS ===` section, branching on the
`use_hybrid` toggle above:

```python
if use_hybrid:
    solver = picmi.HybridPICSolver(
        grid=grid, Te=1.0, n0=p_density, gamma=1,
        plasma_resistivity=0, n_floor=(p_density * plasma_bounding),
    )
else:
    solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.99)
```

| Solver | When chosen | Notes |
|---|---|---|
| `ElectromagneticSolver` (Yee) | `use_hybrid=False` | Full Maxwell; supports external E-field from file |
| `HybridPICSolver` | `use_hybrid=True` | Consumes external B-field as a vector potential A from the openPMD file via `external_vector_potential.polywell.{read_from_file, path}`. Does **not** support external E-fields applied to particles. |
| `ElectrostaticSolver` | (commented out) | Poisson-based; simpler but no EM waves |

---

## Diagnostics

Field and particle diagnostics both use `name="diag"` so openPMD writes them into a **single series** at `diags/diag/openpmd_%T.h5`. Each iteration file carries `meshes/` (fields) and `particles/` (species) groups side-by-side — no more parallel `field_diag/` and `part_diag/` folders to reconcile.

**Field side** (`field_diag`, mesh data):
- Fields: `Bx`, `By`, `Bz`, `Bx_fp_external`, `By_fp_external`, `Bz_fp_external`, `Ex_fp_external`, `Ey_fp_external`, `Ez_fp_external`, `Jx`, `Jy`, `Jz`, `part_per_cell`

**Particle side** (`part_diag`, per-particle records):
- Species: `plasma_i` (electron block commented out by default)
- Data: `x`, `y`, `z`, `ux`, `uy`, `uz`, `weighting`

Both use `warpx_format='openpmd'` with `warpx_openpmd_backend='h5'`. Default period is 10 steps. To split them back into separate files, give each diagnostic a distinct `name=` value.
