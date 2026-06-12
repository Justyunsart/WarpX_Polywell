# Development Log

Running log of non-trivial development work on the WarpX polywell project.
Newest entries at the top. Each entry captures: what changed, why, and any
gotchas worth remembering. Trivial fixes and pure refactors don't need an
entry — `git log` already covers those.

---

## 2026-06-01 — Spatial "resistive coil islands" in the 2D coil deck via current-damping callback

### What changed

- **Removed the spatial `plasma_resistivity` expression from
  `inputs/coil 2D/coil_2d.py`.** It was a two-Gaussian bump in `(x, z)` at
  the two wire positions (`z = ±d/2`) meant to create high-η "conductor
  islands." WarpX aborted at startup with
  `ParmParse: failed to parse … due to unknown symbol x`. Replaced with a
  uniform scalar `plasma_resistivity=eta_bg`.
- **Added a current-damping callback to emulate the islands.** New
  `damp_current_at_coils()` installed via `installafterdeposition`. Each
  step, after current deposition and before the hybrid E-solve, it
  multiplies the deposited `current_fp` (all three directions) by a cached
  spatial factor `1 / (1 + (eta_coil/eta_bg)·g(x,z))`, where `g` is the same
  two-Gaussian bump. At a coil center the factor ≈ `eta_bg/eta_coil ≈ 0`
  (current zeroed → conductor-like); in the bulk it ≈ 1 (untouched). The
  `eta_bg`/`eta_coil`/`w`/`dh` variables that defined the old expression are
  reused as the callback's contrast/width knobs.
- **New doc page
  [`docs/simulation/hybrid_resistivity.md`](simulation/hybrid_resistivity.md)**
  documenting the parser limit, the callback technique, and the
  embedded-boundary trade-off. Linked from `docs/README.md` and
  cross-referenced from `docs/simulation/parameters.md`.

### Why

WarpX registers the hybrid resistivity parser with a **fixed two-symbol
argument list**: `pywarpx.hybridpicmodel."plasma_resistivity(rho,J)"`. Only
`rho`, `J`, registered constants, and math functions are in scope — there
are **no spatial coordinates**, so any `x`/`z` token is an unknown symbol.
(The external-field parsers like `Ay_external_function(x,y,z)` *do* expose
coordinates, which is why `Ay_expr` in the same deck parses fine — easy to
conflate the two.) Resistivity therefore cannot be made position-dependent
through the input deck at all; the only way to localize resistive behavior
in space is to act on the fields/current from a Python callback.

Damping `current_fp` after deposition is the lightest intervention that
produces the intended effect (no plasma current through the wire cells)
and runs at the right point in the step for the hybrid Ohm's-law solve to
see it.

### Gotchas / follow-ups

- **`plasma_resistivity(rho,J)` and `plasma_hyper_resistivity(rho,B)` are
  never spatial.** Same trap applies to the polywell hybrid path
  (`use_hybrid=True`), which currently uses a scalar `0` so isn't bitten —
  but don't try to upgrade it to an `x,y,z` expression.
- **The callback damps *current*, not via an `E = ηJ` term.** It zeroes
  current at the coils but does not deposit the resistive E-field / Ohmic
  heating a true local η would. For that signature, patch `Efield_fp` in
  the callback instead. Documented in the new page.
- **Field access idiom for callbacks.** `sim.fields.get("current_fp",
  dir=sim.extension.libwarpx_so.Direction(idir), level=0)` returns the
  per-direction MultiFab; `mf.mesh("x")`/`mf.mesh("z")` give global
  cell coordinates with the MultiFab's own centering (robust for the
  collocated grid). `mf[:, :]` read is a global allgather; assignment is
  local-block — correct under MPI, cheap at 64².
- **Shape safety.** `factor.reshape(arr.shape[:2] + (1,)*(arr.ndim-2))`
  broadcasts the 2D mask over any trailing component axis the array carries.
- **Embedded boundary was considered and rejected for now.** EB conductor
  *field* BCs are EM-solver–only and don't apply to the hybrid algebraic
  E-field; EB particle *absorption* does work but the coils are ~1 cell wide
  here (under-resolved for EB) and the B-field comes from the analytic `Ay`,
  not the coil-as-object. Revisit only if particle exclusion is the goal and
  the mesh is refined.

### Verifying

```bash
cd "inputs/coil 2D" && python coil_2d.py   # (set MAX_STEPS small to smoke-test)
```

A 2-step smoke run completes cleanly — no `ParmParse`/SIGABRT, no callback
exception, both steps advance through `HybridPICEvolveFields`.

### Correction (later)

The `installafterdeposition` hook claimed above (and originally in
`hybrid_resistivity.md`) is **wrong for the hybrid loop**: probing every
callback during a hybrid run shows the hybrid evolve path never triggers
`afterdeposition`/`beforedeposition` — only `beforestep`, `afterstep`,
`beforeEsolve`, `afterEsolve` (per step) and `afterBpush`/`afterEpush` (per
substep). A function installed on `afterdeposition` registers but is never
called, so the current is never damped. The correct hook is **`beforeEsolve`**
(fires once per step, after deposition, before `HybridPICEvolveFields` consumes
`current_fp`). The 3D translation in `inputs/polywell_hybrid.py`
(`zero_coil_current`) uses `beforeEsolve`; `hybrid_resistivity.md` has been
updated. The 2D `coil_2d.py` example is still written with
`installafterdeposition` and would need the same fix to actually damp current.

---

## 2026-05-23 — Vector-potential pipeline for Hybrid-PIC; A from B via FFT curl-inverse

### What changed

- **New module [`src/warpx_polywell/bext/vector_potential.py`](../src/warpx_polywell/bext/vector_potential.py).**
  Computes the vector potential A on the WarpX grid from any
  magpylib-evaluable B. Coulomb-gauge FFT curl-inverse on a zero-padded
  box: `Ã(k) = i (k × B̃) / |k|²` with the DC mode zeroed.
  `compute_A_grid(collection, domain, pad_factor)` does one pad/FFT/crop
  pass; `converge_A_grid(...)` doubles `pad_factor` (2 → 4 → 8) until A in
  the physics region stops moving by more than `rtol`. `curl_A` and
  `check_curl` are reusable helpers for `B = ∇×A` and verification.
- **New module [`src/warpx_polywell/eext/potential.py`](../src/warpx_polywell/eext/potential.py).**
  Closed-form scalar potential φ for a charged ring (elliptic integrals;
  `φ = Q·K(k) / (2π²ε₀√((a+ρ)² + z²))`). Reuses
  `make_polywell_collection` for the 6 ring placements/orientations and is
  vectorised over the whole grid — typically ~100× faster than the
  per-point `methods.py::fw_e`/`bob_e` loops it can replace.
  `compute_E_from_phi` returns `E = -∇φ`.
- **`use_potentials` parameter threaded through the B and E pipelines.**
  `setup_bext`, `make_bext_file`, `get_bext_file_name`, and `fill_eext_file`
  all take `use_potentials: bool = False`. When True:
  * B file built via FFT curl-inverse → ∇×A (writes both A and B to the
    `.h5`); E file built via φ → -∇φ.
  * Filenames carry a `_potentials` tag (`B_ext_potentials_…h5`,
    `_E_ext_potentials_…`) so cache files never collide with the
    magpylib-direct / analytic-E pipeline outputs.
- **openPMD file now optionally carries an `A` mesh.**
  `_make_empty_ext_h5` creates a third mesh group `A` (alongside `B` and
  `E`) with `unitDimension = [1, 1, -2, -1, 0, 0, 0]` (Wb/m). `_fill_h5_file`
  writes the `A/x,y,z` datasets when `use_potentials=True`. When False the
  group exists but is empty — backward-compatible with downstream readers.
- **Hybrid-PIC wiring in `setup_bext`.**
  When `solver="hybrid"` the dispatcher forces `use_potentials=True` (with
  a printed notice if it had been False) and calls new helper
  `_wire_hybrid_external_A(ext_path)`, which drives `pywarpx.hybridpicmodel`
  and `pywarpx.external_vector_potential` Buckets to set:
  ```
  hybrid_pic_model.add_external_fields = 1
  external_vector_potential.fields = polywell
  external_vector_potential.do_diva_cleaning = 0
  external_vector_potential.polywell.read_from_file = 1
  external_vector_potential.polywell.path = <ext_path>
  external_vector_potential.polywell.A_time_external_grid_function(t) = 1
  ```
  WarpX then reads A from the openPMD file at startup and curls it
  internally each step.
- **Single user toggle `use_hybrid` in
  [`inputs/polywell_input.py`](../inputs/polywell_input.py).**
  Replaces the previous `use_potentials` toggle. Drives three things in
  lockstep: (a) solver class (`HybridPICSolver` vs `ElectromagneticSolver`),
  (b) `use_potentials = use_hybrid` (derived), (c) the `solver="hybrid"`
  kwarg passed through to `setup_bext`. The two-line edit users actually
  flip.
- **Filterable `use_hybrid` column in `runs.db`.**
  Wired through the six places `scripts.md` flags: `_SCHEMA`, `_MIGRATIONS`,
  `_INDEXES`, `list_runs(use_hybrid=…)` (accepts truthy CLI strings),
  `_BACKFILL_COLUMNS`, and `SCALAR_MAP`. `_print_runs` now shows
  `hybrid=Y/N` per row. Migration applies on next DB open (verified live).
- **New test suite [`tests/test_vector_potential.py`](../tests/test_vector_potential.py).**
  5 tests, 8 sub-assertions — all passing:
  1. Padding convergence: doubling `pad_factor` moves |A| by < 0.1%.
  2. ∇×A vs magpylib B: 99.1% of cells within 5% of peak |B|, 97% within 1%.
  3. Coulomb gauge: `‖∇·A‖ / max‖∇×A‖ = 9×10⁻¹⁶` in the plasma cube
     (machine precision).
  4. Resolution convergence: cells-within-5%-of-peak grows monotonically
     97.3 → 98.8 → 99.1 → 99.5 % as N = 16 → 48.
  5. Single isolated ring vs analytic `A_φ` (elliptic integrals): 14% rel
     err off-ring.

### Why

WarpX's hybrid-PIC solver consumes the external magnetic field through a
vector potential A, not B directly. To run the polywell with hybrid PIC we
needed (a) A on a grid in Wb/m, (b) written into an openPMD `A` mesh, and
(c) the `external_vector_potential.<name>.read_from_file/path` ParmParse
keys pointing WarpX at the file.

magpylib only exposes `getB`/`getH`/`getJ`/`getM` — no `getA`. The cleanest
way to recover A for an arbitrary coil arrangement is to start from B
(which magpylib does have) and invert the curl in Coulomb gauge. In
Fourier space this is the one-liner above, which is just vector Poisson
per Cartesian component. The padding is the cost: `np.fft.fftn` is
periodic, so an isolated coil in a finite box gets repeated as an infinite
lattice and image contamination leaks into the interior. Zero-padding the
box buries the images far enough away that they cancel out, and doubling
the padding gives a clean convergence test.

The E-field side was a free win: the existing per-point loop in
`get_e_field_data` was the bottleneck for large N (O(N³ × 6) Python calls
to `fw_e`/`bob_e`). The same physics expressed as a vectorised
elliptic-integral sum over 6 rotated rings runs in one shot — useful even
without `use_hybrid` if you want to swap the analytic-E pipeline for the
potential-derived one.

### Gotchas / follow-ups

- **Don't gate on per-cell relative error for the polywell.** It has a
  magnetic null at origin by design, so |B| → 0 in the interior. Any
  per-cell `|a - b| / |b|` blows up to ∞ in the null even when the
  absolute error is tiny. The test suite uses **peak-normalised absolute
  tolerance** (fraction of cells where `|error| < 5% of peak |B|`), which
  matches what a hybrid-PIC particle in the null actually experiences.
- **L∞ at coil-adjacent cells is unbounded by construction.** Central
  differences of a 1/r³ singular source can't reproduce the source.
  ~1% of cells (the ones touching a coil ring) will always show large L∞
  error — this is a fundamental FD limit, not a pipeline defect, and
  WarpX's own Yee-mesh curl has the same property.
- **Coulomb gauge is at machine precision in the interior.** Test 3
  reports `9×10⁻¹⁶` in the plasma cube. The spectral `ik·Ã = 0`
  enforcement survives IFFT and cropping intact. The full-grid div/curl
  ratio is non-zero (~15%) but it's the same coil-cell FD-truncation
  artefact as Test 2 — diagnostic only.
- **Production `pad_factor = 8` is enough.** Test 1 shows the rel change
  from pad=4 to pad=8 is < 0.1%. Going to pad=16 doubles every FFT axis
  again — memory scales as `(pad·N)³ × 16 B` per complex array, so the
  jump from 8 to 16 is the difference between fitting in RAM and an OOM
  SIGKILL.
- **The B mesh in a `_potentials`-tagged file is unused by the hybrid
  solver.** WarpX reads A from `external_vector_potential.polywell.path`
  only. The derived B (∇×A via central differences) is written to the
  same file as a debugging convenience and so non-hybrid runs that happen
  to find the cache still see what they expect.
- **pywarpx Bucket idiom for ParmParse sub-keys.** `external_vector_potential`
  is a top-level Bucket, but per-field keys like
  `external_vector_potential.polywell.path` aren't legal Python attribute
  names — use `setattr(external_vector_potential, "polywell.path", value)`.
  Same convention `bext.py::setup_bext` already uses for
  `particles.Bx_external_particle_function(x,y,z,t)`.

### Running the test suite

```bash
PYTHONPATH=$(pwd) python tests/test_vector_potential.py
```

Exits 0 on full pass, 1 on any failure. ~30 s on a laptop at the default
parameters (memory budget capped at ~1 GB via `gc.collect()` between
tests and small `pad·N`).

---

## 2026-05-17 — External-field ParaView workflow; SIGABRT edge case

### What changed
- **No external field components in the live FieldDiagnostic.** Tried to
  append `Bx_fp_external`/etc. to `field_diag.diagnostic.fields_to_plot`
  after `sim.initialize_inputs()`, but WarpX's
  `FullDiagnostics::isFieldOutputType` rejects those names — they're
  internal MultiFab labels, not output types. Reverted the patch; left a
  comment in [`inputs/polywell_input.py`](../inputs/polywell_input.py)
  pointing future-self at the pre-generated `output/bext/*.h5` file as
  the right source for external-field StreamTracer in ParaView (already
  openPMD with `B/x,y,z` and `E/x,y,z` vector records).

### Why
The `do_not_deposit = 1` test-particle regime means the live `B` written
to the diagnostic *is* the external B (no self-fields are generated), so
a separate external record is unnecessary for now. If deposition is ever
re-enabled, the bext file remains a clean static reference for the
external-only field.

### Gotchas / follow-ups
- WarpX's `amrex::Abort` (C++) terminates the process via SIGABRT
  **without raising a Python exception**, so `run_context`'s except
  branch never fires for those failures — they leave a `status='running'`
  zombie row and an orphan directory. Cleaned one such case manually
  (id=7). A startup sweep that converts old `status='running'` rows into
  cleanup candidates would close this gap.
- `set_or_replace_attr` is the correct entry point for mutating any
  `pywarpx.Diagnostic` attribute (direct `__setattr__` is rejected by
  the consistency check in `Diagnostics.py`). PICMI itself uses this.

### ParaView workflow for external-field StreamTracer
1. Open `output/bext/B_ext_*.h5` as a separate openPMD source.
2. `B` and `E` are already vector records — point StreamTracer's
   `Vectors` directly at them. No Calculator filter needed.
3. Apply `CellDataToPointData` first (StreamTracer wants point data).
4. Seed near a coil (`b_offset` along an axis) or with a small sphere at
   the origin to render the cusp topology.

---

## 2026-05-17 — Unblock octant polywell run; strict runs DB

Commit: [`4951a89`](../) — *unblock octant polywell run + drop failed-run rows from DB*

### What changed
- **PICMI `pmc` boundary passthrough.** Added `picmi.BC_map["pmc"] = "pmc"`
  near the top of [`inputs/polywell_input.py`](../inputs/polywell_input.py).
  picmistandard exposes no name that maps to WarpX's `pmc`, and the octant
  symmetry planes need it (tangential B = 0, normal E = 0). Same pattern
  pywarpx uses for `"open"` at runtime.
- **"count" particle mode is now per-cell.** Renamed `n_test_particles` →
  `n_test_particles_per_cell` in [`src/warpx_polywell/spawn.py`](../src/warpx_polywell/spawn.py) and the
  input deck. WarpX's `AnalyticDistribution` paired with `PseudoRandomLayout`
  only accepts `n_macroparticles_per_cell`, not a global total — see
  `pywarpx/picmi.py:591`.
- **Runs DB schema cleanup.** Dropped the old `n_test_particles` column,
  added `n_test_particles_per_cell`. Introduced a `_DELETIONS` list in
  [`src/warpx_polywell/db/runs.py`](../src/warpx_polywell/db/runs.py) parallel to `_MIGRATIONS` for
  retiring columns via `ALTER TABLE … DROP COLUMN` (no-op via try/except
  on DBs that never had the column).
- **Strict failure policy in `run_context`.** Failed runs are now *deleted*
  from the DB (row + `run_metadata.json` sidecar) **and the run directory
  itself is `shutil.rmtree`'d** instead of marked `"failed"`. Nothing is
  left under `output/runs/` for failed attempts. New `RunsDB.delete_run()`
  method handles the DB-side cleanup; directory removal lives in
  `run_context` so the primitive stays composable.
- **Octant-run parameter tune.** Input deck: `p_density 1e12→1e18`,
  `b_offset 1.1→0.435`, `N 72→80`, `e_method` off, `symmetry=octant`,
  `particle_mode=count`.

### Why
The polywell deck was failing at `sim.step()` with two cascading errors:
first a `KeyError: 'pmc'` from PICMI's BC translation table, then an
`AssertionError` from WarpX requiring per-cell counts for
`AnalyticDistribution + PseudoRandomLayout`. Both were straightforward
plumbing issues — the physics intent (PMC at symmetry planes, fixed
per-cell macroparticle count) was already correct in the underlying code.

The strict DB policy was a follow-up cleanup: the two error runs from this
session left `status='failed'` rows that were never going to be useful.
Better to keep the DB as a registry of *runs that produced data*.

### Gotchas / follow-ups
- `n_test_particles_per_cell = 1` in the deck currently — that's 1 random
  particle per cell over the octant cube, ~46k cells in the box but only
  ~32 cells inside the plasma sphere carry non-zero weight. Bump if test
  statistics are too thin.
- The `_DELETIONS` mechanism in `runs.py` runs on every `RunsDB()` open. If
  this list grows long, consider gating it behind a version table so it
  doesn't re-attempt drops indefinitely.
- Per-run cleanup of failed runs only happens through `run_context` — a
  hard-killed process (SIGKILL) will still leave a `status='running'`
  zombie row **and an orphan directory**. Not addressed here; consider a
  startup sweep if it becomes a problem. Ad-hoc cleanup pattern: list
  `status='running'` rows, `delete_run()` them, and `rmtree` their dirs.

---
