# Development Log

Running log of non-trivial development work on the WarpX polywell project.
Newest entries at the top. Each entry captures: what changed, why, and any
gotchas worth remembering. Trivial fixes and pure refactors don't need an
entry — `git log` already covers those.

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
  `n_test_particles_per_cell` in [`src/spawn.py`](../src/spawn.py) and the
  input deck. WarpX's `AnalyticDistribution` paired with `PseudoRandomLayout`
  only accepts `n_macroparticles_per_cell`, not a global total — see
  `pywarpx/picmi.py:591`.
- **Runs DB schema cleanup.** Dropped the old `n_test_particles` column,
  added `n_test_particles_per_cell`. Introduced a `_DELETIONS` list in
  [`src/db/runs.py`](../src/db/runs.py) parallel to `_MIGRATIONS` for
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
