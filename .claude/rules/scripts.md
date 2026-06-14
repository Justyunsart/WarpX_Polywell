---
paths:
  - inputs/polywell_input.py
  - src/warpx_polywell/db/runs.py
  - tests/coil_field_analysis/_build_efield_notebook.py
  - tests/coil_field_analysis/_build_polywell_opt_notebook.py
---

# Entry-point scripts

Per-script reference for the four runnable Python files in this repo. Loads on-demand when Claude reads any of the paths above.

## `inputs/polywell_input.py`

The single simulation driver. Reads user parameters at the top of the file (B/E field method, grid `L`/`N`, `symmetry`, `particle_mode`, plasma temps), derives the simulated-domain spec via `derive_domain` and `make_layout`, builds PICMI objects, generates or loads the external B+E HDF5 file via `setup_bext` / `fill_eext_file`, allocates a per-run directory via `new_run_dir()` (groups under `OUTPUT_DIR/<deck>/`), registers the run in `OUTPUT_DIR/runs.db`, `chdir`s into the run dir, then calls `sim.step()`. `OUTPUT_DIR` follows `LOCAL_OUTPUT_DIR` in `.env` (default `<repo>/output`).

- **Run:** `python inputs/polywell_input.py` from the project root (or `mpirun -n N …`).
- **Inputs:** no CLI args. Edit the `=== USER INPUTS ===` block in-place.
- **Outputs:** `OUTPUT_DIR/polywell_input/run_YYYYMMDD_HHMMSS/diags/diag/openpmd_%T.h5` (combined field + particle diagnostic), plus a `run_metadata.json` sidecar and a snapshot copy of the deck itself. One row inserted into `OUTPUT_DIR/runs.db`.
- **Gotcha:** launch from the project root so generated output lands beside the repo when `LOCAL_OUTPUT_DIR` is unset. First run at a new parameter set is slow (field grid generation); subsequent runs hit the `OUTPUT_DIR/bext/` cache.

## `src/warpx_polywell/db/runs.py`

SQLite registry for simulation runs. Doubles as a library (`RunsDB`, `new_run_dir`) and a `__main__` CLI.

- **CLI:**
  - `python -m warpx_polywell.db.runs list [k=v …]` — list rows; filters include `status=`, `b_method=`, `e_method=`, `symmetry=`, `particle_mode=`, etc.
  - `python -m warpx_polywell.db.runs get <id>` — fetch one row as JSON
  - `python -m warpx_polywell.db.runs scan [<runs_root>]` — backfill rows for `OUTPUT_DIR/<deck>/run_*/` that pre-date the DB (walks both the per-deck tree and a flat `<root>/run_*`; pass an explicit root to ingest a legacy/external location)
- **Run dirs / deck lifecycle:** decks should use the `run_session` context manager — `with run_session(__file__, run_params): sim.step()` — which allocates the run dir (MPI-safe: created on rank 0, broadcast), `chdir`s in, snapshots the deck, registers the run, marks it `completed` on clean exit, and removes it on failure. Keep run-relative writes (post-step dumps) inside the `with`. `allocate_run_dir(script, *, copy_from, chdir)` is the lower-level building block (allocate + broadcast + chdir + snapshot, no DB) for decks that need the dir before stepping (e.g. `polywell_hybrid` builds its input file first, then registers via `RunsDB.run_context`). Both derive `<deck>` from the script filename stem; `new_run_dir(script=None)` underlies them and defaults the deck to `sys.argv[0]`'s stem. The `script` DB column records the deck (auto-derived from the run dir's parent in `register_run`).
- **Library use:** `with RunsDB().run_context(run_dir, params) as run_id: …` — context manager handles `completed` / `failed` status on exit.
- **Schema additions:** when adding a new user-input parameter, append to `_SCHEMA`, `_MIGRATIONS`, `_BACKFILL_COLUMNS`, **and** `SCALAR_MAP` together. Migrations apply via `ALTER TABLE` on existing DBs.
- **Gotcha:** `scan_existing()` is best-effort — its AST-parse recognises literal assignments and the `<num> * sc.eV` pattern for Te/Ti only. Non-literal expressions are silently dropped.

## `tests/coil_field_analysis/_build_efield_notebook.py`

One-shot generator for `E_Field_Analysis.ipynb`. Runs at module scope (no `if __name__ == "__main__"` guard) and overwrites the notebook on disk.

- **Run:** `python tests/coil_field_analysis/_build_efield_notebook.py` from project root.
- **Inputs:** none — geometry parameters (`DIA`, `OFFSET`, `Q`, `L`, `N`) are baked into the embedded cell strings.
- **Outputs:** `tests/coil_field_analysis/E_Field_Analysis.ipynb` (overwrites in place). The notebook itself, when later executed in Jupyter, writes PNGs to `tests/coil_field_analysis/plots/`.
- **Gotcha:** imports `warpx_polywell.eext`, `warpx_polywell.bext`, `warpx_polywell.domain` — must run from project root. The embedded notebook code calls `get_e_field_data(..., domain=domain)` after the recent Domain refactor, so old generated notebooks need regeneration.

## `tests/coil_field_analysis/_build_polywell_opt_notebook.py`

One-shot generator for `Polywell_Geometry_Optimization.ipynb`. Same convention as `_build_efield_notebook.py`. The notebook is an **isolated testing environment for sanity-checking design decisions** — e.g. validating the analytical E-field equation, comparing alternative integrands, or checking a geometric tweak in a self-contained scratchpad before it's wired into the production pipeline. Despite the filename, it's a playground, not a fixed optimization sweep.

- **Run:** `python tests/coil_field_analysis/_build_polywell_opt_notebook.py` from project root.
- **Inputs:** none (parameters baked into embedded cells).
- **Outputs:** `tests/coil_field_analysis/Polywell_Geometry_Optimization.ipynb`.
- **Gotcha:** ~1100 lines of cell-string Python; regeneration is the only practical way to edit. Treat the generated notebook as the source of truth for whatever experiment is currently active — its contents change as different design questions get checked.
