# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A WarpX particle-in-cell (PIC) simulation harness for a polywell fusion configuration. This is a single-driver project: `inputs/polywell_input.py` is the entry-point driver, and the `warpx_polywell` package (under `src/`, installed via `poetry install`) contains the field-generation, storage, and run-tracking machinery it depends on. Detailed architecture lives in `docs/`; on-demand per-script guidance is in `.claude/rules/scripts.md`.

## Tech stack

- **Python 3.x** (conda env: `warpx-env`)
- **WarpX / pyWarpX** — Berkeley PIC framework; PICMI Python interface
- **AMReX / pyAMReX** — block-structured AMR backend
- **magpylib** — coil-field calculations for B-field grid generation
- **scipy** — special functions (elliptic integrals) for analytic B-field
- **h5py + openPMD** — HDF5 I/O for external field files and diagnostics
- **SQLite** (stdlib `sqlite3`) — `output/runs.db` run registry
- **MPI** (via `mpirun`) — parallel execution
- **Poetry** — packaging/dependency management. The `warpx_polywell` package is defined in `pyproject.toml` (poetry-core, src-layout) and installed editable with `poetry install`. The conda env (`setup.sh`) still provides the compiled `pywarpx`/`pyamrex` stack, which is not pip-installable.

## Repo layout

```
.
├── inputs/         # Entry-point driver script(s)
├── src/            # Poetry src-layout root
│   └── warpx_polywell/   # Importable package: domain, spawn, bext, eext, db, utils
├── docs/           # In-tree documentation (start at docs/README.md)
├── tests/          # Jupyter notebooks + notebook-builder scripts
├── output/         # Per-run output, cached field files, runs.db (gitignored)
├── .claude/        # Claude Code settings + rules
├── setup.sh        # Conda env bootstrap
└── CLAUDE.md       # This file
```

## How to run

```bash
# First-time setup (creates conda env warpx-env)
./setup.sh                          # add --force to recreate, --dry-run to preview
conda activate warpx-env

# Install the warpx_polywell package (editable) into the active env
poetry install                      # makes `import warpx_polywell` resolve from anywhere

# Run a simulation — launch from project root so output/ resolves next to the repo
python inputs/polywell_input.py
mpirun -n 8 python inputs/polywell_input.py   # parallel; cells/axis must divide rank count

# Query the runs database (output/runs.db)
python -m warpx_polywell.db.runs list                              # most recent
python -m warpx_polywell.db.runs list status=completed b_method=analytic
python -m warpx_polywell.db.runs get <id>
python -m warpx_polywell.db.runs scan                              # backfill existing run dirs
```

No test runner or linter is configured. Validation notebooks under `tests/` run interactively.

## Project-wide conventions

- **`warpx_polywell` is an installed package.** After `poetry install`, `from warpx_polywell.* import ...` resolves from any directory — the repo root no longer needs to be on `sys.path`. Re-run `poetry install` if you move the package.
- **`.env` sets the output base.** `LOCAL_OUTPUT_DIR` (loaded by `utils/config.py`, exposed as `paths.OUTPUT_DIR`) is the single root for *all* generated output — per-deck run dirs (`OUTPUT_DIR/<deck>/run_<timestamp>/`), the `bext/` field cache, and `runs.db`. It defaults to `<repo>/output` when unset. Each driver wraps its run in the shared `run_session(__file__, params)` context manager (in `db/runs.py`) — it allocates the per-deck run dir (MPI-safe), `chdir`s in, snapshots the deck, registers the run, and sets `completed`/cleans up on exit — so a new deck needs almost no output boilerplate. Post-processing resolves runs through `post/reader.py` (`latest_run`, `run_dir`, `chdir_to_run`) rather than hardcoding paths.
- **Don't commit secrets.** `.env` (storage + output config) is gitignored; check before staging.
- **Heavy compute is rank-0 only.** Field-file generation (magpylib, E-field integration) runs serially before `sim.step()`; only the WarpX PIC advance is parallel.
- **Cache filenames are the cache key.** `OUTPUT_DIR/bext/*.h5` names encode every parameter — changing any parameter produces a new file. Don't rename or hand-edit.
- **For library docs, prefer Context7 over web search.**

## Context7 library IDs

When using the context7 MCP server for these packages, call `get-library-docs` directly with the IDs below. Skip `resolve-library-id` for these — the IDs are already known:

- pyAMReX: `/amrex-codes/pyamrex`
- WarpX: `/blast-warpx/warpx`

For any other library, call `resolve-library-id` first as normal.

## Further reading

- `docs/README.md` — full documentation index (architecture, module reference, physics, gotchas)
- `docs/simulation/running.md` — execution flow + cache keys
- `docs/modules/external_particle_fields.md` — B-field pipelines (file vs analytic)
- `docs/modules/domain.md`, `docs/modules/spawn.md` — symmetry + particle-mode toggles
- `.claude/rules/scripts.md` — per-script reference (auto-loads when reading any entry-point script)
