# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A WarpX particle-in-cell (PIC) simulation harness for a polywell fusion configuration. This is a single-driver project, not a library: `inputs/polywell_input.py` is the entry-point driver and `src/` contains the field-generation, storage, and run-tracking machinery it depends on. Detailed architecture lives in `docs/`; on-demand per-script guidance is in `.claude/rules/scripts.md`.

## Tech stack

- **Python 3.x** (conda env: `warpx-env`)
- **WarpX / pyWarpX** — Berkeley PIC framework; PICMI Python interface
- **AMReX / pyAMReX** — block-structured AMR backend
- **magpylib** — coil-field calculations for B-field grid generation
- **scipy** — special functions (elliptic integrals) for analytic B-field
- **h5py + openPMD** — HDF5 I/O for external field files and diagnostics
- **SQLite** (stdlib `sqlite3`) — `output/runs.db` run registry
- **MPI** (via `mpirun`) — parallel execution
- No `pyproject.toml` / `requirements.txt` / `setup.py` — env is conda-defined via `setup.sh`.

## Repo layout

```
.
├── inputs/         # Entry-point driver script(s)
├── src/            # Library modules: domain, spawn, bext, eext, db, utils
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

# Run a simulation — MUST launch from project root so `from src.*` resolves
python inputs/polywell_input.py
mpirun -n 8 python inputs/polywell_input.py   # parallel; cells/axis must divide rank count

# Query the runs database (output/runs.db)
python -m src.db.runs list                              # most recent
python -m src.db.runs list status=completed b_method=analytic
python -m src.db.runs get <id>
python -m src.db.runs scan                              # backfill existing run dirs
```

No test runner or linter is configured. Validation notebooks under `tests/` run interactively.

## Project-wide conventions

- **Always run from the project root.** `from src.* import ...` requires the repo root on `sys.path`.
- **Don't commit secrets.** `.env` (storage backend config) is gitignored; check before staging.
- **Heavy compute is rank-0 only.** Field-file generation (magpylib, E-field integration) runs serially before `sim.step()`; only the WarpX PIC advance is parallel.
- **Cache filenames are the cache key.** `output/bext/*.h5` names encode every parameter — changing any parameter produces a new file. Don't rename or hand-edit.
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
