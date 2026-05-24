# WarpX Polywell Simulation — Documentation

This is the documentation hub for the WarpX polywell fusion simulation project.
The project uses the [WarpX](https://ecp-warpx.github.io/) particle-in-cell (PIC) framework
to simulate plasma confinement in a polywell magnetic configuration.

---

## What This Project Does

1. **Generates external field grids** (B and E) using analytical/numerical methods
2. **Writes them to openPMD-compliant HDF5 files** that WarpX can read at startup
3. **Runs a PIC simulation** of electrons and protons in those fields
4. **Outputs diagnostics** (field snapshots and particle data) every 100 steps

---

## Documentation Index

### Setup
| Document | Description |
|---|---|
| [Installation](setup/installation.md) | Environment setup, conda dependencies, and first-run checklist |
| [Project Structure](setup/project-structure.md) | Directory layout and file purposes |

### Simulation
| Document | Description |
|---|---|
| [Running a Simulation](simulation/running.md) | How to launch and configure a run |
| [Input Parameters](simulation/parameters.md) | Every tunable parameter in `polywell_input.py` |
| [Output Files](simulation/output.md) | Diagnostics format, HDF5 structure, how to read results |

### Source Modules
| Document | Description |
|---|---|
| [domain module](modules/domain.md) | `Domain` dataclass + symmetry reduction (full / octant) — single source of truth for the simulated region |
| [spawn module](modules/spawn.md) | `make_layout` particle-mode selector — density vs exact count |
| [bext module](modules/bext.md) | External B-field generation (`src/bext/`) — file-based and analytic modes |
| [External Particle Field Modes](modules/external_particle_fields.md) | In-depth guide: physics, API reference, and how-to for both B-field pipelines |
| [eext module](modules/eext.md) | External E-field grid generation (`src/eext/`) |
| [utils module](modules/utils.md) | Coordinate helpers and path definitions (`src/utils/`) |

### Physics Background
| Document | Description |
|---|---|
| [Polywell Fusion](physics/polywell.md) | Physical basis of the polywell configuration and simulation choices |

---

## Quick Start

```bash
# 1. Activate your conda environment
conda activate warpx-env

# 2. Run the simulation from the project root
python inputs/polywell_input.py
```

The first run generates the field HDF5 file in `output/bext/`.
Subsequent runs with the same parameters reuse the cached file.

---

## Key Design Decisions

- **Dual B-field modes**: The B-field can be supplied as a pre-computed grid file (`"file"` mode) or as exact analytic expressions evaluated per-particle (`"analytic"` mode). Set `b_method` in the input deck to switch. See [External Particle Field Modes](modules/external_particle_fields.md).
- **Hybrid-PIC via vector potentials**: A single toggle `use_hybrid` in `polywell_input.py` flips the solver to `HybridPICSolver` and reroutes the B / E pipelines through the **potentials** path — B is built from A via Coulomb-gauge FFT curl-inverse (`src/bext/vector_potential.py`), E is built from φ via closed-form ring scalar potential (`src/eext/potential.py`), and the openPMD `A` mesh is wired into WarpX's `external_vector_potential` ParmParse keys so the hybrid solver curls it internally each step. Cache files carry a `_potentials` tag so they never collide with the magpylib-direct / analytic-E outputs. Verified end-to-end by [`tests/test_vector_potential.py`](../tests/test_vector_potential.py).
- **Symmetry reduction**: `symmetry = "octant"` simulates one octant of the cubic polywell domain with PMC + reflecting BCs on the three inner faces — 8× cheaper, validated against a full-domain reference. See [domain module](modules/domain.md).
- **Particle-mode toggle**: `particle_mode = "count"` swaps the default `GriddedLayout` for a `PseudoRandomLayout(n_macroparticles=…)`, giving exactly the requested macroparticle count globally — useful for tracer/orbit studies. See [spawn module](modules/spawn.md).
- **Field caching**: In file mode, B and E fields are computed once, cached as `.h5` files, and reloaded on repeat runs. The file name encodes every parameter including the `symmetry` tag, so full/octant caches never collide.
- **Combined diagnostic**: Field and particle diagnostics share `name="diag"`, producing a single openPMD series with `meshes/` and `particles/` groups per iteration.
- **openPMD format**: All field files follow the openPMD 1.1.0 standard so they are compatible with WarpX's `read_from_file` interface and can be inspected with `openPMD-viewer`.
- **Non-vectorized E-field**: The analytic E-field integration loops over every grid point. This is slow for large `N` (mitigated by octant mode, which does (N/2)³ work); see [eext module](modules/eext.md) for details.
- **Runs database**: Every run is registered in `output/runs.db` with full parameter capture and queryable via `python -m src.db.runs list`.
