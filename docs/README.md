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
- **Field caching**: In file mode, B and E fields are computed once, cached as `.h5` files, and reloaded on repeat runs. The file name encodes all parameters, acting as a cache key.
- **openPMD format**: All field files follow the openPMD 1.1.0 standard so they are compatible with WarpX's `read_from_file` interface and can be inspected with `openPMD-viewer`.
- **Non-vectorized E-field**: The analytic E-field integration loops over every grid point. This is slow for large `N`; see [eext module](modules/eext.md) for details.
