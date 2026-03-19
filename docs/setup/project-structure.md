# Project Structure

```
WarpX/
├── docs/                          # Documentation (this directory)
│   ├── README.md                  # Documentation index
│   ├── packages                   # Conda packages installed during development
│   ├── setup/
│   │   ├── installation.md
│   │   └── project-structure.md   # (this file)
│   ├── simulation/
│   │   ├── running.md
│   │   ├── parameters.md
│   │   └── output.md
│   ├── modules/
│   │   ├── bext.md
│   │   ├── eext.md
│   │   └── utils.md
│   └── physics/
│       └── polywell.md
│
├── inputs/
│   └── polywell_input.py          # Main simulation entry point
│
├── output/
│   └── bext/                      # Generated external field HDF5 files (auto-created)
│       └── *.h5
│
├── src/                           # Python source package
│   ├── bext/
│   │   ├── bext.py                # B-field HDF5 file generation
│   │   └── make_collection.py     # magpylib coil geometry setup
│   ├── eext/
│   │   ├── eext.py                # E-field HDF5 file population
│   │   └── methods.py             # Analytic E-field methods + EMethods enum
│   ├── post/
│   │   └── reader.py              # Post-processing placeholder (empty)
│   └── utils/
│       ├── cyl.py                 # Cartesian <-> Cylindrical coordinate conversion
│       └── paths.py               # Centralised path constants (ROOT_DIR, BEXT_DIR)
│
└── tests/                         # Jupyter notebooks for exploration
    ├── input_fields.ipynb
    ├── magpylib.ipynb
    └── openPMD-viewer.ipynb
```

---

## Key Files Explained

### `inputs/polywell_input.py`
The single entry point for a simulation run. Contains all user-facing parameters
(grid size, resolution, currents, plasma density, etc.) and orchestrates:
1. Field file generation via `src/bext` and `src/eext`
2. WarpX grid, solver, and species setup via PICMI
3. Diagnostic configuration
4. `sim.step()` to launch the run

See [Input Parameters](../simulation/parameters.md) for a full parameter reference.

### `src/bext/bext.py`
Computes the 3D magnetic field from a polywell coil configuration and writes it to
an openPMD HDF5 file. Uses magpylib for the field calculation.
See [bext module](../modules/bext.md).

### `src/bext/make_collection.py`
Constructs the 6-coil polywell formation as a `magpylib.Collection`.
Each coil is a circular current loop placed at ±x, ±y, ±z with alternating current
directions to create the polywell magnetic well.

### `src/eext/eext.py`
Reads an existing B-field HDF5 file and overwrites the (initially zeroed) E-field
datasets with analytically computed values.
See [eext module](../modules/eext.md).

### `src/eext/methods.py`
Contains two analytic implementations of the E-field from a charged ring:
- `fw_e` — numerical integration (Riemann sum over angle)
- `bob_e` — alternative normalised integration

Registered in the `EMethods` enum so `polywell_input.py` can select by name string.

### `src/utils/paths.py`
Defines `ROOT_DIR` and `BEXT_DIR` using `__file__`-relative resolution so paths
are correct regardless of where scripts are invoked from.

### `src/utils/cyl.py`
Two lightweight helper functions: `toCyl(xyz)` and `toCart(r, theta, z)`.
Used by the E-field module to transform between coordinate systems when applying
per-coil local frames.

### `output/bext/`
Auto-generated at runtime. Stores cached HDF5 field files. File names encode all
parameters (current, diameter, offset, grid length, resolution) so the correct file
can be located on repeat runs without recomputation.

### `tests/`
Jupyter notebooks used during development for interactive exploration.
Not part of the production run pipeline.
