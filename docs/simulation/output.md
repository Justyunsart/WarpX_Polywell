# Output Files

The simulation produces two categories of output files: **field grid files** (pre-computed inputs)
and **diagnostic files** (simulation results).

---

## 1. External Field Grid Files (`output/bext/`)

These are the HDF5 files generated before the simulation runs.
They are inputs to WarpX, not results from it.

### File Naming Convention

**B-field only:**
```
B_ext_I-{I}A_D-{dia}m_Off-{offset}m_L-{L}m_N-{N}.h5
```

**B + E combined:**
```
B_ext_I-{I}A_D-{dia}m_Off-{offset}m_L-{L}m_N-{N}_E_ext_Q-{Q}_D-{e_dia}m_offset-{e_offset}m_C_L-{L}m_N-{N}.h5
```

### HDF5 Internal Structure

```
/ (root)
├── attrs:
│   openPMD = "1.1.0"
│   basePath = "/data/%T/"
│   meshesPath = "meshes/"
│   ...
└── data/
    └── 1/
        ├── attrs: time=0.0, dt=0.0, timeUnitSI=1.0
        └── meshes/
            ├── B/
            │   ├── attrs: geometry, gridSpacing, gridGlobalOffset, unitDimension, ...
            │   ├── x   — float64 array, shape (N, N, N), units Tesla
            │   ├── y   — float64 array, shape (N, N, N), units Tesla
            │   └── z   — float64 array, shape (N, N, N), units Tesla
            └── E/
                ├── attrs: geometry, gridSpacing, gridGlobalOffset, unitDimension, ...
                ├── x   — float64 array, shape (N, N, N), units V/m
                ├── y   — float64 array, shape (N, N, N), units V/m
                └── z   — float64 array, shape (N, N, N), units V/m
```

Each component dataset has attributes:
- `unitSI`: 1.0 (SI units)
- `position`: `[0.5, 0.5, 0.5]` (cell-centered)
- `shape`: `(N, N, N)`

### Reading with h5py

```python
import h5py
import numpy as np

with h5py.File("output/bext/B_ext_....h5", "r") as f:
    Bx = f["data/1/meshes/B/x"][:]   # shape (N, N, N)
    By = f["data/1/meshes/B/y"][:]
    Bz = f["data/1/meshes/B/z"][:]
    Ex = f["data/1/meshes/E/x"][:]
    # grid spacing and offset
    spacing = f["data/1/meshes/B"].attrs["gridSpacing"]   # [dx, dy, dz] in metres
    offset  = f["data/1/meshes/B"].attrs["gridGlobalOffset"]  # [-L, -L, -L]
```

### Reading with openPMD-viewer

The `tests/openPMD-viewer.ipynb` notebook shows how to use the `openpmd_viewer` package
to inspect and visualise field data interactively.

```python
from openpmd_viewer import OpenPMDTimeSeries
ts = OpenPMDTimeSeries("output/bext/")
Bz, info = ts.get_field(field="B", coord="z", iteration=1)
```

---

## 2. Simulation Diagnostic Files

WarpX writes these during the run, every `period=100` steps.

### Field Diagnostics (`field_diag/`)

Contains mesh data for each diagnostic period:

| Field | Description |
|---|---|
| `Bx`, `By`, `Bz` | Magnetic field components (T) |
| `Jx`, `Jy`, `Jz` | Current density components (A/m²) |
| `part_per_cell` | Number of macro-particles per cell |

### Particle Diagnostics (`part_diag/`)

Contains particle data for each diagnostic period:

| Quantity | Description |
|---|---|
| `x`, `y`, `z` | Particle positions (m) |
| `ux`, `uy`, `uz` | Normalised momenta (dimensionless, = γv/c) |
| `weighting` | Macro-particle weight (number of real particles represented) |

Species recorded: `plasma_e` (electrons) and `plasma_i` (protons).

### Diagnostic Format

Both diagnostics use `warpx_format='openpmd'` with `warpx_openpmd_backend='h5'`.
This produces standard openPMD HDF5 files compatible with:
- `openpmd_viewer` (Python)
- `VisIt` and `ParaView` (with openPMD plugin)
- `h5py` (direct low-level access)

---

## Post-Processing

`src/post/reader.py` is a placeholder for post-processing utilities.
It is currently empty. Analysis should be done directly via the notebooks in `tests/`
or custom scripts using `h5py` / `openpmd_viewer`.
