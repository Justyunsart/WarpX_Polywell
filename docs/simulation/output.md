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
B_ext_I-{I}A_D-{dia}m_Off-{offset}m_L-{L}m_N-{N}_sym-{symmetry}.h5
```

**B + E combined:**
```
B_ext_I-{I}A_D-{dia}m_Off-{offset}m_L-{L}m_N-{N}_sym-{symmetry}_E_ext_Q-{Q}_D-{e_dia}m_offset-{e_offset}m_C_L-{L}m_N-{N}.h5
```

`L` and `N` are the user-facing full-domain values; `symmetry` discriminates the sampled grid (full vs octant), since the same `L`/`N` yield different actual sampling in each mode.

### HDF5 Internal Structure

Array shapes match the simulated grid, not the full-domain `N`. In octant mode the arrays are `(N/2, N/2, N/2)` and the `gridGlobalOffset` is `(0, 0, 0)`; in full mode the arrays are `(N, N, N)` with `gridGlobalOffset = (-L, -L, -L)`.

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
            │   ├── x   — float64 array, shape (Nx, Ny, Nz), units Tesla
            │   ├── y   — float64 array, shape (Nx, Ny, Nz), units Tesla
            │   └── z   — float64 array, shape (Nx, Ny, Nz), units Tesla
            └── E/
                ├── attrs: geometry, gridSpacing, gridGlobalOffset, unitDimension, ...
                ├── x   — float64 array, shape (Nx, Ny, Nz), units V/m
                ├── y   — float64 array, shape (Nx, Ny, Nz), units V/m
                └── z   — float64 array, shape (Nx, Ny, Nz), units V/m
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

WarpX writes these during the run, every `period=10` steps. Field and particle diagnostics share `name="diag"` and therefore land in a **single openPMD series** at `diags/diag/openpmd_%T.h5`. Each iteration file contains both `meshes/` (field side) and `particles/` (per-particle records).

### Field side (mesh data)

| Field | Description |
|---|---|
| `Bx`, `By`, `Bz` | Magnetic field components (T) |
| `Bx_fp_external`, `By_fp_external`, `Bz_fp_external` | External B-field from the file (T) |
| `Ex_fp_external`, `Ey_fp_external`, `Ez_fp_external` | External E-field from the file (V/m) |
| `Jx`, `Jy`, `Jz` | Current density components (A/m²) |
| `part_per_cell` | Number of macroparticles per cell |

### Particle side (per-particle records)

| Quantity | Description |
|---|---|
| `x`, `y`, `z` | Particle positions (m) |
| `ux`, `uy`, `uz` | Normalised momenta (dimensionless, = γv/c) |
| `weighting` | Macroparticle weight (number of real particles represented) |

Species recorded: `plasma_i` (protons). The electron block is commented out by default in the deck.

### Diagnostic Format

Both diagnostics use `warpx_format='openpmd'` with `warpx_openpmd_backend='h5'`. The shared `name=` is what merges them into a single series — give them distinct names to split back into parallel folders.

Compatible with:
- `openpmd_viewer` (Python)
- `VisIt` and `ParaView` (with openPMD plugin)
- `h5py` (direct low-level access)

```python
from openpmd_viewer import OpenPMDTimeSeries
ts = OpenPMDTimeSeries("output/runs/run_*/diags/diag/")
Bz, info     = ts.get_field(field="B", coord="z", iteration=0)
x, y, z, ux  = ts.get_particle(["x", "y", "z", "ux"], species="plasma_i", iteration=0)
```

---

## Post-Processing

`src/post/reader.py` is a placeholder for post-processing utilities.
It is currently empty. Analysis should be done directly via the notebooks in `tests/`
or custom scripts using `h5py` / `openpmd_viewer`.
