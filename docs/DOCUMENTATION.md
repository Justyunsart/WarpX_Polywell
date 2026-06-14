# WarpX Polywell Simulation — Full Documentation

> **How to use this file:**
> Render on GitHub Pages, convert to PDF via `pandoc DOCUMENTATION.md -o DOCUMENTATION.pdf --toc`, or read as plain Markdown.
> Individual editable files are kept in the sub-folders (`setup/`, `simulation/`, `modules/`, `physics/`) — this file is regenerated from them.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Installation & Environment Setup](#2-installation--environment-setup)
3. [Project Structure](#3-project-structure)
4. [Running a Simulation](#4-running-a-simulation)
5. [Input Parameters Reference](#5-input-parameters-reference)
6. [Output Files](#6-output-files)
7. [Module: `src/warpx_polywell/bext`](#7-module-srcwarpx_polywellbext)
8. [Module: `src/warpx_polywell/eext`](#8-module-srcwarpx_polywelleext)
9. [Module: `src/warpx_polywell/utils`](#9-module-srcwarpx_polywellutils)
10. [Physics Background: Polywell Fusion](#10-physics-background-polywell-fusion)

---

## 1. Project Overview

This project uses the [WarpX](https://ecp-warpx.github.io/) particle-in-cell (PIC) framework to simulate plasma confinement in a **polywell** magnetic configuration. The workflow is:

1. **Generate external field grids** (B and E) using analytical/numerical methods and write them to openPMD-compliant HDF5 files.
2. **Run a PIC simulation** — WarpX reads those grids at startup, then self-consistently evolves particle trajectories and electromagnetic fields.
3. **Output diagnostics** — field snapshots and particle data are written every 100 steps in openPMD/HDF5 format.

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Field pre-computation + caching | B and E fields are expensive to compute. Results are saved to `.h5` files; the filename encodes all parameters as a cache key so repeated runs skip computation. |
| openPMD 1.1.0 format | Ensures compatibility with WarpX's `read_from_file` interface, `openpmd_viewer`, VisIt, and ParaView. |
| Non-vectorised E-field loop | The analytic E-field integrand has per-coil coordinate transforms that are difficult to batch; each of the N³ × 6 evaluations is computed in a plain Python loop. |
| `do_initial_div_cleaning = 0` | The magpylib B-field is analytically divergence-free. Divergence cleaning with open boundary conditions introduces numerical errors at startup and is unnecessary. |

### Quick Start

```bash
conda activate warpx-env
cd /path/to/WarpX
python inputs/polywell_input.py
```

---

## 2. Installation & Environment Setup

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python 3.9+ (managed through conda)
- 8+ CPU cores recommended (`N` must be divisible by the MPI rank count)

### Step 1 — Create a Conda Environment

```bash
conda create -n warpx-env python=3.10
conda activate warpx-env
```

### Step 2 — Install WarpX via conda-forge

```bash
conda install -c conda-forge warpx
```

This installs `pywarpx` (the PICMI Python interface) and all compiled WarpX dependencies (MPI, HDF5, ADIOS2, etc.).

### Step 3 — Install Additional Packages

```bash
conda install -c conda-forge magpylib h5py
```

| Package | Purpose |
|---|---|
| `magpylib` | Magnetic field calculation for current loop coils |
| `h5py` | Read/write HDF5 field grid files |
| `numpy` | Array math (installed as a WarpX transitive dependency) |

### Step 4 — Verify

```python
import pywarpx
from pywarpx import picmi
import magpylib
import h5py
print("All imports OK")
```

### Dependency Summary

| Package | Source | Required For |
|---|---|---|
| `warpx` / `pywarpx` | conda-forge | PIC simulation engine |
| `magpylib` | conda-forge | Coil B-field computation |
| `h5py` | conda-forge | HDF5 file I/O |
| `numpy` | conda-forge (transitive) | Array operations |

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'warpx_polywell'` | Install the package: `poetry install` (with `warpx-env` active) |
| MPI domain decomposition error | `N` must be divisible by rank count. Default `N=72` is divisible by 1,2,3,4,6,8,9,12... |
| Divergence cleaning crash at startup | Keep `warpx.do_initial_div_cleaning = 0` |

---

## 3. Project Structure

```
WarpX/
├── docs/                          # Documentation (this directory)
│   ├── DOCUMENTATION.md           # This file — full combined reference
│   ├── README.md                  # Navigation index
│   ├── packages                   # Conda packages used during development
│   ├── setup/
│   │   ├── installation.md
│   │   └── project-structure.md
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
│   └── polywell_input.py          # Main simulation entry point (edit parameters here)
│
├── output/
│   └── bext/                      # Auto-generated HDF5 field files
│       └── *.h5
│
├── src/
│   └── warpx_polywell/            # Importable package (poetry src-layout)
│       ├── bext/
│       │   ├── bext.py            # B-field HDF5 generation
│       │   └── make_collection.py # magpylib coil geometry
│       ├── eext/
│       │   ├── eext.py            # E-field HDF5 population
│       │   └── methods.py         # Analytic E-field methods + EMethods enum
│       ├── post/
│       │   └── reader.py          # Post-processing placeholder (empty)
│       └── utils/
│           ├── cyl.py             # Cartesian ↔ Cylindrical coordinate conversion
│           └── paths.py           # ROOT_DIR and BEXT_DIR path constants
│
└── tests/                         # Jupyter notebooks for exploration
    ├── input_fields.ipynb
    ├── magpylib.ipynb
    └── openPMD-viewer.ipynb
```

### Key File Roles

| File | Role |
|---|---|
| `inputs/polywell_input.py` | Entry point. Contains all user parameters. Orchestrates field file generation, WarpX setup, and `sim.step()`. |
| `src/warpx_polywell/bext/bext.py` | Computes 3D B-field grid from polywell coils via magpylib; writes openPMD HDF5. |
| `src/warpx_polywell/bext/make_collection.py` | Builds 6-coil polywell `magpylib.Collection`. |
| `src/warpx_polywell/eext/eext.py` | Overwrites zeroed E-field datasets in existing HDF5 with analytically computed values. |
| `src/warpx_polywell/eext/methods.py` | Contains `fw_e`, `bob_e` analytic integrand functions and the `EMethods` enum. |
| `src/warpx_polywell/utils/paths.py` | `ROOT_DIR` and `BEXT_DIR` path constants, resolved relative to `__file__`. |
| `src/warpx_polywell/utils/cyl.py` | `toCyl()` and `toCart()` coordinate helpers. |
| `output/bext/` | Cached HDF5 field files. Filename encodes all parameters as cache key. |

---

## 4. Running a Simulation

### Basic Run

```bash
cd /path/to/WarpX
conda activate warpx-env
python inputs/polywell_input.py
```

### MPI Parallel Run

```bash
mpirun -n 8 python inputs/polywell_input.py
```

> `N` must be divisible by the MPI rank count. Default `N=72` works with 8 ranks.

### Execution Flow

```
polywell_input.py
│
├── 1. Read user parameters (top of file)
│
├── 2. Build WarpX objects
│   ├── picmi.Cartesian3DGrid       — 3D domain [-L,L]³ with open/absorbing BCs
│   ├── picmi.ElectromagneticSolver — Yee scheme, CFL=0.99
│   ├── picmi.UniformDistribution   — plasma initialised in ±(0.11·L)³ region
│   ├── picmi.Species (electron)    — plasma_e
│   └── picmi.Species (proton)      — plasma_i
│
├── 3. Generate / load B-field file
│   └── make_bext_file(I, b_dia, b_offset, L, N)
│       ├── File exists → return cached path immediately
│       └── File absent → compute with magpylib, write HDF5, return path
│
├── 4. (Optional) Generate / load E-field
│   └── fill_eext_file(path, method, e_dia, e_offset, Q, L, N)
│       ├── File exists → return cached path immediately
│       └── File absent → compute analytically, update HDF5, rename file, return path
│
├── 5. warpx.read_fields_from_path = ext_path
│
├── 6. picmi.Simulation(solver, max_steps, verbose=True)
│      + add_species(plasma_e, plasma_i)
│      + add_diagnostic(field_diag, part_diag)
│
└── 7. sim.step()  — runs for max_steps
```

### Field File Caching

Both field generators check for a file with the current-parameter-encoded name before computing:

| File type | Name pattern |
|---|---|
| B only | `B_ext_I-{I}A_D-{dia}m_Off-{offset}m_L-{L}m_N-{N}.h5` |
| B + E | `B_ext_..._E_ext_Q-{Q}_D-{e_dia}m_offset-{e_offset}m_C_L-{L}m_N-{N}.h5` |

Old files are not deleted automatically — clean `output/bext/` manually when needed.

### Monitoring Progress

E-field integration prints progress every 10%:

```
[get_e_field_data] Computing E-field: 0/373248 points (0%) done
[get_e_field_data] Computing E-field: 37324/373248 points (10%) done
...
[get_e_field_data] E-field computation complete. Peak magnitude: 3.1234e+03 V/m
```

---

## 5. Input Parameters Reference

All parameters are in the `=== USER INPUTS ===` block at the top of `inputs/polywell_input.py`.

### Simulation Control

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `max_steps` | `10000` | steps | Total PIC time steps |
| `warpx.const_dt` | `1e-12` | s | Fixed time step size |
| `p_density` | `1e18` | m⁻³ | Number density of each plasma species |

### B-Field — Polywell Coils

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `I` | `1e6` | A | Current through each coil |
| `b_dia` | `1` | m | Coil diameter |
| `b_offset` | `1.1` | m | Distance from origin to coil center |

### E-Field — Charged Rings

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `e_method` | `"FW"` | — | Method name, or `None` to disable E-field |
| `Q` | `1e-9` | C | Total charge per ring |
| `e_dia` | `0.75` | m | Ring diameter |
| `e_offset` | `1.1` | m | Distance from origin to ring center |

#### Available E-field methods

| Name | Function | Quadrature points |
|---|---|---|
| `"FW"` | `fw_e()` | 500 (angular) |
| `"BOB"` | `bob_e()` | 100 (angular) |
| `None` | — | E-field disabled; zeros used |

### Grid

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `L` | `3` | m | Domain half-length; full domain is `[-L, L]³` |
| `N` | `72` | — | Grid cells per axis; must be divisible by MPI rank count |
| `number_per_cell_each_dim` | `[10, 10, 10]` | — | Macro-particles per cell per dimension per species |

### Scale Factors

| Parameter | Default | Description |
|---|---|---|
| `plasma_bounding` | `0.11` | Fraction of `L`; plasma initialised within `±(L × 0.11)` |

### Solver Options

```python
# Active:
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.99)

# Alternatives (commented out in file):
# solver = picmi.ElectrostaticSolver(grid=grid)
# solver = picmi.HybridPICSolver(...)  # Note: does NOT support E-field from file
```

### Diagnostics

Both diagnostics write every 100 steps in openPMD/HDF5 format.

| Diagnostic | Fields / data |
|---|---|
| `field_diag` | `Bx`, `By`, `Bz`, `Jx`, `Jy`, `Jz`, `part_per_cell` |
| `part_diag` | `x`, `y`, `z`, `ux`, `uy`, `uz`, `weighting` for `plasma_e` and `plasma_i` |

---

## 6. Output Files

### 6.1 External Field Files (`output/bext/*.h5`)

Pre-computed inputs to WarpX. Generated once, cached, reused.

#### HDF5 Internal Layout

```
/ (root)
│   attrs: openPMD="1.1.0", basePath="/data/%T/", meshesPath="meshes/", ...
└── data/
    └── 1/
        │   attrs: time=0.0, dt=0.0, timeUnitSI=1.0
        └── meshes/
            ├── B/
            │   attrs: geometry="cartesian", gridSpacing=[dx,dy,dz],
            │          gridGlobalOffset=[-L,-L,-L], unitDimension=[0,1,1,-2,0,0,-1]
            │   ├── x  float64 (N,N,N)  unitSI=1.0 (Tesla)  position=[0.5,0.5,0.5]
            │   ├── y  float64 (N,N,N)
            │   └── z  float64 (N,N,N)
            └── E/
                attrs: geometry="cartesian", gridSpacing=[dx,dy,dz],
                       gridGlobalOffset=[-L,-L,-L], unitDimension=[1,1,-3,-1,0,0,0]
                ├── x  float64 (N,N,N)  unitSI=1.0 (V/m)  position=[0.5,0.5,0.5]
                ├── y  float64 (N,N,N)
                └── z  float64 (N,N,N)
```

#### Reading with h5py

```python
import h5py
with h5py.File("output/bext/B_ext_....h5", "r") as f:
    Bx = f["data/1/meshes/B/x"][:]          # shape (N, N, N), Tesla
    Ex = f["data/1/meshes/E/x"][:]          # shape (N, N, N), V/m
    spacing = f["data/1/meshes/B"].attrs["gridSpacing"]       # [dx, dy, dz]
    offset  = f["data/1/meshes/B"].attrs["gridGlobalOffset"]  # [-L, -L, -L]
```

#### Reading with openPMD-viewer

```python
from openpmd_viewer import OpenPMDTimeSeries
ts = OpenPMDTimeSeries("output/bext/")
Bz, info = ts.get_field(field="B", coord="z", iteration=1)
```

### 6.2 Simulation Diagnostic Files

Written by WarpX every 100 steps during `sim.step()`.

#### Field Diagnostics (`field_diag/`)

| Field | Description |
|---|---|
| `Bx`, `By`, `Bz` | Magnetic field (T) |
| `Jx`, `Jy`, `Jz` | Current density (A/m²) |
| `part_per_cell` | Macro-particles per cell |

#### Particle Diagnostics (`part_diag/`)

| Quantity | Description |
|---|---|
| `x`, `y`, `z` | Positions (m) |
| `ux`, `uy`, `uz` | Normalised momenta γv/c |
| `weighting` | Macro-particle statistical weight |

Both use `warpx_format='openpmd'`, `warpx_openpmd_backend='h5'` and are compatible with `openpmd_viewer`, VisIt, and ParaView.

---

## 7. Module: `src/warpx_polywell/bext`

Generates the 3D B-field grid from polywell coil geometry and writes it as an openPMD HDF5 file.

### Files

| File | Purpose |
|---|---|
| `bext.py` | Top-level HDF5 creation and B-field population |
| `make_collection.py` | Builds the magpylib coil geometry |

### `make_collection.py`

#### `make_polywell_collection(a, dia, d)` → `magpylib.Collection`

Six circular current loops at ±x, ±y, ±z with alternating polarities.

| Coil | Position | Current | Rotation |
|---|---|---|---|
| `s1` | `(-d, 0, 0)` | `+a` | 90° around Y |
| `s2` | `(+d, 0, 0)` | `-a` | 90° around Y |
| `s3` | `(0, -d, 0)` | `-a` | −90° around X |
| `s4` | `(0, +d, 0)` | `+a` | −90° around X |
| `s5` | `(0, 0, -d)` | `+a` | 90° around Z |
| `s6` | `(0, 0, +d)` | `-a` | 90° around Z |

#### `make_helmholtz_collection(a, dia, d)` → `magpylib.Collection`

Two-coil Helmholtz pair for testing. Not used in the production pipeline.

### `bext.py`

#### `get_bext_file_name(I, dia, offset, L, N)` → `str`

Returns the expected filename for a given parameter set. Used to check for cached files.

```python
get_bext_file_name(1e6, 1.0, 1.1, 3, 72)
# → "B_ext_I-1000000.0A_D-1m_Off-1.1m_L-3m_N-72.h5"
```

#### `make_bext_file(I, dia, offset, L, N)` → `Path`

Main public function. Returns cached path if file exists; otherwise computes and writes.

**Pipeline when file is absent:**

1. `make_polywell_collection(I, dia, offset)` → `magpylib.Collection`
2. `np.meshgrid(linspace(-L,L,N), ..., indexing='ij')` → mesh shape `(N,N,N,3)`
3. `collection.getB(mesh)` → `B` shape `(N,N,N,3)`; split into `Bx, By, Bz`
4. `_make_empty_ext_h5(file_path)` → skeleton HDF5 with openPMD metadata
5. `_fill_h5_file(...)` → writes B datasets; initialises E datasets to zero

#### `_make_empty_ext_h5(filename)` — openPMD attributes written

| Attribute | Value |
|---|---|
| `openPMD` | `"1.1.0"` |
| `basePath` | `"/data/%T/"` |
| `iterationEncoding` | `"fileBased"` |
| B `unitDimension` | `[0, 1, 1, -2, 0, 0, -1]` (Tesla) |
| E `unitDimension` | `[1, 1, -3, -1, 0, 0, 0]` (V/m) |

### Data Flow

```
make_bext_file(I, dia, offset, L, N)
   ├─ [cached?] → return path
   └─ make_polywell_collection
         → meshgrid (N,N,N,3)
         → collection.getB()  [vectorised]
         → _make_empty_ext_h5 + _fill_h5_file
         → return path
```

---

## 8. Module: `src/warpx_polywell/eext`

Computes the E-field from charged rings and overwrites the zeroed E-field placeholder
left by `bext.py`.

### Files

| File | Purpose |
|---|---|
| `eext.py` | Grid computation and HDF5 population |
| `methods.py` | Analytic E-field integrands + `EMethods` enum |

### `methods.py`

#### `fw_e(r, z, a, Q, resolution=500)` → `(E_r, E_z)`

Numerical integration of E-field from a charged ring using a Riemann sum over θ ∈ [0, 2π].

Linear charge density: `λ = Q / (2πa)`

```
E_r = (λa / 4πε₀) ∫₀²π  (r − a cosθ) / D³  dθ
E_z = (λa z / 4πε₀) ∫₀²π  1 / D³  dθ
D   = √(r² + a² − 2ar cosθ + z²)
```

#### `bob_e(r, z, a, Q, resolution=100)` → `(E_r, E_z)`

Alternative normalised integration over θ ∈ [0, π] using Coulomb's constant `k = 8.99×10⁹ N·m²/C²`. Clamps `r` to `1e-10` near the axis to avoid singularities.

#### `EMethods` Enum

```python
class EMethods(Enum):
    FW  = (fw_e,)
    BOB = (bob_e,)
```

Usage:

```python
method = EMethods["FW"].value[0]   # → fw_e callable
```

### `eext.py`

#### `fill_eext_file(filepath, method, dia, offset, Q, L, N)` → `Path`

Main public function. Computes E-field grid and writes it into the B-field HDF5.
Returns immediately if the renamed output file already exists.

**Pipeline when file is absent:**

1. `get_e_field_data(method, ...)` → `Ex, Ey, Ez` of shape `(N,N,N)`
2. `_fill_efield_datasets(filepath, ...)` → overwrites E datasets in HDF5
3. `filepath.rename(new_filepath)` → appends E-field parameters to filename

#### `get_e_field_data(...)` — Inner Loop

For each grid point `(i,j,k)` and each of the 6 coils:

```
orient_point(coil, cartesian_point)
    → subtract coil.position
    → apply coil.orientation.inv()
toCyl(local_cartesian)
    → (r, φ, z)
method(r, z, a, Q)
    → (E_r, E_z)
toCart(E_r, φ, E_z)
    → (ΔEx, ΔEy, ΔEz)
accumulate into Ex[i,j,k], Ey[i,j,k], Ez[i,j,k]
```

> **Performance:** O(N³ × 6) non-vectorised calls. At N=72: ~2.2M evaluations. Progress logged every 10%.

### Coordinate Transform Pipeline

```
World Cartesian
    → orient_point  (subtract position, inverse rotation)
    → Local Cartesian  (ring at origin, axis = z)
    → toCyl  → (r, φ, z)
    → method  → (E_r, E_z)   [E_φ = 0 by symmetry]
    → toCart  → (ΔEx, ΔEy, ΔEz)
    → accumulate
```

---

## 9. Module: `src/warpx_polywell/utils`

### `cyl.py`

#### `toCyl(coord)` → `ndarray([ρ, φ, z])`

```
ρ = √(x² + y²)
φ = arctan2(y, x)   [radians]
z = coord[2]
```

#### `toCart(r, theta, z)` → `(x, y, z)`

```
x = r · cos(θ)
y = r · sin(θ)
z = z
```

> `theta` is the azimuthal angle of the **position** point (from `toCyl`), not of the vector. Correct because E_φ = 0.

### `paths.py`

```python
_script_dir = Path(__file__).resolve().parent          # src/warpx_polywell/utils/
ROOT_DIR    = _script_dir.parent.parent.parent          # WarpX/ (repo checkout)
OUTPUT_DIR  = Path(get_config()["LOCAL_OUTPUT_DIR"])    # .env base, else <repo>/output
BEXT_DIR    = OUTPUT_DIR / "bext"                        # field-file cache
```

| Constant | Resolves to | Used by |
|---|---|---|
| `ROOT_DIR` | repo checkout | git metadata; default output base |
| `OUTPUT_DIR` | `LOCAL_OUTPUT_DIR` (`.env`), else `<repo>/output` | runs, `runs.db`, `BEXT_DIR` |
| `BEXT_DIR` | `OUTPUT_DIR/bext/` | `src/warpx_polywell/bext/bext.py` |

**Adding a new path** (anchor on `OUTPUT_DIR` so it follows `.env`):

```python
# In paths.py
NEW_DIR = OUTPUT_DIR / "new_dir"

# Elsewhere
from warpx_polywell.utils.paths import NEW_DIR
```

---

## 10. Physics Background: Polywell Fusion

### The Polywell Concept

A polywell is an inertial electrostatic confinement (IEC) fusion device combining:

1. **Magnetic cusps** from a polyhedral coil arrangement — create a magnetic well confining electrons near the centre
2. **Electrostatic potential well** — the confined electron cloud builds negative space charge that accelerates positive ions inward

### Magnetic Configuration

Six coils on cube faces carry alternating currents. The resulting field forms magnetic mirrors at the cusps; electrons are reflected back by the stronger field regions.

```
         +y
    s4(↑) | s4 at (0, +d, 0)
          |
s1(↑)----[O]----s2(↓)    s1 at (-d,0,0), s2 at (+d,0,0)
          |
    s3(↓) | s3 at (0, -d, 0)

s5(↑) at (0,0,-d),  s6(↓) at (0,0,+d)  [along z, not shown]
```

### Electrostatic Field

Modelled as six uniformly charged rings (same geometry as coils). Field equations for ring of radius `a`, charge `Q`, at position `(r,z)` in cylindrical:

```
E_r = (λa / 4πε₀) ∫₀²π (r − a cosθ)/D³ dθ
E_z = (λa z / 4πε₀) ∫₀²π 1/D³ dθ

λ = Q/(2πa),   D = √(r² + a² − 2ar cosθ + z²)
```

### PIC Simulation Setup

| Property | Value |
|---|---|
| Grid | 3D Cartesian `[-L, L]³` |
| Field BCs | Open |
| Particle BCs | Absorbing |
| Solver | Electromagnetic, Yee scheme, CFL = 0.99 |
| Species | Electrons + Protons |
| Initial distribution | Uniform within `±0.11L` per axis |
| Initial velocity | 0.9c in z-direction |
| External fields | Pre-computed B and E, fixed (read-only) |
| Divergence cleaning | Disabled (`do_initial_div_cleaning = 0`) |

### Key Physical Scales

| Quantity | Value |
|---|---|
| Domain size | 6 m × 6 m × 6 m |
| Coil radius | 0.5 m |
| Coil current | 1 MA |
| Plasma density | 10¹⁸ m⁻³ |
| Time step | 1 ps |
| Total simulated time (default) | 10 ns |

### References

- Bussard, R.W. (1991). "Some Physics Considerations of Magnetic Inertial-Electrostatic Confinement." *Fusion Technology*, 19(2).
- WarpX documentation: <https://ecp-warpx.github.io/>
- magpylib documentation: <https://magpylib.readthedocs.io/>
- openPMD standard: <https://github.com/openPMD/openPMD-standard>

---

*Last updated: 2026-03-18*
