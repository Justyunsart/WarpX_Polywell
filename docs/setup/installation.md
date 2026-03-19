# Installation & Environment Setup

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Python 3.9+ (managed through conda)
- A machine with at least 8 CPU cores recommended (the grid resolution `N` must be divisible by the MPI rank count)

---

## 1. Create a Conda Environment

WarpX and its Python bindings are distributed through `conda-forge`.
Create a dedicated environment:

```bash
conda create -n warpx-env python=3.10
conda activate warpx-env
```

---

## 2. Install WarpX (Python bindings via conda-forge)

```bash
conda install -c conda-forge warpx
```

This installs `pywarpx` (the PICMI Python interface) along with all compiled WarpX dependencies
(MPI, HDF5, ADIOS2, etc.).

---

## 3. Install Additional Python Packages

```bash
conda install -c conda-forge magpylib h5py
```

| Package | Purpose |
|---|---|
| `magpylib` | Magnetic field calculation for current loop coils |
| `h5py` | Read/write HDF5 field grid files |
| `numpy` | Array math (installed as a WarpX dependency) |

> **Note**: `pywarpx` and `picmi` are installed as part of the `warpx` conda package.
> You do not need to install them separately.

---

## 4. Verify the Install

```python
# Quick sanity check — run this in a Python shell
import pywarpx
from pywarpx import picmi
import magpylib
import h5py
print("All imports OK")
```

---

## 5. Clone / Open the Project

The project root is expected to be the working directory when running any script:

```bash
cd /path/to/WarpX
python inputs/polywell_input.py
```

Imports like `from src.bext.bext import make_bext_file` rely on the project root being on
`sys.path`. If you run scripts from a subdirectory, you will get `ModuleNotFoundError`.

---

## Dependency Summary

| Package | Source | Required For |
|---|---|---|
| `warpx` / `pywarpx` | conda-forge | PIC simulation engine |
| `magpylib` | conda-forge | Coil B-field computation |
| `h5py` | conda-forge | HDF5 file I/O |
| `numpy` | conda-forge (transitive) | Array operations throughout |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Run the script from the project root, not from inside `inputs/` or `src/`.

**`ValueError: N must be divisible by number of MPI ranks`**
Change `N` in `polywell_input.py` to a value divisible by your MPI rank count.
On an 8-core machine, `N = 72` (divisible by 8) is the default.

**WarpX crashes on first divergence cleaning step**
`warpx.do_initial_div_cleaning = 0` is already set in `polywell_input.py` to avoid this.
Do not re-enable it when using open boundary conditions with a pre-computed divergence-free B-field.
