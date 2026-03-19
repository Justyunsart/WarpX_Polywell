# `src/utils` — Utilities Module

Lightweight helpers used across the codebase for coordinate conversion and path management.

---

## Files

| File | Purpose |
|---|---|
| `src/utils/cyl.py` | Cartesian ↔ Cylindrical coordinate conversion |
| `src/utils/paths.py` | Centralised path constants |

---

## `cyl.py`

Used by `src/eext/eext.py` to convert field vectors between coordinate systems
when applying coil-local frames.

### `toCyl(coord)` → `ndarray([r, φ, z])`

Converts a 3D Cartesian coordinate to cylindrical.

```
Parameters:
    coord : array-like of shape (3,) — [x, y, z] in metres

Returns:
    ndarray([ρ, φ, z])
        ρ = sqrt(x² + y²)        — radial distance from z-axis
        φ = arctan2(y, x)        — azimuthal angle in radians
        z = coord[2]             — axial component (unchanged)
```

### `toCart(r, theta, z)` → `(x, y, z)`

Converts cylindrical field components back to Cartesian.

```
Parameters:
    r     : float — radial component of a vector (e.g. E_r)
    theta : float — azimuthal angle of the point in radians (φ from toCyl)
    z     : float — axial component

Returns:
    (x, y, z) : tuple of floats
        x = r · cos(θ)
        y = r · sin(θ)
        z = z (unchanged)
```

> **Important:** `theta` here is the azimuthal angle of the *position*, not of the vector.
> This is correct because the cylindrical E-field has no φ-component by symmetry,
> so only `E_r` needs to be projected back using the position angle.

---

## `paths.py`

Defines project-wide path constants using `__file__`-relative resolution,
so all scripts find the correct directories regardless of where they are invoked from.

```python
from pathlib import Path

_script_dir = Path(__file__).resolve().parent   # → .../src/utils/
ROOT_DIR    = _script_dir.parent.parent          # → .../WarpX/

BEXT_DIR    = ROOT_DIR / "output" / "bext"      # → .../WarpX/output/bext/
```

### Constants

| Name | Value (relative to project root) | Used by |
|---|---|---|
| `ROOT_DIR` | `.` (project root) | `BEXT_DIR` derivation |
| `BEXT_DIR` | `output/bext/` | `src/bext/bext.py` — where `.h5` field files are stored |

### Adding New Paths

To add a new output directory, extend `paths.py`:

```python
NEW_OUTPUT_DIR = ROOT_DIR / "output" / "new_dir"
NEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # optional auto-creation
```

Then import it wherever needed:

```python
from src.utils.paths import NEW_OUTPUT_DIR
```
