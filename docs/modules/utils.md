# `src/warpx_polywell/utils` — Utilities Module

Lightweight helpers used across the codebase for coordinate conversion and path management.

---

## Files

| File | Purpose |
|---|---|
| `src/warpx_polywell/utils/cyl.py` | Cartesian ↔ Cylindrical coordinate conversion |
| `src/warpx_polywell/utils/paths.py` | Centralised path constants |

---

## `cyl.py`

Used by `src/warpx_polywell/eext/eext.py` to convert field vectors between coordinate systems
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
from warpx_polywell.utils.config import get_config

_script_dir = Path(__file__).resolve().parent          # .../src/warpx_polywell/utils/
ROOT_DIR    = _script_dir.parent.parent.parent          # .../WarpX/ (repo checkout)

# Configured output base — follows LOCAL_OUTPUT_DIR in .env, defaults to
# <repo>/output. Single source of truth for runs, the bext cache, and runs.db.
OUTPUT_DIR  = Path(get_config()["LOCAL_OUTPUT_DIR"])
BEXT_DIR    = OUTPUT_DIR / "bext"                        # field-file cache
```

### Constants

| Name | Value | Used by |
|---|---|---|
| `ROOT_DIR` | repo checkout | git metadata; default output base |
| `OUTPUT_DIR` | `LOCAL_OUTPUT_DIR` (`.env`), else `<repo>/output` | runs, `runs.db`, `BEXT_DIR` |
| `BEXT_DIR` | `OUTPUT_DIR/bext/` | `src/warpx_polywell/bext/bext.py` — where `.h5` field files are stored |

### Adding New Paths

To add a new output directory, extend `paths.py` (anchor it on `OUTPUT_DIR` so
it follows `.env`):

```python
NEW_OUTPUT_DIR = OUTPUT_DIR / "new_dir"
NEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # optional auto-creation
```

Then import it wherever needed:

```python
from warpx_polywell.utils.paths import NEW_OUTPUT_DIR
```
