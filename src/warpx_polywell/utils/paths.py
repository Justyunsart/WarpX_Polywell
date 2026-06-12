"""
When any function requires outputting or reading a file, it will generally reference the paths set in this file.
"""
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
# paths.py lives at src/warpx_polywell/utils/, so the repo root is three levels up.
ROOT_DIR = _script_dir.parent.parent.parent

# after getting the root directory, you can derive specific dirs inside the project
BEXT_DIR = ROOT_DIR / "output" / "bext" # location of external B-field .h5 files