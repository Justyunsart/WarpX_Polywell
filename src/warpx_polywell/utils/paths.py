"""
When any function requires outputting or reading a file, it will generally reference the paths set in this file.

ROOT_DIR is the repo checkout (used for git metadata and as the default output
location). OUTPUT_DIR is the *configured* output base — it follows the
LOCAL_OUTPUT_DIR setting in .env (see utils/config.py) and is the single source
of truth for where runs, the bext cache, and runs.db are written. It defaults to
<repo>/output when LOCAL_OUTPUT_DIR is unset.
"""
from pathlib import Path

from warpx_polywell.utils.config import get_config

_script_dir = Path(__file__).resolve().parent
# paths.py lives at src/warpx_polywell/utils/, so the repo root is three levels up.
ROOT_DIR = _script_dir.parent.parent.parent

# Configured output base (from .env: LOCAL_OUTPUT_DIR). All generated output is
# rooted here so a single .env edit relocates the whole tree.
OUTPUT_DIR = Path(get_config()["LOCAL_OUTPUT_DIR"])

# Derived output locations. BEXT_DIR matches what the storage backend already
# uses (LocalBackend with subdir="bext"), so the field-file cache and this
# constant never diverge.
BEXT_DIR = OUTPUT_DIR / "bext"  # location of external B-field .h5 files