"""
Locate simulation run directories for post-processing.

Runs are grouped per deck under the configured output base (see
``utils/paths.OUTPUT_DIR``, driven by ``LOCAL_OUTPUT_DIR`` in ``.env``):

    OUTPUT_DIR/<deck>/run_<timestamp>/

The diagnostics inside a run dir (``diags/diag/openpmd_%T.h5``,
``diags/segment_flux.npz``, ``warpx_used_inputs``, …) are read with paths
*relative to the run dir*. These helpers find the right run dir so notebooks and
scripts never hardcode the output location. Typical use from a notebook:

    from warpx_polywell.post.reader import chdir_to_run
    chdir_to_run("coil_2d")          # cd into the latest coil_2d run
    # ...then the existing relative reads (diags/...) just work.

Or open the openPMD series directly:

    from warpx_polywell.post.reader import open_series
    series = open_series(script="coil_2d")
"""
import os
from pathlib import Path

from warpx_polywell.utils.paths import OUTPUT_DIR
from warpx_polywell.db.runs import RunsDB


def _run_dirs(script: str | None = None) -> list[Path]:
    """All run_* dirs for *script* (deck stem), or across every deck if None,
    sorted chronologically by directory name (run_YYYYMMDD_HHMMSS[_N])."""
    if script:
        decks = [OUTPUT_DIR / Path(script).stem]
    elif OUTPUT_DIR.is_dir():
        decks = [d for d in OUTPUT_DIR.iterdir() if d.is_dir()]
    else:
        decks = []
    runs = [p for deck in decks if deck.is_dir()
            for p in deck.glob("run_*") if p.is_dir()]
    return sorted(runs, key=lambda p: p.name)


def latest_run(script: str | None = None) -> Path:
    """Return the newest run dir for *script* (e.g. ``"coil_2d"``), or the
    newest across all decks when *script* is None.

    Raises FileNotFoundError if no matching run exists.
    """
    runs = _run_dirs(script)
    if not runs:
        where = f"{OUTPUT_DIR}/{script}/run_*" if script else f"{OUTPUT_DIR}/*/run_*"
        raise FileNotFoundError(f"no runs found under {where}")
    return runs[-1]


def run_dir(run_id: int | None = None, script: str | None = None) -> Path:
    """Resolve a run directory.

    With *run_id*, look it up in the runs database (authoritative — works even
    for runs stored outside the current OUTPUT_DIR). Otherwise fall back to the
    latest run for *script* (or the latest overall).
    """
    if run_id is not None:
        db = RunsDB()
        try:
            row = db.get_run(run_id)
        finally:
            db.close()
        if row is None:
            raise KeyError(f"no run with id={run_id} in the database")
        return Path(row["run_dir"])
    return latest_run(script)


def chdir_to_run(script: str | None = None, run_id: int | None = None) -> Path:
    """``os.chdir`` into the resolved run dir and return it. Lets cwd-relative
    post-processing (``diags/...``) run against any chosen run."""
    rd = run_dir(run_id=run_id, script=script)
    os.chdir(rd)
    return rd


def open_series(script: str | None = None, run_id: int | None = None,
                name: str = "diag"):
    """Open the openPMD diagnostic series for a run as read-only.

    *name* selects the diagnostic subdirectory under ``diags/`` (e.g. "diag",
    "scrape"). Imports ``openpmd_api`` lazily so importing this module stays
    dependency-light.
    """
    from openpmd_api import Series, Access
    rd = run_dir(run_id=run_id, script=script)
    pattern = str(rd / "diags" / name / "openpmd_%T.h5")
    return Series(pattern, Access.read_only)
