"""
SQLite database for tracking WarpX simulation runs, their parameters, and
output locations.

Mirrors the pattern used in the ICF_Neutronics project (db/runs.py) but with
a schema tailored to the polywell PIC setup in inputs/polywell_input.py.

Usage
-----
    from warpx_polywell.db.runs import RunsDB, new_run_dir

    # One-shot:
    db = RunsDB()
    run_dir = new_run_dir()
    run_id = db.register_run(run_dir, params)
    try:
        sim.step()
        db.update_status(run_id, "completed")
    except Exception:
        db.update_status(run_id, "failed")
        raise

    # Or with the context manager (recommended):
    with RunsDB().run_context(run_dir, params) as run_id:
        sim.step()     # on success: status -> "completed".
                       # on failure: DB row + run_dir are deleted, exception re-raised.

    # Querying:
    db = RunsDB()
    recent = db.list_runs(status="completed", limit=20)
    pick   = db.list_runs(b_method="analytic", solver_type="ElectromagneticSolver")

    # Backfill previously-completed runs that pre-date the DB:
    db.scan_existing()
"""

import ast
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from warpx_polywell.utils.paths import ROOT_DIR, OUTPUT_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Runs are grouped per deck directly under the configured output base:
#   OUTPUT_DIR/<deck>/run_<timestamp>/
# RUNS_DIR is that base (the parent of the per-deck dirs); the DB lives beside
# them. Both follow LOCAL_OUTPUT_DIR via paths.OUTPUT_DIR.
RUNS_DIR = OUTPUT_DIR
DB_PATH = OUTPUT_DIR / "runs.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_dir             TEXT    UNIQUE NOT NULL,
    timestamp           TEXT,
    status              TEXT    DEFAULT 'running',

    -- simulation control
    max_steps           INTEGER,
    const_dt            REAL,

    -- plasma
    p_density           REAL,
    Te_eV               REAL,
    Ti_eV               REAL,
    plasma_bounding     REAL,

    -- B-field
    b_method            TEXT,
    coil_current        REAL,
    b_dia               REAL,
    b_offset            REAL,

    -- E-field
    e_method            TEXT,
    e_charge            REAL,
    e_dia               REAL,
    e_offset            REAL,

    -- grid
    grid_L              REAL,
    grid_N              INTEGER,
    particles_per_cell  TEXT,
    symmetry            TEXT    NOT NULL DEFAULT 'full',
    particle_mode       TEXT    NOT NULL DEFAULT 'density',
    n_test_particles_per_cell INTEGER,

    -- solver
    solver_type         TEXT,
    solver_method       TEXT,
    cfl                 REAL,
    use_hybrid          INTEGER NOT NULL DEFAULT 0,

    -- diagnostics
    diag_period         INTEGER,
    diag_path           TEXT,

    -- misc
    script              TEXT,
    notes               TEXT,
    git_commit          TEXT
);
"""

# Indexes are created after migrations so they don't fail on pre-existing
# databases that are missing a column.
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_runs_timestamp  ON runs(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_runs_status     ON runs(status)",
    "CREATE INDEX IF NOT EXISTS idx_runs_b_method   ON runs(b_method)",
    "CREATE INDEX IF NOT EXISTS idx_runs_e_method   ON runs(e_method)",
    "CREATE INDEX IF NOT EXISTS idx_runs_use_hybrid ON runs(use_hybrid)",
    "CREATE INDEX IF NOT EXISTS idx_runs_script     ON runs(script)",
]

# Columns added after the original schema. (column_name, ddl_type) -- applied
# in order by the migration block in __init__. Each is wrapped in its own
# try/except so a fresh schema (which already has the column) is a no-op.
_MIGRATIONS: list[tuple[str, str]] = [
    ("symmetry", "TEXT NOT NULL DEFAULT 'full'"),
    ("particle_mode", "TEXT NOT NULL DEFAULT 'density'"),
    ("n_test_particles_per_cell", "INTEGER"),
    ("use_hybrid", "INTEGER NOT NULL DEFAULT 0"),
    ("script", "TEXT"),
]

# Columns removed from the schema. Dropped via ALTER TABLE on existing DBs;
# wrapped in try/except so a DB that never had the column is a no-op.
_DELETIONS: list[str] = [
    "n_test_particles",
]


# ---------------------------------------------------------------------------
# RunsDB
# ---------------------------------------------------------------------------
class RunsDB:
    """Thin wrapper around a SQLite database that indexes WarpX runs."""

    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        # Migrate existing databases: add any post-creation columns missing.
        for col, ddl in _MIGRATIONS:
            try:
                self.conn.execute(f"SELECT {col} FROM runs LIMIT 1")
            except sqlite3.OperationalError:
                self.conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {ddl}")
                self.conn.commit()
        # Drop columns that have been retired from the schema.
        for col in _DELETIONS:
            try:
                self.conn.execute(f"ALTER TABLE runs DROP COLUMN {col}")
                self.conn.commit()
            except sqlite3.OperationalError:
                pass
        # Best-effort indexes: skip silently if a column is missing on an
        # older DB (migrations above should normally add it first).
        for ddl in _INDEXES:
            try:
                self.conn.execute(ddl)
            except sqlite3.OperationalError:
                pass
        self.conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def register_run(self, run_dir, params):
        """Insert a new run at simulation start. Returns the row id.

        Parameters
        ----------
        run_dir : str | Path
            Absolute path to the run directory.
        params : dict
            Keys matching column names. List / tuple / dict values are
            automatically JSON-serialised (e.g. particles_per_cell).
        """
        p = dict(params)

        # JSON-serialise any structured values
        for k, v in list(p.items()):
            if isinstance(v, (list, tuple, dict)):
                p[k] = json.dumps(v)

        # Derive timestamp from directory name if not given
        if "timestamp" not in p or p["timestamp"] is None:
            p["timestamp"] = _timestamp_from_dirname(os.path.basename(str(run_dir)))

        # Capture git commit
        if "git_commit" not in p or p["git_commit"] is None:
            p["git_commit"] = _git_short_sha()

        # Default diag_path to the run dir itself
        if "diag_path" not in p or p["diag_path"] is None:
            p["diag_path"] = str(run_dir)

        # Derive the originating deck from the run dir's parent (the per-deck
        # grouping created by new_run_dir, e.g. .../polywell_input/run_*).
        if "script" not in p or p["script"] is None:
            p["script"] = Path(run_dir).parent.name

        columns = ["run_dir"] + list(p.keys())
        placeholders = ", ".join(["?"] * len(columns))
        col_str = ", ".join(columns)
        values = [str(run_dir)] + [p[k] for k in columns[1:]]

        cur = self.conn.execute(
            f"INSERT OR IGNORE INTO runs ({col_str}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()

        # If INSERT OR IGNORE skipped (run_dir already present) return
        # the existing row's id so callers always get a usable handle.
        if cur.rowcount == 0:
            existing = self.get_run_by_dir(run_dir)
            if existing:
                return existing["id"]

        run_id = cur.lastrowid

        # Write a sidecar run_metadata.json that mirrors the row. It's the
        # preferred source for scan_existing() backfills. Best-effort: if the
        # directory doesn't exist or isn't writable we just skip.
        try:
            rd = Path(run_dir)
            if rd.is_dir():
                meta = dict(p)
                meta["id"] = run_id
                meta.setdefault("status", "running")
                (rd / "run_metadata.json").write_text(
                    json.dumps(meta, indent=2, default=str)
                )
        except Exception:
            pass

        return run_id

    def update_status(self, run_id, status):
        """Set the status of a run (running, completed, failed, incomplete)."""
        self.conn.execute(
            "UPDATE runs SET status = ? WHERE id = ?", (status, run_id)
        )
        self.conn.commit()
        self._refresh_sidecar(run_id)

    def delete_run(self, run_id):
        """Remove a run row and its run_metadata.json sidecar.

        Used by run_context() to keep failed runs out of the DB. The run
        directory itself is left intact so the input snapshot and any partial
        diagnostics remain available for debugging.
        """
        row = self.conn.execute(
            "SELECT run_dir FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return
        try:
            rd = Path(row["run_dir"])
            meta = rd / "run_metadata.json"
            if meta.exists():
                meta.unlink()
        except Exception:
            pass
        self.conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        self.conn.commit()

    def update_run(self, run_id, **fields):
        """Update arbitrary columns on a run row."""
        if not fields:
            return
        # Auto JSON-serialise structured values
        for k, v in list(fields.items()):
            if isinstance(v, (list, tuple, dict)):
                fields[k] = json.dumps(v)
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [run_id]
        self.conn.execute(f"UPDATE runs SET {set_clause} WHERE id = ?", values)
        self.conn.commit()
        self._refresh_sidecar(run_id)

    def update_run_dir(self, run_id, new_run_dir):
        """Update run_dir after moving output to a new location."""
        self.conn.execute(
            "UPDATE runs SET run_dir = ?, diag_path = ? WHERE id = ?",
            (str(new_run_dir), str(new_run_dir), run_id),
        )
        self.conn.commit()
        self._refresh_sidecar(run_id)

    def _refresh_sidecar(self, run_id):
        """Rewrite run_metadata.json from the current DB row. Best-effort:
        silently skip if the run_dir no longer exists or isn't writable."""
        row = self.get_run(run_id)
        if not row:
            return
        try:
            rd = Path(row["run_dir"])
            if rd.is_dir():
                (rd / "run_metadata.json").write_text(
                    json.dumps(row, indent=2, default=str)
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_run(self, run_id):
        """Fetch a single run by id. Returns a dict or None."""
        row = self.conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_run_by_dir(self, run_dir):
        """Fetch a run by its directory path."""
        row = self.conn.execute(
            "SELECT * FROM runs WHERE run_dir = ?", (str(run_dir),)
        ).fetchone()
        return dict(row) if row else None

    def list_runs(
        self,
        status=None,
        b_method=None,
        e_method=None,
        solver_type=None,
        use_hybrid=None,
        since=None,
        limit=100,
    ):
        """Return runs matching optional filters, most recent first.

        Parameters
        ----------
        status : str, optional
            Filter by status column (e.g. "completed", "failed", "running").
        b_method : str, optional
            Filter by B-field method ("analytic", "file").
        e_method : str, optional
            Filter by E-field method (e.g. "FW", "None").
        solver_type : str, optional
            Filter by solver class name ("ElectromagneticSolver", ...).
        use_hybrid : bool | int | str, optional
            Filter by whether the run used the Hybrid-PIC solver. Accepts
            truthy strings from the CLI ("1", "true", "yes", "y", "t").
        since : str, optional
            ISO-8601 timestamp; returns runs newer than this.
        limit : int
            Max rows to return.
        """
        clauses, params = [], []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if b_method is not None:
            clauses.append("b_method = ?")
            params.append(b_method)
        if e_method is not None:
            clauses.append("e_method = ?")
            params.append(e_method)
        if solver_type is not None:
            clauses.append("solver_type = ?")
            params.append(solver_type)
        if use_hybrid is not None:
            if isinstance(use_hybrid, str):
                use_hybrid = use_hybrid.strip().lower() in ("1", "true", "yes", "y", "t")
            clauses.append("use_hybrid = ?")
            params.append(1 if use_hybrid else 0)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self.conn.execute(
            f"SELECT * FROM runs{where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def run_context(self, run_dir, params):
        """Register a run, yield its id, auto-update status on exit.

        On clean exit the status is set to "completed"; on any exception
        the row is deleted from the DB (along with its run_metadata.json
        sidecar) **and the run directory itself is removed** so failed
        runs leave no trace under output/runs/. The exception is re-raised.
        """
        run_id = self.register_run(run_dir, params)
        try:
            yield run_id
        except BaseException:
            self.delete_run(run_id)
            try:
                shutil.rmtree(run_dir)
            except FileNotFoundError:
                pass
            raise
        else:
            self.update_status(run_id, "completed")

    # ------------------------------------------------------------------
    # Backfill from existing run directories
    # ------------------------------------------------------------------

    def scan_existing(self, runs_root=None, verbose=True):
        """Walk the per-deck run tree and insert any runs not yet in the DB.

        Looks for run_* dirs both one level down (``<root>/<deck>/run_*``, the
        current layout) and directly under *root* (``<root>/run_*``, a legacy or
        externally-pointed flat layout). Parameter extraction order
        (first hit wins per key):
          1. run_metadata.json sidecar (written by register_run)
          2. AST parse of the polywell_input.py snapshot in the run dir
        Status is inferred from diags/ contents when not already known.

        Returns the number of new rows added.
        """
        root = Path(runs_root) if runs_root else RUNS_DIR
        if not root.is_dir():
            if verbose:
                print(f"scan_existing: {root} does not exist; nothing to scan.")
            return 0

        added = 0
        for sub in _iter_run_dirs(root):
            if self.get_run_by_dir(sub):
                continue  # already registered

            params = _extract_params_from_dir(sub)
            try:
                self.register_run(sub, params)
                added += 1
                if verbose:
                    print(
                        f"scan_existing: added {sub.name} "
                        f"(status={params.get('status','?')})"
                    )
            except Exception as e:
                if verbose:
                    print(f"scan_existing: skipped {sub.name}: {e}")

        if verbose:
            if added:
                print(f"scan_existing: added {added} run(s) to the database")
            else:
                print("scan_existing: no new runs found")
        return added

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def new_run_dir(script: str | None = None, parent: Path | None = None) -> Path:
    """Create OUTPUT_DIR/<deck>/run_YYYYMMDD_HHMMSS/ and return its abs Path.

    The deck name groups runs by the driver that produced them. It defaults to
    the stem of the running script (``sys.argv[0]``) so callers need no
    boilerplate — ``python inputs/polywell_input.py`` lands runs under
    ``OUTPUT_DIR/polywell_input/``. Pass *script* to override (a bare name or a
    filename like ``"coil_2d.py"`` both work — only the stem is used).

    If a directory with that name already exists (e.g. two runs launched in
    the same second), an integer suffix _2, _3, ... is appended so each call
    returns a fresh, empty directory. *parent* overrides the base dir entirely.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if parent is not None:
        base = Path(parent)
    else:
        deck = Path(script).stem if script else (Path(sys.argv[0]).stem or "unknown")
        base = OUTPUT_DIR / deck
    base.mkdir(parents=True, exist_ok=True)

    run_dir = base / f"run_{timestamp}"
    suffix = 2
    while run_dir.exists():
        run_dir = base / f"run_{timestamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _mpi_comm_rank():
    """Return (comm, rank) using mpi4py if available, else (None, 0).

    Imported lazily so the package never hard-depends on mpi4py; the conda
    WarpX stack provides it, and a non-MPI run falls back to a single rank.
    """
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank()
    except Exception:
        return None, 0


def allocate_run_dir(script: str | None = None, *, copy_from=None,
                     chdir: bool = True) -> Path:
    """MPI-safe allocation of a per-deck run directory.

    Creates ``OUTPUT_DIR/<deck>/run_<timestamp>/`` once on rank 0 and broadcasts
    the path so every rank agrees. Optionally ``chdir`` into it (all ranks — so
    WarpX's relative diagnostic writes land inside) and snapshot the deck file
    into it (rank 0 only).

    *script* names the deck (its stem is the grouping dir); pass ``__file__``.
    *copy_from* overrides what gets snapshotted (defaults to *script* when it is
    an existing file). Returns the run dir, identical on all ranks.
    """
    comm, rank = _mpi_comm_rank()
    run_dir = new_run_dir(script=script) if rank == 0 else None
    if comm is not None:
        run_dir = comm.bcast(run_dir, root=0)
    if chdir:
        os.chdir(run_dir)
    if rank == 0:
        src = copy_from or (script if script and os.path.isfile(str(script)) else None)
        if src:
            try:
                shutil.copy2(src, run_dir / Path(str(src)).name)
            except Exception:
                pass
    return run_dir


@contextmanager
def run_session(script: str | None = None, params: dict | None = None, *,
                copy_from=None):
    """Full deck-run lifecycle as a context manager — the shared entry point
    every deck should use.

        with run_session(__file__, run_params) as run_dir:
            sim.step()

    Allocates the run dir (MPI-safe), chdir's into it, snapshots the deck, and
    registers the run when *params* is given. On clean exit the run is marked
    ``completed``; on any exception its DB row and directory are removed (rank 0
    only) so failed runs leave no trace. The original cwd is always restored, so
    keep run-relative work (post-step dumps, etc.) inside the ``with`` block.
    """
    comm, rank = _mpi_comm_rank()
    prev_cwd = os.getcwd()
    run_dir = allocate_run_dir(script, copy_from=copy_from, chdir=True)

    db = RunsDB() if rank == 0 else None
    run_id = db.register_run(run_dir, params) if (rank == 0 and params is not None) else None
    if rank == 0:
        print(f"[runs.db] run id={run_id} at {run_dir}")
    try:
        yield run_dir
    except BaseException:
        os.chdir(prev_cwd)
        if rank == 0:
            if run_id is not None:
                db.delete_run(run_id)
            shutil.rmtree(run_dir, ignore_errors=True)
            db.close()
        raise
    else:
        if rank == 0:
            if run_id is not None:
                db.update_status(run_id, "completed")
            db.close()
        os.chdir(prev_cwd)


def _iter_run_dirs(root: Path):
    """Yield run_* directories under *root*.

    Supports both the per-deck layout (``root/<deck>/run_*``) and a flat layout
    (``root/run_*``). Non-run subdirectories (e.g. ``bext/``) are descended one
    level to find their run_* children; we never recurse deeper.
    """
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("run_"):
            yield child
            continue
        for sub in sorted(child.iterdir()):
            if sub.is_dir() and sub.name.startswith("run_"):
                yield sub


def _timestamp_from_dirname(dirname: str):
    """Parse 'run_YYYYMMDD_HHMMSS' into ISO-8601. Returns None on mismatch."""
    try:
        ts = dirname.replace("run_", "")
        return datetime.strptime(ts, "%Y%m%d_%H%M%S").isoformat()
    except ValueError:
        return None


def _git_short_sha():
    """Return the current HEAD short SHA, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(ROOT_DIR),
        )
        return result.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Backfill helpers
# ---------------------------------------------------------------------------

# Schema columns that scan_existing is allowed to populate from a
# run_metadata.json sidecar or an AST parse. Anything else in the JSON
# (e.g. "id" from a previous DB) is ignored.
_BACKFILL_COLUMNS = {
    "timestamp", "status",
    "max_steps", "const_dt",
    "p_density", "Te_eV", "Ti_eV", "plasma_bounding",
    "b_method", "coil_current", "b_dia", "b_offset",
    "e_method", "e_charge", "e_dia", "e_offset",
    "grid_L", "grid_N", "particles_per_cell", "symmetry",
    "particle_mode", "n_test_particles_per_cell",
    "solver_type", "solver_method", "cfl", "use_hybrid",
    "diag_period", "diag_path",
    "script", "notes", "git_commit",
}


def _extract_params_from_dir(run_dir):
    """Best-effort parameter extraction from a run directory.

    Tries the run_metadata.json sidecar first, then falls back to an AST
    parse of polywell_input.py. Status is inferred from diags/ contents if
    neither source supplies a terminal status.
    """
    run_dir = Path(run_dir)
    params = {}

    # --- 1. sidecar JSON (preferred) ---
    meta_path = run_dir / "run_metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            for k, v in meta.items():
                if k in _BACKFILL_COLUMNS and v is not None:
                    params[k] = v
        except Exception:
            pass

    # --- 2. AST parse of snapshotted input script (fallback) ---
    py_snapshot = run_dir / "polywell_input.py"
    if py_snapshot.is_file():
        snap = _extract_from_python_snapshot(py_snapshot)
        for k, v in snap.items():
            if k in _BACKFILL_COLUMNS:
                params.setdefault(k, v)

    # --- 3. Status: prefer sidecar, else infer from diags/ ---
    sidecar_status = params.get("status")
    if sidecar_status in (None, "running"):
        params["status"] = _infer_status(run_dir)

    # --- 4. Always refresh diag_path to the actual location ---
    params["diag_path"] = str(run_dir)

    # --- 5. Deck name from the per-deck grouping dir (sidecar wins if present) ---
    params.setdefault("script", run_dir.parent.name)
    return params


def _infer_status(run_dir):
    """Heuristic: 'completed' if diags/ contains any non-empty subdirectory,
    else 'incomplete'. A run that crashed before producing diagnostics is
    indistinguishable from one interrupted mid-run without a sidecar JSON."""
    diags = Path(run_dir) / "diags"
    if diags.is_dir():
        for sub in diags.iterdir():
            if sub.is_dir() and any(sub.iterdir()):
                return "completed"
    return "incomplete"


def _extract_from_python_snapshot(py_path):
    """Pull literal assignments out of a snapshotted polywell_input.py.

    Recognises:
      - scalar literals:  max_steps, p_density, I, b_dia, b_offset, Q,
                          e_dia, e_offset, L, N, plasma_bounding,
                          b_method, e_method
      - list literals:    number_per_cell_each_dim -> particles_per_cell
      - attribute assign: warpx.const_dt
      - unit-aware expr:  Te, Ti when written as `<num> * sc.eV`
      - solver call:      solver = picmi.<Class>(method=..., cfl=...)

    Anything that fails to parse is silently skipped — this is a
    best-effort backfill, not a validator.
    """
    params = {}
    try:
        src = Path(py_path).read_text()
        tree = ast.parse(src)
    except Exception:
        return params

    # name-in-source -> column-in-DB
    SCALAR_MAP = {
        "max_steps":                 "max_steps",
        "p_density":                 "p_density",
        "b_method":                  "b_method",
        "I":                         "coil_current",
        "b_dia":                     "b_dia",
        "b_offset":                  "b_offset",
        "e_method":                  "e_method",
        "Q":                         "e_charge",
        "e_dia":                     "e_dia",
        "e_offset":                  "e_offset",
        "L":                         "grid_L",
        "N":                         "grid_N",
        "symmetry":                  "symmetry",
        "particle_mode":             "particle_mode",
        "n_test_particles_per_cell": "n_test_particles_per_cell",
        "plasma_bounding":           "plasma_bounding",
        "number_per_cell_each_dim":  "particles_per_cell",
        "use_hybrid":                "use_hybrid",
    }

    def _eV_literal(node):
        """If *node* is `<num> * sc.eV` (either order), return the numeric.
        Returns None if the pattern doesn't match."""
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
            return None
        for num_side, attr_side in ((node.left, node.right),
                                    (node.right, node.left)):
            if (isinstance(attr_side, ast.Attribute)
                    and isinstance(attr_side.value, ast.Name)
                    and attr_side.value.id in ("sc", "scipy_constants")
                    and attr_side.attr == "eV"):
                try:
                    return float(ast.literal_eval(num_side))
                except Exception:
                    return None
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]

        # --- simple name = literal / list / Te-Ti expr / solver Call ---
        if isinstance(tgt, ast.Name):
            name = tgt.id
            if name in SCALAR_MAP:
                try:
                    val = ast.literal_eval(node.value)
                    if isinstance(val, tuple):
                        val = list(val)
                    params[SCALAR_MAP[name]] = val
                except Exception:
                    pass
            elif name in ("Te", "Ti"):
                ev = _eV_literal(node.value)
                if ev is not None:
                    params[f"{name}_eV"] = ev
            elif name == "solver" and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Attribute):
                    params["solver_type"] = func.attr
                elif isinstance(func, ast.Name):
                    params["solver_type"] = func.id
                for kw in node.value.keywords:
                    if kw.arg == "method":
                        try:
                            params["solver_method"] = ast.literal_eval(kw.value)
                        except Exception:
                            pass
                    elif kw.arg == "cfl":
                        try:
                            params["cfl"] = ast.literal_eval(kw.value)
                        except Exception:
                            pass

        # --- attribute target, e.g. warpx.const_dt = 1e-9 ---
        elif (isinstance(tgt, ast.Attribute)
              and isinstance(tgt.value, ast.Name)
              and tgt.value.id == "warpx"
              and tgt.attr == "const_dt"):
            try:
                params["const_dt"] = ast.literal_eval(node.value)
            except Exception:
                pass

    return params


# ---------------------------------------------------------------------------
# CLI helper: `python -m src.db.runs list`
# ---------------------------------------------------------------------------
def _print_runs(rows):
    if not rows:
        print("(no runs)")
        return
    # Compact one-line-per-run dump; full dicts are available via get_run().
    for r in rows:
        hybrid_flag = "Y" if r.get("use_hybrid") else "N"
        print(
            f"[{r['id']:>4}] {r.get('timestamp') or '?':<19} "
            f"{r.get('status','?'):<10} "
            f"b={r.get('b_method')} e={r.get('e_method')} "
            f"solver={r.get('solver_type')} hybrid={hybrid_flag} "
            f"steps={r.get('max_steps')} N={r.get('grid_N')} L={r.get('grid_L')} "
            f"sym={r.get('symmetry')} mode={r.get('particle_mode')} "
            f"  {r.get('run_dir')}"
        )


if __name__ == "__main__":
    import sys
    db = RunsDB()
    args = sys.argv[1:]
    if not args or args[0] == "list":
        kwargs = {}
        for a in args[1:]:
            if "=" in a:
                k, v = a.split("=", 1)
                kwargs[k] = v
        _print_runs(db.list_runs(**kwargs))
    elif args[0] == "get" and len(args) >= 2:
        row = db.get_run(int(args[1]))
        print(json.dumps(row, indent=2, default=str) if row else "(not found)")
    elif args[0] == "scan":
        # Optional positional arg: root directory to scan (defaults to RUNS_DIR)
        root = args[1] if len(args) >= 2 else None
        db.scan_existing(runs_root=root)
    else:
        print(
            "Usage:\n"
            "  python -m src.db.runs list [status=completed] [b_method=analytic] ...\n"
            "  python -m src.db.runs get <id>\n"
            "  python -m src.db.runs scan [<runs_root>]\n"
        )
    db.close()
