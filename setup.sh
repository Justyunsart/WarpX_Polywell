#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# WarpX polywell simulation: one-shot environment setup.
#
# Creates a conda environment and installs WarpX + all runtime dependencies
# from conda-forge. Mirrors docs/setup/installation.md but scripted end-to-end.
#
# Usage:
#     ./setup.sh                    # default env name 'warpx-env', Python 3.10
#     ./setup.sh --env my-env       # custom env name
#     ./setup.sh --python 3.11      # custom Python version
#     ./setup.sh --force            # remove existing env and recreate
#     ./setup.sh --dry-run          # print what would run, do nothing
#     ./setup.sh -h | --help        # show usage
#
# Environment variables (overridden by matching flags if provided):
#     WARPX_ENV          env name  (default: warpx-env)
#     WARPX_PYTHON       python version (default: 3.10)
# -----------------------------------------------------------------------------
set -euo pipefail

# ------------------------------ defaults -------------------------------------
ENV_NAME="${WARPX_ENV:-warpx-env}"
PY_VERSION="${WARPX_PYTHON:-3.10}"
FORCE=0
DRY_RUN=0

CONDA_FORGE_PKGS=(
    warpx          # pywarpx + picmi + MPI + HDF5 + ADIOS2
    magpylib       # coil B-field calculation
    h5py           # HDF5 I/O for external field grids
    numpy          # array math (usually transitive, pinned for safety)
    scipy          # scipy.constants, scipy.special.ellipk/ellipe
    matplotlib     # plotting (tests/, notebooks/)
)

# ------------------------------ helpers --------------------------------------
log()   { printf "\033[1;34m[setup]\033[0m %s\n"  "$*"; }
ok()    { printf "\033[1;32m[ ok ]\033[0m %s\n"   "$*"; }
warn()  { printf "\033[1;33m[warn]\033[0m %s\n"   "$*" >&2; }
die()   { printf "\033[1;31m[err ]\033[0m %s\n"   "$*" >&2; exit 1; }

run() {
    # Print then execute; in --dry-run mode only print.
    printf "\033[1;90m\$ %s\033[0m\n" "$*"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        "$@"
    fi
}

usage() {
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

# ------------------------------ arg parsing ----------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env|--env-name)
            [[ $# -ge 2 ]] || die "--env requires a value"
            ENV_NAME="$2"; shift 2 ;;
        --python)
            [[ $# -ge 2 ]] || die "--python requires a value"
            PY_VERSION="$2"; shift 2 ;;
        --force)   FORCE=1;   shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage ;;
        *)         die "unknown option: $1  (try --help)" ;;
    esac
done

# ------------------------------ choose a manager -----------------------------
# Prefer mamba (much faster solver). Fall back to conda. Error if neither.
if command -v mamba >/dev/null 2>&1; then
    MGR="mamba"
elif command -v conda >/dev/null 2>&1; then
    MGR="conda"
else
    die "Neither mamba nor conda was found on PATH.
     Install Miniforge (recommended) or Miniconda first:
       https://github.com/conda-forge/miniforge
       https://docs.conda.io/en/latest/miniconda.html"
fi

# Ensure `conda activate` works inside this shell for the post-install check.
# `mamba` does not ship `activate`; we always source conda's hook.
CONDA_BIN="$(command -v conda || true)"
if [[ -n "$CONDA_BIN" ]]; then
    # shellcheck disable=SC1091
    source "$("$CONDA_BIN" info --base)/etc/profile.d/conda.sh"
fi

log "Using package manager: $MGR"
log "Target env name      : $ENV_NAME"
log "Python version       : $PY_VERSION"
log "Packages             : ${CONDA_FORGE_PKGS[*]}"
[[ "$DRY_RUN" -eq 1 ]] && warn "DRY RUN — no changes will be made"

# ------------------------------ env create -----------------------------------
env_exists() {
    "$MGR" env list 2>/dev/null | awk '{print $1}' | grep -Fxq "$ENV_NAME"
}

if env_exists; then
    if [[ "$FORCE" -eq 1 ]]; then
        log "Environment '$ENV_NAME' exists — removing (--force)"
        run "$MGR" env remove -n "$ENV_NAME" -y
    else
        warn "Environment '$ENV_NAME' already exists. Reusing it."
        warn "Pass --force to delete and recreate from scratch."
    fi
fi

if ! env_exists; then
    log "Creating env '$ENV_NAME' with python=$PY_VERSION"
    run "$MGR" create -n "$ENV_NAME" -c conda-forge -y "python=$PY_VERSION"
fi

# ------------------------------ install packages -----------------------------
log "Installing conda-forge packages into '$ENV_NAME'"
run "$MGR" install -n "$ENV_NAME" -c conda-forge -y "${CONDA_FORGE_PKGS[@]}"

# ------------------------------ install local package ------------------------
# Editable-install the warpx_polywell package so `import warpx_polywell` resolves
# from any working directory. `--no-deps` because every runtime dependency is
# already provided by conda-forge above — we only want pip to wire up the
# editable link, not layer PyPI builds over the conda scientific stack.
log "Installing the warpx_polywell package (editable) into '$ENV_NAME'"
run conda run -n "$ENV_NAME" python -m pip install -e . --no-deps

# ------------------------------ verify ---------------------------------------
if [[ "$DRY_RUN" -eq 0 ]]; then
    log "Verifying imports"
    # `conda run` works even from a non-activated shell and across managers.
    if ! conda run -n "$ENV_NAME" python - <<'PY'
import importlib, sys
needed = [
    ("pywarpx", "WarpX Python bindings"),
    ("pywarpx.picmi", "PICMI interface"),
    ("magpylib", "Coil B-field"),
    ("h5py", "HDF5 I/O"),
    ("numpy", "Array math"),
    ("scipy", "Scientific utilities"),
    ("scipy.constants", "Physical constants"),
    ("scipy.special", "Elliptic integrals"),
    ("matplotlib", "Plotting"),
    ("warpx_polywell", "Polywell simulation package (editable install)"),
]
bad = []
for mod, why in needed:
    try:
        importlib.import_module(mod)
        print(f"  OK   {mod:<22}  ({why})")
    except Exception as e:
        bad.append((mod, e))
        print(f"  FAIL {mod:<22}  {e}")
if bad:
    sys.exit(1)
PY
    then
        die "one or more packages failed to import; see output above"
    fi
    ok "All dependencies import cleanly."
fi

# ------------------------------ done -----------------------------------------
cat <<EOF

$(log "Setup complete.")

Next steps:
  conda activate $ENV_NAME
  cd "$(pwd)"
  python inputs/polywell_input.py

Tip: 'warpx_polywell' is installed editable, so imports resolve from anywhere.
     Still launch from the project root so generated output/ lands beside the repo.
EOF
