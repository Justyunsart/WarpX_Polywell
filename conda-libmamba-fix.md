# Fixing the `conda-libmamba-solver` Error on macOS

## What you're seeing

Two messages appear when running the WarpX install script on the other Mac:

```
Error while loading conda entry point: conda-libmamba-solver
(module 'libmambapy' has no attribute 'QueryFormat')
```

```
CondaValueError: You have chosen a non-default solver backend (libmamba)
but it was not recognized. Choose one of: classic
```

The first is a **warning** — conda keeps running, which is why `conda create` succeeded.
The second is the **actual failure** — `conda install -c conda-forge warpx` aborted because conda's default solver (`libmamba`) failed to load and no fallback was configured.

## Why it happens

The base conda environment on that Mac has mismatched versions of two packages:

- `conda-libmamba-solver` — the plugin that registers the libmamba solver with conda.
- `libmambapy` — the Python bindings to libmamba that the plugin calls into.

The newer plugin expects `libmambapy.QueryFormat`, but the installed `libmambapy` is too old to have it. The plugin crashes at load time, so libmamba never gets registered as a solver. Conda's config still requests libmamba by default, and the install fails.

Common causes: a partial `conda update`, mixing `pkgs/main` and `conda-forge` packages in `base`, or a stale base conda (24.11.3 here, with 26.3.2 available).

## Fix options

Pick one — try them in order.

### 1. Realign the libmamba packages

```bash
conda update -n base -c defaults conda conda-libmamba-solver libmambapy
```

Or force a reinstall so the two pieces agree:

```bash
conda install -n base -c defaults conda-libmamba-solver libmambapy --force-reinstall
```

### 2. Switch the default solver to classic

If the plugin still won't load, sidestep it:

```bash
conda config --set solver classic
```

Every subsequent `conda` command will use the classic solver — slower, but reliable — without needing `--solver=classic` on each call.

### 3. Use a conda-forge-native install for WarpX

WarpX is only published on conda-forge. Mixing channels in `base` is what causes most of these ABI mismatches. Cleanest path on a fresh machine:

```bash
conda create -n warpx-env -c conda-forge python=3.10 warpx
conda activate warpx-env
```

Or install [Miniforge](https://github.com/conda-forge/miniforge) instead of Anaconda so the entire stack is conda-forge from the start.

## Recommended order on the other Mac

1. Run option 1 to realign libmamba.
2. If it still errors, run option 2 to fall back to the classic solver.
3. Re-run the install script.
