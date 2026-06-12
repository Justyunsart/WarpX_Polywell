# Running a Simulation

## Quick Run

```bash
cd /path/to/WarpX
conda activate warpx-env
python inputs/polywell_input.py
```

---

## Execution Flow

When `polywell_input.py` runs, it follows this sequence:

```
polywell_input.py
│
├── 1. Read user parameters (incl. `symmetry`, `particle_mode`, `n_test_particles`)
│
├── 2. Derive simulated-domain spec
│   ├── domain = derive_domain(symmetry, L, N)     [src/warpx_polywell/domain.py]
│   │   → Domain dataclass: bounds, n_cells, field/particle BCs
│   └── layout = make_layout(particle_mode, …)     [src/warpx_polywell/spawn.py]
│       → picmi.GriddedLayout or picmi.PseudoRandomLayout
│
├── 3. Build WarpX objects (all read from `domain`)
│   ├── picmi.Cartesian3DGrid       — bounds + BCs from domain.*
│   ├── picmi.ElectromagneticSolver — Yee scheme, CFL=0.99
│   ├── picmi.UniformDistribution   — plasma bounds via plasma_bounds(domain, …)
│   ├── picmi.Species (electron)    — plasma_e
│   └── picmi.Species (proton)      — plasma_i
│
├── 4. Generate / load B-field file
│   └── setup_bext(method, …, domain=domain)
│       ├── If file exists in output/bext/ → return cached path
│       └── If not → compute with magpylib over domain.lower..upper, write HDF5
│
├── 5. (Optional) Generate / load E-field
│   └── fill_eext_file(path, method, e_dia, e_offset, Q, domain)
│       ├── If combined file exists → return cached path
│       └── If not → compute analytically over domain, update HDF5, rename file
│
├── 6. Point WarpX at the field file
│   └── particles.read_fields_from_path = ext_path
│
├── 7. Build simulation, add species (with derived layout) + diagnostics
│   └── picmi.Simulation(solver, max_steps, verbose=True)
│
├── 8. Allocate run dir + register in runs DB
│   └── new_run_dir() + RunsDB().run_context(run_dir, run_params)
│
└── 9. sim.step()  — runs for max_steps; status → 'completed' or 'failed' on exit
```

---

## Field File Caching

The field grid computation can be expensive (especially the E-field for large `N`).
To avoid redundant computation, both `make_bext_file` and `fill_eext_file` check
whether an output file matching the current parameters already exists before doing any work.

**B-field cache key** (encoded in filename):
```
B_ext_I-{I}A_D-{b_dia}m_Off-{b_offset}m_L-{L}m_N-{N}_sym-{symmetry}.h5
```

**B+E combined cache key**:
```
B_ext_..._sym-{symmetry}_E_ext_Q-{Q}_D-{e_dia}m_offset-{e_offset}m_C_L-{L}m_N-{N}.h5
```

The `_sym-…` token discriminates octant from full-domain caches — they sample the field on different grids (`[-L, +L]^3` with N points vs `[0, +L]^3` with N/2 points), so the files are not interchangeable. The E-field filename does not carry its own `_sym-…` token because the B stem it appends to already does.

If you change any parameter, a new file is generated with the updated name. Old files are not deleted automatically — clean `output/bext/` manually if needed.

---

## Changing Parameters

Edit the `=== USER INPUTS ===` section at the top of `inputs/polywell_input.py`.
See [Input Parameters](parameters.md) for a full reference.

Common changes:

```python
# Increase resolution (also increase memory/time budget)
N = 144   # must be divisible by MPI rank count

# Reduce to one octant (8× faster, requires N even and N/2 rank-divisible)
symmetry = "octant"

# Tracer-ensemble run: 1000 macroparticles globally instead of density-based loading
particle_mode = "count"
n_test_particles = 1000

# Disable E-field
e_method = None

# Shorten test run
max_steps = 100
```

---

## Parallel Execution (MPI)

WarpX uses MPI automatically when launched with `mpirun`/`mpiexec`:

```bash
mpirun -n 8 python inputs/polywell_input.py
```

**Important**: the *simulated* cells per axis must be divisible by the number of MPI ranks. In full mode that is `N`; in octant mode it is `N/2`. The default `N = 72` is divisible by 1/2/3/4/6/8/9/12 (full mode) and `N/2 = 36` is divisible by 1/2/3/4/6/9/12 — note **not 8** in octant. Use `N = 144` to run octant on 8 ranks.

> The field file generation (magpylib, E-field integration) runs on rank 0 only
> and is not parallelised. Only the WarpX PIC advance (`sim.step()`) is parallel.

---

## Monitoring Progress

WarpX prints a step summary to stdout at each time step when `verbose=True`.
The E-field integration also prints progress every 10% of grid points:

```
[get_e_field_data] Computing E-field: 0/373248 points (0%) done
[get_e_field_data] Computing E-field: 37324/373248 points (10%) done
...
```

---

## Common Issues

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: warpx_polywell` | Package not installed in active env | `poetry install` (with `warpx-env` active) |
| WarpX MPI domain error | `N` not divisible by rank count | Adjust `N` or change rank count |
| Very slow startup | E-field being computed from scratch | Wait, or pre-generate the file; will be cached after |
| Divergence cleaning crash | `do_initial_div_cleaning = 1` | Keep it `0` with open BCs |
