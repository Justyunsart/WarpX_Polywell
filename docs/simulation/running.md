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
├── 1. Read user parameters
│
├── 2. Build WarpX objects
│   ├── picmi.Cartesian3DGrid       — 3D domain with open/absorbing BCs
│   ├── picmi.ElectromagneticSolver — Yee scheme, CFL=0.99
│   ├── picmi.UniformDistribution   — plasma initialisation region
│   ├── picmi.Species (electron)    — plasma_e
│   └── picmi.Species (proton)      — plasma_i
│
├── 3. Generate / load B-field file
│   └── make_bext_file(I, b_dia, b_offset, L, N)
│       ├── If file exists in output/bext/ → return cached path
│       └── If not → compute with magpylib, write HDF5, return path
│
├── 4. (Optional) Generate / load E-field
│   └── fill_eext_file(path, method, e_dia, e_offset, Q, L, N)
│       ├── If combined file exists → return cached path
│       └── If not → compute analytically, update HDF5, rename file, return path
│
├── 5. Point WarpX at the field file
│   └── warpx.read_fields_from_path = ext_path
│
├── 6. Build simulation, add species + diagnostics
│   └── picmi.Simulation(solver, max_steps, verbose=True)
│
└── 7. sim.step()  — runs for max_steps
```

---

## Field File Caching

The field grid computation can be expensive (especially the E-field for large `N`).
To avoid redundant computation, both `make_bext_file` and `fill_eext_file` check
whether an output file matching the current parameters already exists before doing any work.

**B-field cache key** (encoded in filename):
```
B_ext_I-{I}A_D-{b_dia}m_Off-{b_offset}m_L-{L}m_N-{N}.h5
```

**B+E combined cache key**:
```
B_ext_..._E_ext_Q-{Q}_D-{e_dia}m_offset-{e_offset}m_C_L-{L}m_N-{N}.h5
```

If you change any parameter, a new file is generated with the updated name.
Old files are not deleted automatically — clean `output/bext/` manually if needed.

---

## Changing Parameters

Edit the `=== USER INPUTS ===` section at the top of `inputs/polywell_input.py`.
See [Input Parameters](parameters.md) for a full reference.

Common changes:

```python
# Increase resolution (also increase memory/time budget)
N = 144   # must be divisible by MPI rank count

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

**Important**: `N` must be divisible by the number of MPI ranks.
The default `N = 72` is divisible by 1, 2, 3, 4, 6, 8, 9, 12, etc.

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
| `ModuleNotFoundError: src` | Not running from project root | `cd /path/to/WarpX` |
| WarpX MPI domain error | `N` not divisible by rank count | Adjust `N` or change rank count |
| Very slow startup | E-field being computed from scratch | Wait, or pre-generate the file; will be cached after |
| Divergence cleaning crash | `do_initial_div_cleaning = 1` | Keep it `0` with open BCs |
