# Using `inputs/polywell_hybrid.py` with `configs/polywell_hybrid_config.py`

#### Notes: 
1) Please bring up any concerns you have about the parameters used
2) Smaller scale

- Smaller system scale for quick iterations
- Slows down greatly with higher `N`
- Large `L` compared to system for stability
- Ensured I was resolving ion scales
```
L = 0.6
N = 64
b_diameter = 0.1
b_offset = 0.1
```
3) Boundary conditions

- The solver really likes field BC's of `neumann` or `periodic`
    - Neumann seemed more relevant to our system, hence we use them
    - Neumann is equivalent to PMC in warpx
    - Please call me out if these are unacceptable 
- Particle boundary conditions remain as `absorbing`

4) Plasma bounding
- This is by far one of the strongest stabilizing factors in the run
- Spawning closer to the core (beta >> 1, low B-field) is very stable
    - Con: Greater initial particle burst and hence particle loss
- Spawning closer to the coils (beta ~ 1) can lead to instability quickly 
    - Pro: Smaller particle burst, immediate B-field influence from particles
    - Con: Couldn't get a run to complete in this setting
- As of now, I spawn at about `0.5*b_offset`, 
- Logic for automatic plasma_bounding scale found in 
`inputs/polywell_hybrid.py::store_config_params(self, cfg)`

5) Changing the scale of the system
- I made sure to print out all relevant values that come about from our inputs
- It would be good to verify these remain as desired when changing the scale of the system 

6) `n_floor` in `HybridPICSolver`
- This sets a floor density for the simulation
- High values are less physical but provide stability (for the 1 / rho in the solver)
- This should be tuned further, right now it's at 0.5 * p_density

## `configs/polywell_hybrid_config.py`

This is where you can define all of your parameters. `inputs/polywell_hybrid.py` will import this and use it to instantiate your simulation in two ways.

### Using the `configs/polywell_hybrid_config.py`

You have two options here

1) [Easiest] Adjust the pre-built config in `configs/polywell_hybrid_config.py`

Automatically imported when running the input script

It should run as is with the following:

`python -m inputs.polywell_hybrid`

If you would like the flat disc version add arguments for
- `--nturns int`
- `--r1 float`
- `--r2 float`

For example:

`python -m inputs.polywell_hybrid --nturns 10 --r1 0.02 --r2 0.09`

### Using a saved `.json` file

Define a `<config_name>.json` file with the following form:
```
{
    "symmetry": "full"/"octant",
    "particle_mode": "density"/"count",
    "p_density": 2e+19,
    "n_test_particles_per_cell": > 0 if "particle_mode": "count",
    "Ti_eV": 1000,
    "Te_eV": 1000,
    "plasma_bounding": 0.1, 
    "mass": 1,
    "L": 0.5,
    "N": 64,
    "n_per_cell_each_dim": [3,3,3],
    "b_dia": 0.1,
    "b_offset": 0.1,
    "I": 8000.0,
    "b_method": "file"
}
```

If you would like to use the disc coils then ensure you have the following entries in your file:

```
{
    ... from above
    "n_turns": 10,
    "r1": 0.02,
    "r2": 0.09
}
```

Call the script as follows:

`python -m inputs.polywell_hybrid --cfg <config_filepath>`

This can be made easier by creating a bash script with the above command.