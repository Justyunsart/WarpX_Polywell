# Switching to most recent dev branch

In root directory `./WarpX_Polywell`

1) run: `git pull origin dev`

- This will pull the most recent files in from this branch
- Ensure you don't have any commits or files staged via `git add`

2) run: `git switch dev`
- You now have the updated `inputs/polywell_input.py` file
- You have the correct B-Field magpylib creation as well
- This will not effect your `output/` directory

3) You can now run simulations in module-mode via:

`python -m inputs.polywell_input`
- Non-module mode:

`python inputs/polywell_input.py`

### **Note**: The current simulation has some flaws namely:
- coil offset `b_offset` is too small, with a coil diameter `b_dia` = 1.0 (m), the coils intersect

### Required changes to `inputs/polywell_input.py`

1) `b_offset` = 1.0 (m)
2) `L` = 1.0 (m)
    - This cuts off right at the coils, may need to be adjusted
3) `N`, where `N` must be divisible by thread count

    Or if using `octant` mode, a thread count that divides `N/2`

### Outputs
- directory: `output/runs/{most_recent runs_123456_123456}/diags` where both `123456` are dictated by the date and time ran, respectively
    - Here you will find `part_diag` and a `field_diag` folders

### Paraview pipeline
**Note**: It helps to rename the `paraview.pmd` files after importing for clarity, this can be done via a double-click
#### Field Pipeline
1) Import the `paraview.pmd` file from the `output/runs/{run_folder}/diags/field_diag/paraview.pmd`
2) Click `Apply`
##### Visualizing field lines
3) Right-click the `paraview.pmd` file in the Pipeline Browser window
4) `Add Filter` -> `Alphabetical` -> `Stream Tracer`
5) Before `Apply`, scroll down to find `Seed Type`, and choose `Point Cloud`, adjust as needed

#### Particle Pipeline
1) Import `paraview.pmd` file from `output/runs/{run_folder}/diags/part_diag/paraview.pmd` 
2) Click `Apply`
#### Visualizing path lines
3) Right-click `Particles` from the `paraview.pmd` file just imported via the `part_diag` folder in the Pipeline Browser Window
4) `Add Filer` -> `Alphabetical` -> `Merge Blocks` -> `Apply`
5) Right-click the `Merge Blocks` pipeline just added
6) `Add Filer` -> `Alphabetical` -> `Temporal Particles to Pathlines`
7) *Important*: While in properties of `Temporal Particles to Pathlines` ensure that `Id channel array` is set to `id` (else we use presets that can swap particle ID's and hence give us big leaps across the system)