# `src/warpx_polywell/spawn.py` — Particle Spawning Layouts

Single helper that selects between density-based and count-based particle loading. The distribution object (`picmi.UniformDistribution(density=p_density, …)`) is unchanged in either mode — only the *layout* differs, so the represented physical density is identical across modes.

---

## `make_layout(mode, *, grid, n_macroparticle_per_cell=None, n_test_particles=None)` → picmi layout

| Mode | Returns | Required param | Total macroparticles |
|---|---|---|---|
| `"density"` | `picmi.GriddedLayout(grid=grid, n_macroparticle_per_cell=…)` | `n_macroparticle_per_cell` (sequence of ints) | implicit: `cells × ppc × plasma_bounding³` (per species) |
| `"count"` | `picmi.PseudoRandomLayout(grid=grid, n_macroparticles=…)` | `n_test_particles` (int) | exact `n_test_particles` globally |

Unknown `mode` or missing required param raises `ValueError`.

---

## Weight semantics

PICMI sets `weight = density × plasma_volume / N_macro` for either layout, so a density-mode run and a count-mode run at the same `p_density` represent the **same** physical plasma — count mode is a Monte Carlo undersampling of the same density.

A macroparticle in count mode is the same kind of object as in density mode (a weighted phase-space packet); the weight just scales up to keep the integrated density honest. Total represented physical population, total charge, total kinetic energy in the ensemble all match across modes in expectation.

---

## Choosing a mode

- **Density mode** (default): use when you care about field self-consistency, density-deposited diagnostics, or when statistical noise per cell matters. Both species in the polywell deck have `do_not_deposit = 1` today, so the field self-consistency aspect is moot — but the noise argument still applies if you later turn deposition back on.
- **Count mode**: use for tracer studies, single-particle orbit sampling, confinement-time scans, or any "I want N independent samples drawn from this distribution" problem. With `do_not_deposit = 1`, count mode behaves as a clean tracer ensemble: each macroparticle moves under the prescribed external fields without backreacting.

Noise scales as `1/√N_macro` for moment-based diagnostics, so count mode at small `n_test_particles` will be louder than density mode at default `ppc = 10×10×10`.

---

## `VALID_PARTICLE_MODES`

Tuple `("density", "count")`. The input deck asserts `particle_mode in VALID_PARTICLE_MODES` next to the symmetry assertion so misspelled modes fail before any PICMI work begins.
