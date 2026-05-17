"""
Particle-spawning layout selector for polywell runs.

Two modes are supported, picked by a single user knob in the input deck:

- "density": GriddedLayout(n_macroparticle_per_cell=...). Total macroparticles
  is implicit (N_cells * ppc); weights match the requested physical density.

- "count":   PseudoRandomLayout(n_macroparticles_per_cell=...). Each cell gets
  a fixed number of randomly-placed macroparticles; weights match the requested
  physical density via the AnalyticDistribution's density_expression. WarpX
  requires the per-cell form (not a global total) when paired with
  AnalyticDistribution.

Both modes use the same UniformDistribution (density-aware) — only the layout
swaps, so a density-mode run and a count-mode run with the same physics
parameters represent the same plasma with different sampling resolution.
"""
from pywarpx import picmi


VALID_PARTICLE_MODES = ("density", "count")


def make_layout(mode, *, grid, n_macroparticle_per_cell=None, n_test_particles_per_cell=None):
    """Return the PICMI layout object for the requested mode.

    Parameters
    ----------
    mode : str
        "density" or "count" — see module docstring.
    grid : picmi grid instance
        The simulation grid the layout follows.
    n_macroparticle_per_cell : sequence of int, required for "density"
        Particles per cell along each axis. Same shape PICMI's GriddedLayout
        expects.
    n_test_particles_per_cell : int, required for "count"
        Macroparticles to load per cell (randomly placed). Cells outside the
        plasma region get zero-weight particles via density_expression.
    """
    if mode == "density":
        if n_macroparticle_per_cell is None:
            raise ValueError(
                "particle_mode='density' requires n_macroparticle_per_cell"
            )
        return picmi.GriddedLayout(
            grid=grid,
            n_macroparticle_per_cell=n_macroparticle_per_cell,
        )
    if mode == "count":
        if n_test_particles_per_cell is None:
            raise ValueError(
                "particle_mode='count' requires n_test_particles_per_cell"
            )
        return picmi.PseudoRandomLayout(
            grid=grid,
            n_macroparticles_per_cell=int(n_test_particles_per_cell),
        )
    raise ValueError(
        f"Unknown particle_mode '{mode}' (use one of {VALID_PARTICLE_MODES})"
    )
