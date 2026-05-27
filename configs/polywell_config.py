"""
A polywell dataclass that handles initialization via a separate class that
includes all relevant parameters in the experiments.

Example:

# configs/my_run.py
from inputs.config import PolywellConfig

cfg = PolywellConfig(
    p_density = 1e21,
    N         = 32,
    I         = 8e6,
)

polywell = PollywellSixCoil(cfg=cfg)

# Production run
cfg = PolywellConfig(p_density=1e21, N=24, I=8e6)
polywell = PollywellSixCoil(cfg=cfg)
polywell.run()

# Test run
cfg = PolywellConfig(
    p_density = 1e12,
    Ti_eV     = 1e3,
    N         = 72,
    b_method  = "file",
    I         = 1e6,
    e_method  = "FW",
    Q         = 1e-9,
    e_dia     = 0.75,
    e_offset  = 1.1,
)
polywell = PollywellSixCoil(cfg=cfg)
polywell.run()
"""

VALID_SYMMETRIES = ("full", "octant")
VALID_PARTICLE_MODES = ("density", "count")
VALID_SOLVERS = ("EM", "hybrid")
VALID_B_METHODS = ("file", "analytic")

from dataclasses import dataclass, field
from src.domain import Domain, derive_domain

@dataclass
class PolywellConfig:

    # geometry
    symmetry:                       str             = "full"
    domain:                         Domain          = None

    # count/density 
    particle_mode:                  str             = "density"

    # solver            
    solver_type:                    str             = ""

    # Plasma    
    p_density:                      float           = 1e21
    n_test_particles_per_cell:      int             = 0
    Ti_eV:                          float           = 85e3
    Te_eV:                          float           = 50e3
    plasma_bounding:                float           = 0.11

    # Grid  
    L:                              float           = 2.0
    N:                              int             = 24
    n_per_cell_each_dim:            list            = field(default_factory=lambda: [10, 10, 10])

    # B-field           
    b_dia:                          float           = 1.0
    b_offset:                       float           = 1.1
    I:                              float           = 8e6
    b_method:                       str             = "hybrid"
    substeps:                       int             = -1

    # n-turn coil
    n_turns:                        int             = 1
    b_inner_radius:                 float           = None  # defaults to b_dia/2
    b_outer_radius:                 float           = None  # defaults to b_dia/2

    # E-field           
    e_method:                       str             = None
    Q:                              float           = 0.0
    e_dia:                          float           = 0.0
    e_offset:                       float           = 0.0

    # Derived quantities — computed post-init, not set by user
    Ti_J:                       float           = field(init=False)
    Te_J:                       float           = field(init=False)
    dx:                         float           = field(init=False)
    B_coil:                     float           = field(init=False)
    beta:                       float           = field(init=False)

    def __post_init__(self):

        if self.symmetry not in VALID_SYMMETRIES:
            raise ValueError(f"`symmetry` must be one of {VALID_SYMMETRIES}")
        if self.particle_mode not in VALID_PARTICLE_MODES:
            raise ValueError(f"`particle_mode` must be one of {VALID_PARTICLE_MODES}")
        if self.solver_type not in VALID_SOLVERS:
            raise ValueError(f"`solver_type` must be one of {VALID_SOLVERS}")
        if self.b_method not in VALID_B_METHODS:
            raise ValueError(f"`b_method` must be one of {VALID_B_METHODS}")
        if not self.N or not self.L:
            raise ValueError(f"`L` and `N` must be provided for proper Domain initialization")
        if self.solver_type == "hybrid" and self.substeps < 0:
            raise ValueError(f"`substeps`: B-field solve substeps, needs to be provided in config when using `HybridPICSolver`")
        
        # If single turn coil, a = b
        if self.b_inner_radius is None:
            self.b_inner_radius = self.b_dia / 2
        if self.b_outer_radius is None:
            self.b_outer_radius = self.b_dia / 2

        import scipy.constants as sc
        import numpy as np
        MU0 = 4 * np.pi * 1e-7
        self.Ti_J = self.Ti_eV * sc.eV
        self.Te_J = self.Te_eV * sc.eV
        # NOW CALCULATING B_COIL (@ center) FROM COIL CONFIG
        MU0 = 4 * np.pi * 1e-7
        self.B_coil = (MU0 * self.I) / self.b_dia      # field at coil center, Tesla
        self.domain = derive_domain(self.symmetry, self.L, self.N)
        self.dx   = 2*self.L / self.N if self.symmetry == "full" else self.L / (self.N / 2)

# configs/polywell_config.py

TEST_CONFIG = PolywellConfig(
    symmetry                    = "octant",
    particle_mode               = "count",
    solver_type                 = "EM",
    p_density                   = 1e18,
    n_test_particles_per_cell   = 10,
    Ti_eV                       = 1e4,
    Te_eV                       = 50e3,
    plasma_bounding             = 0.25,
    L                           = 1.0,
    N                           = 208,
    n_per_cell_each_dim         = [10, 10, 10],
    b_dia                       = 1.0,
    b_offset                    = 1.0,
    I                           = 1e6,
    b_method                    = "file",
    e_method                    = None,
    Q                           = 1e-9,
    e_dia                       = 0.75,
    e_offset                    = 1.1,
)

HYBRID_CONFIG = PolywellConfig(
    symmetry                = "full",
    particle_mode           = "density",
    solver_type             = "hybrid",
    p_density               = 4.4e21,
    Ti_eV                   = 1e3,
    Te_eV                   = 1e3,
    plasma_bounding         = 0.05, # this needs to actually be within the null zone
    L                       = 0.3,
    N                       = 128,
    n_per_cell_each_dim     = [4, 4, 4],
    b_dia                   = 0.2,
    b_offset                = 0.16,
    I                       = 73e3,
    b_method                = "file",
    substeps                = 5,
    e_method                = None,
    Q                       = 0.0,
    e_dia                   = 0.0,
)