"""
A polywell dataclass that handles initialization of relevant simulation parameters

Provided at the bottom is HYBRID_CONFIG

This is the base config being used to run simulations at the moment.
"""

VALID_SYMMETRIES = ("full", "octant")
VALID_PARTICLE_MODES = ("density", "count")
VALID_SOLVERS = ("EM", "hybrid")
VALID_B_METHODS = ("file", "analytic")

from dataclasses import dataclass, field
from warpx_polywell.domain import Domain, derive_domain
import scipy.constants as sc
import numpy as np

@dataclass
class PolywellHybridConfig:

    # geometry
    symmetry:                       str             = "full"

    # count/density 
    particle_mode:                  str             = "density"

    # Plasma    
    p_density:                      float           = 1e21
    n_test_particles_per_cell:      int             = 0
    Ti_eV:                          float           = 85e3
    Te_eV:                          float           = None
    plasma_bounding:                float           = 0.11
    mass:                           float           = 1.0

    # Grid  
    L:                              float           = 2.0
    N:                              int             = 24
    n_per_cell_each_dim:            list            = field(default_factory=lambda: [10, 10, 10])

    # B-field           
    b_dia:                          float           = 1.0
    b_offset:                       float           = 1.1
    I:                              float           = 8e6
    b_method:                       str             = "hybrid"

    # n-turn coil
    n_turns:                        int             = 1
    r1:                             float           = None  # defaults to b_dia/2
    r2:                             float           = None  # defaults to b_dia/2

    # Derived quantities — computed post-init, not set by user
    domain:                         Domain          = field(init=False)
    Ti_J:                           float           = field(init=False)
    Te_J:                           float           = field(init=False)
    dx:                             float           = field(init=False)
    B_coil:                         float           = field(init=False)

    def __post_init__(self):

        if self.symmetry not in VALID_SYMMETRIES:
            raise ValueError(f"`symmetry` must be one of {VALID_SYMMETRIES}")
        if self.particle_mode not in VALID_PARTICLE_MODES:
            raise ValueError(f"`particle_mode` must be one of {VALID_PARTICLE_MODES}")
        if self.b_method not in VALID_B_METHODS:
            raise ValueError(f"`b_method` must be one of {VALID_B_METHODS}")
        if not self.N or not self.L:
            raise ValueError(f"`L` and `N` must be provided for proper Domain initialization")
        
        # initialize Te if none, assumed equal to Ti
        if not self.Te_eV:
            self.Te_eV = self.Ti_eV
        
        # If single turn coil, r1 = r2
        if self.r1 is None:
            self.r1 = self.b_dia / 2
        if self.r2 is None:
            self.r2 = self.b_dia / 2

        self.Ti_J = self.Ti_eV * sc.eV
        self.Te_J = self.Te_eV * sc.eV
        # NOW CALCULATING B_COIL (@ center) FROM COIL CONFIG
        MU0 = 4 * np.pi * 1e-7
        self.domain = derive_domain(self.symmetry, self.L, self.N, hybrid=True)
        self.dx   = 2*self.L / self.N if self.symmetry == "full" else self.L / (self.N / 2)

    def compute_b(self):
        MU0 = 4 * np.pi * 1e-7
        if self.n_turns != 1:
            rs = np.linspace(self.r1, self.r2, self.n_turns)
            self.B_coil = sum((MU0 * self.I) / (2 * r) for r in rs)
        else:
            self.B_coil = (MU0 * self.I) / self.b_dia      # field at coil center, Tesla

# configs/polywell_hybrid_config.py

HYBRID_CONFIG = PolywellHybridConfig(
    symmetry                = "full",
    particle_mode           = "density",
    mass                    = 1,
    # TODO: increase density x10 -> pressure x 10
    # Larger n -> smaller alfven (whistler still binding)
    # Larger n -> larger plasma freq -> smaller skin depth
    # Larger n -> 
    p_density               = 2e21,
    Ti_eV                   = 100,
    Te_eV                   = 100,
    # This is applied automatically in the script, ensuring plasma is spawned in the core of the system
    # NOTE --- Spawning at high-beta leads to instability, a concern to think about
    # NOTE --- hence spawns closer to center of core where beta is lower
    plasma_bounding         = 1, # this needs to actually be within the null zone, L * bounding <= offset
    L                       = 0.2,
    N                       = 128,
    n_per_cell_each_dim     = [5, 5, 5],
    b_dia                   = 0.1,
    b_offset                = 0.1,
    I                       = 8e3,
    b_method                = "file",
)