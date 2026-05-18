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

from dataclasses import dataclass, field
from src.domain import Domain

@dataclass
class PolywellConfig:

    # geometry
    symmetry:           str             = "full"
    domain:             Domain          = None

    # spawn type (particle mode)
    particle_mode:      str             = "density"

    # solver
    solver_type:        str             = ""

    # Plasma
    p_density:          float           = 1e21
    plasma_count:       int             = 0
    Ti_eV:              float           = 85e3
    Te_eV:              float           = 50e3
    plasma_bounding:    float           = 0.11

    # Grid
    L:                  float           = 2.0
    N:                  int             = 24
    n_per_cell:         list            = field(default_factory=lambda: [10, 10, 10])

    # B-field
    b_dia:              float           = 1.0
    b_offset:           float           = 1.1
    I:                  float           = 8e6
    b_method:           str             = "hybrid"

    # E-field
    e_method:           str             = None
    Q:                  float           = 0.0
    e_dia:              float           = 0.0
    e_offset:           float           = 0.0

    # Derived quantities — computed post-init, not set by user
    Ti_J:               float           = field(init=False)
    Te_J:               float           = field(init=False)
    dx:                 float           = field(init=False)
    beta:               float           = field(init=False)

    def __post_init__(self):
        import scipy.constants as sc
        import numpy as np
        MU0 = 4 * np.pi * 1e-7
        self.Ti_J = self.Ti_eV * sc.eV
        self.Te_J = self.Te_eV * sc.eV
        self.dx   = self.L / self.N
        self.beta = self.p_density * self.Ti_J * 2 * MU0 / self.B_coil**2

# configs/polywell_config.py

TEST_CONFIG = PolywellConfig(
    symmetry        = "full",
    particle_mode   = "density",
    solver_type     = "EM",
    p_density       = 1e12,
    Ti_eV           = 1e3,
    Te_eV           = 50e3,
    plasma_bounding = 0.11,
    L               = 2.0,
    N               = 72,
    n_per_cell      = [10, 10, 10],
    # unused here since we use a test current I = 1e6 (needs to be 8e6 for 10T B-Field)
    b_dia           = 1.0,
    b_offset        = 1.1,
    I               = 1e6,
    b_method        = "file",
    e_method        = "FW",
    Q               = 1e-9,
    e_dia           = 0.75,
    e_offset        = 1.1,
)

HYBRID_CONFIG = PolywellConfig(
    symmetry        = "full",
    particle_mode   = "density",
    solver_type     = "hybrid",
    p_density       = 1e21,
    Ti_eV           = 85e3,
    Te_eV           = 50e3,
    plasma_bounding = 0.11,
    L               = 2.0,
    N               = 24,
    n_per_cell      = [10, 10, 10],
    b_dia           = 1.0,
    b_offset        = 1.1,
    I               = 1e6,
    b_method        = "file",
    e_method        = None,
    Q               = 0.0,
    e_dia           = 0.0,
)