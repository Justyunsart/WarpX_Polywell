# src/aext/aext.py

import numpy as np
from scipy.special import ellipk, ellipe
from src.warpx_polywell.domain import Domain

MU0 = 4e-7 * np.pi

POLYWELL_COILS = [
    ('x', -1, -1),   # s1: -x axis, current -I
    ('x', +1, +1),   # s2: +x axis, current +I
    ('y', -1, -1),   # s3: -y axis, current -I
    ('y', +1, +1),   # s4: +y axis, current +I
    ('z', -1, -1),   # s5: -z axis, current -I
    ('z', +1, +1),   # s6: +z axis, current +I
]

EPS_SAFE = 1e-30

def _Aphi_vec(rho, zeta, R, I):
    """
    Vectorized Aphi for a single circular filament of radius R, current I.
    rho, zeta : arrays of cylindrical coords (radial, axial from coil center)
    Returns Aphi array, shape matching rho.
    """
    rho_safe = np.where(rho < EPS_SAFE, EPS_SAFE, rho)
    k2 = np.clip(4 * R * rho_safe / ((R + rho_safe)**2 + zeta**2), 0, 1 - EPS_SAFE)
    K = ellipk(k2)
    E = ellipe(k2)
    # analytic.py
    # (I * MU0 / pi)*sqrt(R / (r + 1e-30))*((1.0 - 0.5 * k2) * K - E) / sqrt(k2**2 + 1e-30)
    # (I * MU0 *)
    pref = MU0 * I / (np.pi * np.sqrt(k2 + 1e-30))
    Aphi = pref * np.sqrt(R / rho_safe) * ((1 - 0.5 * k2) * K - E)
    # MU0 * I * sqrt(R / r) * ((1.0 - 0.5 * k2) * K - E) / (pi * sqrt(k2**2 + 1e-30))

    # MU0 * I * sqrt(R / r) * ((1.0 - 0.5 * k2) * K - E) / (pi * sqrt(k2 + 1e-30))
    # zero exactly on axis (rho=0 has no phi direction)
    return np.where(rho < EPS_SAFE, 0.0, Aphi)

def _n_turn_Aphi_vec(rho, zeta, I, a, b, n):
    Rs = np.linspace(a, b, n)
    # implies shape = same as rho
    Aphi = np.sum([_Aphi_vec(rho, zeta, R, I) for R in Rs], axis=0)
    assert Aphi.shape == rho.shape, f"[_n_turn_Aphi_vec] Mismatch in shape {Aphi.shape} != {rho.shape}"
    return Aphi

def _n_turn_Aphi_vec_linear_dropoff_current(rho, zeta, I, a, b, n):
    """
    Identical to _n_turn_Aphi_vec except it drops current off linearily as radius increases

    Implements a basic linear dropoff dependent upon 1/turn_i, where turn_i is an integer
    denoting the turn number, within [1, n], inclusive. 
    """
    Rs = np.linspace(a, b, n)
    # implies shape = same as rho
    Aphi = np.zeros_like(rho)
    Aphi = np.sum([_Aphi_vec(rho, zeta, R, I / (k+1)) for k, R in enumerate(Rs)], axis=0)
    assert Aphi.shape == rho.shape, f"[_n_turn_Aphi_vec] Mismatch in shape {Aphi.shape} != {rho.shape}"
    return Aphi

def _n_turn_Aphi_vec_exponential_dropoff_current(rho, zeta, I, a, b, n):
    """
    Implements a dropoff via the function:

    J(r) = exp(1 - (2r)**2)(2r)**2 (A)

    In order to implement this, I convert radius into n, such that at turn n_turns / 2, we reach a maximum
    as the function adheres to.

    J(n) = exp(1 - (2*turn_i / n)**2) * (2*turn_i / n)**2

    This necessitates n_turns % 2 == 0
    """
    assert n % 2 == 0 and n > 3, "The function as implemented requires an even number of turns to properly map the max current to turn number"
    if n == 3:
        print("WARNING::aext::_n_turn_vec_exponential_dropoff_current::Recommend higher n for current distribution to better adhere to the function")

    def exponential_current(turn_i, n):
        t1 = np.exp(1 - (2 * turn_i / n)**2)
        t2 = (2 * turn_i / n)**3
        return t1 * t2

    Rs = np.linspace(a, b, n)
    Aphi = np.zeros_like(rho)
    Aphi = np.sum([_Aphi_vec(rho, zeta, R, I * exponential_current(k, n)) for k, R in enumerate(Rs)], axis=0)
    return Aphi
    

def _A_single_n_turn_coil(X, Y, Z, axis, pos, I, a, b, n, fn_Aphi_vec=_n_turn_Aphi_vec):
    """
    Cartesian A contribution from one filament coil.
    axis: 'x', 'y', or 'z' — coil symmetry axis
    pos : coil center coordinate along that axis
    I   : signed current
    a   : inner radius
    b   : outer radius
    n   : n-turns
    Returns Ax, Ay, Az arrays of shape matching X.

    Using right-hand rule: x -> y -> z -> x

    rho_s denotes a safe rho using EPS_SAFE = 1e-12
    """

    # [x / p, y / p, 0.0] -> [-y / p, x / p, 0.0]
    if axis == 'z':
        # radial part
        rho   = np.sqrt(X**2 + Y**2)
        # retrieve axial part
        zeta  = Z - pos
        Aphi  = fn_Aphi_vec(rho, zeta, I, a, b, n)
        # phi_hat = (-sin(phi), cos(phi), 0) = (-y/rho, x/rho, 0)
        rho_s = np.where(rho < EPS_SAFE, EPS_SAFE, rho)
        return -Aphi * Y / rho_s, Aphi * X / rho_s, np.zeros_like(X)

    # [0.0, y / p, z / p] -> [0.0, -z / p, y / p]
    elif axis == 'x':
        rho   = np.sqrt(Y**2 + Z**2)
        zeta  = X - pos
        Aphi  = fn_Aphi_vec(rho, zeta, I, a, b, n)
        rho_s = np.where(rho < EPS_SAFE, EPS_SAFE, rho)
        # coil axis is x: phi rotates in y-z plane
        # phi_hat in local (y,z) plane = (-z/rho, y/rho) mapped to global
        return np.zeros_like(X), -Aphi * Z / rho_s, Aphi * Y / rho_s

    # [z / p, 0, 0, x / p] -> [-x / p, 0.0, z / p]
    elif axis == 'y':
        rho   = np.sqrt(Z**2 + X**2)
        zeta  = Y - pos
        Aphi  = fn_Aphi_vec(rho, zeta, I, a, b, n)
        rho_s = np.where(rho < EPS_SAFE, EPS_SAFE, rho)
        # coil axis is y: phi rotates in z-x plane
        return Aphi * Z / rho_s, np.zeros_like(X), -Aphi * X / rho_s
    
def _A_single_n_turn_coil_with_linear_dropoff_current(X, Y, Z, axis, pos, I, a, b, n):
    """
    Cartesian A contribution from one filament coil.
    axis: 'x', 'y', or 'z' — coil symmetry axis
    pos : coil center coordinate along that axis
    I   : signed current
    a   : inner radius
    b   : outer radius
    n   : n-turns
    Returns Ax, Ay, Az arrays of shape matching X.

    Using right-hand rule: x -> y -> z -> x

    rho_s denotes a safe rho using EPS_SAFE = 1e-12
    """

    return _A_single_n_turn_coil(X, Y, Z, axis, pos, I, a, b, n, _n_turn_Aphi_vec_linear_dropoff_current)

    # # [x / p, y / p, 0.0] -> [-y / p, x / p, 0.0]
    # if axis == 'z':
    #     # radial part
    #     rho   = np.sqrt(X**2 + Y**2)
    #     # retrieve axial part
    #     zeta  = Z - pos
    #     Aphi  = _n_turn_Aphi_vec_linear_dropoff_current(rho, zeta, I, a, b, n)
    #     # phi_hat = (-sin(phi), cos(phi), 0) = (-y/rho, x/rho, 0)
    #     rho_s = np.where(rho < EPS_SAFE, EPS_SAFE, rho)
    #     return -Aphi * Y / rho_s, Aphi * X / rho_s, np.zeros_like(X)

    # # [0.0, y / p, z / p] -> [0.0, -z / p, y / p]
    # elif axis == 'x':
    #     rho   = np.sqrt(Y**2 + Z**2)
    #     zeta  = X - pos
    #     Aphi  = _n_turn_Aphi_vec_linear_dropoff_current(rho, zeta, I, a, b, n)
    #     rho_s = np.where(rho < EPS_SAFE, EPS_SAFE, rho)
    #     # coil axis is x: phi rotates in y-z plane
    #     # phi_hat in local (y,z) plane = (-z/rho, y/rho) mapped to global
    #     return np.zeros_like(X), -Aphi * Z / rho_s, Aphi * Y / rho_s

    # # [z / p, 0, 0, x / p] -> [-x / p, 0.0, z / p]
    # elif axis == 'y':
    #     rho   = np.sqrt(Z**2 + X**2)
    #     zeta  = Y - pos
    #     Aphi  = _n_turn_Aphi_vec_linear_dropoff_current(rho, zeta, I, a, b, n)
    #     rho_s = np.where(rho < EPS_SAFE, EPS_SAFE, rho)
    #     # coil axis is y: phi rotates in z-x plane
    #     return Aphi * Z / rho_s, np.zeros_like(X), -Aphi * X / rho_s

def compute_A_polywell(X, Y, Z, I, offset, a, b, n):
    """
    Total vector potential A from all 6 polywell coils.
    X, Y, Z : coordinate arrays (any shape, must broadcast together)
    I       : coil current magnitude (A)
    offset  : coil center distance from origin (m)
    a       : inner radius (m)
    b       : outer radius (m)
    n       : n-turns
    Returns Ax, Ay, Az — same shape as inputs.
    """
    Ax = np.zeros_like(X, dtype=float)
    Ay = np.zeros_like(Y, dtype=float)
    Az = np.zeros_like(Z, dtype=float)
    for axis, pos_sign, I_sign in POLYWELL_COILS:
        ax, ay, az = _A_single_n_turn_coil(X, Y, Z, axis, pos_sign * offset, I_sign * I, a, b, n)
        Ax += ax
        Ay += ay
        Az += az
    return Ax, Ay, Az

def curlA(Ax, Ay, Az, dx, dy, dz):
    # Curl of A =
    # Bx = dAz/dy - dAy/dz
    # By = dAx/dz - dAz/dx
    # Bz = dAy/dx - dAx/dy
    dAz_dy = np.gradient(Az, dy, axis=1, edge_order=2)
    dAy_dz = np.gradient(Ay, dz, axis=2, edge_order=2)

    dAx_dz = np.gradient(Ax, dz, axis=2, edge_order=2)
    dAz_dx = np.gradient(Az, dx, axis=0, edge_order=2)

    dAy_dx = np.gradient(Ay, dx, axis=0, edge_order=2)
    dAx_dy = np.gradient(Ax, dy, axis=1, edge_order=2)
    
    B_curlA = {
        'x': dAz_dy - dAy_dz,
        'y': dAx_dz - dAz_dx,
        'z': dAy_dx - dAx_dy
    }

    return B_curlA

def get_B_disk(X, Y, Z, I, r1, r2, n_turns, ring_r, dx, dy, dz):
    N = X.shape[0]
    Axd, Ayd, Azd = _A_single_n_turn_coil(X, Y, Z, 'x', 0.0, I, r1, r2, n_turns)
    Bd = curlA(Axd, Ayd, Azd, dx, dy, dz)
    Bxd, Byd, Bzd = Bd['x'], Bd['y'], Bd['z']

    ring_Bx, _, _ = get_B_ring(X, Y, Z, I, ring_r)

    Bxd_center = Bxd[N//2, N//2, N//2]
    Bxr_center = ring_Bx[N//2, N//2, N//2]

    scale = Bxr_center / Bxd_center

    Axd, Ayd, Azd = _A_single_n_turn_coil(X, Y, Z, 'x', 0.0, I * scale, r1, r2, n_turns)
    Bd = curlA(Axd, Ayd, Azd, dx, dy, dz)
    disk_Bx, disk_By, disk_Bz = Bd['x'], Bd['y'], Bd['z']

    return disk_Bx, disk_By, disk_Bz

def get_B_ring(X, Y, Z, I, r, dx, dy, dz):
    Axr, Ayr, Azr = _A_single_n_turn_coil(X, Y, Z, 'x', 0.0, I, r, r, 1)
    Br = curlA(Axr, Ayr, Azr, dx, dy, dz)
    ring_Bx, ring_By, ring_Bz = Br['x'], Br['y'], Br['z']
    return ring_Bx, ring_By, ring_Bz