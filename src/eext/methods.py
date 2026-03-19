"""
Analytical equations for the external E-field, expected to be called to fill the external .h5 file.
"""
import numpy as np
from enum import Enum # for storing available methods

def fw_e(r, z, a, Q, resolution=500, **kwargs):
    """
    coord: input field coordinates (in cylindrical coordinates)
    a: radius of the ring, m
    Q: total charge on the ring, Coulombs
    """
    # constants and derived constants
    epsilon_0 = 8.854e-12  # Vacuum permittivity (F/m)
    lambda_ = Q / (2 * np.pi * a)  # Linear charge density

    # Define the electric field components
    def integrand_Er(theta, r, z):
        D = np.sqrt(r ** 2 + a ** 2 - 2 * a * r * np.cos(theta) + z ** 2)
        return (r - a * np.cos(theta)) / D ** 3

    def integrand_Ez(theta, r, z):
        D = np.sqrt(r ** 2 + a ** 2 - 2 * a * r * np.cos(theta) + z ** 2)
        return 1 / D ** 3
    theta = np.linspace(0, 2 * np.pi, resolution)
    dtheta = theta[1] - theta[0]
    int_Er = np.sum(integrand_Er(theta, r, z)) * dtheta
    int_Ez = np.sum(integrand_Ez(theta, r, z)) * dtheta
    E_r = (1 / (4 * np.pi * epsilon_0)) * lambda_ * a * int_Er
    E_z = (1 / (4 * np.pi * epsilon_0)) * lambda_ * a * z * int_Ez
    return E_r, E_z


def bob_e(r, z, a=1.0, Q=1e-9, resolution=100, **kwargs):
    """
    Implementation of Bob's E-field from a coil loop function.

    Assumes that the input coord is in cylindrical coordinates (rho, phi, z)

    Because the value is the same for all phi, this function returns the rho, zeta components only.
    """
    # print(f"q is: {q}")
    # print(f"coord is: {coord}")
    # print(f'FieldMethods_Impl.bob_e_impl.at: bob_e called with charge {q} and radius {radius}')
    # Parameters
    k = 8.99e9  # Coulomb's constant, N * m^2/C^2
    kq_a2 = (k * Q) / (a ** 2)

    # Coordinate Constants
    z = z / a
    r = r / a
    if abs(r) < 1e-10:
        r = 1e-10

    # Integral Constants - pg.3 of document
    mag = (r ** 2 + z ** 2 + 1)
    mag_3_2 = mag ** (3 / 2)
    ## Fzeta
    Fzeta_c = (z) / (mag_3_2 * (a ** 2))
    ## Frho
    Frho_c = (r) / (mag_3_2 * (a ** 2))

    # Integration
    # Circle is broken into {resolution} slices; with each result being appended to the lists below.

    thetas = np.linspace(0, np.pi, resolution,
                         dtype=np.float64)  # np.array of all the theta values used in the integration (of shape {resolution})
    cosines = np.cos(thetas)  # np.array of all the cosine values of the thetas
    denominators = (1 - ((2 * r * cosines) / mag)) ** (3 / 2)  # shared denominator values of fzeta and frho

    # replace zeros with a really small decimal.
    denominators[denominators == 0] = 1e-20

    fzeta = 1 / denominators
    frho = (1 - cosines / r) / denominators

    # Final - fzeta, frho is summed, multiplied by the integration constant and kq.a^2.
    E_z = np.asarray(fzeta).sum() * Fzeta_c * kq_a2
    E_r = np.asarray(frho).sum() * Frho_c * kq_a2
    return E_r, E_z

"""
Registry of available methods. This is generally what the user interfaces with
when selecting the E-field method to use.
"""
class EMethods(Enum):
    FW = (fw_e,)
    BOB = (bob_e,)