"""
Math helpers
"""
import numpy as np

def toCyl(coord):
    """
    toCyl: a helper function.
        - Takes in an input coordinate in cartesian space
        - Outputs its equivalent in cylindrical space.
    """
    rho = np.sqrt(coord[0]**2 + coord[1]**2)
    azimuth = np.arctan2(coord[1], coord[0])
    return np.array([rho, azimuth, coord[2]])

def toCart(r, theta, z):
    """
    Takes an input of a cylindrical coordinate and converts it to
    cartesian coordinates
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z

