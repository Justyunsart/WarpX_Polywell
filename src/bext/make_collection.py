'''
Script that handles the creation of the current coils used in the simulation.

'''
from magpylib.current import Circle as C
from magpylib import Collection
import numpy as np

#==============#4
# CONSTRUCTION #
#==============#
# creates a square box of Loop coils
def make_polywell_collection(a, dia, d):
    """
    returns a magpylib Collection object with 6 circular current loops in a polywell formation.

    a: current in Amperes
    dia: diameter in cm
    d: distance from origin in cm
    """
    # current Loop creation, superimpose Loops and their fields
    s1 = C(current=-a, diameter=dia).move([-(d),0,0]).rotate_from_angax(90, [0, 1, 0])
    s2 = C(current=a, diameter=dia).move([(d),0,0]).rotate_from_angax(90, [0, 1, 0])
    s3 = C(current=a, diameter=dia).move([0,-(d),0]).rotate_from_angax(90, [1, 0, 0])
    s4 = C(current=-a, diameter=dia).move([0,(d),0]).rotate_from_angax(90, [1, 0, 0])
    s5 = C(current=-a, diameter=dia).move([0,0,-(d)])
    s6 = C(current=a, diameter=dia).move([0,0,(d)])

    c = Collection(s1,s2,s3,s4,s5,s6, style_color='black')
    return c

def return_n_turn_coil(a, offset, axis, r1, r2, n):
    rs = np.linspace(r1, r2, n)
    n_turn_coil = Collection()
    if axis == 0:
        coils = [C(current=a, diameter=r*2).move([offset,0,0]).rotate_from_angax(90, [0, 1, 0]) for r in rs]
    elif axis == 1:
        coils = [C(current=a, diameter=r*2).move([0,offset,0]).rotate_from_angax(-90, [1, 0, 0]) for r in rs]
    else:
        coils = [C(current=a, diameter=r*2).move([0,0,offset]) for r in rs]

    for coil in coils:
        n_turn_coil.add(coil)

    return n_turn_coil

def make_polywell_collection_n_turn(a, dia, d, r1, r2, n):
    """
    returns a magpylib Collection object with 6 circular current loops in a polywell formation.

    a: current in Amperes
    dia: diameter in cm
    d: distance from origin in cm
    """

    c1 = return_n_turn_coil(-a, -d, 0, r1, r2, n)
    c2 = return_n_turn_coil(a, d, 0, r1, r2, n)
    c3 = return_n_turn_coil(-a, -d, 1, r1, r2, n)
    c4 = return_n_turn_coil(a, d, 1, r1, r2, n)
    c5 = return_n_turn_coil(-a, -d, 2, r1, r2, n)
    c6 = return_n_turn_coil(a, d, 2, r1, r2, n)

    c = Collection(c1, c2, c3, c4, c5, c6, style_color='black')
    return c

# helmholtz setup for a test
def make_helmholtz_collection(a, dia, d):
    # helmholtz test
    s7 = C(current=a, diameter=dia).move([-(d),0,0]).rotate_from_angax(90, [0, 1, 0])
    s8 = C(current=a, diameter=dia).move([(d),0,0]).rotate_from_angax(90, [0, 1, 0])

    c = Collection (s7, s8)
    return c
