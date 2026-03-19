'''
Script that handles the creation of the current coils used in the simulation.

'''
from magpylib.current import Circle as C
from magpylib import Collection

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
    s1 = C(current=a, diameter=dia).move([-(d),0,0]).rotate_from_angax(90, [0, 1, 0])
    s2 = C(current=-a, diameter=dia).move([(d),0,0]).rotate_from_angax(90, [0, 1, 0])
    s3 = C(current=-a, diameter=dia).move([0,-(d),0]).rotate_from_angax(-90, [1, 0, 0])
    s4 = C(current=a, diameter=dia).move([0,(d),0]).rotate_from_angax(-90, [1, 0, 0])
    s5 = C(current=a, diameter=dia).move([0,0,-(d)]).rotate_from_angax(90, [0, 0, 1])
    s6 = C(current=-a, diameter=dia).move([0,0,(d)]).rotate_from_angax(90, [0, 0, 1])

    c = Collection(s1,s2,s3,s4,s5,s6, style_color='black')
    return c

# helmholtz setup for a test
def make_helmholtz_collection(a, dia, d):
    # helmholtz test
    s7 = C(current=a, diameter=dia).move([-(d),0,0]).rotate_from_angax(90, [0, 1, 0])
    s8 = C(current=a, diameter=dia).move([(d),0,0]).rotate_from_angax(90, [0, 1, 0])

    c = Collection (s7, s8)
    return c
