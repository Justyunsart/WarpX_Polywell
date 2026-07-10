'''
Script that handles the creation of the current coils used in the simulation.

'''
from magpylib.current import Circle as C
from magpylib import Collection

from warpx_polywell.coils.primitives import Loop, Polywell


#==========#
# ADAPTER  #
#==========#
def _loop_to_circle(loop: Loop):
    """
    Build one magpylib Circle from a Loop, in the canonical right-hand-rule
    convention: a loop with axis `k` and positive current produces axial B
    along +k (matching analytic._eval_loop_cartesian).

    A magpylib Circle has normal +z; rotate it so +z maps onto the loop axis:
      x-axis: rotate +90 about y  (+z -> +x)
      y-axis: rotate -90 about x  (+z -> +y)
      z-axis: no rotation
    Rotation is about the loop's own centre, so move/rotate order is irrelevant.
    """
    c = C(current=loop.current, diameter=2.0 * loop.radius)
    if loop.axis == "x":
        c = c.rotate_from_angax(90, [0, 1, 0]).move([loop.position, 0, 0])
    elif loop.axis == "y":
        c = c.rotate_from_angax(-90, [1, 0, 0]).move([0, loop.position, 0])
    else:  # "z"
        c = c.move([0, 0, loop.position])
    return c


def to_collection(loops, *, style_color="black"):
    """
    Adapter: turn a `list[Loop]` into a magpylib Collection for `.getB()`.

    This is the single bridge from the canonical coil representation to
    magpylib; every composite (Polywell, Washer, ...) reaches magpylib through
    here. Units are SI (metres, amperes) throughout.
    """
    return Collection(*[_loop_to_circle(lp) for lp in loops], style_color=style_color)

#==============#4
# CONSTRUCTION #
#==============#
# All constructors below are thin wrappers: geometry lives in the Loop/Polywell
# primitives (warpx_polywell.coils), and to_collection is the only bridge to
# magpylib. Units are SI (metres, amperes).
def make_polywell_collection(a, dia, d):
    """
    returns a magpylib Collection object with 6 circular current loops in a
    polywell formation.

    a   : current in amperes
    dia : coil diameter in metres
    d   : distance from origin to coil centre in metres
    """
    return to_collection(Polywell(current=a, diameter=dia, offset=d).expand())


# helmholtz setup for a test: two coaxial x-loops with same-direction current
def make_helmholtz_collection(a, dia, d):
    r = dia / 2
    return to_collection([Loop("x", -d, r, a), Loop("x", d, r, a)])
