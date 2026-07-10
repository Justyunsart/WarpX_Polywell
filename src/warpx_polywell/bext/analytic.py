"""
Analytic B-field module for WarpX polywell simulations.

Provides two interfaces:
  1. NumPy evaluation — for testing, plotting, and validation
  2. WarpX parser expressions — for runtime per-particle evaluation without grid files

The field from each circular current loop is computed using the exact elliptic
integral solution. For the WarpX parser path, AMReX's native comp_ellint_1(k)
and comp_ellint_2(k) functions are used directly — no polynomial approximations.

Convention note:
  - scipy.special.ellipk(m) takes parameter m = k²
  - AMReX comp_ellint_1(k) takes modulus k = sqrt(m)
  So: ellipk(m) ↔ comp_ellint_1(sqrt(m))

Usage (analytic mode):
    from warpx_polywell.bext.analytic import build_bext_expressions

    exprs = build_bext_expressions(I=1e6, dia=1.0, offset=1.1)
    # exprs['Bx'], exprs['By'], exprs['Bz'] are parser-ready strings
"""

import numpy as np
from scipy.special import ellipk, ellipe

MU0 = 4e-7 * np.pi

# The polywell coil layout now lives in warpx_polywell.coils.Polywell — it is the
# single source of truth. NumPy and parser-expression builders below all consume
# a list[Loop], so any coil configuration (single coil, washer, ...) is expressed
# by passing the appropriate loops rather than mutating a module-level layout.


# ================================================================
# 1. NumPy evaluation (for testing and plotting)
# ================================================================

def B_single_loop(rho, zeta, a, I):
    """
    Magnetic field (B_rho, B_zeta) from a single circular current loop.

    Parameters
    ----------
    rho   : radial distance from coil axis
    zeta  : axial distance from coil center
    a     : coil radius (m)
    I     : current (A)

    Returns
    -------
    B_rho, B_zeta : field components in cylindrical coordinates
    """
    rho = np.asarray(rho, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    rho_safe = np.where(np.abs(rho) < 1e-15, 1e-15, rho)

    alpha2 = (rho_safe - a)**2 + zeta**2
    beta2  = (rho_safe + a)**2 + zeta**2
    k2 = np.clip(4 * a * rho_safe / beta2, 0, 1 - 1e-15)

    K = ellipk(k2)
    E = ellipe(k2)

    C = MU0 * I / (2 * np.pi)
    sqrt_b2 = np.sqrt(beta2)

    B_zeta = C / sqrt_b2 * (K + (a**2 - rho_safe**2 - zeta**2) / alpha2 * E)
    B_rho  = C * zeta / (rho_safe * sqrt_b2) * (-K + (a**2 + rho_safe**2 + zeta**2) / alpha2 * E)
    B_rho  = np.where(np.abs(rho) < 1e-12, 0.0, B_rho)

    return B_rho, B_zeta


def _eval_loop_cartesian(X, Y, Z, axis, pos, a, I):
    """
    Evaluate a single loop's B-field in Cartesian components.

    Parameters
    ----------
    X, Y, Z : observation coordinates (arrays)
    axis    : 'x', 'y', or 'z' — coil symmetry axis
    pos     : coil center position along that axis
    a       : coil radius
    I       : current
    """
    if axis == 'z':
        rho   = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        zeta  = Z - pos
        Br, Bax = B_single_loop(rho, zeta, a, I)
        return Br * np.cos(theta), Br * np.sin(theta), Bax

    elif axis == 'x':
        rho   = np.sqrt(Z**2 + Y**2)
        theta = np.arctan2(Y, Z)
        zeta  = X - pos
        Br, Bax = B_single_loop(rho, zeta, a, I)
        return Bax, Br * np.sin(theta), Br * np.cos(theta)

    elif axis == 'y':
        rho   = np.sqrt(Z**2 + X**2)
        theta = np.arctan2(X, Z)
        zeta  = Y - pos
        Br, Bax = B_single_loop(rho, zeta, a, I)
        return Br * np.sin(theta), Bax, Br * np.cos(theta)


def B_from_loops(X, Y, Z, loops):
    """
    Total B-field from an arbitrary `list[Loop]` (NumPy evaluation).

    Parameters
    ----------
    X, Y, Z : observation coordinates (scalars or arrays)
    loops   : iterable of warpx_polywell.coils.Loop

    Returns
    -------
    Bx, By, Bz : Cartesian field components
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    Bx = np.zeros_like(X)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)

    for lp in loops:
        bx, by, bz = _eval_loop_cartesian(X, Y, Z, lp.axis, lp.position, lp.radius, lp.current)
        Bx += bx
        By += by
        Bz += bz

    return Bx, By, Bz


def B_polywell(X, Y, Z, I, dia, offset):
    """
    Total B-field from a 6-coil polywell (NumPy evaluation) — a convenience
    wrapper over `B_from_loops`.

    Parameters
    ----------
    X, Y, Z : observation coordinates (scalars or arrays)
    I       : coil current (A)
    dia     : coil diameter (m)
    offset  : coil center distance from origin (m)
    """
    from warpx_polywell.coils import Polywell
    return B_from_loops(X, Y, Z, Polywell(I, dia, offset).expand())


# ================================================================
# 2. WarpX parser expression builder
# ================================================================
# Uses AMReX's native comp_ellint_1(k) and comp_ellint_2(k) —
# no polynomial approximations needed.
#
# AMReX parser local-variable syntax:
#   "var1=expr1; var2=expr2; final_expression"
# The last expression (after the final semicolon) is the returned value.
# All local variables share the same flat namespace within one expression,
# so each coil uses uniquely-suffixed names (e.g. r2_1, K_1, r2_2, K_2, …).
#
# Convention: comp_ellint_1(k) takes modulus k, NOT parameter m=k².
#   scipy:  K = ellipk(m)        where m = k²
#   AMReX:  K = comp_ellint_1(k) where k = sqrt(m)
# ================================================================

def _coil_var_defs(tag, axis, pos, a, eps=1e-30):
    """
    Build local-variable assignment strings for a single coil.

    Parameters
    ----------
    tag  : str — unique suffix for variable names (e.g. "1", "2", …)
    axis : 'x', 'y', or 'z' — coil symmetry axis
    pos  : float — coil center position along that axis (m)
    a    : float — coil radius (m)

    Returns
    -------
    list of "varname=expression" strings (to be joined with "; ")
    """
    a2 = a**2

    # Cylindrical coordinate mapping depends on coil axis
    if axis == 'z':
        r2_rhs  = "x*x+y*y"
        z_rhs   = f"z-({pos:.15e})"
        ct_rhs  = f"x/(r_{tag}+{eps})"
        st_rhs  = f"y/(r_{tag}+{eps})"
    elif axis == 'x':
        r2_rhs  = "z*z+y*y"
        z_rhs   = f"x-({pos:.15e})"
        ct_rhs  = f"z/(r_{tag}+{eps})"
        st_rhs  = f"y/(r_{tag}+{eps})"
    elif axis == 'y':
        r2_rhs  = "z*z+x*x"
        z_rhs   = f"y-({pos:.15e})"
        ct_rhs  = f"z/(r_{tag}+{eps})"
        st_rhs  = f"x/(r_{tag}+{eps})"

    return [
        f"r2_{tag}={r2_rhs}",
        f"r_{tag}=sqrt(r2_{tag}+{eps**2})",
        f"z_{tag}={z_rhs}",
        f"ct_{tag}={ct_rhs}",
        f"st_{tag}={st_rhs}",
        f"a2_{tag}=(r_{tag}-{a:.15e})**2+z_{tag}**2",          # alpha²
        f"b2_{tag}=(r_{tag}+{a:.15e})**2+z_{tag}**2",          # beta²
        f"k_{tag}=sqrt(min(4.0*{a:.15e}*r_{tag}/b2_{tag}, 0.9999999))",  # modulus k
        f"K_{tag}=comp_ellint_1(k_{tag})",
        f"E_{tag}=comp_ellint_2(k_{tag})",
        f"sb_{tag}=sqrt(b2_{tag})",
        # Pre-compute the axial and radial field components
        f"Bax_{tag}={MU0:.15e}*{0.5/np.pi:.15e}/sb_{tag}*(K_{tag}+({a2:.15e}-r2_{tag}-z_{tag}*z_{tag})/a2_{tag}*E_{tag})",
        f"Br_{tag}=if(r2_{tag}>1e-24, {MU0:.15e}*{0.5/np.pi:.15e}*z_{tag}/(r_{tag}*sb_{tag})*(-K_{tag}+({a2:.15e}+r2_{tag}+z_{tag}*z_{tag})/a2_{tag}*E_{tag}), 0.0)",
    ]


def _coil_cartesian_term(tag, axis, component, I_val):
    """
    Return the expression fragment for one coil's contribution to a
    Cartesian component (Bx, By, or Bz), given that the coil's local
    variables (Bax_{tag}, Br_{tag}, ct_{tag}, st_{tag}) are already defined.

    The current I is folded in as a numeric prefactor.
    """
    # Determine which cylindrical component maps to which Cartesian component
    # axis='z': Bx = Br*ct, By = Br*st, Bz = Bax
    # axis='x': Bx = Bax,  By = Br*st, Bz = Br*ct
    # axis='y': Bx = Br*st, By = Bax,  Bz = Br*ct
    mapping = {
        'z': {'Bx': f"Br_{tag}*ct_{tag}", 'By': f"Br_{tag}*st_{tag}", 'Bz': f"Bax_{tag}"},
        'x': {'Bx': f"Bax_{tag}",         'By': f"Br_{tag}*st_{tag}", 'Bz': f"Br_{tag}*ct_{tag}"},
        'y': {'Bx': f"Br_{tag}*st_{tag}", 'By': f"Bax_{tag}",         'Bz': f"Br_{tag}*ct_{tag}"},
    }
    base_expr = mapping[axis][component]
    return f"{I_val:.15e}*({base_expr})"


def build_bext_from_loops(loops):
    """
    Build WarpX parser expression strings for Bx(x,y,z), By(x,y,z), Bz(x,y,z)
    from a `list[Loop]` (the canonical coil representation).

    Uses AMReX's native comp_ellint_1(k)/comp_ellint_2(k) and local-variable
    syntax for compact, exact expressions (~3-4 KB each instead of ~28 KB).

    Returns
    -------
    dict with keys 'Bx', 'By', 'Bz' : parser expression strings (of x, y, z).
    """
    all_var_defs = []      # list of "varname=expr" strings
    Bx_terms = []
    By_terms = []
    Bz_terms = []

    for idx, loop in enumerate(loops):
        tag = str(idx + 1)   # "1", "2", ...
        a = loop.radius

        # Variable definitions for this coil (axis-dependent geometry,
        # but current-independent — current is factored in at the end)
        var_defs = _coil_var_defs(tag, loop.axis, loop.position, a)
        all_var_defs.extend(var_defs)

        # Cartesian projection terms (with current prefactor)
        Bx_terms.append(_coil_cartesian_term(tag, loop.axis, 'Bx', loop.current))
        By_terms.append(_coil_cartesian_term(tag, loop.axis, 'By', loop.current))
        Bz_terms.append(_coil_cartesian_term(tag, loop.axis, 'Bz', loop.current))

    # Assemble: "var1=...; var2=...; ...; sum_term1 + sum_term2 + ..."
    preamble = "; ".join(all_var_defs)

    return {
        'Bx': preamble + "; " + "+".join(Bx_terms),
        'By': preamble + "; " + "+".join(By_terms),
        'Bz': preamble + "; " + "+".join(Bz_terms),
    }


def build_bext_expressions(I, dia, offset):
    """
    Polywell convenience wrapper over `build_bext_from_loops` — see there.

    Parameters
    ----------
    I      : coil current (A)
    dia    : coil diameter (m)
    offset : coil center distance from origin (m)
    """
    from warpx_polywell.coils import Polywell
    return build_bext_from_loops(Polywell(I, dia, offset).expand())

def _coil_aext_term(tag, axis, component):
    """
    Return the expression fragment for one coil's contribution to a
    Cartesian component (Ax, Ay, or Az) using phi_hat projection.
    Current I is folded in by the caller as a prefactor.
    """
    # phi_hat projection (cyclic ordering, current-independent geometry):
    # axis='z': Ax=-Aphi*y/r,  Ay=+Aphi*x/r,  Az=0
    # axis='x': Ax=0,          Ay=-Aphi*z/r,  Az=+Aphi*y/r
    # axis='y': Ax=+Aphi*z/r,  Ay=0,          Az=-Aphi*x/r
    # Note, r_tag is already eps adjusted
    mapping = {
        'z': {'Ax': f"-Aphi_{tag}*y/(r_{tag})",  'Ay': f"Aphi_{tag}*x/(r_{tag})",   'Az': "0.0"},
        'x': {'Ax': "0.0",                              'Ay': f"-Aphi_{tag}*z/(r_{tag})",  'Az': f"Aphi_{tag}*y/(r_{tag})"},
        'y': {'Ax': f"Aphi_{tag}*z/(r_{tag})",   'Ay': "0.0",                              'Az': f"-Aphi_{tag}*x/(r_{tag})"},
    }
    return mapping[axis][component]

def build_aext_from_loops(loops, eps=1e-30):
    """
    Returns a dict of coil entries for WarpX's A_external nested dict, built from
    a `list[Loop]`. Each coil has its own self-contained Ax, Ay, Az parser
    expressions with only the geometric variables it needs in the preamble.

    Coils are named coil_1 .. coil_N in list order. The result is accepted by
    HybridPICSolver.
    """
    coils = {}

    for idx, loop in enumerate(loops):
        tag = str(idx + 1)
        axis = loop.axis
        pos = loop.position
        a = loop.radius
        coil_I = loop.current

        B_only_prefixes = ('ct_', 'st_', 'Bax_', 'Br_')
        var_defs = [v for v in _coil_var_defs(tag, axis, pos, a, eps)
                    if not any(v.startswith(p) for p in B_only_prefixes)]
        # A_phi = (mu0 I)/(pi k) * sqrt(a/rho) * [(1 - k^2/2) K(k) - E(k)]
        # (I folded in by the caller). The geometric prefactor sqrt(a/rho) is
        # dimensionless; an earlier version used 1/beta (=1/sb) here, which has
        # units of 1/length and made B=curl(A) wrong by a position-dependent
        # factor (~2x, and ~3x too weak on-axis).
        var_defs.append(
            f"Aphi_{tag}={MU0:.15e}/{np.pi:.15e}"
            f"*sqrt({a:.15e}/(r_{tag}+1e-30))"
            f"*((1.0-0.5*k_{tag}**2)*K_{tag}-E_{tag})"
            f"/sqrt(k_{tag}**2+1e-30)"
        )
        preamble = "; ".join(var_defs)

        ax_term = _coil_aext_term(tag, axis, 'Ax')
        ay_term = _coil_aext_term(tag, axis, 'Ay')
        az_term = _coil_aext_term(tag, axis, 'Az')

        coils[f'coil_{tag}'] = {
            'Ax_external_function': preamble + "; " + (f"{coil_I:.15e}*({ax_term})" if ax_term != "0.0" else "0.0"),
            'Ay_external_function': preamble + "; " + (f"{coil_I:.15e}*({ay_term})" if ay_term != "0.0" else "0.0"),
            'Az_external_function': preamble + "; " + (f"{coil_I:.15e}*({az_term})" if az_term != "0.0" else "0.0"),
            'A_time_external_function': '1.0',
        }

    return coils


def build_aext_expressions(I, dia, offset, eps=1e-30):
    """
    Polywell convenience wrapper over `build_aext_from_loops`.

    Parameters
    ----------
    I      : coil current (A)
    dia    : coil diameter (m)
    offset : coil center distance from origin (m)
    eps    : geometric softening for the r->0 axis (m)
    """
    from warpx_polywell.coils import Polywell
    return build_aext_from_loops(Polywell(I, dia, offset).expand(), eps=eps)


def build_n_turn_aext_expression(I, offset, a, b, n):
    """
    Build expressions of all 6*n coils, where coils are named via coil_ij
        i denotes which of the original six coils
        j denotes which turn the coil is
    """
    rs = np.linspace(a, b, n)
    turns_tag = [i + 1 for i in range(n)]
    discs = {}

    # construct coils of increasing radii from inner most to outer most
    for r, turn_tag in zip(rs, turns_tag):
        coils_at_current_turn = build_aext_expressions(I, r*2, offset)

        # append these to the total set of coils (discs)
        # warpx accepts multiple a_ext as long as they have their own name and
        # axis-specified solutions
        for name, val in coils_at_current_turn.items():
            discs[f"{name}_{turn_tag}"] = val

    return discs