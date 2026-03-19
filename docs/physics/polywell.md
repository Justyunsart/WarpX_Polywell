# Physics Background: Polywell Fusion

## The Polywell Concept

A polywell is an inertial electrostatic confinement (IEC) fusion device that combines
two field sources to trap and accelerate plasma:

1. **Magnetic cusps** from a polyhedral arrangement of current-carrying coils — create
   a magnetic "well" that confines electrons near the center
2. **Electrostatic potential well** — the confined electron cloud builds a negative
   space charge that electrostatically accelerates positive ions inward

The goal is to achieve sufficient ion density and energy at the center to sustain
fusion reactions without the large plasma instabilities that afflict tokamaks.

---

## Magnetic Configuration

### Polywell Coil Geometry

Six circular coils are placed on the faces of a cube (at ±x, ±y, ±z from the origin).
Adjacent coils carry current in alternating directions so that the magnetic field
forms a closed structure with cusps along the cube edges and corners.

```
         +y
          |
    s4 (↑ current)
          |
  s1 ---- O ---- s2     (-x, 0, 0) and (+x, 0, 0)
(↑ current)  (↓ current)
          |
    s3 (↓ current)
          |
         -y

(s5 and s6 along z-axis, not shown)
```

The alternating polarity creates a magnetic mirror geometry where field lines
near the center are weaker than those toward the coils. Electrons on magnetic
field lines cannot escape easily — they are reflected back by the stronger field
at the cusps.

### Parameters in Code

```python
I        = 1e6   # A  — coil current (drives field strength)
b_dia    = 1.0   # m  — coil diameter
b_offset = 1.1   # m  — coil distance from origin
```

The field is computed via magpylib's Biot-Savart implementation for circular loops,
then sampled onto the 3D simulation grid.

---

## Electrostatic Field

### Charged Ring Approximation

The electrostatic potential well is modelled by superimposing the fields of six
uniformly charged rings (same geometry as the magnetic coils). This approximates
the space-charge potential that would form as electrons accumulate near the center.

```python
Q        = 1e-9  # C  — total charge per ring
e_dia    = 0.75  # m  — ring diameter
e_offset = 1.1   # m  — ring distance from origin
```

### Field Equations

For a ring of radius `a` carrying total charge `Q` at the origin in the xy-plane,
the field at cylindrical point `(r, z)` is:

```
E_r = (λa / 4πε₀) · ∫₀²π  (r - a·cosθ) / D³  dθ
E_z = (λa · z / 4πε₀) · ∫₀²π  1 / D³  dθ

λ = Q / (2πa)      linear charge density
D = sqrt(r² + a² - 2ar·cosθ + z²)
```

These integrals have no closed-form solution and are evaluated numerically.
Two implementations are provided (`fw_e`, `bob_e`) — see [eext module](../modules/eext.md).

---

## Particle-in-Cell Simulation

### Why PIC?

PIC methods self-consistently evolve both the particle trajectories and the
electromagnetic fields. This captures:
- Collective plasma effects (waves, instabilities)
- Particle–field feedback (the plasma's own fields modifying particle motion)
- Non-Maxwellian velocity distributions that arise in IEC devices

### Simulation Setup in WarpX

| Property | Value |
|---|---|
| Grid type | 3D Cartesian, `[-L, L]³` metres |
| Boundary conditions (fields) | Open |
| Boundary conditions (particles) | Absorbing |
| Solver | Electromagnetic (Yee scheme, CFL=0.99) |
| Species | Electrons (`plasma_e`) + Protons (`plasma_i`) |
| Initial distribution | Uniform within `±(0.11L)` per axis |
| Initial velocity | 0.9c in z-direction |

### External Fields

WarpX reads the pre-computed B and E grids at initialisation via:

```python
warpx.B_ext_field_init_style = "read_from_file"
warpx.E_ext_field_init_style = "read_from_file"
warpx.read_fields_from_path  = ext_path
```

These fields are fixed throughout the simulation (they do not self-consistently
respond to the plasma). The self-consistent field evolution from the simulation is
computed separately and superimposed.

### Divergence Cleaning

`warpx.do_initial_div_cleaning = 0` is set because:
- The magpylib solution is analytically divergence-free (∇·B = 0 exactly)
- Divergence cleaning with open boundary conditions causes numerical errors at startup

---

## Key Physical Scales

| Quantity | Typical Value | Notes |
|---|---|---|
| Domain size | 6 m × 6 m × 6 m | `2L` per side |
| Coil radius | 0.5 m | `b_dia / 2` |
| Coil offset | 1.1 m | Distance from origin |
| Coil current | 1 MA | Drives B-field strength |
| Plasma density | 10¹⁸ m⁻³ | |
| Plasma region | ~0.66 m × 0.66 m × 0.66 m | `2 × 0.11 × L` per side |
| Time step | 1 ps | `const_dt = 1e-12 s` |
| Simulation time | 10 ns | `max_steps × const_dt` |

---

## References

- Bussard, R.W. (1991). "Some Physics Considerations of Magnetic Inertial-Electrostatic Confinement: A New Concept for Spherical Converging-Flow Fusion." *Fusion Technology*, 19(2).
- WarpX documentation: https://ecp-warpx.github.io/
- magpylib documentation: https://magpylib.readthedocs.io/
- openPMD standard: https://github.com/openPMD/openPMD-standard
