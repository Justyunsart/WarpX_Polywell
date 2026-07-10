# Design note — coil construct architecture (`Loop` primitive + composites)

**Status:** approved for implementation (full unification) — code not yet started
**Author:** drafted with Claude Code, 2026-07-09
**Motivation:** a growing need for parameterized coil constructs (starting with a
**washer**) that must feed *both* the magpylib pipeline and the analytic
vector-potential pipeline.

### Decisions locked in

- **Scope:** full unification — `Loop` primitive as single source of truth,
  refactor `make_collection.py` and `analytic.py`'s `POLYWELL_COILS` onto it,
  then add `Washer`/`Polywell` composites.
- **Units: strict SI everywhere.** Lengths in **metres**, current in **amperes**.
  `Loop` stores metres; adapters must not reintroduce cm. This resolves the
  `make_collection.py` (docstrings claim cm) vs `analytic.py` (already m) split —
  see §6 and the unit trap in §5.
- **Workflow:** all work happens on branch `feature/coil-primitives` (forked from
  `dev` at `21d5c1c`). A pull request into `dev` is opened **only after** the
  verification gates in §5 pass. `dev` is not modified in the meantime.

---

## 1. Problem

We want a `washer` construct: same arguments as a coil loop, plus an inner
radius, outer radius, and a resolution (number of discretization loops). A
washer is physically a thick annular winding, discretized as many coaxial
circular loops with radii spanning `[r_inner, r_outer]`.

The obvious implementation — a `magpylib.Collection` of `Circle`s — is **not
sufficient**, because it is incompatible with one of the two "vector potential"
code paths. Understanding *why* is the whole basis for the design.

### Two things are called "vector potential" in this repo

| Path | Entry point | How it uses coils | Washer-compatible as a `Collection`? |
|---|---|---|---|
| **FFT curl-inverse** | `bext/vector_potential.py::compute_A_grid` | calls `collection.getB(mesh)`, then FFTs B → A on a grid | **Yes** — only needs `getB()` |
| **Analytic A_ext parser** | `bext/analytic.py::build_aext_expressions` → `build_n_turn_aext_expression` | emits one `A_phi` AMReX parser string **per loop**, from that loop's `(axis, position, radius, signed current)` | **No** |

The analytic-A path (used by the Hybrid-PIC solver, which consumes **A**, not B)
needs each loop's geometry as *live numbers* to synthesize its `A_phi(x,y,z)`
expression. A `magpylib.Collection` is a black box: once a loop is built as
`Circle(...).move(...).rotate_from_angax(...)`, magpylib's only supported
contract is `getB()`/`getA()` on a mesh — there is no clean way to recover
"this loop is on the +y axis at offset d with radius r." **That** is the
incompatibility the washer runs into.

## 2. Root cause — dual source of truth

Coil geometry is currently defined **twice**, in two representations that must
be hand-synchronized:

- **magpylib form** — `bext/make_collection.py:22-27`, the polywell as six
  `Circle().move().rotate_from_angax()` calls.
- **tuple form** — `bext/analytic.py:31-38`, the same polywell as
  `POLYWELL_COILS = [(axis, pos_sign, I_sign), …]`.

The discretized-washer sweep is *already* written a third and fourth time:

- `bext/make_collection.py::return_n_turn_coil` (radii `linspace(r1, r2, n)`,
  magpylib form)
- `bext/analytic.py::build_n_turn_aext_expression` (radii `linspace(a, b, n)`,
  tuple form)

Adding `Washer` naively means a fifth and sixth copy. The two forms have already
drifted: `make_collection.py` docstrings say **cm**, `analytic.py` works in
**m**. Dual source of truth is the underlying defect; the washer just exposes it.

## 3. Proposed architecture — a coil primitive as intermediate representation

Introduce **one canonical, magpylib-independent primitive** in
`warpx_polywell/coils/` and make every existing consumer an *adapter* off it.

```
                  ┌─────────────────────────────────────┐
   composite      │   Loop(axis, position, radius,       │   ← single source of truth
   builders  ───► │        current)   [SI, dataclass]    │      (plain data, no magpylib)
 (Polywell,       └─────────────────────────────────────┘
  Washer,                     │            │            │
  n-turn)                     ▼            ▼            ▼
                       to_collection()  build_bext_   build_aext_
                       → magpylib       expressions   expressions
                         Collection     (analytic B)  (analytic A_phi)
```

- **`Loop`** — a frozen dataclass carrying exactly `(axis, position, radius,
  current)` in SI. The *only* description of a single coil. Knows nothing about
  magpylib or WarpX.
- **A composite (`Washer`, `Polywell`) `.expand()`s to a `list[Loop]`.** A
  washer is *n* coaxial loops sharing one axis/position/current with radii
  `linspace(r_inner, r_outer, resolution)` — literally the existing sweep,
  written once.
- **Three adapters consume `list[Loop]`:**
  - `to_collection(loops) -> magpylib.Collection` — replaces `make_collection.py`;
    feeds FFT-A and direct-`getB` file mode.
  - `build_bext_expressions(loops)` — analytic per-particle B.
  - `build_aext_expressions(loops)` — analytic Hybrid-PIC A_ext (the path that
    needs per-loop params). `analytic.py:264` and `:317` already iterate over
    `POLYWELL_COILS` doing exactly this — they would take `loops` instead of the
    hardcoded tuple. Near drop-in.

### Sketch

```python
# coils/primitives.py
@dataclass(frozen=True)
class Loop:
    axis: str          # 'x' | 'y' | 'z'
    position: float    # signed center along axis (m)
    radius: float      # m
    current: float     # signed A

# coils/washer.py
@dataclass(frozen=True)
class Washer:
    axis: str
    position: float
    current: float
    r_inner: float
    r_outer: float
    resolution: int
    def expand(self) -> list[Loop]:
        return [Loop(self.axis, self.position, r, self.current)
                for r in np.linspace(self.r_inner, self.r_outer, self.resolution)]
```

`Polywell(I, dia, offset)` returns six `Loop`s (from the current `POLYWELL_COILS`
layout). A washer-polywell returns six `Washer`s and flattens their loops. Same
`list[Loop]`, same three adapters — no per-construct A/B/magpylib code.

## 4. Justification (supported arguments)

1. **Directly dissolves the blocker.** Making `Loop` the source of truth gives
   the analytic-A path structured per-loop data *and* the magpylib path its
   `Collection`, from one object. Neither pipeline is privileged.
2. **Kills existing duplication.** Polywell layout: 2 definitions → 1. The
   washer/n-turn sweep: 2 → 1. `Washer` and `Polywell` *compose* instead of
   needing an `_n_turn` twin of every function.
3. **Minimal thing that generalizes.** A washer and an "n-turn coil" are the
   same object (coaxial loops over a radius range) — the codebase already needed
   it twice. The next construct (helical, racetrack, thick solenoid) is another
   `.expand() -> list[Loop]` with **zero** new adapter code.
4. **Respects magpylib's actual contract.** magpylib is a *field evaluator*, not
   a geometry database. Keeping geometry in our own primitive and treating
   magpylib as one downstream consumer is the idiomatic separation and avoids the
   black-box problem.
5. **Low blast radius.** Adapters wrap code that already exists. `setup_bext`
   keeps its signature; only its internals swap
   `make_polywell_collection(I, dia, offset)` for
   `to_collection(Polywell(I, dia, offset).expand())`, and the analytic builders
   take `loops` instead of `POLYWELL_COILS`.

## 5. Migration & verification

Principle: **pin current behavior first, refactor underneath the pin, add the
washer last.** "Works as intended" becomes a mechanical `assert new == old`, not
a judgment call.

### Two verification levers this codebase provides

- **Analytic builders emit strings.** `build_bext_expressions`,
  `build_aext_expressions`, `build_n_turn_aext_expression` return dicts of AMReX
  parser strings → **exact, tolerance-free** regression via string equality. This
  covers the highest-risk path (analytic-A / Hybrid-PIC) provably.
- **Three independent B computations** (magpylib `getB`, analytic elliptic
  integrals, `curl` of analytic A) → a **differential oracle**: pin one test that
  evaluates all three on a fixed grid and asserts agreement to physical
  tolerance. Independent paths don't stay mutually consistent by accident.

### Phased gates (each must pass before the next)

| Phase | Action | Gate |
|---|---|---|
| **0. Freeze** | Snapshot golden parser strings + a golden magpylib B array (`.npy`) for a fixed param set; distill notebook asserts into a runnable test. **Run the differential oracle on current code to expose any pre-existing cm/m disagreement (see trap below).** | Golden fixtures committed; baseline suite green against *old* code. |
| **1. Primitive** | Add `Loop` + `Polywell(...).expand()`; change nothing else. | `Polywell(...).expand()` reproduces `POLYWELL_COILS` (derive + assert equal). |
| **2. Adapters** | Write `to_collection`; rewrite analytic builders to take `list[Loop]`. | String equality vs golden strings; `np.allclose` vs golden magpylib array. |
| **3. Rewire** | Point `setup_bext` + notebooks at new builders; delete old `make_collection` funcs and `POLYWELL_COILS`. | Full suite green; **cache filenames unchanged**; differential oracle green. |
| **4. Feature** | Add `Washer` + washer-polywell composite. | `Washer(...).expand()` → analytic adapter reproduces `build_n_turn_aext_expression` exactly; differential oracle green on a washer config. |

The washer is **last**: reach single-source-of-truth on the known-good polywell
(where golden values exist) before building the new construct on the now-trusted
foundation.

### Repo-specific traps

1. **cm/m unit landmine (now being fixed to SI).** Enforcing metres is the
   resolution of the `make_collection.py`-cm vs `analytic.py`-m split. **In Phase
   0, run the differential oracle on the *current* code first**: if magpylib-B and
   analytic-B disagree today, the magpylib path was being fed mislabeled units —
   a pre-existing bug. Fix it consciously and re-baseline the magpylib golden;
   don't let it hide inside the refactor.
2. **Cache keys are behavior.** `OUTPUT_DIR/bext/*.h5` filenames encode every
   parameter and are reused. Assert the refactored path yields the *same* cache
   filename for the same physical config, or every cached field file and
   `runs.db` correlation is silently invalidated.
3. **Loop order, signs, orientation.** `to_collection` must reproduce loop order,
   signed current, and orientation exactly — the golden magpylib array is what
   catches a regression here.

### Commit hygiene

Keep the pure refactor (Phases 1–3, outputs unchanged) in commits separate from
the feature (Phase 4). A test failure in the refactor range is then unambiguously
a refactor bug and `git bisect` lands on the exact commit.

### Branch / PR flow

Forked `feature/coil-primitives` from `dev` @ `21d5c1c`. Open the PR into `dev`
**only after all Phase gates pass**. `dev` is untouched until then.

## 6. Open questions for implementation

- **Units. — RESOLVED: strict SI (metres, amperes).** `Loop` stores metres;
  adapters must not reintroduce cm. magpylib v5 is SI-native, so `to_collection`
  passes metres straight through. See the cm/m trap in §5.
- **Orientation model.** `Loop` currently encodes orientation as an axis label
  (`'x'|'y'|'z'`), matching `analytic.py`. Do we need arbitrary orientations
  (tilted coils)? If so, `axis` should generalize to a unit normal vector, and
  the analytic `A_phi` projection (`_coil_aext_term`) would need the general
  rotation rather than the three hardcoded cyclic cases.
- **Where composites live.** `warpx_polywell/coils/` (currently a placeholder
  package whose docstring already names "washers").
- **Should `Polywell`/`Washer` be dataclasses or plain factory functions?**
  Dataclasses give run-registry-friendly `repr`/serialization for `runs.db`.