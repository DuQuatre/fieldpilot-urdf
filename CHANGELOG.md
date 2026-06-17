# Changelog

All notable changes to `fieldpilot-urdf` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

_Nothing yet._

## [1.3.0] — 2026-06-17

Rounds out the **diagnostics** symptom set. Adds a third symptom,
`reduced_workspace`, bringing the `(fault_mode, symptom)` registry to seven live
combinations (a near-complete 3×3 — only the unsound `limit_misconfig` ×
`self_collision` pair is intentionally absent). First use of `sample_workspace`
inside `diagnose`. No breaking changes; additive over the 1.2 public API. 225 tests.

### Added
- **Third symptom in `diagnose` — `reduced_workspace`.** Diagnose a shrunken
  reachable envelope: the simulator samples the end-effector's workspace
  (`sample_workspace`) on the healthy robot vs the faulted one and compares the
  bounding-box reach. CONFIRMED when a fault shrinks it by at least the symptom's
  `min_shrinkage` (default 0.1); REFUTED below that. All three fault modes apply —
  freezing a DOF (`motor_dead`/`joint_stuck`) or clipping a limit
  (`limit_misconfig`) all reduce the envelope — bringing the registry to seven
  live `(fault_mode, symptom)` combinations. An inapplicable fault (e.g. a
  limitless joint) is INCONCLUSIVE, not an abort.

## [1.2.0] — 2026-06-17

Continues filling out the **diagnostics** layer (Layer 4). Adds a third fault
mode, `limit_misconfig` — the first *non-lock* fault (the joint still moves, its
travel `<limit>` is just wrong), proving the `(fault_mode, symptom)` registry
extends to faults that aren't a frozen axis. No breaking changes; additive over
the 1.1 public API. 218 tests.

### Added
- **Third fault mode in `diagnose` — `limit_misconfig`.** A mis-set joint travel
  `<limit>` that clips the joint's range without freezing it — the first *non-lock*
  fault (`motor_dead`/`joint_stuck` both just hold an axis fixed). `Hypothesis`
  gains `bad_lower` / `bad_upper` (at least one required, enforced by a validator);
  the simulator narrows the limit on a copy and re-runs IK. Registered for
  `cant_reach` only — a static range change can't alter a fixed commanded pose, so
  it has no sound mapping to `self_collision` (passing that pair returns
  INCONCLUSIVE). A suspect joint with no `<limit>` yields INCONCLUSIVE rather than
  aborting the loop.
- **`misconfigure_limit(robot, joint, *, lower=None, upper=None)`** (in `faults`)
  — overwrite a joint's travel bound(s); the injection primitive behind
  `limit_misconfig`. Raises `KeyError` if the joint is unknown or has no `<limit>`.

## [1.1.0] — 2026-06-17

Deepens the **diagnostics** layer (Layer 4). 1.0 shipped a single vertical slice
— `motor_dead` → `cant_reach`; 1.1 fills out the grid to **two symptoms × two
fault modes**, all four combinations dispatched through one `(fault_mode,
symptom)` registry that is now the sole extension point. No breaking changes; the
1.0 public API is unchanged and additive only. 212 tests.

### Added
- **Second symptom in `diagnose` — `self_collision`.** Beyond `cant_reach`, the
  loop now diagnoses a robot that self-collides at a commanded pose: a `Symptom`
  carries the commanded `at_config` (and optionally the reported `colliding_links`
  pair), and each fault mode is tested by holding the suspect joint at its lock
  value and re-checking `detect_self_collisions`. CONFIRMED when the fault drives
  a non-adjacent pair into contact that was clear on the healthy robot; the
  commanded pose colliding on the healthy robot is INCONCLUSIVE (not attributable).
  Both fault modes (`motor_dead`, `joint_stuck`) work against it, so the
  `(fault_mode, symptom)` registry now covers all four combinations. Tier-0's
  static scan is scoped to `cant_reach` (a static rule can't explain a
  configuration-dependent collision).
- **Second fault mode in `diagnose` — `joint_stuck`.** The diagnosis loop now
  handles a joint *jammed at a reported angle* alongside the existing
  `motor_dead` (dead actuator at the zero pose). `Hypothesis` gains a `stuck_at`
  field (rad / m; defaults to 0, where `joint_stuck` reduces to `motor_dead`),
  and the verdict records it in `evidence`. Both modes share one symbolic
  "locked axis → can't reach" simulator, so the `(fault_mode, symptom)` registry
  stays the single extension point.
- **`freeze_joint_at(robot, joint, angle)`** (in `faults`) — lock a joint at a
  non-zero pose by baking its motion into a fixed-joint origin (vs `freeze_joint`,
  which locks at zero). Underpins `joint_stuck`.
- **`R_to_rpy`** (in `fk`) — inverse of `rpy_to_R`, recovering URDF fixed-axis
  `(roll, pitch, yaw)` from a rotation matrix (round-trips to ~1e-13, with
  gimbal-lock handling).

## [1.0.0] — 2026-06-17

**First stable release.** `fieldpilot-urdf` declares its API stable under
[SemVer](https://semver.org/): the open robotics core of FieldPilot, complete as
a four-layer ladder you climb from a URL to a confirmed fault — and you install
only the weight you use.

1. **Model** (core) — import any ROS robot from an HTTPS URL (`import_urdf`
   expands xacro / `$(find)` / `<xacro:include>` in-process, with SSRF defences),
   parse URDF ⇄ a pydantic model, validate against 8 lint rules (R001–R008), and
   deterministically auto-`repair` the fixable ones.
2. **Kinematics** (core) — forward kinematics, numerical joint-limit-aware
   inverse kinematics, AABB + mesh self-collision, and workspace / trajectory
   sampling.
3. **Dynamics & simulation** (`[dynamics]`, `[sim]`) — `SymbolicDynamics` builds
   Kane's-method equations of motion for tree robots (symbolic `M(q)`, forcing,
   and a NumPy forward-dynamics callable); `LoopClosure` +
   `constrained.constrained_dynamics` extend it to closed loops with Baumgarte
   stabilisation; `sim.PyBulletSim` drives the same model numerically, fed by the
   import pipeline and cross-validated against the symbolic dynamics to ~1e-5.
4. **Diagnostics** (core) — localise a fault on the kinematic graph
   (`rank_root_causes`, `affected_links`, `criticality`) and prove it with the
   two-tier hypothesis-and-test loop (`diagnose`), which injects a fault on a copy
   and checks it reproduces the symptom before returning CONFIRMED / REFUTED /
   INCONCLUSIVE.

The core install stays pure-Python and light (`pydantic` + `numpy` + `scipy` +
`networkx`); mesh, viz, dynamics, and sim are opt-in extras. 197 tests across
Python 3.10–3.13.

This release adds no new physics over `0.9.0` — it's the **stability + story**
milestone:

### Added
- **Tutorial** — `docs/tutorial.md` walks all four layers end to end (import →
  diagnose), with `examples/full_stack_tour.py` as its runnable companion (runs
  on the core install; lights up Layer 3 when `[dynamics,sim]` are present).
- **Public API reference** — `docs/api.md` documents the 1.0 stability contract:
  what's public (the top-level `__all__` + each optional submodule's surface)
  versus internal (`_`-prefixed names, the `_dyn_adapter` shim).

### Changed
- **Development Status** classifier `3 - Alpha` → `5 - Production/Stable`.
- **README** restructured around the four-layer ladder (model → kinematics →
  dynamics+sim → diagnostics) instead of a flat feature list, leading with the
  import → diagnose arc and a stability/public-API section.
- **CI** now smoke-runs the bundled examples (under all extras), not just the
  unit tests.

## [0.9.0] — 2026-06-17

### Added
- **Example: `examples/ducky_diagnosis.py`** — a "Rubber Ducky" robot
  fault-diagnosis transcript: a scripted engineer↔assistant dialogue where every
  conclusion is backed by a real call (`run_all` → `solve_ik` → `rank_root_causes`
  → `diagnose` → `affected_links`/`criticality`). Shows the deterministic MDG
  reasoning core localising and confirming a dead base-yaw motor end to end.
  Core install only.

## [0.8.0] — 2026-06-17

### Added
- **Numerical simulation (`[sim]` extra)** — `sim.PyBulletSim` drives a URDF
  `Robot` in PyBullet: load (DIRECT/GUI), step, position/velocity control, and
  joint/link state readout. Fed by the import pipeline — `sim.rewrite_mesh_paths`
  rewrites `package://` mesh URIs to the absolute paths `fetch_meshes` wrote, so
  a robot imported from a URL drops straight into the simulator. Loads with
  `URDF_USE_INERTIA_FROM_FILE` so its free-fall dynamics match the symbolic
  `SymbolicDynamics` to ~1e-5 (cross-validated in the tests). PyBullet is a
  compiled engine behind the optional extra, imported lazily so the core stays
  pure-Python. Deliberately thin — for richer simulation use PyBullet/MuJoCo/Drake
  directly on the imported URDF.

## [0.7.0] — 2026-06-17

### Added
- **Example: `examples/closed_loop_sim.py`** — end-to-end closed-loop dynamics
  simulation over time. Builds a mobility-1 4R spatial mechanism (tip pinned to
  a point), integrates it under gravity, and shows the constraint residual
  staying at ~1e-13 with Baumgarte + projection vs ~2e-2 unstabilized — while
  the mechanism swings 1.4 rad and the loop stays closed to machine precision.

## [0.6.0] — 2026-06-16

### Added
- **Closed-loop DAE drift stabilization** — `ConstrainedDynamics` now keeps a
  closed-loop simulation on its constraint manifold two ways:
  `lambdify_forward_dynamics(alpha=, beta=)` adds **Baumgarte** feedback
  (`A q̈ + Ȧ q̇ = −2α(A q̇) − β² c`) so drift is damped cheaply each step, and
  `project(q, q̇)` snaps a drifted state back exactly (`q ← q − A⁺ c`,
  `q̇ ← q̇ − A⁺(A q̇)`) — the pseudo-inverse also tolerates redundant constraints
  (e.g. a planar `point` closure). Completes the closed-loop dynamics path.

## [0.5.0] — 2026-06-16

### Added
- **Closed-loop (constrained) dynamics** — `constrained.ConstrainedDynamics`
  (Lagrange multipliers over SymPy's `LagrangesMethod`) plus the high-level
  `constrained.constrained_dynamics(robot)`, which wires a `Robot` with `loops`
  end-to-end: tree Lagrangian + loop-closure constraints → augmented system
  `[M, -Aᵀ; A, 0][q̈; λ] = [F; -Ȧ q̇]` with a `(q, q̇) → (q̈, λ)` forward-dynamics
  callable. Reduces to the unconstrained tree dynamics when there are no loops.
  Index-3 DAE: requires full-rank (non-redundant) constraints at the evaluated
  state; integration-time drift stabilization is a follow-up. Ported from the
  MecAI project (MIT). Completes the closed-loop chain (model → constraints →
  Lagrangian → solver).
- **Tree Lagrangian builder** — `SymbolicDynamics.lagrangian()` returns the
  tree's `L = T − V` as a SymPy expression (kinetic energy of the rigid bodies +
  gravitational potential), cross-validated to match the Kane-based forward
  dynamics. This is the input the Lagrange-multiplier solver needs to consume the
  loop-closure constraints from `loops` (see the constrained-dynamics entry).
- **Closed-loop modelling + constraint deriver** — `LoopClosure` / `FrameRef`
  model a closed kinematic loop as a spanning-tree URDF plus frame-coincidence
  constraints (`point` removes 3 DOF, `fixed` removes 6). `loops.derive_loop_constraints`
  turns them into symbolic `c(q) = 0` in the tree coordinates (reusing
  `SymbolicDynamics`' frames — no duplicate FK), with `loops.lambdify_loop_residual`
  (assembly check) and `loops.mobility` (closed-loop DOF). `loops` defaults to
  `[]`, isn't parsed from standard URDF, and leaves the tree FK / `is_tree` /
  XML round-trip untouched. Needs the `[dynamics]` extra; consumed by the
  constrained-dynamics solver (see entry above).

## [0.4.0] — 2026-06-16

### Added
- **Graph fault propagation & root-cause ranking** — `fault_propagation`
  module: `affected_links(robot, faulty_id)` (downstream links of a faulty
  joint/link), `criticality(robot, faulty_id)` (mass-weighted impact), and
  `rank_root_causes(robot, observed_links)` (ranks suspect joints by
  precision × recall over their downstream set, with a specificity tie-breaker).
  Pure NetworkX, in the core install — no new dependency. Pairs with
  `diagnose`: ranked suspects can be fed in as hypotheses. Ported from the
  MecAI project (MIT) and re-targeted from sensors onto links.

## [0.3.0] — 2026-06-16

### Added
- **Symbolic dynamics** — `dynamics.SymbolicDynamics(robot)` builds Kane's-method
  equations of motion for tree (serial) robots: symbolic `mass_matrix` `M(q)`,
  `forcing` `F(q, q̇, τ)`, and `lambdify_forward_dynamics()` → a NumPy
  `(q, u, tau) → q̈` callable for `scipy.integrate.solve_ivp`. `link_pose()`
  resolves a link's world transform for cross-checking against
  `forward_kinematics`. Ported from the MecAI project (MIT) and re-targeted onto
  the URDF `Robot` model via a small validating adapter.
- **`[dynamics]` optional extra** (`sympy`), lazy-imported so the core
  kinematics path never pulls in SymPy.

### Fixed
- `__version__` in `fieldpilot_urdf/__init__.py` was stuck at `0.1.0` while the
  packaged version had advanced to `0.2.x`; it now tracks the real version.

### Notes
- Dynamics v1 is **tree-only**. Joint-origin frames use URDF's
  `Rz(yaw)·Ry(pitch)·Rx(roll)` (space-fixed) convention, so `link_pose` matches
  `forward_kinematics` to machine precision. Closed-loop mechanisms, a non-zero
  `<inertial>` origin `rpy`, and multi-DOF joints (`floating`/`planar`/
  `spherical`) raise `UnsupportedSystemError`.

## [0.2.1] — 2026-06-14

### Fixed
- `render_pose_mesh` no longer reports a missing `[meshviz]` extra when the
  extra is installed but the selected GL backend's system library
  (`libOSMesa`/`libEGL`) can't load. It now checks the packages first and, on
  a backend load failure, raises a clear error naming the backend and the
  library to install.

## [0.2.0] — 2026-06-14

### Added
- **Mesh-accurate pose render** — `viz.render_pose_mesh(robot, q?, mesh_dir?)`
  renders the robot's actual visual meshes offscreen (urchin → pyrender) to PNG
  bytes, complementing the existing mesh-free `render_pose_3d` stick figure.
  Resolves link meshes against the on-disk layout written by
  `importer.fetch_meshes`; robots with only primitive geometry render with no
  `mesh_dir`.
- **`[meshviz]` optional extra** (`urchin`, `pyrender`) for the mesh renderer,
  kept out of `[viz]` so the light tree/pose renderers don't pull in the GL
  stack. Needs a headless GL backend at runtime (EGL by default; set
  `FIELDPILOT_URDF_RENDER_BACKEND=osmesa` for pure software).

## [0.1.0] — 2026-06-13

First public release: the open robotics core of FieldPilot, extracted into a
standalone, pure-Python, pip-installable package (AGPL-3.0).

### Added
- **Import** any ROS robot from an HTTPS URL — `import_urdf` expands `xacro`,
  `$(find)`, and `<xacro:include>` in-process (no ROS, no build).
- **Parse** URDF ⇄ Pydantic model — `from_xml`, `from_file`, `to_xml`.
- **Kinematics** — forward kinematics (`forward_kinematics`) and numerical,
  joint-limit-aware inverse kinematics (`solve_ik`).
- **Self-collision** — AABB + optional mesh-aware detection
  (`detect_self_collisions`; meshes via the `[mesh]` extra).
- **Workspace / trajectory** sampling — `sample_workspace`, `check_trajectory`.
- **Validation** — 8 symbolic lint rules R001–R008 (`run_all`, `summary`).
- **Auto-repair** — deterministic fixes for the repairable rules (`repair`).
- **Symbolic fault diagnosis** — the pure two-tier hypothesis-and-test loop
  (`diagnose`, `Symptom`, `Hypothesis`, `Verdict`).
- **Visualisation** — kinematic-tree and 3D-pose renderers
  (`fieldpilot_urdf.viz`, via the `[viz]` extra).
- **Local registry** — file-based robot storage (`save_robot`, `load_robot`, …).
- Security: `import_urdf` ships SSRF defences (HTTPS-only, host allowlist,
  5 MB cap, timeout, redirect re-validation) — see `SECURITY.md`.
- CI across Python 3.10–3.13 (plus an older-`xacro` guard); 144 tests.

### Notes
- Configuration env vars use the `FIELDPILOT_URDF_*` namespace; the legacy
  `MECHDIAG_*` names are still read as a deprecated fallback.
- The LLM robot chat, the natural-language fault-diagnosis front-end, the
  spare-parts BOM, and multi-tenant hosting are **not** part of this package —
  they live in FieldPilot SaaS.

[1.3.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.3.0
[1.2.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.2.0
[1.1.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.1.0
[1.0.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.0.0
[0.1.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v0.1.0
