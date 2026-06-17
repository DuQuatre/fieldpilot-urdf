# Changelog

All notable changes to `fieldpilot-urdf` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[0.1.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v0.1.0
