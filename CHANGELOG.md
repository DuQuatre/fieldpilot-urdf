# Changelog

All notable changes to `fieldpilot-urdf` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

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
