# Public API reference

This is the **1.0 stability contract**. Everything listed here is public and
covered by [Semantic Versioning](https://semver.org/): it won't break without a
major-version bump. Anything not listed — in particular any `_`-prefixed name
and the `fieldpilot_urdf._dyn_adapter` module — is internal and may change at
any time.

The contract is also machine-readable: the top-level surface is exactly
`fieldpilot_urdf.__all__`, and each optional submodule defines its own `__all__`
(or, for `loops`/`viz`, exports only its non-underscore names).

---

## Core (`from fieldpilot_urdf import ...`)

Installed by `pip install fieldpilot-urdf`. No optional extra required.

### Layer 1 — Model

**Parse & serialise** — `from_file`, `from_xml`, `to_xml`

**Import from a URL** — `import_urdf`, and the lower-level pieces it composes:
`fetch_urdf`, `expand_xacro`, `resolve_includes`, `substitute_find`,
`fetch_meshes`, `infer_package_root`, `package_uri_parts`

**Model types** (pydantic) — `Robot`, `Link`, `Joint`, `JointLimit`, `Origin`,
`Inertial`, `Inertia`, `Visual`, `Collision`, `Geometry`, `Box`, `Cylinder`,
`Sphere`, `Mesh`, `FrameRef`, `LoopClosure`

**Validate** — `run_all`, `summary`, `RULES`, `Finding`

**Repair** — `repair`, `Patch`, `REPAIRS`, `UNFIXABLE_CODES`

**Graph** — `build_graph`, `chain`, `is_tree`, `joints_on_path`, `leaf_links`,
`root_links`, `subtree`

**Local registry** — `save_robot`, `load_robot`, `delete_robot`, `list_robots`,
`load_meta`, `load_resolver`

### Layer 2 — Kinematics

**Forward kinematics** — `forward_kinematics`, and the transform helpers
`joint_motion`, `origin_to_T`, `rotation_around_axis`, `rpy_to_R`, `R_to_rpy`

**Inverse kinematics** — `solve_ik`, `solve_ik_multi`, `IKResult`
(`solve_ik` takes `n_restarts`/`seed` for random-restart robustness on hard
targets; `solve_ik_multi` returns multiple *distinct* solutions, e.g. elbow-up /
elbow-down)

**Self-collision** — `detect_self_collisions`, `aabb_overlap`,
`link_collision_aabbs`, `transform_aabb`, `MeshResolver`, `clear_mesh_cache`,
`unresolved_meshes`

**Workspace & trajectory** — `sample_workspace`, `WorkspaceResult`,
`check_trajectory`, `StepFinding`, `trajectory_summary`

**Motion planning** — `plan_path`, `PlanResult`, `shorten_path`, `path_length`
(RRT-Connect: collision-free joint-space path between two configs; a returned
`path` feeds straight into `check_trajectory` / `forward_kinematics`)

### Layer 4 — Diagnostics

**Localise** — `rank_root_causes`, `RootCauseCandidate`, `affected_links`,
`criticality`

**Hypothesis-test** — `diagnose`, `Symptom`, `Hypothesis`, `Verdict`,
`DiagnoseReport`

**Fault injection** (used by `diagnose`, also callable directly) —
`inject_motor_fault`, `freeze_joint`, `freeze_joint_at`, `misconfigure_limit`,
`fault_source_tag`

### Misc

`__version__`

---

## Optional extras

Each layer below is lazily imported, so a core install never pays for it.

### Layer 3 — Symbolic dynamics — `fieldpilot_urdf.dynamics` (`[dynamics]`)

`SymbolicDynamics`, `UnsupportedSystemError`

### Layer 3 — Closed-loop dynamics — `fieldpilot_urdf.constrained` / `.loops` (`[dynamics]`)

- `fieldpilot_urdf.constrained`: `ConstrainedDynamics`, `constrained_dynamics`
- `fieldpilot_urdf.loops`: `derive_loop_constraints`, `lambdify_loop_residual`,
  `mobility`

### Layer 3 — Numerical simulation — `fieldpilot_urdf.sim` (`[sim]`)

`PyBulletSim`, `rewrite_mesh_paths`

### Visualisation — `fieldpilot_urdf.viz` (`[viz]` / `[meshviz]`)

`render_kinematic_tree`, `render_pose_3d` (`[viz]`); `render_pose_mesh`
(`[meshviz]`)

> The visualisation renderers are intentionally **not** re-exported from the
> top-level package, so `import fieldpilot_urdf` stays light. Import them from
> `fieldpilot_urdf.viz`.

---

## Not public

- Any name beginning with `_`.
- The `fieldpilot_urdf._dyn_adapter` module (the shim that maps a `Robot` onto
  the dynamics engine) — use `fieldpilot_urdf.dynamics` instead.
- Internal helpers in otherwise-public modules that aren't named above.
