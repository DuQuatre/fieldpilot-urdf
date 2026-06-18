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

**Import from disk** — `import_urdf_file` (the network-free twin of
`import_urdf`: same `$(find)` → includes → xacro → parse pipeline against the
local filesystem, with `package://` resolved via a `package_roots` map), plus
`resolve_includes_local`, `expand_xacro_local`

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

**Velocity kinematics** — `geometric_jacobian`, `jacobian_joints`,
`joint_velocity_to_twist`, `manipulability`, `singularity_report`,
`SingularityReport` (the 6×n geometric Jacobian mapping `qdot` → world twist
`[v; w]`; from it: forward velocity, the Yoshikawa manipulability measure, and a
singular-value singularity report. `manipulability` / `singularity_report` take
`rows=` to restrict to a task subspace, e.g. `rows=(0,1,2)` for a position-only
measure on a sub-6-DoF arm)

**Self-collision** — `detect_self_collisions`, `aabb_overlap`,
`link_collision_aabbs`, `transform_aabb`, `MeshResolver`, `clear_mesh_cache`,
`unresolved_meshes`

**Environment collision** — `Obstacle`, `box_obstacle`, `sphere_obstacle`,
`detect_obstacle_collisions` (static world-frame AABB obstacles; the
world-collision counterpart of `detect_self_collisions`. `plan_path`,
`shorten_path`, and `check_trajectory` take an `obstacles=` list to plan /
validate against the world, not just the robot itself)

**Mesh bounds** — `read_mesh_bounds`, `SUPPORTED_FORMATS` (pure-Python vertex
AABB for `.stl` / `.obj` / `.ply` — no `[mesh]` extra; mesh self-collision uses
this and falls back to `trimesh` only for other formats like `.dae`)

**Workspace & trajectory** — `sample_workspace`, `WorkspaceResult`,
`check_trajectory`, `StepFinding`, `trajectory_summary`

**Motion planning** — `plan_path`, `PlanResult`, `shorten_path`, `path_length`
(RRT-Connect: collision-free joint-space path between two configs; a returned
`path` feeds straight into `check_trajectory` / `forward_kinematics`)

**Cartesian planning** — `plan_cartesian_path`, `CartesianPlanResult`,
`interpolate_pose`, `pose_error` (task-space complement to `plan_path`: a
joint-space path whose link follows a straight line in SE(3), via a resolved-rate
servo over the geometric Jacobian with singular-direction-only damping; `path`
feeds straight into `check_trajectory` / `forward_kinematics`)

**Time-parameterization** — `time_parameterize`, `TimedTrajectory` (assign a
geometric path a schedule under joint velocity limits — a trapezoidal velocity
profile over the path's arc length, with an optional `max_acceleration` for
smooth ramps; returns positions + velocities sampled over time, `as_dicts()`
feeds `check_trajectory` and the result mirrors `simulate.Trajectory`)

### Layer 4 — Diagnostics

**Localise** — `rank_root_causes`, `RootCauseCandidate`, `affected_links`,
`criticality` (structural: downstream-link overlap)

**Localise (kinematic)** — `localize_joint_fault`, `JointFaultCandidate` (given a
link's observed vs. commanded pose, rank the chain's joints by how well a single
calibration offset on each explains the deviation, via the geometric Jacobian)

**Calibrate (kinematic)** — `calibrate_joint_offsets`, `PoseObservation`,
`CalibrationResult` (multi-pose generalization: from a set of commanded/observed
pose measurements, estimate the per-joint calibration offsets that explain them
all, via Gauss-Newton — resolves the ambiguity one pose can't and handles large
offsets)

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

### Layer 3 — Dynamics simulation — `fieldpilot_urdf.simulate` (`[dynamics]`)

`integrate_dynamics`, `Trajectory`, `inverse_dynamics`, `gravity_torques`
(roll the symbolic forward dynamics forward in time — RK4 / semi-implicit Euler,
pure NumPy — plus inverse dynamics and gravity-compensation torques)

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
