# Changelog

All notable changes to `fieldpilot-urdf` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **End-to-end diagnostics example — `examples/diagnostics_workflow.py`.** A
  runnable, offline, deterministic walkthrough of the whole Layer-4 loop on a
  robot with one miscalibrated joint: **symptom** (tool measured off the
  commanded pose) → **localise** (`localize_joint_fault`) → **prior** from past
  cases (`fault_priors`) → **dialog** (`rank_questions` / `update_beliefs` narrow
  to one fault) → **calibrate** (`calibrate_joint_offsets` recovers the exact
  offset) → **recommend** (`recommend_solution`) → **record** (`save_case`) →
  **visualise** (3D fault-motion GIF + oscilloscope PNG, when `[viz]` is
  installed). Every number comes from a real call. Covered by a smoke test
  (`tests/test_example_diagnostics.py`) so it stays runnable. Ties together the
  1.14–1.20 diagnostics features into one reference workflow.

## [1.20.0] — 2026-06-18

Final slice of the interactive diagnostics direction: **oscilloscope parameter
visuals**. `fieldpilot_urdf.viz` gains `render_scope` and
`render_trajectory_scope` (the `[viz]` extra) — stacked time-series panels of
joint parameters with the expected (simulated) signal overlaid against the
observed one and the divergence shaded. Together with the 1.19 3D fault-motion
video this completes the visual diagnostics story (3D motion + quantitative
traces). No breaking changes; additive over the 1.19 public API. 407 tests.

### Added
- **Oscilloscope parameter visuals — `render_scope` / `render_trajectory_scope`**
  (`fieldpilot_urdf.viz`, the `[viz]` extra). Stacked time-series panels — a
  multi-channel "scope" — for joint parameters (position / velocity / effort) over
  time, with an **expected (simulated)** signal overlaid against the **observed**
  one so the tech sees *where* and *how much* the real robot diverges from the
  model. `render_scope(panels)` is the general plotter over `ScopePanel` /
  `ScopeSeries` (each panel a stacked subplot; with two same-grid series and
  `shade_divergence`, the gap is shaded and the max `|Δ|` annotated);
  `render_trajectory_scope(nominal, observed=…)` is the convenience for any
  `TimedTrajectory` / `simulate.Trajectory` (anything with `joint_ids` / `times`
  / `q` / `u`), building one panel per (joint, signal). PNG or SVG bytes. Pairs
  with the 1.19 3D fault-motion video — together they complete the diagnostics
  visual story (3D motion + quantitative traces).

## [1.19.0] — 2026-06-18

Third slice of the interactive diagnostics direction: **3D fault-motion
visuals**. `fieldpilot_urdf.viz` gains `render_motion` and
`render_motion_comparison` (the `[viz]` extra) — animate a robot's motion as a
3D stick-figure video and play a simulated faulted motion against the nominal
one, so the technician can compare reality to the model. No breaking changes;
additive over the 1.18 public API. 400 tests.

### Added
- **3D fault-motion visuals — `render_motion` / `render_motion_comparison`**
  (`fieldpilot_urdf.viz`, the `[viz]` extra). Animate a robot through a trajectory
  (a list of `{joint: value}` configs — `plan_path` output, `TimedTrajectory.as_dicts()`,
  a sim trajectory) as a 3D stick-figure video, so the tech can compare reality to
  the simulation. `render_motion` plays one motion (with an optional `track_link`
  end-effector trail); `render_motion_comparison` plays a **nominal** motion
  against a **faulted** one — `layout="overlay"` draws both in one axes (nominal
  solid blue, faulted dashed red) with a dotted line marking the end-effector
  divergence and its magnitude in the title, or `layout="sidebyside"` for two
  panels. The camera box is fixed across all frames so the robot doesn't drift,
  and the two trajectories are paired frame-by-frame. `fmt="gif"` returns
  animated-GIF bytes (assembled via Pillow, a matplotlib dependency);
  `fmt="frames"` returns the per-frame PNG bytes. Mesh-free stick figures, like
  `render_pose_3d`. First of the diagnostics visual pair (oscilloscope traces next).

## [1.18.0] — 2026-06-18

Second slice of the interactive diagnostics direction: a **case knowledge base**.
`fieldpilot_urdf.case_base` persists each resolved diagnosis and mines the
history into fault-frequency priors and per-fix success statistics — the priors
feed the 1.17 differential-diagnosis engine, so the dialog starts each new case
from what the fleet has actually shown and sharpens with use. Pure Python. No
breaking changes; additive over the 1.17 public API. 392 tests.

### Added
- **Diagnostic case knowledge base — `fieldpilot_urdf.case_base`.** Remember
  every resolved diagnosis (symptom, questions/answers, the confirmed fault, the
  fix that worked) and turn the history into statistics that sharpen future
  diagnoses. Two halves:
  - **Persistence** — `save_case` / `load_case` / `load_cases` / `list_cases` /
    `delete_case` store one `DiagnosticCase` JSON per case under a directory
    (the `FIELDPILOT_URDF_CASE_DIR` env var, default `/data/diagnostic-cases`, or
    an explicit `root=`).
  - **Aggregation** (pure functions over a `list[DiagnosticCase]`) —
    `fault_priors` returns the empirical fault frequencies as a normalized,
    Laplace-smoothed prior (optionally per `machine`, or forced over a given
    candidate set); `solution_stats` ranks each `(fault, solution)` pair by its
    success record (`attempts` / `successes` / `success_rate`); and
    `recommend_solution` picks the best-proven fix for a fault.
  - **The payoff:** `fault_priors` drops straight into
    `differential_diagnosis.candidates_from_scores`, so the dialog engine starts
    each new case from what the fleet has actually shown — it gets sharper the
    more it's used. Pure Python, core install.

## [1.17.0] — 2026-06-18

Opens an **interactive** diagnostics direction. When several faults could explain
a symptom, the new `fieldpilot_urdf.differential_diagnosis` engine narrows them
by asking the technician the most discriminating question (highest information
gain) and Bayesian-updating the candidate posterior as answers arrive — the
"twenty questions" loop, with the maths kept stateless so an n8n / LLM front-end
owns the dialog. Pure Python. First of a planned series (case statistics, then
3D / oscilloscope visuals). No breaking changes; additive over the 1.16 public
API. 381 tests.

### Added
- **Differential-diagnosis dialog engine — `fieldpilot_urdf.differential_diagnosis`.**
  When several faults could explain a symptom, narrow them by asking the
  technician the *most discriminating* question. Given competing `Candidate`
  hypotheses (with priors — seed them from `rank_root_causes` /
  `localize_joint_fault` via `candidates_from_scores`) and a set of `Question`s
  (each with the predicted outcome likelihood per candidate), `rank_questions`
  scores every unanswered question by **expected information gain** (mutual
  information, bits) — a perfectly discriminating yes/no on a 50/50 pair scores
  ~1 bit, an uninformative one ~0 — and `update_beliefs` Bayesian-updates the
  posterior over candidates as answers arrive (an impossible outcome drives a
  candidate to zero). `next_question` returns the single best one; `BeliefState`
  reports the posterior, entropy, leading candidate, and whether it's `resolved`
  (leading probability past a threshold). The engine is **stateless** — every
  call is a pure function of `(candidates, questions, answers-so-far)` — so an
  n8n / LLM front-end owns the dialog and persistence while this owns the maths.
  Naive-Bayes independence across questions. Pure Python, core install.

## [1.16.0] — 2026-06-18

Makes inverse kinematics **collision-aware**. `solve_ik` ignores collision
entirely — it can return a pose-reaching configuration that self-collides or
sits inside an obstacle. The new `solve_ik_collision_free` enumerates the
distinct IK branches and returns the one that's actually usable: self-collision
free and clear of the 1.13 world obstacles — a valid `plan_path` endpoint.
Composes IK, self-collision, and obstacles; pure reuse. No breaking changes;
additive over the 1.15 public API. 367 tests.

### Added
- **Collision-free inverse kinematics — `solve_ik_collision_free`.** Plain
  `solve_ik` returns a pose-reaching configuration with no regard for collision —
  it may self-collide or sit inside an obstacle, making it useless as, e.g., a
  planning endpoint. The new solver enumerates the *distinct* IK branches (via
  `solve_ik_multi` — elbow-up / elbow-down and other postures reaching the same
  pose) and returns the best one that neither self-collides nor hits any
  `obstacles` (the 1.13 world-obstacle type). Different postures collide
  differently, so this picks the one that's clear — and its output drops straight
  into `plan_path` as a valid start/goal. Returns the existing `IKResult`:
  `converged=True` means a collision-free pose-reaching solution was found; if the
  pose is reachable only in collision, the best colliding configuration is
  returned with `converged=False` and a message saying so. Composes IK (1.6) ×
  self-collision × obstacles (1.13); pure reuse, additive.

## [1.15.0] — 2026-06-18

Generalizes 1.14's single-pose localizer to **multi-pose calibration**.
`calibrate_joint_offsets` (in `fieldpilot_urdf.kinematic_diagnosis`) takes a set
of commanded/observed link poses and jointly estimates the per-joint calibration
offsets that explain them all — resolving the ambiguity a single pose can't, and
via Gauss-Newton recovering large offsets, not just the small-signal regime.
Pure NumPy. No breaking changes; additive over the 1.14 public API. 361 tests.

### Added
- **Multi-pose kinematic calibration — `calibrate_joint_offsets`.** The
  multi-pose generalization of 1.14's `localize_joint_fault`, in
  `fieldpilot_urdf.kinematic_diagnosis`. Given a *set* of `PoseObservation`s
  (each a commanded configuration paired with where the link was actually
  measured), it estimates the per-joint calibration offsets that best explain
  **all** of them at once — jointly resolving the ambiguity a single pose can't.
  A **Gauss-Newton** iteration (re-evaluating FK and the Jacobian each step, and
  solving the stacked least-squares system `A·Δδ = b` over every sample's
  Jacobian / residual twist) recovers *large* offsets, not just the small-signal
  regime of `localize_joint_fault`. Position-only observations (no
  `observed_rpy`) contribute position rows only; `orientation_weight` balances
  the mixed metre/radian metric. Returns a `CalibrationResult` — the per-joint
  `offsets` plus position / orientation RMS pose error before and after, and
  `iterations` / `converged`. Only the chain joints to the link are observable
  and calibrated; with too few poses the system is underdetermined and the
  minimum-norm offset is returned (add varied poses to pin it down). Pure NumPy.

## [1.14.0] — 2026-06-18

Brings the kinematics toolkit to the **diagnostics** layer. Every diagnostics
tool so far reasoned structurally (downstream-link overlap) or by simulation;
the new `fieldpilot_urdf.kinematic_diagnosis` module reasons *geometrically* —
given a link's observed vs. commanded pose, `localize_joint_fault` ranks the
chain's joints by how well a single calibration offset on each explains the
deviation, via the 1.10 geometric Jacobian. Pure NumPy. No breaking changes;
additive over the 1.13 public API. 353 tests.

### Added
- **Kinematic fault localization — `localize_joint_fault`.** New
  `fieldpilot_urdf.kinematic_diagnosis` module: the first diagnostics tool that
  reasons *geometrically* rather than structurally. Given a link's commanded
  configuration and its *observed* pose (the tool isn't where the model says),
  it ranks the chain's joints by how well a single calibration offset on each
  explains the deviation — using the 1.10 geometric Jacobian. A small offset
  `δqⱼ` shifts the pose by the twist `J[:, j]·δqⱼ`; the best single-joint
  explanation is the least-squares projection of the observed deviation twist
  onto each Jacobian column. Returns a ranked `list[JointFaultCandidate]`
  (`joint`, `estimated_offset` in rad/m, `explained_fraction` ∈ [0, 1],
  `residual_position`, `residual_orientation`), best-first. Pass only
  `observed_xyz` for a position-only fit (orientation excluded from the fit),
  or add `observed_rpy` for full pose; `orientation_weight` balances the mixed
  metre/radian metric, `min_explained` / `max_candidates` filter the list. It is
  a linearization — exact for small offsets, with `explained_fraction` flagging
  when a single-joint story no longer fits. Pure NumPy.

## [1.13.0] — 2026-06-18

Adds **environment collision**. Until now every collision query was
*self*-collision — the robot could be planned against itself but never the
world. World-frame `Obstacle`s and `detect_obstacle_collisions` close that gap,
and `plan_path` / `shorten_path` / `check_trajectory` gain an `obstacles=`
argument so the RRT-Connect planner routes around them. Reuses the existing
AABB collision model. No breaking changes; `obstacles=None` reproduces prior
behaviour byte-for-byte. Additive over the 1.12 public API. 345 tests.

### Added
- **Environment collision + obstacle-aware planning.** Until now all collision
  checking was *self*-collision; the robot could be planned only against itself,
  never the world. New world-obstacle support in `fieldpilot_urdf.collisions`:
  - **`Obstacle`** — a static world-frame axis-aligned box, built with
    **`box_obstacle(name, center, size)`** or **`sphere_obstacle(name, center,
    radius)`** (the sphere bounded by its enclosing AABB, matching the module's
    AABB-only model), or directly from world `lower` / `upper` corners.
  - **`detect_obstacle_collisions(robot, obstacles, q=…)`** — the world-collision
    counterpart of `detect_self_collisions`; returns the overlapping
    `(link_name, obstacle_name)` pairs. Takes a `tol` to inflate the test.

### Changed
- **`plan_path`, `shorten_path`, and `check_trajectory` gain an `obstacles=`
  argument.** Pass a list of `Obstacle` and the planner routes around the world
  (every config and edge is checked against the obstacles as well as for
  self-collision), short-cutting respects them, and trajectory validation flags
  per-step obstacle hits (`code="collision"`, detail says obstacle). With
  `obstacles=None` (default) behaviour — and the self-collision failure messages —
  are byte-for-byte unchanged. Fully additive.

## [1.12.0] — 2026-06-18

Adds **trajectory time-parameterization**. The planners (`plan_path`,
`plan_cartesian_path`) produce *geometric* paths with no timing; the new
`fieldpilot_urdf.retime` module assigns a path a schedule that respects the
joints' velocity limits, turning waypoints into motion over time — the bridge
from a kinematic path to executable motion, feeding the dynamics / simulation
layer. Pure NumPy (no scipy). No breaking changes; additive over the 1.11
public API. 336 tests.

### Added
- **Trajectory time-parameterization — `time_parameterize`.** New
  `fieldpilot_urdf.retime` module: the bridge from a *geometric* path to an
  *executable* motion. The planners (`plan_path`, `plan_cartesian_path`) emit
  untimed waypoints; `time_parameterize` lays a **trapezoidal velocity profile**
  over the path's joint-space arc length, honouring each joint's velocity limit
  (`JointLimit.velocity` — until now unused for motion). Cruise speed is set by
  the most velocity-constraining segment (conservative, but the per-joint limits
  are *always* respected) and scaled by `velocity_scale`; an optional
  `max_acceleration` adds smooth accel/decel ramps (omit it for a rectangular
  profile). Returns a **`TimedTrajectory`** (`joint_ids`, `times`, `q`, `u`,
  plus `duration`, `as_dicts()`, `final_q()`, and arbitrary-time `sample(t)`) —
  mirroring `simulate.Trajectory`'s shape, so the two interoperate, and feeding
  the dynamics / simulation layer. Continuous joints take the shortest wrapped
  path (matching the planners). Pure NumPy. Single-profile retiming — full
  per-segment time-optimal parameterization (TOPP) is out of scope.

## [1.11.0] — 2026-06-17

Adds **Cartesian (task-space) path planning**. Where 1.5's `plan_path` plans in
joint space (RRT-Connect), the new `fieldpilot_urdf.cartesian` module plans in
task space: `plan_cartesian_path` drives a link along a straight line in SE(3)
via a resolved-rate servo over the 1.10 geometric Jacobian — standing the new
layer directly on the previous one. Pure NumPy (no scipy). No breaking changes;
additive over the 1.10 public API. 323 tests.

### Added
- **Cartesian path planning — `plan_cartesian_path`.** New
  `fieldpilot_urdf.cartesian` module: the task-space complement to
  `plan_path` (which plans in joint space). `plan_cartesian_path` returns a
  joint-space path whose chosen link follows a **straight line** in SE(3) — lerp
  position, slerp orientation — from its current pose to a target pose. The
  engine is a **resolved-rate** servo over the 1.10 geometric Jacobian, stepping
  with an SVD pseudo-inverse that **damps only the singular directions**
  (well-conditioned directions are inverted exactly, so it reaches tight
  tolerance, while directions collapsing toward a singularity are smoothly
  damped instead of blowing up). Pass `target_rpy=None` to hold orientation (a
  pure translation). Returns a `CartesianPlanResult` (`path`, `success`,
  `reached_fraction`, `position_error`, `orientation_error`, `message`); an
  unreachable target / joint limit / singularity yields `success=False` with the
  partial path and the fraction of the line followed. The `path` drops straight
  into `check_trajectory` / `forward_kinematics`, like `plan_path`'s result.
  Also exported: `interpolate_pose` (SE(3) lerp+slerp) and `pose_error`
  (position / orientation error between two poses). Pure NumPy.

## [1.10.0] — 2026-06-17

Adds the **velocity-kinematics** layer. The package had FK (*where* a link is)
and IK (*how to get there*) but not the derivative that connects them; the new
`fieldpilot_urdf.kinematics` module supplies the geometric Jacobian and what it
unlocks — forward velocity, the Yoshikawa manipulability measure, and a
singular-value singularity report. Pure NumPy (no scipy). No breaking changes;
additive over the 1.9 public API. 313 tests.

### Added
- **Velocity kinematics — `fieldpilot_urdf.kinematics`.** The derivative bridge
  between FK (*where* a link is) and IK (*how to get there*): the geometric
  Jacobian and what it unlocks. Pure NumPy — no scipy.
  - **`geometric_jacobian`** returns the 6×n matrix `J` mapping joint velocities
    to a link's world-frame twist (`[v; w] = J @ qdot`, linear rows over angular).
    Columns follow the movable joints on the root→link path; `jacobian_joints`
    gives that ordering. Revolute / continuous / prismatic joints; fixed joints
    contribute no column.
  - **`joint_velocity_to_twist`** is the forward-velocity convenience: the
    world twist of a link for a `{joint: velocity}` dict.
  - **`manipulability`** gives the Yoshikawa measure (volume of the
    manipulability ellipsoid, `√det(JJᵀ)`) — a scalar dexterity / distance-from-
    singularity score.
  - **`singularity_report`** (→ `SingularityReport`) reports the Jacobian's
    singular values, `sigma_min`/`sigma_max`, condition number, manipulability,
    and an `is_singular` flag.
  - `manipulability` and `singularity_report` take `rows=` to restrict the
    measure to a task subspace of the 6-row twist (e.g. `rows=(0, 1, 2)` for a
    position-only measure) — needed for sub-6-DoF arms, whose positional
    singularities the full 6×n Jacobian masks.

## [1.9.0] — 2026-06-17

Adds native **mesh format support**. A new `fieldpilot_urdf.mesh` module reads
vertex bounding boxes for STL / OBJ / PLY in pure Python, so mesh self-collision
no longer needs the `[mesh]` (trimesh) extra for those common formats — trimesh
stays only as the fallback for the long tail (COLLADA `.dae`, glTF). No breaking
changes; results are identical. Additive over the 1.8 public API. 302 tests.

### Added
- **Pure-Python mesh bounds reader — `read_mesh_bounds`.** New
  `fieldpilot_urdf.mesh` module reads a mesh file's vertex bounding box in pure
  Python (NumPy only) for the formats that dominate URDFs: **STL** (binary *and*
  ASCII, auto-detected), **OBJ**, and **PLY** (ASCII + binary little/big endian).
  Returns `((minx, miny, minz), (maxx, maxy, maxz))` or `None` for an
  unsupported/corrupt file; `SUPPORTED_FORMATS` lists the handled extensions.

### Changed
- **Mesh self-collision no longer requires the `[mesh]` extra for STL/OBJ/PLY.**
  `collisions._load_mesh_aabb` now resolves AABBs via `read_mesh_bounds` first
  and only falls back to `trimesh` for other formats (COLLADA `.dae`, glTF). So
  `detect_self_collisions` with mesh geometry works on a plain
  `pip install fieldpilot-urdf` for the common formats. No API change; results
  are identical (the native STL reader matches trimesh's bounds).

## [1.8.0] — 2026-06-17

Broadens the **importer**. `import_urdf` could already pull a robot (with full
`$(find)` / `<xacro:include>` / xacro expansion) from an HTTPS URL; now
`import_urdf_file` does the same against the local filesystem — a checked-out ROS
package or exported xacro imports with no server and no network. Strictly local
(remote includes are refused). No breaking changes; additive over the 1.7 public
API. 284 tests.

### Added
- **Local-filesystem import — `import_urdf_file`.** The network-free twin of
  `import_urdf`: runs the same `$(find)` → `<xacro:include>` → xacro → parse
  pipeline against a `.urdf`/`.xacro` on disk, so a checked-out ROS package or
  exported xacro imports without standing up an HTTPS server (until now,
  `from_file` only parsed *plain* URDF — no includes, no macros). `package://`
  references resolve via a caller-supplied `package_roots` ({package: directory})
  map; relative includes resolve against the file's directory; `${load_yaml(...)}`
  reads configs from disk. Strictly local — a remote `<xacro:include>` is a hard
  error (use `import_urdf` for network sources). `expand_macros=False` skips the
  pipeline for an already-plain URDF. Also exported: `resolve_includes_local` and
  `expand_xacro_local` (the lower-level pieces, mirroring their HTTPS twins).

## [1.7.0] — 2026-06-17

Deepens the **dynamics** layer. `SymbolicDynamics` produced the instantaneous
equations of motion; the new `fieldpilot_urdf.simulate` module rolls them forward
in time (`integrate_dynamics`) and inverts them (`inverse_dynamics`,
`gravity_torques`) — closing the loop from model to motion to control. Pure
NumPy, no SciPy required, under the existing `[dynamics]` extra. No breaking
changes; additive over the 1.6 public API. 270 tests.

### Added
- **Dynamics simulation — `fieldpilot_urdf.simulate`.** A numerical layer over
  `SymbolicDynamics` (the `[dynamics]` extra) that turns the instantaneous
  equations of motion into motion over time and inverts them:
  - **`integrate_dynamics`** rolls the forward dynamics from an initial state
    under an applied-torque law (passive, a constant per-joint dict/vector, or a
    `(t, q, u) -> torque` callable), returning a sampled `Trajectory`. Methods:
    `"rk4"` (default) and symplectic `"euler"`. Pure NumPy — no SciPy needed.
  - **`inverse_dynamics`** returns a callable `tau(q, u, qdd)` giving the joint
    torques that realise a desired acceleration (`tau = M(q)·qdd − F(q, q̇, 0)`).
  - **`gravity_torques`** gives the static holding torques at a configuration
    (gravity compensation — inverse dynamics with `q̇ = q̈ = 0`).
  - **`Trajectory`** carries `joint_ids`, `times`, `q`, `u` with `as_dicts()`,
    `final_q()`, `final_u()` helpers. Degenerate (all-fixed) robots are handled.

## [1.6.0] — 2026-06-17

Strengthens **inverse kinematics**. `solve_ik_multi` surfaces *all* distinct
solutions to a pose (elbow-up / elbow-down and other joint flips) rather than
just one, and `solve_ik` gains opt-in random-restart robustness for hard targets
— both via cheap multi-seed restarts over the existing solver. No new
dependencies; the default single-shot `solve_ik` is unchanged. Additive over the
1.5 public API. 256 tests.

### Added
- **Multi-solution IK — `solve_ik_multi`.** Many arms reach a pose more than one
  way (elbow-up / elbow-down, joint flips); `solve_ik` returns one, this returns
  all *distinct* ones. Runs the solver from the midpoint seed plus `n_restarts`
  random in-bounds seeds (default 24), then collapses results landing on the same
  configuration (joint-space distance below `dedup_tol`, continuous joints
  compared on the wrapped arc). Returns the distinct solutions as a
  `list[IKResult]` sorted best-first; `require_converged=True` (default) keeps
  only converged ones (empty list = none found), `max_solutions` caps the count,
  `seed` makes it reproducible.

### Changed
- **`solve_ik` gains random-restart robustness.** New `n_restarts` / `seed`
  arguments: when the primary solve (from `q_init`/midpoint) doesn't converge,
  the solver retries from up to `n_restarts` random in-bounds seeds and keeps the
  best result — a cheap way past local minima on hard targets. `n_restarts=0`
  (default) leaves the single-shot behaviour exactly as before.

## [1.5.0] — 2026-06-17

Adds a **motion-planning** layer. Where the 1.x line could *validate* a path
(`check_trajectory`), `plan_path` now *generates* a collision-free one between
two configurations via RRT-Connect — and the result feeds straight back into the
existing kinematics/validation calls. Pure-Python, no new dependencies. No
breaking changes; additive over the 1.4 public API. 247 tests.

### Added
- **Motion planning — `plan_path`.** New `fieldpilot_urdf.planning` module: an
  RRT-Connect planner that *generates* a collision-free joint-space path between
  a start and goal configuration, complementing `check_trajectory` (which only
  *validates* a path you already have). Bidirectional trees grow toward random
  samples within joint limits and link up; edges are collision-checked at
  `step_size` resolution against `detect_self_collisions`. Endpoints are
  validated up front (out-of-limits or self-colliding start/goal fail with an
  explanatory message). Continuous joints are handled on the shortest wrapped
  arc. The result is post-processed by greedy short-cutting (`smooth=True`).
  Returns a `PlanResult` whose `path` is a list of waypoint dicts that feeds
  straight into `check_trajectory` / `forward_kinematics`. Also exported:
  `shorten_path` (short-cut any waypoint path) and `path_length` (joint-space
  length, continuous-aware). `seed` makes runs reproducible.

## [1.4.0] — 2026-06-17

Closes the **diagnostics** loop into a self-bootstrapping engine. `diagnose` now
generates its own hypotheses from the symptom — `diagnose(robot, symptom)` works
with zero supplied candidates by chaining graph root-cause ranking into the
hypothesis-and-test loop. No breaking changes; an explicit `hypotheses` list
behaves exactly as before. Additive over the 1.3 public API. 230 tests.

### Added
- **Auto-hypothesis generation in `diagnose`.** `hypotheses` is now optional —
  when omitted (or empty), `diagnose` derives candidates from the symptom by
  ranking suspect joints with `rank_root_causes` (the affected links come from
  the symptom: `target_link` for `cant_reach`/`reduced_workspace`,
  `colliding_links` for `self_collision`) and tests a parameter-free `motor_dead`
  on each (up to `max_auto`, default 5), best-first. Fixed joints are excluded
  (no motor to kill). The returned `DiagnoseReport` carries `auto_generated=True`
  and an `evidence["auto_candidates"]` list. Connects the two core diagnostic
  features (graph root-cause ranking → hypothesis-and-test) into one call; an
  explicit `hypotheses` list behaves exactly as before. The parametric modes
  (`joint_stuck`, `limit_misconfig`) still need caller-supplied values.

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

[1.20.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.20.0
[1.19.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.19.0
[1.18.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.18.0
[1.17.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.17.0
[1.16.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.16.0
[1.15.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.15.0
[1.14.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.14.0
[1.13.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.13.0
[1.12.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.12.0
[1.11.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.11.0
[1.10.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.10.0
[1.9.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.9.0
[1.8.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.8.0
[1.7.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.7.0
[1.6.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.6.0
[1.5.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.5.0
[1.4.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.4.0
[1.3.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.3.0
[1.2.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.2.0
[1.1.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.1.0
[1.0.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v1.0.0
[0.1.0]: https://github.com/DuQuatre/fieldpilot-urdf/releases/tag/v0.1.0
