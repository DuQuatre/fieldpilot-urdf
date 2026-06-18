# fieldpilot-urdf

[![CI](https://github.com/DuQuatre/fieldpilot-urdf/actions/workflows/ci.yml/badge.svg)](https://github.com/DuQuatre/fieldpilot-urdf/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

> Import any ROS robot from a URL and run **FK / IK / collision / validation /
> repair** in pure Python. No ROS install, no build.

`fieldpilot-urdf` is the **open core** of [FieldPilot](https://github.com/DuQuatre)'s
robotics toolkit — small, self-contained, pure-Python. Point it at a robot on
GitHub and get a working kinematic model in three lines.

> **Status: `1.23.0` — stable, published on [PyPI](https://pypi.org/project/fieldpilot-urdf/).**
> 419 passing tests, a documented [public API](#public-api--stability) under
> SemVer. `pip install fieldpilot-urdf` (see [`RELEASING.md`](RELEASING.md) for
> how releases are cut).

## The four-layer ladder

`fieldpilot-urdf` is built as four layers, each standing on the one below — and
you install only the weight you use. The arc runs **import → diagnose**:

| Layer | What it does | Install |
|------:|--------------|---------|
| **1 · Model** | get a robot (URL / file / code), validate, repair | core |
| **2 · Kinematics** | FK, IK, self-collision, workspace, trajectory, motion planning | core |
| **3 · Dynamics + Sim** | symbolic equations of motion, time integration, inverse dynamics, PyBullet | `[dynamics]`, `[sim]` |
| **4 · Diagnostics** | localise a fault, then hypothesis-test it | core |

New here? The **[tutorial](docs/tutorial.md)** climbs all four layers end to end,
and **[`examples/full_stack_tour.py`](examples/full_stack_tour.py)** is its
runnable companion. The sections below mirror the ladder. For the diagnostics
loop end to end — localise → narrow by dialog → calibrate → recommend → record →
report — run **[`examples/diagnostics_workflow.py`](examples/diagnostics_workflow.py)**
or read the illustrated **[diagnostics guide](docs/diagnostics-guide.md)**
(scenarios, dialogs, spare parts, field interactions).

## Install

```bash
pip install fieldpilot-urdf                # core (parse, FK, IK, validation)
pip install "fieldpilot-urdf[mesh]"        # + mesh-aware self-collision (trimesh)
pip install "fieldpilot-urdf[viz]"         # + kinematic-tree / 3D-pose renderers
pip install "fieldpilot-urdf[dynamics]"    # + Kane's-method symbolic dynamics (sympy)
pip install "fieldpilot-urdf[sim]"         # + PyBullet numerical simulation
pip install "fieldpilot-urdf[all]"         # everything
```

## Layer 1 — Model: import any ROS robot in 3 lines

```python
from fieldpilot_urdf import import_urdf, run_all, summary

# Point at any ROS robot on GitHub — xacro, $(find), and <xacro:include> expand
robot, _ = import_urdf(
    "https://raw.githubusercontent.com/ros-industrial/universal_robot/melodic-devel/"
    "ur_description/urdf/ur5.urdf.xacro"
)

print(robot.name, len(robot.links), "links", len(robot.joints), "joints")
print(summary(run_all(robot)))   # validate: {'total': 0, 'error': 0, ...}
```

No URL? `from_file` / `from_xml` parse a local URDF, and `to_xml` round-trips the
model back out.

### Validate & auto-repair

```python
from fieldpilot_urdf import from_file, run_all, summary, repair

robot = from_file("maybe_broken.urdf")
findings = run_all(robot)                 # 8 lint rules (R001–R008)
print(summary(findings))                  # {'total': 3, 'error': 1, 'warning': 2, ...}

fixed, patches, unfixable = repair(robot)  # deterministic fixes for the repairable rules
print([p.code for p in patches])           # e.g. ['R003', 'R005']
print("left for a human:", unfixable)      # rule codes that can't be auto-fixed
print(summary(run_all(fixed)))             # fewer (often zero) findings
```

## Layer 2 — Kinematics: FK / IK / collision / workspace

```python
from fieldpilot_urdf import forward_kinematics, solve_ik, detect_self_collisions

poses = forward_kinematics(robot)                       # {link: 4x4 world transform}
ik = solve_ik(robot, "tool0", target_xyz=(0.4, 0.1, 0.5))
print(ik.converged, ik.position_error)                  # numerical IK, honours limits
print(detect_self_collisions(robot))                    # [(link_a, link_b), ...]
```

`sample_workspace` and `check_trajectory` round out the layer — reachable
envelope and a per-step limit/collision check along a path.

**Planning around the world**, not just the robot: pass `obstacles` and the
planner routes around them (and `check_trajectory` flags any that are hit).

```python
from fieldpilot_urdf import plan_path, box_obstacle

wall = box_obstacle("wall", center=(1.0, 0.0, 0.5), size=(0.3, 2.0, 1.0))
res = plan_path(robot, start, goal, obstacles=[wall])    # RRT-Connect, obstacle-aware
print(res.success, res.n_waypoints)                      # detours around the wall
```

Plain `solve_ik` ignores collision — `solve_ik_collision_free` picks the IK
*branch* (elbow-up / elbow-down …) that's actually usable: self-collision-free
and clear of the obstacles, so it feeds straight in as a `plan_path` endpoint.

```python
from fieldpilot_urdf import solve_ik_collision_free

ik = solve_ik_collision_free(robot, "tool0", target_xyz=(0.5, 0.0, 0.4), obstacles=[wall])
print(ik.converged, ik.message)          # converged == a collision-free posture was found
```

**Velocity kinematics** sits between FK and IK: the geometric Jacobian maps joint
velocities to the end-effector twist, and from it you read off dexterity and
proximity to a singularity.

```python
from fieldpilot_urdf import geometric_jacobian, manipulability, singularity_report

J = geometric_jacobian(robot, q, "tool0")               # 6×n: [v; w] = J @ qdot
print(manipulability(robot, q, "tool0"))                # Yoshikawa dexterity measure
rep = singularity_report(robot, q, "tool0")             # σ_min, condition number, is_singular
print(rep.is_singular, rep.condition_number)
```

**Cartesian motion** plans in task space where `plan_path` plans in joint space:
drive the tool along a straight line, get back a joint path that feeds the same
validators.

```python
from fieldpilot_urdf import plan_cartesian_path

res = plan_cartesian_path(robot, "tool0", target_xyz=(0.5, 0.0, 0.4), start_q=q)
print(res.success, res.reached_fraction)                # straight-line move in SE(3)
print(res.path[-1])                                     # final joint config
```

A planned path is purely geometric — `time_parameterize` gives it a schedule
that respects the joints' velocity limits, turning waypoints into motion over
time (positions *and* velocities) ready for the dynamics layer.

```python
from fieldpilot_urdf import time_parameterize

traj = time_parameterize(robot, res.path, max_acceleration=2.0)  # trapezoidal profile
print(traj.duration, len(traj.times))                            # timed, velocity-limited
print(traj.sample(traj.duration / 2))                           # joint config at any t
```

### Render (needs the `[viz]` extra)

```python
from fieldpilot_urdf.viz import render_kinematic_tree, render_pose_3d

open("tree.png", "wb").write(render_kinematic_tree(robot))      # graphviz
open("pose.png", "wb").write(render_pose_3d(robot, fmt="png"))  # matplotlib
```

## Layer 3 — Dynamics & simulation

### Symbolic dynamics (needs the `[dynamics]` extra)

```python
from fieldpilot_urdf.dynamics import SymbolicDynamics

dyn = SymbolicDynamics(robot)                 # Kane's method on the kinematic tree
print(dyn.n_dof)                              # actuated DOF
print(dyn.mass_matrix)                        # symbolic M(q)
print(dyn.forcing)                            # symbolic F(q, q̇, τ) = τ − C(q,q̇)q̇ − G(q)

# Forward dynamics as a NumPy callable, ready for scipy.integrate.solve_ivp:
fwd = dyn.lambdify_forward_dynamics()         # (q, u, tau) -> q̈   (solves M·q̈ = F)
qdd = fwd([0.0] * dyn.n_dof, [0.0] * dyn.n_dof, [0.0] * dyn.n_dof)
```

Tree (serial) robots only in this release. Joint-origin frames follow URDF's
`Rz(yaw)·Ry(pitch)·Rx(roll)` convention, so `dyn.link_pose(link, q)` matches
`forward_kinematics` to machine precision. Closed-loop mechanisms and
multi-DOF joints (`floating`/`planar`/`spherical`) raise `UnsupportedSystemError`
— closed loops have their own path (`LoopClosure` + `constrained.constrained_dynamics`).

### Numerical simulation (needs the `[sim]` extra)

```python
from pathlib import Path
from fieldpilot_urdf import import_urdf
from fieldpilot_urdf.importer import fetch_meshes
from fieldpilot_urdf.sim import PyBulletSim

robot, url = import_urdf("https://.../ur5.urdf.xacro")      # URDF -> model
fetch_meshes(robot, url, Path("/tmp/ur5"))                  # download package:// meshes
with PyBulletSim(robot, mesh_dir="/tmp/ur5") as sim:        # straight into PyBullet
    sim.set_position_targets({"shoulder_pan_joint": 0.5})
    sim.step(240)
    print(sim.joint_states())                               # {joint: (pos, vel)}
```

A thin PyBullet wrapper — load, step, control, read state — fed by the import
pipeline (`package://` mesh paths are rewritten to the fetched files). It honours
the URDF's `<inertia>` (`URDF_USE_INERTIA_FROM_FILE`), so its free-fall dynamics
match the symbolic `SymbolicDynamics` to ~1e-5. For richer simulation, use
PyBullet / MuJoCo / Drake directly on the URDF this package imports.

## Layer 4 — Diagnostics: localise, then prove

First **localise** — which joint best explains the links a tech reports as dead?
Pure NetworkX graph reasoning, deterministic, in the core install:

```python
from fieldpilot_urdf import affected_links, criticality, rank_root_causes

# Which links does a faulty joint drag down, and how much mass is at stake?
affected_links(robot, "shoulder_pan_joint")   # {'upper_arm_link', 'forearm_link', 'wrist_1_link', ...}
criticality(robot, "shoulder_pan_joint")      # 0.0–1.0, mass-weighted downstream impact

# Reverse: a tech reports the wrist + tool went limp — which joint best explains it?
ranked = rank_root_causes(robot, ["wrist_3_link", "tool0"])
print(ranked[0].target, round(ranked[0].score, 3))   # suspect joint, precision×recall score
```

Or localise **kinematically** — the tool measured *off* where the model says it
should be. Which joint is miscalibrated, and by how much? Ranked via the
geometric Jacobian:

```python
from fieldpilot_urdf import localize_joint_fault

# commanded q, but the tool was measured at observed_xyz (optionally + observed_rpy)
cands = localize_joint_fault(robot, "tool0", commanded_q, observed_xyz=(0.41, 0.10, 0.52))
print(cands[0].joint, round(cands[0].estimated_offset, 4), round(cands[0].explained_fraction, 2))
# -> e.g. 'wrist_2_joint' 0.03 0.99   (a 0.03 rad miscalibration explains 99% of the deviation)
```

A single pose can be ambiguous; with **several** measurements, `calibrate_joint_offsets`
solves for every joint's offset at once (Gauss-Newton, so even large offsets) —
turning a set of measured poses into a calibrated model:

```python
from fieldpilot_urdf import calibrate_joint_offsets, PoseObservation

obs = [PoseObservation(commanded_q=q, observed_xyz=p) for q, p in measurements]
cal = calibrate_joint_offsets(robot, "tool0", obs)
print(cal.offsets)                                   # {joint: estimated offset}
print(cal.position_rms_before, "->", cal.position_rms_after)   # error collapses
```

When several faults still compete, **narrow by dialog** — ask the technician the
*most discriminating* question (highest information gain) and fold the answer
back in, turn by turn, until one fault dominates. Stateless, so the bot/LLM owns
the conversation:

```python
from fieldpilot_urdf import Candidate, Question, rank_questions, update_beliefs

candidates = [Candidate(name="motor_dead"), Candidate(name="backlash")]
questions = [Question(id="hold", text="Does the joint hold position under load?",
                      outcomes=["yes", "no"],
                      likelihoods={"motor_dead": {"yes": 0.0, "no": 1.0},
                                   "backlash":   {"yes": 0.9, "no": 0.1}})]

print(rank_questions(candidates, questions)[0].info_gain)     # ~1 bit: ask this first
state = update_beliefs(candidates, questions, {"hold": "no"}) # tech answers
print(state.leading, round(state.leading_prob, 2), state.resolved)  # motor_dead 1.0 True
```

**Remember every case** so the diagnosis sharpens over time — store the resolved
fault and the fix that worked, then mine the history for fault-frequency priors
(which seed the dialog above) and for the fix with the best track record:

```python
from fieldpilot_urdf import (DiagnosticCase, save_case, load_cases,
                             fault_priors, recommend_solution, candidates_from_scores)

save_case(DiagnosticCase(id="INT-2026-0142", machine="robotA",
                         confirmed_fault="motor_dead", solution="replace_motor",
                         resolved=True))
cases = load_cases()
priors = fault_priors(cases, machine="robotA")            # learned fault frequencies
seeded = candidates_from_scores(priors)                   # -> start the next dialog smarter
print(recommend_solution(cases, "motor_dead"))            # the best-proven fix
```

Then **prove it** — inject the hypothesis on a copy, re-test, and confirm or
refute. The ranked suspect feeds straight into `diagnose` as the hypothesis:

```python
from fieldpilot_urdf import diagnose, Symptom, Hypothesis

# "tool can't reach this pose" — is a dead shoulder motor the cause?
report = diagnose(
    robot,
    Symptom(kind="cant_reach", target_link="tool0", target_xyz=(0.4, 0.1, 0.5)),
    [Hypothesis(suspect_joint="shoulder_pan_joint", fault_mode="motor_dead")],
)
print(report.verdict, "—", report.summary)   # CONFIRMED / REFUTED / INCONCLUSIVE
```

Then **show the tech** — animate the simulated faulted motion against the
nominal one so reality can be compared to the model (needs the `[viz]` extra):

```python
from fieldpilot_urdf.viz import render_motion_comparison

gif = render_motion_comparison(robot, nominal_path, faulted_path,
                               layout="overlay", track_link="tool0")  # 3D GIF bytes
open("fault.gif", "wb").write(gif)   # nominal solid / faulted dashed, divergence marked
```

Then **assemble the report** — ask the tech for the right photos, attach the
simulation illustrations *once confirmed*, and emit a self-contained French HTML
document (ready for HTML→PDF):

```python
from fieldpilot_urdf import (DiagnosticReport, ReportImage, photo_requests,
                             attach_simulation_illustrations, render_report_html)

print(photo_requests(joints=["shoulder_pan_joint"]))   # what pictures to ask for (French)

report = DiagnosticReport(reference="INT-2026-0042", confirmed=True,
                          fault="shoulder_pan_joint", confidence=0.97,
                          photos=[ReportImage.from_bytes("p1.jpg", tech_photo_bytes)])
report = attach_simulation_illustrations(report, robot, nominal_path, faulted_path,
                                         track_link="tool0")   # only when confirmed
open("rapport.html", "w").write(render_report_html(report))    # → Gotenberg → PDF
```

…and the **scope traces** — joint position / velocity over time, expected vs
observed, with the divergence shaded — the quantitative half of the comparison:

```python
from fieldpilot_urdf.viz import render_trajectory_scope

png = render_trajectory_scope(expected_traj, observed_traj,        # TimedTrajectory / sim
                              signals=("position", "velocity"))
open("scope.png", "wb").write(png)   # stacked panels, max Δ annotated per channel
```

*(The natural-language front-end that turns a free-text or voice symptom into
these hypotheses via an LLM is part of [FieldPilot](https://github.com/DuQuatre)
SaaS. The reasoning core above stays local and deterministic.)*

## What you can do

| Capability | API |
|---|---|
| Parse URDF ⇄ model | `from_xml`, `from_file`, `to_xml` |
| Import a robot from a URL (xacro/includes/meshes) | `import_urdf` |
| Forward kinematics | `forward_kinematics` |
| Inverse kinematics (numerical, limit-aware) | `solve_ik` |
| Self-collision (AABB + mesh) | `detect_self_collisions` |
| Workspace / trajectory sampling | `sample_workspace`, `check_trajectory` |
| 8 lint rules (R001–R008) | `run_all`, `summary` |
| Deterministic auto-repair | `repair` |
| Two-tier symbolic fault diagnosis | `diagnose` |
| Fault propagation & root-cause ranking | `affected_links`, `criticality`, `rank_root_causes` |
| Symbolic dynamics (Kane's method) | `SymbolicDynamics` |
| Closed-loop modelling & constraint deriver | `LoopClosure`, `loops.derive_loop_constraints` |
| Closed-loop (constrained) dynamics | `constrained.constrained_dynamics` |
| Numerical simulation (PyBullet) | `sim.PyBulletSim` |
| Render kinematic tree / 3D pose | `render_kinematic_tree`, `render_pose_3d` |
| Local robot registry | `save_robot`, `load_robot`, `list_robots` |

## Public API & stability

As of **1.0.0**, the public API follows [Semantic Versioning](https://semver.org/):
breaking changes to anything documented as public wait for a major bump.

- **Stable** — everything exported from the top-level package (`fieldpilot_urdf.__all__`),
  plus the public names of the optional submodules: `fieldpilot_urdf.dynamics`,
  `.constrained`, `.loops`, `.sim`, and `.viz`. If you can `from fieldpilot_urdf import X`
  (or `from fieldpilot_urdf.dynamics import SymbolicDynamics`), it's covered.
- **Internal** — any name prefixed with `_` and the `_dyn_adapter` module. These
  may change without notice; don't import them.

See **[`docs/api.md`](docs/api.md)** for the full surface, layer by layer.

## How this compares

The Python URDF ecosystem already has good *parsers*. `fieldpilot-urdf` is not
trying to replace them — it sits one layer up, as an **analysis** toolkit.

- **[`urchin`](https://github.com/fishbotics/urchin)** is the maintained fork of
  the classic `urdfpy` (unmaintained since 2020, won't install on Python 3.10+).
  Reach for it if you want the original `urdfpy` API and mesh-heavy
  visualization on a modern Python.
- **[`yourdfpy`](https://github.com/clemense/yourdfpy)** is the most robust
  *loader* of real-world URDFs and ships an excellent visualization CLI. Reach
  for it if your priority is parsing messy URDFs found in the wild.

Reach for **`fieldpilot-urdf`** when parsing is the *start*, not the goal — when
you also want to **solve IK** (numerical, joint-limit-aware), **import a robot
straight from a URL** (`$(find)` / `<xacro:include>` / xacro expansion, with
SSRF defenses), **lint** a URDF (8 rules, R001–R008) and **deterministically
auto-repair** the fixable ones, run **symbolic fault diagnosis** ("is a dead
shoulder motor why the tool can't reach?"), and go past geometry into
**dynamics** — both parsers stop at kinematics, whereas this one builds Kane's-method
equations of motion and cross-checks them against PyBullet. The core install stays light —
`pydantic` + `numpy` + `scipy` + `networkx` — with mesh/viz as optional extras,
so you never pull `pyrender` or `pycollada` unless you ask for them.

> Need to *load* a difficult URDF more than analyze it? `yourdfpy` is probably
> the better fit — and `fieldpilot-urdf` happily consumes anything it can export.

## ⭐ Want more?

The open toolkit gives you the robotics. **[FieldPilot](https://github.com/DuQuatre)**
(the hosted SaaS) adds the parts you can't easily self-host:

- a **natural-language fault-diagnosis** front-end (describe a symptom → ranked hypotheses),
- a **13-tool LLM chat** over your robot,
- a **spare-parts BOM** with pricing, and
- **multi-tenant hosting**, Telegram bots, and the agro-food field-service pipeline.

→ **Star this repo** and check out FieldPilot SaaS.

## Security

`import_urdf` fetches a user-supplied URL, so it ships SSRF defences (HTTPS-only,
host allowlist, 5 MB cap, timeout, redirect re-validation). See
[`SECURITY.md`](SECURITY.md) for how to configure the allowlist and report issues.

## License

**AGPL-3.0-only.** Free to self-host, modify, and use; network use obliges source
disclosure. A **commercial license** is available for closed/embedded use — see
FieldPilot.
