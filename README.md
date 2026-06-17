# fieldpilot-urdf

[![CI](https://github.com/DuQuatre/fieldpilot-urdf/actions/workflows/ci.yml/badge.svg)](https://github.com/DuQuatre/fieldpilot-urdf/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

> Import any ROS robot from a URL and run **FK / IK / collision / validation /
> repair** in pure Python. No ROS install, no build.

`fieldpilot-urdf` is the **open core** of [FieldPilot](https://github.com/DuQuatre)'s
robotics toolkit — small, self-contained, pure-Python. Point it at a robot on
GitHub and get a working kinematic model in three lines.

> **Status: `0.7.0` — published on [PyPI](https://pypi.org/project/fieldpilot-urdf/).**
> 19 open modules + 188 passing tests. `pip install fieldpilot-urdf` (see
> [`RELEASING.md`](RELEASING.md) for how releases are cut).

## Install

```bash
pip install fieldpilot-urdf                # core (parse, FK, IK, validation)
pip install "fieldpilot-urdf[mesh]"        # + mesh-aware self-collision (trimesh)
pip install "fieldpilot-urdf[viz]"         # + kinematic-tree / 3D-pose renderers
pip install "fieldpilot-urdf[dynamics]"    # + Kane's-method symbolic dynamics (sympy)
pip install "fieldpilot-urdf[sim]"         # + PyBullet numerical simulation
pip install "fieldpilot-urdf[all]"         # everything
```

## Import any ROS robot in 3 lines

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

Then do something real:

```python
from fieldpilot_urdf import forward_kinematics, solve_ik, detect_self_collisions

poses = forward_kinematics(robot)                       # {link: 4x4 world transform}
ik = solve_ik(robot, "tool0", target_xyz=(0.4, 0.1, 0.5))
print(ik.converged, ik.position_error)                  # numerical IK, honours limits
print(detect_self_collisions(robot))                    # [(link_a, link_b), ...]
```

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

### Render (needs the `[viz]` extra)

```python
from fieldpilot_urdf.viz import render_kinematic_tree, render_pose_3d

open("tree.png", "wb").write(render_kinematic_tree(robot))      # graphviz
open("pose.png", "wb").write(render_pose_3d(robot, fmt="png"))  # matplotlib
```

### Symbolic fault diagnosis

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

*(The natural-language front-end that generates hypotheses from a free-text
symptom via an LLM is part of [FieldPilot](https://github.com/DuQuatre) SaaS.)*

### Localise a fault on the kinematic graph

```python
from fieldpilot_urdf import affected_links, criticality, rank_root_causes

# Which links does a faulty joint drag down, and how much mass is at stake?
affected_links(robot, "shoulder_pan_joint")   # {'upper_arm_link', 'forearm_link', 'wrist_1_link', ...}
criticality(robot, "shoulder_pan_joint")      # 0.0–1.0, mass-weighted downstream impact

# Reverse: a tech reports the wrist + tool went limp — which joint best explains it?
ranked = rank_root_causes(robot, ["wrist_3_link", "tool0"])
print(ranked[0].target, round(ranked[0].score, 3))   # suspect joint, precision×recall score
```

Pure NetworkX graph reasoning — deterministic, in the core install. The ranked
suspects can feed straight into `diagnose` as hypotheses. *(The LLM/NL symptom
front-end stays in [FieldPilot](https://github.com/DuQuatre) SaaS.)*

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
multi-DOF joints (`floating`/`planar`/`spherical`) raise `UnsupportedSystemError`.

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
auto-repair** the fixable ones, and run **symbolic fault diagnosis** ("is a dead
shoulder motor why the tool can't reach?"). The core install stays light —
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
