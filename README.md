# fieldpilot-urdf

> Import any ROS robot from a URL and run **FK / IK / collision / validation /
> repair** in pure Python. No ROS install, no build.

`fieldpilot-urdf` is the **open core** of [FieldPilot](https://github.com/DuQuatre)'s
robotics toolkit — small, self-contained, pure-Python. Point it at a robot on
GitHub and get a working kinematic model in three lines.

> **Status: pre-release (`0.1.0.dev0`).** The 14 open modules + their test suite
> (144 tests, passing) are in place. Not yet published to PyPI — install from
> source for now (`pip install -e .`). API is stabilising toward v0.1.

## Install

```bash
pip install fieldpilot-urdf                # core (parse, FK, IK, validation)
pip install "fieldpilot-urdf[mesh]"        # + mesh-aware self-collision (trimesh)
pip install "fieldpilot-urdf[viz]"         # + kinematic-tree / 3D-pose renderers
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
| Render kinematic tree / 3D pose | `render_kinematic_tree`, `render_pose_3d` |
| Local robot registry | `save_robot`, `load_robot`, `list_robots` |

## ⭐ Want more?

The open toolkit gives you the robotics. **[FieldPilot](https://github.com/DuQuatre)**
(the hosted SaaS) adds the parts you can't easily self-host:

- a **natural-language fault-diagnosis** front-end (describe a symptom → ranked hypotheses),
- a **13-tool LLM chat** over your robot,
- a **spare-parts BOM** with pricing, and
- **multi-tenant hosting**, Telegram bots, and the agro-food field-service pipeline.

→ **Star this repo** and check out FieldPilot SaaS.

## License

**AGPL-3.0-only.** Free to self-host, modify, and use; network use obliges source
disclosure. A **commercial license** is available for closed/embedded use — see
FieldPilot.
