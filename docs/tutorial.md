# Tutorial — from a URL to a confirmed fault

`fieldpilot-urdf` is built as a **four-layer ladder**. Each layer stands on the
one below it, and you only install the weight you use:

| Layer | What it does | Install |
|------:|--------------|---------|
| **1 · Model** | get a robot (URL / file / code), validate, repair | core |
| **2 · Kinematics** | FK, IK, self-collision, workspace, trajectory | core |
| **3 · Dynamics + Sim** | symbolic equations of motion, PyBullet | `[dynamics]`, `[sim]` |
| **4 · Diagnostics** | localise a fault, then hypothesis-test it | core |

This tutorial climbs the ladder once, end to end — point at a real ROS robot,
get a kinematic model, reason about its dynamics, then diagnose a fault. Every
snippet runs as written. The runnable companion that prints real numbers for the
whole tour is [`examples/full_stack_tour.py`](../examples/full_stack_tour.py):

```bash
pip install "fieldpilot-urdf[dynamics,sim]"   # all four layers
python examples/full_stack_tour.py
```

---

## Layer 1 — get a robot, and trust it

The headline entry point is `import_urdf`: hand it an HTTPS URL and it expands
`xacro`, `$(find pkg)`, and `<xacro:include>` **in-process** — no ROS, no
`catkin`/`colcon` build, no workspace.

```python
from fieldpilot_urdf import import_urdf

robot, url = import_urdf(
    "https://raw.githubusercontent.com/ros-industrial/universal_robot/"
    "melodic-devel/ur_description/urdf/ur5.urdf.xacro"
)
print(robot.name, len(robot.links), "links", len(robot.joints), "joints")
```

`robot` is a plain [pydantic](https://docs.pydantic.dev) model — `robot.links`,
`robot.joints`, `joint.origin`, `joint.limit`, link `inertial`/`visual`/
`collision` are all there to read and mutate. You can also skip the network:

```python
from fieldpilot_urdf import from_file, from_xml, to_xml

robot = from_file("my_robot.urdf")     # parse a local file
robot = from_xml(xml_string)           # parse a string
to_xml(robot)                          # serialise back to URDF XML (round-trips)
```

`import_urdf` fetches a user-supplied URL, so it ships **SSRF defences** —
HTTPS-only, a host allowlist, a 5 MB cap, a timeout, and redirect
re-validation. See [`../SECURITY.md`](../SECURITY.md) to configure the allowlist.

### Validate and auto-repair

Real-world URDFs are messy. Before you compute anything, lint the model against
8 deterministic rules (`R001`–`R008`):

```python
from fieldpilot_urdf import run_all, summary, repair

findings = run_all(robot)
print(summary(findings))     # {'total': 3, 'error': 1, 'warning': 2, 'info': 0}
for f in findings:
    print(f.code, f.severity, f.message)

# Deterministic fixes for the repairable rules — no heuristics, no guessing:
fixed, patches, unfixable = repair(robot)
print([p.code for p in patches])   # e.g. ['R003', 'R005']
print("left for a human:", unfixable)
print(summary(run_all(fixed)))     # fewer (often zero) findings
```

`repair` never invents data it can't derive; anything it can't fix safely comes
back in `unfixable` for a human. That contract is what lets you put it in a
pipeline.

---

## Layer 2 — kinematics

With a trusted model, do the geometry. **Forward kinematics** maps joint values
to a world transform per link:

```python
from fieldpilot_urdf import forward_kinematics

poses = forward_kinematics(robot, {"shoulder_pan_joint": 0.5})  # {link: 4x4}
tool_xyz = poses["tool0"][:3, 3]
```

**Inverse kinematics** is numerical and joint-limit-aware (`scipy` least-squares
under the hood). It returns a result object, not just an answer — you always
know whether it actually converged:

```python
from fieldpilot_urdf import solve_ik

ik = solve_ik(robot, "tool0", target_xyz=(0.4, 0.1, 0.5))
print(ik.converged, ik.position_error, ik.n_iter)   # honours <limit> bounds
if ik.converged:
    poses = forward_kinematics(robot, ik.q)
```

**Self-collision** is AABB-based out of the box, and mesh-accurate with the
`[mesh]` extra (`trimesh`, lazily imported):

```python
from fieldpilot_urdf import detect_self_collisions

detect_self_collisions(robot, q={"elbow_joint": 2.0})   # [(link_a, link_b), ...]
```

**Workspace and trajectory** sampling close out the layer — reachable envelope
and a collision/limit check along a path:

```python
from fieldpilot_urdf import sample_workspace, check_trajectory

ws = sample_workspace(robot, "tool0", n_samples=500, seed=0)
print(ws.reachable_count, ws.bbox_min, ws.bbox_max)

check_trajectory(robot, [q0, q1, q2])   # per-step limit + collision findings
```

---

## Layer 3 — dynamics and simulation

Geometry tells you *where* the robot can be; dynamics tells you *how it moves*.
This layer is optional — `pip install "fieldpilot-urdf[dynamics]"` — and lazily
imported, so the kinematics path never pulls in SymPy.

`SymbolicDynamics` builds the equations of motion for a **tree (serial) robot**
with Kane's method and hands you symbolic matrices plus a fast NumPy callable:

```python
from fieldpilot_urdf.dynamics import SymbolicDynamics

dyn = SymbolicDynamics(robot)
dyn.n_dof                       # actuated DOF
dyn.mass_matrix                 # symbolic M(q)
dyn.forcing                     # symbolic F(q, q̇, τ) = τ − C(q,q̇)q̇ − G(q)

# Forward dynamics as a NumPy callable, ready for scipy.integrate.solve_ivp:
fwd = dyn.lambdify_forward_dynamics()        # (q, u, tau) -> q̈   (solves M·q̈ = F)
qdd = fwd(q, qdot, tau)
```

Joint-origin frames follow URDF's `Rz(yaw)·Ry(pitch)·Rx(roll)` convention, so
`dyn.link_pose(link, q)` matches `forward_kinematics` to machine precision.
Closed-loop mechanisms and multi-DOF joints (`floating`/`planar`/`spherical`)
raise `UnsupportedSystemError` — but closed loops have their own path:

```python
from fieldpilot_urdf import LoopClosure, FrameRef
from fieldpilot_urdf.constrained import constrained_dynamics

# A Robot carrying loop closures → constrained (Lagrange-multiplier) dynamics,
# with Baumgarte stabilisation + projection to stay on the constraint manifold.
cdyn = constrained_dynamics(robot_with_loops)
```

For **numerical** simulation, `[sim]` wraps PyBullet. The import pipeline feeds
it directly — `package://` mesh URIs are rewritten to the files `fetch_meshes`
downloaded — and it loads with `URDF_USE_INERTIA_FROM_FILE`, so its free-fall
dynamics match the symbolic model to ~1e-5:

```python
from fieldpilot_urdf.sim import PyBulletSim

with PyBulletSim(robot) as sim:       # mesh_dir=... for a mesh robot
    sim.free()                        # disable motors → pure gravity/inertia
    sim.step(240)                     # 1 s at 240 Hz
    print(sim.joint_states())         # {joint: (pos, vel)}
```

This wrapper is deliberately thin — load, step, control, read state. For richer
work, point PyBullet / MuJoCo / Drake straight at the URDF this package imports.

---

## Layer 4 — diagnostics

This is the layer the rest of the ladder exists to serve, and the core of
FieldPilot's MDG (robot-diagnostics) assistant. It answers: *a robot is
misbehaving — which joint is the culprit, and can we prove it?*

**Localise first.** Given the links a technician reports as dead, rank the
joints whose downstream subtree best explains the observation — pure NetworkX
graph reasoning, no dependency beyond the core:

```python
from fieldpilot_urdf import rank_root_causes, affected_links, criticality

ranked = rank_root_causes(robot, ["wrist_3_link", "tool0"])
suspect = ranked[0].target                       # best precision×recall match
affected_links(robot, suspect)                   # everything it drags down
criticality(robot, suspect)                      # 0–1, mass-weighted impact
```

**Then prove it.** Don't trust the ranking — inject the hypothesised fault on a
*copy* of the robot, re-run the relevant analysis, and check it reproduces the
symptom. That's `diagnose`, a two-tier hypothesis-and-test loop:

```python
from fieldpilot_urdf import diagnose, Symptom, Hypothesis

report = diagnose(
    robot,
    Symptom(kind="cant_reach", target_link="tool0", target_xyz=(0.4, 0.1, 0.5)),
    [Hypothesis(suspect_joint=suspect, fault_mode="motor_dead")],
)
print(report.verdict, report.confidence)   # CONFIRMED / REFUTED / INCONCLUSIVE
print(report.summary)                       # human-readable reasoning
```

The verdict is grounded: `CONFIRMED` means the target was reachable on the
healthy robot and became unreachable once the fault was injected. `INCONCLUSIVE`
means the symptom couldn't even be reproduced on the healthy model (e.g. the
target was never reachable) — so the fault can't be blamed. No hand-waving.

`rank_root_causes` → `diagnose` chains cleanly: the top-ranked suspect becomes
the hypothesis you test. See [`examples/ducky_diagnosis.py`](../examples/ducky_diagnosis.py)
for the full engineer↔assistant transcript.

---

## Where FieldPilot SaaS picks up

This package is the **deterministic core**. The natural-language layer — describe
a symptom in free text or by voice, and an LLM turns it into the `Symptom` /
`Hypothesis` objects above, then narrates the verdict — lives in
[FieldPilot](https://github.com/DuQuatre) SaaS, along with the 13-tool robot
chat, the spare-parts BOM, and multi-tenant hosting. Everything in *this*
tutorial stays local, offline-capable, and dependency-light.
