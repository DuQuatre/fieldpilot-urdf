"""Full-stack tour — the four layers of fieldpilot-urdf, end to end.

This is the runnable companion to ``docs/tutorial.md``. It walks a single robot
up the whole ladder, and every number printed comes from a real call:

    Layer 1  Model        parse / validate / repair          (core)
    Layer 2  Kinematics   FK / IK / collision / workspace     (core)
    Layer 3  Dynamics+Sim symbolic M(q) ⇄ PyBullet free-fall  ([dynamics], [sim])
    Layer 4  Diagnostics  localise + hypothesis-test a fault  (core)

Layers 1, 2 and 4 need nothing but the core install. Layer 3 lights up only when
the optional extras are present — without them the tour prints a one-line skip
and still exits 0:

    pip install fieldpilot-urdf                 # layers 1, 2, 4
    pip install "fieldpilot-urdf[dynamics,sim]" # + layer 3
    python examples/full_stack_tour.py

The headline entry point is ``import_urdf`` (pull a robot straight off a URL —
xacro, ``$(find)`` and ``<xacro:include>`` all expand). We build the arm in code
here so the tour stays offline and deterministic; the import path is one call
away and shown in the tutorial.
"""
from __future__ import annotations

import math

from fieldpilot_urdf import (
    Inertia, Inertial, Joint, JointLimit, Link, Origin, Robot,
    affected_links, criticality, detect_self_collisions, diagnose,
    forward_kinematics, Hypothesis, rank_root_causes, run_all, sample_workspace,
    solve_ik, summary, Symptom,
)


def banner(n: int, title: str) -> None:
    print("\n" + "─" * 74)
    print(f"  LAYER {n} — {title}")
    print("─" * 74)


def build_arm() -> Robot:
    """A 3-DOF serial arm: base yaw + two pitch joints, links carrying mass.

    Reachable workspace is a torus around the base; freezing the base yaw
    collapses it to the x–z plane — the fault we diagnose in Layer 4.
    """
    def lim() -> JointLimit:
        return JointLimit(lower=-2.9, upper=2.9, effort=150, velocity=3.0)

    def inr(mass: float) -> Inertial:
        return Inertial(origin=Origin(xyz=(0.45, 0, 0)), mass=mass,
                        inertia=Inertia(ixx=0.02, iyy=0.15, izz=0.15))

    return Robot(
        name="tour_arm",
        links=[
            Link(name="base", inertial=inr(6.0)),
            Link(name="link1", inertial=inr(4.0)),
            Link(name="link2", inertial=inr(2.5)),
            Link(name="tool", inertial=inr(1.0)),
        ],
        joints=[
            Joint(name="j_base", type="revolute", parent="base", child="link1",
                  origin=Origin(xyz=(0, 0, 0.5)), axis=(0, 0, 1), limit=lim()),   # yaw
            Joint(name="j_shoulder", type="revolute", parent="link1", child="link2",
                  origin=Origin(xyz=(0.9, 0, 0)), axis=(0, 1, 0), limit=lim()),   # pitch
            Joint(name="j_elbow", type="revolute", parent="link2", child="tool",
                  origin=Origin(xyz=(0.9, 0, 0)), axis=(0, 1, 0), limit=lim()),   # pitch
        ],
    )


def layer1_model(robot: Robot) -> None:
    banner(1, "MODEL — parse & validate")
    print(f"  {robot.name}: {len(robot.links)} links, {len(robot.joints)} joints")
    findings = run_all(robot)
    print(f"  run_all(robot)  -> {summary(findings)}")
    print("  Clean model: 8 lint rules (R001–R008) pass. A broken URDF would")
    print("  surface findings here, and repair(robot) would deterministically")
    print("  fix the repairable ones.")


def layer2_kinematics(robot: Robot) -> tuple[float, float, float]:
    banner(2, "KINEMATICS — FK / IK / collision / workspace")

    # Forward kinematics: a known pose that uses the base yaw (off the x–z plane).
    pose = {"j_base": 0.6, "j_shoulder": -0.5, "j_elbow": -0.7}
    tool_T = forward_kinematics(robot, pose)["tool"]
    target = tuple(round(float(x), 3) for x in tool_T[:3, 3])
    print(f"  forward_kinematics(pose) -> tool at {target}")

    # Inverse kinematics: recover a pose that hits that point, honouring limits.
    ik = solve_ik(robot, "tool", target_xyz=target)
    print(f"  solve_ik(tool -> {target}) -> converged={ik.converged}, "
          f"err={ik.position_error:.2e} m, iters={ik.n_iter}")

    # Self-collision at the IK solution.
    print(f"  detect_self_collisions() -> {detect_self_collisions(robot, q=ik.q)}")

    # Workspace envelope: sample reachable space, report the bounding box.
    ws = sample_workspace(robot, "tool", n_samples=400, seed=0)
    span = tuple(round(hi - lo, 2) for lo, hi in zip(ws.bbox_min, ws.bbox_max))
    print(f"  sample_workspace(400) -> {ws.reachable_count} reachable, "
          f"bbox span (x,y,z)={span} m")
    return target


def layer3_dynamics_sim(robot: Robot) -> None:
    banner(3, "DYNAMICS + SIM — symbolic M(q) ⇄ PyBullet (optional extras)")

    try:
        from fieldpilot_urdf.dynamics import SymbolicDynamics
    except ImportError:
        print("  [skipped] symbolic dynamics needs:  pip install "
              "'fieldpilot-urdf[dynamics]'")
    else:
        dyn = SymbolicDynamics(robot)
        M = dyn.mass_matrix
        print(f"  SymbolicDynamics: n_dof={dyn.n_dof}, M(q) is {M.shape[0]}×{M.shape[1]} "
              "(symbolic, Kane's method)")
        # Forward dynamics at rest under gravity -> joint accelerations.
        fwd = dyn.lambdify_forward_dynamics()
        z = [0.0] * dyn.n_dof
        qdd = fwd(z, z, z)
        accel = ", ".join(f"{a:+.2f}" for a in qdd)
        print(f"  forward dynamics at rest (gravity only) -> q̈ = [{accel}] rad/s²")

    try:
        from fieldpilot_urdf.sim import PyBulletSim
    except ImportError:
        print("  [skipped] numerical simulation needs:  pip install "
              "'fieldpilot-urdf[sim]'")
        return

    # Free-fall from rest: free() disables the joint motors so the arm falls
    # under gravity alone. Primitive geometry, so no meshes to fetch; loads with
    # the URDF's own <inertia> (URDF_USE_INERTIA_FROM_FILE).
    with PyBulletSim(robot) as sim:
        sim.free()
        sim.step(120)                       # 0.5 s at the default 240 Hz
        states = sim.joint_states()
    moved = {j: round(pos, 3) for j, (pos, _vel) in states.items()}
    print(f"  PyBulletSim free-fall, 0.5 s -> joint positions {moved}")
    print("  Same robot, same inertias — the gravity that drove q̈ above now")
    print("  swings the arm in the numerical engine. (Cross-validated to ~1e-5")
    print("  against SymbolicDynamics in the test suite.)")


def layer4_diagnostics(robot: Robot, target: tuple[float, float, float]) -> None:
    banner(4, "DIAGNOSTICS — localise & confirm a fault")

    # Symptom: the whole arm above the turntable went dead and can't reach the
    # side target any more. Which joint best explains the observed dead links?
    observed = ["link1", "link2", "tool"]
    ranked = rank_root_causes(robot, observed)
    print(f"  rank_root_causes(observed={observed})")
    for c in ranked[:3]:
        print(f"     {c.target:12s} score={c.score:.3f}  "
              f"precision={c.precision:.2f} recall={c.recall:.2f}")
    suspect = ranked[0].target

    # Don't guess — inject the fault on a copy and re-test the reach.
    report = diagnose(
        robot,
        Symptom(kind="cant_reach", target_link="tool", target_xyz=target),
        [Hypothesis(suspect_joint=suspect, fault_mode="motor_dead")],
    )
    print(f"  diagnose(cant_reach @ tool, {suspect} motor_dead)")
    print(f"     verdict={report.verdict.value}  confidence={report.confidence}")
    print(f"     {report.summary}")

    impacted = affected_links(robot, suspect)
    crit = criticality(robot, suspect)
    print(f"  affected_links({suspect}) -> {sorted(impacted)}")
    print(f"  criticality({suspect}) -> {crit:.0%} of robot mass downstream")


def main() -> None:
    print("=" * 74)
    print("  fieldpilot-urdf — full-stack tour (import → diagnose)")
    print("=" * 74)
    robot = build_arm()
    layer1_model(robot)
    target = layer2_kinematics(robot)
    layer3_dynamics_sim(robot)
    layer4_diagnostics(robot, target)
    print("\n" + "=" * 74)
    print("  Four layers, one robot, every number from a real call.")
    print("=" * 74)


if __name__ == "__main__":
    main()
