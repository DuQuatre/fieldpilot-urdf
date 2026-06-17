"""End-to-end closed-loop dynamics simulation over time.

Builds a closed-loop mechanism as a spanning-tree URDF + a loop closure, derives
its constraints and Lagrangian, solves the constrained (Lagrange-multiplier)
dynamics, and integrates it forward — showing that DAE drift stabilization keeps
the mechanism on its constraint manifold.

The mechanism is a 4R spatial chain whose tip is pinned to a fixed point: the
three pin constraints are independent (full rank), leaving mobility 1, so gravity
drives an internal self-motion while the tip must stay put. We integrate three
ways and track the constraint residual ||c(q)|| over time:

    (a) unstabilized            — drifts off the manifold (index-3 DAE)
    (b) Baumgarte feedback      — drift damped each step, bounded
    (c) Baumgarte + projection  — held at ~machine precision

Run (needs the dynamics extra):

    pip install "fieldpilot-urdf[dynamics]"
    python examples/closed_loop_sim.py

Typical output (dt=2 ms, 1.5 s):

    method                        max ||c||    final ||c||
    (a) unstabilized              2.1e-02       2.1e-02
    (b) Baumgarte a=b=20          4.1e-03       2.5e-03
    (c) Baumgarte+projection      9.8e-13       1.9e-13
    max joint excursion: 1.43 rad   max tip drift: 9.9e-13 m
"""
from __future__ import annotations

import numpy as np

from fieldpilot_urdf.constrained import constrained_dynamics
from fieldpilot_urdf.fk import forward_kinematics, origin_to_T
from fieldpilot_urdf.models import (
    FrameRef, Inertia, Inertial, Joint, JointLimit, Link, LoopClosure, Origin, Robot,
)

TIP = Origin(xyz=(1, 0, 0))                          # tool point in the last link
QSTAR = {"j1": 0.3, "j2": 0.5, "j3": -0.8, "j4": 0.6}  # an assembled configuration


def _lim() -> JointLimit:
    return JointLimit(lower=-6.3, upper=6.3, effort=50, velocity=20)


def _inertial() -> Inertial:
    return Inertial(origin=Origin(xyz=(0.5, 0, 0)), mass=1.0,
                    inertia=Inertia(ixx=0.02, iyy=0.05, izz=0.05))


def build_robot() -> Robot:
    """A 4R spatial chain (base yaw + three parallel pitch joints) whose tip is
    pinned to the fixed world point it occupies at ``QSTAR`` — a mobility-1
    closed loop with full-rank constraints."""
    robot = Robot(
        name="4r_pinned",
        links=[Link(name="ground")] + [
            Link(name=f"l{i}", inertial=_inertial()) for i in (1, 2, 3, 4)
        ],
        joints=[
            Joint(name="j1", type="revolute", parent="ground", child="l1",
                  origin=Origin(xyz=(0, 0, 0)), axis=(0, 0, 1), limit=_lim()),
            Joint(name="j2", type="revolute", parent="l1", child="l2",
                  origin=Origin(xyz=(0, 0, 0)), axis=(0, 1, 0), limit=_lim()),
            Joint(name="j3", type="revolute", parent="l2", child="l3",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 1, 0), limit=_lim()),
            Joint(name="j4", type="revolute", parent="l3", child="l4",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 1, 0), limit=_lim()),
        ],
    )
    tip = (forward_kinematics(robot, q=QSTAR)["l4"] @ origin_to_T(TIP))[:3, 3]
    robot.loops = [LoopClosure(
        name="pin", kind="point",
        a=FrameRef(link="l4", origin=TIP),
        b=FrameRef(link="ground", origin=Origin(xyz=tuple(float(x) for x in tip))),
    )]
    return robot


def integrate(cd, q0, dt, steps, *, alpha=0.0, beta=0.0, project=False):
    """Semi-implicit Euler. Returns the per-step constraint residual norms."""
    fwd = cd.lambdify_forward_dynamics(alpha=alpha, beta=beta)
    residual = cd.lambdify_constraint_residual()
    q = np.array(q0, dtype=float)
    qd = np.zeros_like(q)
    drift = [float(np.linalg.norm(residual(q)))]
    for _ in range(steps):
        qdd, _ = fwd(q, qd)
        qd = qd + dt * qdd
        q = q + dt * qd
        if project:
            q, qd = cd.project(q, qd)
        drift.append(float(np.linalg.norm(residual(q))))
    return np.array(drift)


def main() -> None:
    robot = build_robot()
    cd = constrained_dynamics(robot, gravity=(0, 0, -9.81))
    order = cd.actuated_joint_ids
    q0 = [QSTAR[j] for j in order]

    A0 = cd.lambdify_constraint_jacobian()(q0)
    rank = int(np.linalg.matrix_rank(A0))
    print(f"mechanism: {cd.n_q} joints, {cd.n_constraints} constraints, "
          f"rank {rank} -> mobility {cd.n_q - rank}")
    print(f"initial residual: {np.linalg.norm(cd.lambdify_constraint_residual()(q0)):.2e}\n")

    dt, steps = 2e-3, 750
    print(f"semi-implicit Euler, dt={dt}, {steps} steps ({dt * steps:.1f}s)")
    print(f"{'method':28s}{'max ||c||':>12s}{'final ||c||':>13s}")
    runs = {
        "(a) unstabilized": dict(alpha=0.0, beta=0.0, project=False),
        "(b) Baumgarte a=b=20": dict(alpha=20.0, beta=20.0, project=False),
        "(c) Baumgarte+projection": dict(alpha=20.0, beta=20.0, project=True),
    }
    for name, kw in runs.items():
        d = integrate(cd, q0, dt, steps, **kw)
        print(f"{name:28s}{d.max():12.1e}{d[-1]:13.1e}")

    # Confirm it genuinely moves while the loop stays closed (stabilized run).
    fwd = cd.lambdify_forward_dynamics(alpha=20.0, beta=20.0)
    pin = np.array(robot.loops[0].b.origin.xyz, dtype=float)
    q = np.array(q0, dtype=float); qd = np.zeros_like(q)
    excursion = tip_drift = 0.0
    for _ in range(steps):
        qdd, _ = fwd(q, qd)
        qd = qd + dt * qdd; q = q + dt * qd
        q, qd = cd.project(q, qd)
        excursion = max(excursion, float(np.max(np.abs(q - q0))))
        tip = (forward_kinematics(robot, q=dict(zip(order, q)))["l4"] @ origin_to_T(TIP))[:3, 3]
        tip_drift = max(tip_drift, float(np.linalg.norm(tip - pin)))
    print(f"\nmax joint excursion: {excursion:.2f} rad   max tip drift: {tip_drift:.1e} m")


if __name__ == "__main__":
    main()
