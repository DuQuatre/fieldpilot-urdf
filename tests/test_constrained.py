"""Constrained (closed-loop) dynamics tests. Needs SymPy (the [dynamics] extra)."""
from __future__ import annotations

import numpy as np
import pytest

from fieldpilot_urdf.fk import forward_kinematics, origin_to_T
from fieldpilot_urdf.models import (
    FrameRef, Inertia, Inertial, Joint, JointLimit, Link, LoopClosure, Origin, Robot,
)

pytest.importorskip("sympy")  # noqa

from fieldpilot_urdf.dynamics import SymbolicDynamics  # noqa: E402
from fieldpilot_urdf.constrained import (  # noqa: E402
    ConstrainedDynamics, constrained_dynamics,
)


def _lim():
    return JointLimit(lower=-6.3, upper=6.3, effort=10, velocity=5)


def _inr(com=(0.5, 0, 0)):
    return Inertial(origin=Origin(xyz=com), mass=1.0,
                    inertia=Inertia(ixx=0.1, iyy=0.1, izz=0.1))


# --- no-loop reduction: must match the validated Kane forward dynamics ------

def _two_link_arm():
    return Robot(
        name="arm",
        links=[Link(name="base"), Link(name="l1", inertial=_inr()),
               Link(name="l2", inertial=_inr())],
        joints=[
            Joint(name="j1", type="revolute", parent="base", child="l1",
                  origin=Origin(xyz=(0, 0, 0)), axis=(0, 1, 0), limit=_lim()),
            Joint(name="j2", type="revolute", parent="l1", child="l2",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 1, 0), limit=_lim()),
        ],
    )


def test_no_loop_reduces_to_kane():
    robot = _two_link_arm()
    cd = constrained_dynamics(robot, gravity=(0, 0, -9.81))
    assert cd.n_constraints == 0
    fwd_c = cd.lambdify_forward_dynamics()

    dyn = SymbolicDynamics(robot, gravity=(0, 0, -9.81))
    fwd_k = dyn.lambdify_forward_dynamics()
    zero_tau = [0.0] * dyn.n_dof

    for qv, uv in [([0.3, -0.4], [0.5, 0.2]), ([1.0, 0.7], [-0.3, 0.6])]:
        qdd_c, lam = fwd_c(qv, uv)
        qdd_k = fwd_k(qv, uv, zero_tau)
        assert lam.shape == (0,)
        assert np.allclose(qdd_c, qdd_k, atol=1e-9), (qv, uv, qdd_c, qdd_k)


# --- closed loop: a spatial 3R chain pinned to a fixed point (0 mobility) ---

def _pinned_3r():
    return Robot(
        name="pinned",
        links=[Link(name="ground"), Link(name="l1", inertial=_inr()),
               Link(name="l2", inertial=_inr()), Link(name="l3", inertial=_inr())],
        joints=[
            # Classic 3-DOF positioner: base yaw (z) + two parallel pitch joints
            # (y). The planar 2R places the tip in a plane; the base rotates that
            # plane → the tip Jacobian is generically full rank 3.
            Joint(name="j1", type="revolute", parent="ground", child="l1",
                  origin=Origin(xyz=(0, 0, 0)), axis=(0, 0, 1), limit=_lim()),
            Joint(name="j2", type="revolute", parent="l1", child="l2",
                  origin=Origin(xyz=(0, 0, 0)), axis=(0, 1, 0), limit=_lim()),
            Joint(name="j3", type="revolute", parent="l2", child="l3",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 1, 0), limit=_lim()),
        ],
    )


_QSTAR = {"j1": 0.3, "j2": 0.5, "j3": -0.4}
_TIP = Origin(xyz=(1, 0, 0))   # tool point in l3's frame (off the pitch axis)


def _pinned_robot():
    """Pin l3's tip to the fixed world point it occupies at q*."""
    robot = _pinned_3r()
    tip_world = (forward_kinematics(robot, q=_QSTAR)["l3"] @ origin_to_T(_TIP))[:3, 3]
    robot.loops = [LoopClosure(
        name="pin", kind="point",
        a=FrameRef(link="l3", origin=_TIP),
        b=FrameRef(link="ground", origin=Origin(xyz=tuple(float(x) for x in tip_world))),
    )]
    return robot


def _qvec(cd):
    return [_QSTAR[jid] for jid in cd.actuated_joint_ids]


def test_constraint_shapes():
    cd = constrained_dynamics(_pinned_robot())
    assert cd.n_q == 3 and cd.n_constraints == 3
    # augmented (n+m) square system
    assert cd.mass_matrix.shape == (6, 6)
    assert cd.forcing.shape == (6, 1)


def test_residual_zero_at_assembled_config():
    cd = constrained_dynamics(_pinned_robot())
    r = cd.lambdify_constraint_residual()(_qvec(cd))
    assert np.allclose(r, 0.0, atol=1e-9)


def test_constraints_full_rank():
    cd = constrained_dynamics(_pinned_robot())
    A = cd.lambdify_constraint_jacobian()(_qvec(cd))
    assert np.linalg.matrix_rank(A) == 3   # 3 independent constraints → 0 mobility


def test_pinned_structure_has_zero_acceleration():
    # A fully-constrained (0-DOF) structure at rest cannot move: q̈ = 0, and the
    # multipliers λ carry the gravity reaction.
    cd = constrained_dynamics(_pinned_robot(), gravity=(0, 0, -9.81))
    qdd, lam = cd.lambdify_forward_dynamics()(_qvec(cd), [0.0, 0.0, 0.0])
    assert np.allclose(qdd, 0.0, atol=1e-7)
    assert lam.shape == (3,)
    assert np.linalg.norm(lam) > 1e-6   # gravity is actually being reacted


# --- DAE drift stabilization: Baumgarte + projection -----------------------

def test_baumgarte_feedback_rhs():
    """At an off-manifold state the stabilized acceleration must satisfy the
    Baumgarte ODE: A·(q̈_stab − q̈_free) = −2α(A q̇) − β²·c. Since the free solve
    enforces A·q̈_free = −Ȧ q̇, this is checkable without Ȧ."""
    cd = constrained_dynamics(_pinned_robot(), gravity=(0, 0, -9.81))
    A_fn, c_fn = cd.lambdify_constraint_jacobian(), cd.lambdify_constraint_residual()
    q = [v + 0.02 for v in _qvec(cd)]      # perturb off the manifold
    qd = np.array([0.1, -0.2, 0.3])
    A, c = A_fn(q), c_fn(q)
    assert np.linalg.norm(c) > 1e-3        # genuinely off-manifold

    alpha, beta = 8.0, 8.0
    qdd0, _ = cd.lambdify_forward_dynamics()(q, qd)
    qdd1, _ = cd.lambdify_forward_dynamics(alpha=alpha, beta=beta)(q, qd)
    lhs = A @ (qdd1 - qdd0)
    rhs = -2.0 * alpha * (A @ qd) - beta ** 2 * c
    assert np.allclose(lhs, rhs, atol=1e-7)


def test_baumgarte_noop_on_manifold():
    # On the manifold at rest the feedback terms vanish → same as unstabilized.
    cd = constrained_dynamics(_pinned_robot(), gravity=(0, 0, -9.81))
    q, qd = _qvec(cd), [0.0, 0.0, 0.0]
    a = cd.lambdify_forward_dynamics()(q, qd)[0]
    b = cd.lambdify_forward_dynamics(alpha=10.0, beta=10.0)(q, qd)[0]
    # Both are ≈ 0 (a 0-DOF structure at rest doesn't accelerate); the feedback
    # adds nothing meaningful on-manifold at rest.
    assert np.allclose(a, 0.0, atol=1e-9) and np.allclose(b, 0.0, atol=1e-9)


def _parallelogram():
    A, B, C, D = 1.0, 2.0, 1.0, 2.0
    inr = lambda: Inertial(mass=1.0, inertia=Inertia(ixx=0.05, iyy=0.05, izz=0.05))
    return Robot(
        name="parallelogram",
        links=[Link(name="ground"), Link(name="crank", inertial=inr()),
               Link(name="coupler", inertial=inr()), Link(name="rocker", inertial=inr())],
        joints=[
            Joint(name="j1", type="revolute", parent="ground", child="crank",
                  origin=Origin(xyz=(0, 0, 0)), axis=(0, 0, 1), limit=_lim()),
            Joint(name="j2", type="revolute", parent="crank", child="coupler",
                  origin=Origin(xyz=(A, 0, 0)), axis=(0, 0, 1), limit=_lim()),
            Joint(name="j3", type="revolute", parent="ground", child="rocker",
                  origin=Origin(xyz=(D, 0, 0)), axis=(0, 0, 1), limit=_lim()),
        ],
        loops=[LoopClosure(name="close", kind="point",
                           a=FrameRef(link="coupler", origin=Origin(xyz=(B, 0, 0))),
                           b=FrameRef(link="rocker", origin=Origin(xyz=(C, 0, 0))))],
    )


def test_projection_removes_position_drift():
    # Redundant planar 'point' closure (rank-2 of 3): pinv-based projection still
    # snaps a drifted config back onto the manifold.
    cd = constrained_dynamics(_parallelogram())
    res = cd.lambdify_constraint_residual()
    order = {j: i for i, j in enumerate(cd.actuated_joint_ids)}
    theta = 0.4
    closed = np.zeros(3)
    for jid, val in {"j1": theta, "j2": -theta, "j3": theta}.items():
        closed[order[jid]] = val
    drifted = closed + np.array([0.05, -0.03, 0.04])
    assert np.linalg.norm(res(drifted)) > 1e-2          # off the manifold

    q_proj, qd_proj = cd.project(drifted, np.zeros(3))
    assert np.linalg.norm(res(q_proj)) < 1e-9           # back on the manifold
    assert np.linalg.norm(q_proj - drifted) < 0.2       # minimal correction


def test_projection_removes_velocity_drift():
    cd = constrained_dynamics(_pinned_robot())
    q = _qvec(cd)
    _, qd_proj = cd.project(q, [0.5, -0.3, 0.7])
    A = cd.lambdify_constraint_jacobian()(q)
    assert np.linalg.norm(A @ qd_proj) < 1e-9           # velocity now tangent


def test_projection_noop_without_constraints():
    cd = constrained_dynamics(_two_link_arm())   # tree, no loops
    q, qd = [0.3, -0.4], [0.5, 0.2]
    qp, qdp = cd.project(q, qd)
    assert np.allclose(qp, q) and np.allclose(qdp, qd)
