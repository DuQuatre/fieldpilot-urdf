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
