"""Loop-closure model + constraint-deriver tests.

The deriver needs SymPy (the [dynamics] extra); the model-validation tests
don't and run regardless.
"""
from __future__ import annotations

import numpy as np
import pytest

from fieldpilot_urdf.fk import forward_kinematics, origin_to_T
from fieldpilot_urdf.models import (
    FrameRef, Joint, JointLimit, Link, LoopClosure, Origin, Robot,
)


def _lim():
    return JointLimit(lower=-6.3, upper=6.3, effort=10, velocity=5)


# --- model + validation (no SymPy needed) ----------------------------------

def test_robot_defaults_to_no_loops():
    r = Robot(name="r", links=[Link(name="base")], joints=[])
    assert r.loops == []


def test_loop_rejects_unknown_link():
    with pytest.raises(ValueError):
        Robot(name="r",
              links=[Link(name="base"), Link(name="l1")],
              joints=[Joint(name="j", type="fixed", parent="base", child="l1")],
              loops=[LoopClosure(name="bad", a=FrameRef(link="l1"), b=FrameRef(link="ghost"))])


def test_loop_rejects_same_link():
    with pytest.raises(ValueError):
        Robot(name="r",
              links=[Link(name="base"), Link(name="l1")],
              joints=[Joint(name="j", type="fixed", parent="base", child="l1")],
              loops=[LoopClosure(name="bad", a=FrameRef(link="l1"), b=FrameRef(link="l1"))])


# --- deriver (needs SymPy) -------------------------------------------------

sympy = pytest.importorskip("sympy")  # noqa: F841

from fieldpilot_urdf.dynamics import SymbolicDynamics  # noqa: E402
from fieldpilot_urdf.loops import (  # noqa: E402
    derive_loop_constraints, lambdify_loop_residual, mobility,
)

# Planar parallelogram four-bar as a spanning tree + one point closure.
# Links: crank (len A), coupler (len B); rocker (len C) on the other ground pivot
# at x=D. With A=C and B=D it's a parallelogram, closed for any θ at q=(θ,-θ,θ).
_A, _B, _C, _D = 1.0, 2.0, 1.0, 2.0


def _parallelogram():
    return Robot(
        name="parallelogram",
        links=[Link(name="ground"), Link(name="crank"),
               Link(name="coupler"), Link(name="rocker")],
        joints=[
            Joint(name="j1", type="revolute", parent="ground", child="crank",
                  origin=Origin(xyz=(0, 0, 0)), axis=(0, 0, 1), limit=_lim()),
            Joint(name="j2", type="revolute", parent="crank", child="coupler",
                  origin=Origin(xyz=(_A, 0, 0)), axis=(0, 0, 1), limit=_lim()),
            Joint(name="j3", type="revolute", parent="ground", child="rocker",
                  origin=Origin(xyz=(_D, 0, 0)), axis=(0, 0, 1), limit=_lim()),
        ],
        loops=[LoopClosure(
            name="close", kind="point",
            a=FrameRef(link="coupler", origin=Origin(xyz=(_B, 0, 0))),
            b=FrameRef(link="rocker", origin=Origin(xyz=(_C, 0, 0))),
        )],
    )


def _q_vec(dyn, mapping):
    return [mapping[jid] for jid in dyn.actuated_joint_ids]


def test_point_closure_has_three_constraints():
    dyn = SymbolicDynamics(_parallelogram())
    assert len(derive_loop_constraints(dyn)) == 3


def test_residual_zero_on_closed_family():
    robot = _parallelogram()
    dyn = SymbolicDynamics(robot)
    residual = lambdify_loop_residual(dyn)
    for theta in (0.0, 0.5, -0.9):
        q = _q_vec(dyn, {"j1": theta, "j2": -theta, "j3": theta})
        assert np.allclose(residual(q), 0.0, atol=1e-9), theta


def test_residual_nonzero_off_manifold():
    dyn = SymbolicDynamics(_parallelogram())
    residual = lambdify_loop_residual(dyn)
    q = _q_vec(dyn, {"j1": 0.5, "j2": 0.0, "j3": 0.0})  # not a closed config
    assert np.linalg.norm(residual(q)) > 1e-3


def test_mobility_is_one():
    # A four-bar has exactly 1 DOF: 3 tree joints − rank-2 constraint set.
    dyn = SymbolicDynamics(_parallelogram())
    assert mobility(dyn) == 1


def test_frameref_world_matches_fk():
    # The deriver's symbolic frame pose must agree with numeric FK + offset.
    from fieldpilot_urdf.loops import _frameref_world
    robot = _parallelogram()
    dyn = SymbolicDynamics(robot)
    ref = FrameRef(link="coupler", origin=Origin(xyz=(_B, 0, 0)))
    theta = 0.5
    q = {"j1": theta, "j2": -theta, "j3": theta}
    qv = _q_vec(dyn, q)
    R_sym, p_sym = _frameref_world(dyn, ref)
    subs = {qsym: float(v) for qsym, v in zip(dyn.q, qv)}
    p_dyn = np.array(p_sym.subs(subs).evalf(), dtype=float).ravel()

    tf = forward_kinematics(robot, q=q)
    p_fk = (tf["coupler"] @ origin_to_T(ref.origin))[:3, 3]
    assert np.allclose(p_dyn, p_fk, atol=1e-9)


def test_fixed_closure_has_six_constraints():
    robot = _parallelogram()
    robot.loops[0].kind = "fixed"
    dyn = SymbolicDynamics(robot)
    assert len(derive_loop_constraints(dyn)) == 6
