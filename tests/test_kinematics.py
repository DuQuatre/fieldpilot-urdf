"""Velocity-kinematics tests: geometric Jacobian, forward velocity,
manipulability, singularity report. The anchor check is the Jacobian's linear
rows against a finite-difference of FK position; the rest builds on known
closed forms for a planar 2R arm.
"""
from __future__ import annotations

import numpy as np

from fieldpilot_urdf.fk import forward_kinematics
from fieldpilot_urdf.kinematics import (
    geometric_jacobian, jacobian_joints, joint_velocity_to_twist,
    manipulability, singularity_report,
)
from fieldpilot_urdf.models import Joint, JointLimit, Link, Origin, Robot


def _lim(lo=-3.0, hi=3.0):
    return JointLimit(lower=lo, upper=hi, effort=1.0, velocity=1.0)


def _planar_2r(l1=1.0, l2=1.0):
    """Two revolute joints about +Z, links along +X; a fixed tip carries the
    second link's length so the tool sits at radius l1+l2 (a real 2-link arm).
    Classic planar 2R: closed-form FK/Jacobian make this a strong oracle."""
    return Robot(
        name="rr",
        links=[Link(name="base"), Link(name="l1"), Link(name="l2"), Link(name="tool")],
        joints=[
            Joint(name="j1", type="revolute", parent="base", child="l1",
                  origin=Origin(xyz=(0, 0, 0)), axis=(0, 0, 1), limit=_lim()),
            Joint(name="j2", type="revolute", parent="l1", child="l2",
                  origin=Origin(xyz=(l1, 0, 0)), axis=(0, 0, 1), limit=_lim()),
            Joint(name="tip", type="fixed", parent="l2", child="tool",
                  origin=Origin(xyz=(l2, 0, 0))),
        ],
    )


def _spatial_6r():
    """A non-planar 6R chain with varied axes/offsets — a generic full-rank arm
    so the 6×6 Jacobian, manipulability and condition number are non-degenerate."""
    axes = [(0, 0, 1), (0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0)]
    offs = [(0, 0, 0.2), (0.1, 0, 0.3), (0.4, 0, 0), (0.3, 0, 0.05),
            (0.2, 0, 0), (0.1, 0.05, 0)]
    links = [Link(name="base")] + [Link(name=f"l{i}") for i in range(1, 7)]
    joints = []
    parent = "base"
    for i in range(6):
        child = f"l{i+1}"
        joints.append(Joint(name=f"j{i+1}", type="revolute", parent=parent,
                            child=child, origin=Origin(xyz=offs[i]), axis=axes[i],
                            limit=_lim(-3.0, 3.0)))
        parent = child
    return Robot(name="r6", links=links, joints=joints)


def _fd_position_jacobian(robot, q, link, joints, eps=1e-6):
    """Finite-difference of the link's world origin w.r.t. each joint angle."""
    cols = []
    for jn in joints:
        qp = dict(q); qp[jn] = q.get(jn, 0.0) + eps
        qm = dict(q); qm[jn] = q.get(jn, 0.0) - eps
        pp = forward_kinematics(robot, qp)[link][:3, 3]
        pm = forward_kinematics(robot, qm)[link][:3, 3]
        cols.append((pp - pm) / (2 * eps))
    return np.array(cols).T  # 3×n


# --- column ordering --------------------------------------------------------

def test_jacobian_joints_root_first_skips_fixed():
    r = _planar_2r()
    # insert a fixed wrist after the tool
    r.links.append(Link(name="flange"))
    r.joints.append(Joint(name="wf", type="fixed", parent="tool", child="flange",
                          origin=Origin(xyz=(0.2, 0, 0))))
    assert jacobian_joints(r, "flange") == ["j1", "j2"]
    assert jacobian_joints(r, "l1") == ["j1"]


# --- linear rows vs finite difference (the anchor) --------------------------

def test_jacobian_linear_matches_finite_difference():
    rng = np.random.default_rng(0)
    for robot, link in [(_planar_2r(), "tool"), (_spatial_6r(), "l6")]:
        cols = jacobian_joints(robot, link)
        for _ in range(20):
            q = {jn: float(rng.uniform(-2.5, 2.5)) for jn in cols}
            J = geometric_jacobian(robot, q, link)
            fd = _fd_position_jacobian(robot, q, link, cols)
            assert np.allclose(J[0:3], fd, atol=1e-6), (robot.name, link)


def test_jacobian_shape_and_empty_chain():
    r = _planar_2r()
    assert geometric_jacobian(r, {}, "tool").shape == (6, 2)
    # a link reached only through a fixed joint -> 6×0
    r2 = Robot(name="f", links=[Link(name="base"), Link(name="tip")],
               joints=[Joint(name="f", type="fixed", parent="base", child="tip",
                             origin=Origin(xyz=(1, 0, 0)))])
    J = geometric_jacobian(r2, {}, "tip")
    assert J.shape == (6, 0)


# --- closed-form planar 2R --------------------------------------------------

def test_planar_2r_jacobian_closed_form():
    r = _planar_2r(1.0, 1.0)
    q = {"j1": 0.3, "j2": 0.5}
    J = geometric_jacobian(r, q, "tool")
    s1, s12 = np.sin(0.3), np.sin(0.3 + 0.5)
    c1, c12 = np.cos(0.3), np.cos(0.3 + 0.5)
    # vx = -(s1 + s12) q̇1 - s12 q̇2 ; vy = (c1 + c12) q̇1 + c12 q̇2 ; both about +Z
    assert np.allclose(J[0:2, 0], [-(s1 + s12), (c1 + c12)], atol=1e-9)
    assert np.allclose(J[0:2, 1], [-s12, c12], atol=1e-9)
    assert np.allclose(J[5, :], [1.0, 1.0], atol=1e-9)  # wz = 1 per revolute about Z
    assert np.allclose(J[2:5, :], 0.0, atol=1e-9)        # no vz, wx, wy


def test_prismatic_column_is_translation_axis():
    r = Robot(name="p", links=[Link(name="base"), Link(name="slider")],
              joints=[Joint(name="s", type="prismatic", parent="base", child="slider",
                            origin=Origin(xyz=(0, 0, 0)), axis=(1, 0, 0), limit=_lim())])
    J = geometric_jacobian(r, {"s": 0.4}, "slider")
    assert np.allclose(J[:, 0], [1, 0, 0, 0, 0, 0], atol=1e-12)


# --- forward velocity -------------------------------------------------------

def test_joint_velocity_to_twist_matches_J_dot_qdot():
    r = _spatial_6r()
    cols = jacobian_joints(r, "l6")
    rng = np.random.default_rng(1)
    q = {jn: float(rng.uniform(-2, 2)) for jn in cols}
    qdot = {jn: float(rng.uniform(-1, 1)) for jn in cols}
    twist = joint_velocity_to_twist(r, q, qdot, "l6")
    J = geometric_jacobian(r, q, "l6", joints=cols)
    expect = J @ np.array([qdot[jn] for jn in cols])
    assert np.allclose(twist, expect, atol=1e-12)
    assert twist.shape == (6,)


def test_twist_linear_matches_fd_of_position_along_qdot():
    """v should equal d/dt of the tool position when joints move at qdot."""
    r = _planar_2r()
    q = {"j1": 0.2, "j2": -0.4}
    qdot = {"j1": 0.7, "j2": -0.3}
    eps = 1e-6
    qp = {k: q[k] + qdot[k] * eps for k in q}
    qm = {k: q[k] - qdot[k] * eps for k in q}
    v_fd = (forward_kinematics(r, qp)["tool"][:3, 3]
            - forward_kinematics(r, qm)["tool"][:3, 3]) / (2 * eps)
    v = joint_velocity_to_twist(r, q, qdot, "tool")[0:3]
    assert np.allclose(v, v_fd, atol=1e-6)


# --- manipulability + singularity ------------------------------------------

def test_manipulability_planar_2r_drops_to_zero_when_straight():
    r = _planar_2r(1.0, 1.0)
    # Positional (rows 0,1,2) measure: folded/elbow-bent is dexterous; straight
    # (j2=0) is fully extended -> positionally singular. The full 6-DoF measure
    # would NOT see this (the angular rows keep the columns independent).
    pos = dict(rows=(0, 1, 2))
    bent = manipulability(r, {"j1": 0.0, "j2": 1.2}, "tool", **pos)
    straight = manipulability(r, {"j1": 0.0, "j2": 0.0}, "tool", **pos)
    assert bent > 1e-3
    assert straight < 1e-9
    # Closed form for planar 2R positional manipulability: w = l1*l2*|sin(j2)|.
    assert np.isclose(bent, 1.0 * 1.0 * abs(np.sin(1.2)), rtol=1e-6)
    # The full 6-DoF Jacobian, by contrast, stays full-rank when straight:
    assert manipulability(r, {"j1": 0.0, "j2": 0.0}, "tool") > 1e-3


def test_singularity_report_flags_extended_arm():
    r = _planar_2r(1.0, 1.0)
    pos = dict(rows=(0, 1, 2))
    rep = singularity_report(r, {"j1": 0.4, "j2": 0.0}, "tool", **pos)
    assert rep.is_singular
    assert rep.sigma_min < 1e-4
    assert rep.condition_number > 1e3
    good = singularity_report(r, {"j1": 0.4, "j2": 1.0}, "tool", **pos)
    assert not good.is_singular
    assert np.isfinite(good.condition_number)
    assert len(good.singular_values) == 2  # svd of a 3×2 -> min(3,2)=2 values
    assert good.sigma_max >= good.sigma_min > 0


def test_singularity_report_fixed_chain():
    r = Robot(name="f", links=[Link(name="base"), Link(name="tip")],
              joints=[Joint(name="f", type="fixed", parent="base", child="tip",
                            origin=Origin(xyz=(1, 0, 0)))])
    rep = singularity_report(r, {}, "tip")
    assert rep.is_singular and rep.manipulability == 0.0
    assert rep.condition_number == float("inf")
    assert manipulability(r, {}, "tip") == 0.0


def test_full_rank_6r_is_nonsingular_and_manipulable():
    r = _spatial_6r()
    rng = np.random.default_rng(7)
    q = {f"j{i+1}": float(rng.uniform(-1.0, 1.0)) for i in range(6)}
    rep = singularity_report(r, q, "l6")
    assert not rep.is_singular
    assert rep.manipulability > 0
    # manipulability == product of singular values == sqrt(det(J Jᵀ)) for n>=6
    J = geometric_jacobian(r, q, "l6")
    assert np.isclose(rep.manipulability, np.sqrt(np.linalg.det(J @ J.T)), rtol=1e-6)
