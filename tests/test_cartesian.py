"""Cartesian (task-space) path planning: straight-line motion via a
resolved-rate servo over the geometric Jacobian. The anchor checks are that
every emitted config's end-effector lies on the planned straight line (verified
through FK) and that unreachable targets fail gracefully with partial progress.
"""
from __future__ import annotations

import numpy as np

from fieldpilot_urdf.cartesian import (
    CartesianPlanResult, interpolate_pose, plan_cartesian_path, pose_error,
)
from fieldpilot_urdf.fk import forward_kinematics, rpy_to_R
from fieldpilot_urdf.models import Joint, JointLimit, Link, Origin, Robot
from fieldpilot_urdf.trajectory import check_trajectory


def _lim(lo=-3.0, hi=3.0):
    return JointLimit(lower=lo, upper=hi, effort=1.0, velocity=1.0)


def _planar_3r(l=1.0):
    """Planar 3R about +Z, three unit links — redundant in the plane, so it can
    follow a 2-D line comfortably away from singularities."""
    links = [Link(name="base")] + [Link(name=f"l{i}") for i in range(1, 4)] + [Link(name="tool")]
    joints = [
        Joint(name="j1", type="revolute", parent="base", child="l1",
              origin=Origin(xyz=(0, 0, 0)), axis=(0, 0, 1), limit=_lim()),
        Joint(name="j2", type="revolute", parent="l1", child="l2",
              origin=Origin(xyz=(l, 0, 0)), axis=(0, 0, 1), limit=_lim()),
        Joint(name="j3", type="revolute", parent="l2", child="l3",
              origin=Origin(xyz=(l, 0, 0)), axis=(0, 0, 1), limit=_lim()),
        Joint(name="tip", type="fixed", parent="l3", child="tool",
              origin=Origin(xyz=(l, 0, 0))),
    ]
    return Robot(name="r3", links=links, joints=joints)


def _spatial_6r():
    """Anthropomorphic 6R with a spherical wrist and realistic link lengths —
    well-conditioned away from the stretched-arm / wrist singularities."""
    axes = [(0, 0, 1), (0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0)]
    offs = [(0, 0, 0.3), (0, 0, 0.2), (0.5, 0, 0), (0.5, 0, 0), (0.1, 0, 0), (0.1, 0, 0)]
    links = [Link(name="base")] + [Link(name=f"l{i}") for i in range(1, 7)]
    joints, parent = [], "base"
    for i in range(6):
        child = f"l{i+1}"
        joints.append(Joint(name=f"j{i+1}", type="revolute", parent=parent,
                            child=child, origin=Origin(xyz=offs[i]), axis=axes[i],
                            limit=_lim(-3.0, 3.0)))
        parent = child
    return Robot(name="r6", links=links, joints=joints)


# A dexterous, well-conditioned configuration (σ_min ≈ 0.17, cond ≈ 12) — clear
# of the stretched-arm singularity the all-equal-angle poses sit near.
_DEXTROUS_6R = {"j1": 0.3, "j2": -0.6, "j3": 0.8, "j4": 0.2, "j5": 0.7, "j6": 0.1}


# --- pose interpolation -----------------------------------------------------

def test_interpolate_pose_endpoints_and_midpoint():
    T0 = np.eye(4)
    T1 = np.eye(4); T1[:3, 3] = [1, 2, 3]; T1[:3, :3] = rpy_to_R((0.0, 0.0, 1.0))
    assert np.allclose(interpolate_pose(T0, T1, 0.0), T0)
    assert np.allclose(interpolate_pose(T0, T1, 1.0), T1)
    mid = interpolate_pose(T0, T1, 0.5)
    assert np.allclose(mid[:3, 3], [0.5, 1.0, 1.5])          # lerp translation
    assert np.allclose(mid[:3, :3], rpy_to_R((0.0, 0.0, 0.5)))  # slerp -> half angle


def test_interpolate_pose_is_geodesic_constant_speed():
    T0 = np.eye(4)
    T1 = np.eye(4); T1[:3, :3] = rpy_to_R((0.0, 0.0, 1.2))
    # equal s-steps -> equal angular steps along the geodesic
    angs = [pose_error(interpolate_pose(T0, T1, s), interpolate_pose(T0, T1, s + 0.1))[1]
            for s in (0.0, 0.3, 0.6, 0.8)]
    assert np.allclose(angs, angs[0], atol=1e-9)


def test_pose_error_zero_for_identical():
    T = np.eye(4); T[:3, 3] = [0.3, -0.2, 0.5]; T[:3, :3] = rpy_to_R((0.2, 0.3, -0.4))
    assert pose_error(T, T) == (0.0, 0.0)


# --- the anchor: every config lies on the straight line ---------------------

def _tcp(robot, q, link):
    return forward_kinematics(robot, q)[link][:3, 3]


def test_path_tcp_stays_on_straight_line():
    r = _planar_3r()
    start = {"j1": 0.4, "j2": 0.5, "j3": 0.3}
    p0 = _tcp(r, start, "tool")
    target = (p0[0] - 0.4, p0[1] - 0.3, 0.0)   # a reachable in-plane translation
    res = plan_cartesian_path(r, "tool", target, start_q=start, n_waypoints=24)
    assert res.success and res.reached_fraction == 1.0
    p1 = np.array(target)
    line = p1 - p0
    L = np.linalg.norm(line)
    for q in res.path:
        p = _tcp(r, q, "tool")
        t = np.dot(p - p0, line) / (L * L)             # projection parameter
        perp = np.linalg.norm((p - p0) - t * line)     # distance off the line
        assert perp < 5e-3, (t, perp)
    assert np.allclose(_tcp(r, res.path[-1], "tool"), p1, atol=1e-3)


def test_endpoints_match_start_and_target():
    r = _spatial_6r()
    start = dict(_DEXTROUS_6R)
    p0 = _tcp(r, start, "l6")
    target = tuple(p0 + np.array([0.05, -0.04, 0.03]))
    res = plan_cartesian_path(r, "l6", target, start_q=start, n_waypoints=20)
    assert res.success
    assert np.allclose(_tcp(r, res.path[0], "l6"), p0, atol=1e-9)
    assert np.allclose(_tcp(r, res.path[-1], "l6"), np.array(target), atol=1e-3)
    assert res.n_waypoints == len(res.path) == 21        # start + 20 waypoints


def test_orientation_target_is_reached():
    r = _spatial_6r()
    start = dict(_DEXTROUS_6R)
    tf0 = forward_kinematics(r, start)["l6"]
    from fieldpilot_urdf.fk import R_to_rpy
    rpy0 = R_to_rpy(tf0[:3, :3])
    target_xyz = tuple(tf0[:3, 3] + np.array([0.02, 0.0, -0.02]))
    target_rpy = (rpy0[0] + 0.15, rpy0[1], rpy0[2])      # small reorientation
    res = plan_cartesian_path(r, "l6", target_xyz, target_rpy, start_q=start)
    assert res.success
    Tf = forward_kinematics(r, res.path[-1])["l6"]
    pos, rot = pose_error(Tf, _goal_T(target_xyz, target_rpy))
    assert pos < 1e-3 and rot < 1e-3


def _goal_T(xyz, rpy):
    T = np.eye(4); T[:3, 3] = xyz; T[:3, :3] = rpy_to_R(rpy)
    return T


# --- limits / output integration / failure ---------------------------------

def test_path_respects_joint_limits_and_feeds_check_trajectory():
    r = _planar_3r()
    start = {"j1": 0.3, "j2": 0.4, "j3": 0.2}
    p0 = _tcp(r, start, "tool")
    target = (p0[0] - 0.3, p0[1] + 0.2, 0.0)
    res = plan_cartesian_path(r, "tool", target, start_q=start)
    assert res.success
    # The emitted path must pass the project's own trajectory validator clean.
    assert check_trajectory(r, res.path) == []


def test_unreachable_target_fails_with_partial_progress():
    r = _planar_3r(1.0)             # max reach = 3.0 from origin
    start = {"j1": 0.0, "j2": 0.0, "j3": 0.0}
    res = plan_cartesian_path(r, "tool", (10.0, 0.0, 0.0), start_q=start, n_waypoints=20)
    assert not res.success
    assert 0.0 <= res.reached_fraction < 1.0
    assert "out of reach" in res.message or "stalled" in res.message
    assert res.position_error > 1e-3
    # Whatever progress it made is still a valid, limit-respecting path.
    assert check_trajectory(r, res.path) == []


def test_pure_translation_holds_orientation():
    r = _spatial_6r()
    start = dict(_DEXTROUS_6R)
    R0 = forward_kinematics(r, start)["l6"][:3, :3]
    p0 = _tcp(r, start, "l6")
    res = plan_cartesian_path(r, "l6", tuple(p0 + np.array([0.03, 0.0, 0.0])), start_q=start)
    assert res.success
    Rf = forward_kinematics(r, res.path[-1])["l6"][:3, :3]
    assert pose_error(_wrapR(R0), _wrapR(Rf))[1] < 2e-3   # orientation preserved


def _wrapR(R):
    T = np.eye(4); T[:3, :3] = R
    return T


def test_fixed_chain_returns_gracefully():
    r = Robot(name="f", links=[Link(name="base"), Link(name="tip")],
              joints=[Joint(name="f", type="fixed", parent="base", child="tip",
                            origin=Origin(xyz=(1, 0, 0)))])
    # target == current pose -> trivially successful, no motion
    here = plan_cartesian_path(r, "tip", (1.0, 0.0, 0.0))
    assert here.success and here.reached_fraction == 1.0
    away = plan_cartesian_path(r, "tip", (2.0, 0.0, 0.0))
    assert not away.success and away.position_error > 0.5
    assert isinstance(here, CartesianPlanResult)
