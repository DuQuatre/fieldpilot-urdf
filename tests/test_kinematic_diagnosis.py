"""Kinematic fault localization: recover a miscalibrated joint from an observed
end-effector pose deviation. The anchor is a round-trip — inject a known offset
on one joint, compute the resulting observed pose, and confirm the localizer
names that joint with the right offset and a high explained fraction.
"""
from __future__ import annotations

import numpy as np

from fieldpilot_urdf.fk import R_to_rpy, forward_kinematics
from fieldpilot_urdf.kinematic_diagnosis import (
    JointFaultCandidate, localize_joint_fault,
)
from fieldpilot_urdf.models import Joint, JointLimit, Link, Origin, Robot


def _lim(lo=-3.0, hi=3.0):
    return JointLimit(lower=lo, upper=hi, effort=1.0, velocity=1.0)


def _planar_3r(l=1.0):
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
    axes = [(0, 0, 1), (0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0)]
    offs = [(0, 0, 0.3), (0, 0, 0.2), (0.5, 0, 0), (0.5, 0, 0), (0.1, 0, 0), (0.1, 0, 0)]
    links = [Link(name="base")] + [Link(name=f"l{i}") for i in range(1, 7)]
    joints, parent = [], "base"
    for i in range(6):
        child = f"l{i+1}"
        joints.append(Joint(name=f"j{i+1}", type="revolute", parent=parent,
                            child=child, origin=Origin(xyz=offs[i]), axis=axes[i],
                            limit=_lim()))
        parent = child
    return Robot(name="r6", links=links, joints=joints)


def _observed(robot, link, expected_q, faulted_joint, offset):
    """Pose of `link` when `faulted_joint` is off by `offset` from expected."""
    actual = dict(expected_q)
    actual[faulted_joint] = actual.get(faulted_joint, 0.0) + offset
    T = forward_kinematics(robot, actual)[link]
    return tuple(T[:3, 3]), R_to_rpy(T[:3, :3])


# --- the anchor: round-trip recovery ----------------------------------------

def test_recovers_injected_offset_small():
    r = _planar_3r()
    expected = {"j1": 0.3, "j2": -0.5, "j3": 0.4}
    xyz, rpy = _observed(r, "tool", expected, "j2", 0.02)   # small offset on j2
    cands = localize_joint_fault(r, "tool", expected, xyz, rpy)
    top = cands[0]
    assert top.joint == "j2"
    assert np.isclose(top.estimated_offset, 0.02, atol=2e-3)
    assert top.explained_fraction > 0.99                    # small -> nearly exact
    assert top.residual_position < 1e-3


def test_each_joint_is_recovered_in_turn():
    r = _spatial_6r()
    expected = {"j1": 0.3, "j2": -0.6, "j3": 0.8, "j4": 0.2, "j5": 0.7, "j6": 0.1}
    for jt in ["j1", "j2", "j3", "j4", "j5", "j6"]:
        xyz, rpy = _observed(r, "l6", expected, jt, 0.015)
        cands = localize_joint_fault(r, "l6", expected, xyz, rpy)
        assert cands[0].joint == jt, f"{jt} not top: {[c.joint for c in cands]}"
        assert np.isclose(cands[0].estimated_offset, 0.015, atol=3e-3)


def test_larger_offset_still_ranks_first_but_explains_less():
    r = _planar_3r()
    expected = {"j1": 0.2, "j2": 0.3, "j3": -0.4}
    small = _observed(r, "tool", expected, "j1", 0.02)
    big = _observed(r, "tool", expected, "j1", 0.5)          # well outside the linear regime
    c_small = localize_joint_fault(r, "tool", expected, *small)
    c_big = localize_joint_fault(r, "tool", expected, *big)
    assert c_small[0].joint == "j1" and c_big[0].joint == "j1"
    # linearization degrades with offset magnitude
    assert c_small[0].explained_fraction > c_big[0].explained_fraction
    assert c_big[0].explained_fraction < 1.0


# --- behaviour & edges ------------------------------------------------------

def test_no_deviation_returns_empty():
    r = _planar_3r()
    expected = {"j1": 0.3, "j2": -0.5, "j3": 0.4}
    T = forward_kinematics(r, expected)["tool"]
    cands = localize_joint_fault(r, "tool", expected,
                                 tuple(T[:3, 3]), R_to_rpy(T[:3, :3]))
    assert cands == []


def test_position_only_when_rpy_omitted():
    r = _planar_3r()
    expected = {"j1": 0.3, "j2": -0.5, "j3": 0.4}
    xyz, _ = _observed(r, "tool", expected, "j3", 0.02)
    cands = localize_joint_fault(r, "tool", expected, xyz)   # no observed_rpy
    assert cands[0].joint == "j3"
    assert cands[0].residual_orientation == 0.0 or cands[0].residual_orientation < 1e-6


def test_prismatic_offset_units_recovered():
    r = Robot(name="p",
              links=[Link(name="base"), Link(name="mid"), Link(name="tool")],
              joints=[
                  Joint(name="s", type="prismatic", parent="base", child="mid",
                        origin=Origin(xyz=(0, 0, 0)), axis=(1, 0, 0), limit=_lim(-1, 3)),
                  Joint(name="r", type="revolute", parent="mid", child="tool",
                        origin=Origin(xyz=(0.5, 0, 0)), axis=(0, 0, 1), limit=_lim()),
              ])
    expected = {"s": 0.4, "r": 0.3}
    xyz, rpy = _observed(r, "tool", expected, "s", 0.05)     # 5 cm slide error
    cands = localize_joint_fault(r, "tool", expected, xyz, rpy)
    assert cands[0].joint == "s"
    assert np.isclose(cands[0].estimated_offset, 0.05, atol=1e-3)   # metres


def test_filters_and_caps():
    r = _spatial_6r()
    expected = {f"j{i+1}": 0.2 for i in range(6)}
    xyz, rpy = _observed(r, "l6", expected, "j3", 0.02)
    all_c = localize_joint_fault(r, "l6", expected, xyz, rpy)
    assert len(all_c) == 6 and all(isinstance(c, JointFaultCandidate) for c in all_c)
    # sorted best-first
    fr = [c.explained_fraction for c in all_c]
    assert fr == sorted(fr, reverse=True)
    capped = localize_joint_fault(r, "l6", expected, xyz, rpy, max_candidates=2)
    assert len(capped) == 2 and capped[0].joint == all_c[0].joint
    strong = localize_joint_fault(r, "l6", expected, xyz, rpy, min_explained=0.9)
    assert all(c.explained_fraction >= 0.9 for c in strong)
    assert strong[0].joint == "j3"


def test_fixed_chain_returns_empty():
    r = Robot(name="f", links=[Link(name="base"), Link(name="tip")],
              joints=[Joint(name="f", type="fixed", parent="base", child="tip",
                            origin=Origin(xyz=(1, 0, 0)))])
    assert localize_joint_fault(r, "tip", {}, (1.5, 0.0, 0.0)) == []
