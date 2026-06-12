"""FK + collisions tests. Run: python3 -m app.urdf.test_fk (from pydexpi-server/)."""
from __future__ import annotations

import numpy as np

from fieldpilot_urdf.collisions import aabb_overlap, detect_self_collisions, link_collision_aabbs
from fieldpilot_urdf.diagnostics import run_all
from fieldpilot_urdf.fk import forward_kinematics, joint_motion, origin_to_T, rotation_around_axis, rpy_to_R
from fieldpilot_urdf.models import (
    Box, Collision, Joint, JointLimit, Link, Origin, Robot, Sphere,
)


def _almost(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol)


# --- FK math ---------------------------------------------------------------

def test_rpy_zero_is_identity():
    assert _almost(rpy_to_R((0, 0, 0)), np.eye(3))


def test_rpy_yaw_90():
    R = rpy_to_R((0, 0, np.pi / 2))
    # +X should map to +Y
    assert _almost(R @ np.array([1, 0, 0]), np.array([0, 1, 0]))


def test_rodrigues_around_z():
    T = rotation_around_axis((0, 0, 1), np.pi / 2)
    assert _almost(T[:3, :3] @ np.array([1, 0, 0]), np.array([0, 1, 0]))


def test_origin_to_T_translation_only():
    T = origin_to_T(Origin(xyz=(1, 2, 3), rpy=(0, 0, 0)))
    assert _almost(T[:3, 3], np.array([1, 2, 3]))
    assert _almost(T[:3, :3], np.eye(3))


def test_joint_motion_neutral_is_identity():
    j = Joint(name="j", type="revolute", parent="a", child="b",
              axis=(0, 0, 1),
              limit=JointLimit(lower=-1, upper=1, effort=1, velocity=1))
    assert _almost(joint_motion(j, 0.0), np.eye(4))


def test_joint_motion_prismatic():
    j = Joint(name="j", type="prismatic", parent="a", child="b",
              axis=(1, 0, 0),
              limit=JointLimit(lower=-1, upper=1, effort=1, velocity=1))
    T = joint_motion(j, 0.5)
    assert _almost(T[:3, 3], np.array([0.5, 0, 0]))


# --- FK on a chain ---------------------------------------------------------

def _chain_robot():
    """base -[1m +x]-> l1 -[1m +x revolute about z]-> l2."""
    return Robot(
        name="chain",
        links=[Link(name="base"), Link(name="l1"), Link(name="l2")],
        joints=[
            Joint(name="j1", type="fixed", parent="base", child="l1",
                  origin=Origin(xyz=(1, 0, 0))),
            Joint(name="j2", type="revolute", parent="l1", child="l2",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 0, 1),
                  limit=JointLimit(lower=-3.14, upper=3.14, effort=5, velocity=1)),
        ],
    )


def test_fk_neutral_pose():
    tf = forward_kinematics(_chain_robot())
    assert _almost(tf["base"][:3, 3], np.array([0, 0, 0]))
    assert _almost(tf["l1"][:3, 3], np.array([1, 0, 0]))
    assert _almost(tf["l2"][:3, 3], np.array([2, 0, 0]))


def test_fk_with_joint_angle():
    # Bend j2 by +90deg around z: l2 origin should rotate around l1
    tf = forward_kinematics(_chain_robot(), q={"j2": np.pi / 2})
    # After j2's origin (+1 along l1's x), then no further translation -> still at l1 + (1,0,0)
    # The rotation affects orientation, not the origin offset (motion is applied AFTER origin)
    assert _almost(tf["l2"][:3, 3], np.array([2, 0, 0]))
    # But the x-axis of l2 should now point in +y
    assert _almost(tf["l2"][:3, :3] @ np.array([1, 0, 0]),
                   np.array([0, 1, 0]), tol=1e-9)


def test_fk_rejects_multi_root():
    r = Robot(name="x", links=[Link(name="a"), Link(name="b")], joints=[])
    try:
        forward_kinematics(r)
    except ValueError:
        return
    raise AssertionError("expected ValueError for multi-root tree")


# --- Collisions ------------------------------------------------------------

def _two_overlapping_boxes():
    # Two free-floating links would fail R001; instead: a base + two siblings
    # that share the base and have overlapping boxes (not adjacent to each other).
    return Robot(
        name="bug",
        links=[
            Link(name="base"),
            Link(name="armA",
                 collisions=[Collision(geometry=Box(size=(1, 1, 1)))]),
            Link(name="armB",
                 collisions=[Collision(geometry=Box(size=(1, 1, 1)))]),
        ],
        joints=[
            Joint(name="jA", type="fixed", parent="base", child="armA",
                  origin=Origin(xyz=(0, 0, 0))),
            Joint(name="jB", type="fixed", parent="base", child="armB",
                  origin=Origin(xyz=(0.5, 0, 0))),  # overlaps with armA
        ],
    )


def _two_disjoint_boxes():
    return Robot(
        name="ok",
        links=[
            Link(name="base"),
            Link(name="armA",
                 collisions=[Collision(geometry=Box(size=(1, 1, 1)))]),
            Link(name="armB",
                 collisions=[Collision(geometry=Box(size=(1, 1, 1)))]),
        ],
        joints=[
            Joint(name="jA", type="fixed", parent="base", child="armA",
                  origin=Origin(xyz=(0, 0, 0))),
            Joint(name="jB", type="fixed", parent="base", child="armB",
                  origin=Origin(xyz=(5, 0, 0))),  # 5m away, no overlap
        ],
    )


def test_aabb_overlap_basic():
    a = (np.array([0, 0, 0]), np.array([1, 1, 1]))
    b = (np.array([0.5, 0.5, 0.5]), np.array([2, 2, 2]))
    c = (np.array([2, 2, 2]), np.array([3, 3, 3]))
    assert aabb_overlap(a, b)
    assert not aabb_overlap(a, c)


def test_self_collision_detected():
    hits = detect_self_collisions(_two_overlapping_boxes())
    assert ("armA", "armB") in hits or ("armB", "armA") in hits


def test_self_collision_clean():
    assert detect_self_collisions(_two_disjoint_boxes()) == []


def test_adjacent_links_ignored():
    # base + child with a small box at parent's frame -> AABBs would touch,
    # but they're joined by a joint and must be skipped.
    r = Robot(
        name="adj",
        links=[
            Link(name="base", collisions=[Collision(geometry=Box(size=(1, 1, 1)))]),
            Link(name="child", collisions=[Collision(geometry=Box(size=(1, 1, 1)))]),
        ],
        joints=[Joint(name="j", type="fixed", parent="base", child="child",
                      origin=Origin(xyz=(0.5, 0, 0)))],
    )
    assert detect_self_collisions(r) == []


def test_mesh_skipped_silently():
    # Mesh contributes no AABB; should not raise
    boxes = link_collision_aabbs(_chain_robot())
    assert boxes == {}  # _chain_robot has no <collision> elements


# --- Rule integration ------------------------------------------------------

def test_r007_zero_axis():
    r = Robot(
        name="x",
        links=[Link(name="a"), Link(name="b")],
        joints=[Joint(name="j", type="revolute", parent="a", child="b",
                      axis=(0, 0, 0),
                      limit=JointLimit(lower=-1, upper=1, effort=1, velocity=1))],
    )
    assert any(f.code == "R007" for f in run_all(r))


def test_r008_in_run_all():
    findings = run_all(_two_overlapping_boxes())
    assert any(f.code == "R008" for f in findings)


def test_r008_silent_on_clean():
    findings = run_all(_two_disjoint_boxes())
    assert all(f.code != "R008" for f in findings)


if __name__ == "__main__":
    test_rpy_zero_is_identity()
    test_rpy_yaw_90()
    test_rodrigues_around_z()
    test_origin_to_T_translation_only()
    test_joint_motion_neutral_is_identity()
    test_joint_motion_prismatic()
    test_fk_neutral_pose()
    test_fk_with_joint_angle()
    test_fk_rejects_multi_root()
    test_aabb_overlap_basic()
    test_self_collision_detected()
    test_self_collision_clean()
    test_adjacent_links_ignored()
    test_mesh_skipped_silently()
    test_r007_zero_axis()
    test_r008_in_run_all()
    test_r008_silent_on_clean()
    print("OK — 17 tests passed")
