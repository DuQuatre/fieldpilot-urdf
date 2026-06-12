"""Tests for app/urdf/trajectory.py.

Run: python3 -m pytest app/urdf/test_trajectory.py -q  (from pydexpi-server/)
"""
from __future__ import annotations

import math

import pytest

from fieldpilot_urdf import from_xml
from fieldpilot_urdf.trajectory import check_trajectory, sample_workspace, trajectory_summary


# Three-link arm: prismatic j_slide along z (range 0–0.5), revolute j_yaw
# (range -pi..pi), then a fixed `tool_tip` offset (0.3, 0, 0) from gripper so
# the tip is NOT on the axis of rotation — that's what gives it a non-trivial
# workspace under j_yaw.
ARM = """\
<robot name="arm">
  <link name="base"/>
  <link name="lift"/>
  <link name="gripper"/>
  <link name="tool_tip"/>
  <joint name="j_slide" type="prismatic">
    <parent link="base"/><child link="lift"/>
    <axis xyz="0 0 1"/><origin xyz="0 0 0"/>
    <limit lower="0" upper="0.5" effort="20" velocity="0.1"/>
  </joint>
  <joint name="j_yaw" type="revolute">
    <parent link="lift"/><child link="gripper"/>
    <axis xyz="0 0 1"/><origin xyz="0 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  </joint>
  <joint name="j_tip" type="fixed">
    <parent link="gripper"/><child link="tool_tip"/>
    <origin xyz="0.3 0 0"/>
  </joint>
</robot>
"""


@pytest.fixture
def arm():
    return from_xml(ARM)


# --- check_trajectory ------------------------------------------------------

def test_trajectory_clean(arm):
    qs = [{"j_slide": 0.1, "j_yaw": 0.0},
          {"j_slide": 0.2, "j_yaw": 0.5},
          {"j_slide": 0.3, "j_yaw": -1.0}]
    assert check_trajectory(arm, qs) == []


def test_trajectory_flags_over_upper(arm):
    qs = [{"j_slide": 0.4},
          {"j_slide": 0.6}]  # > upper=0.5
    findings = check_trajectory(arm, qs)
    assert len(findings) == 1
    assert findings[0].step == 1
    assert findings[0].code == "limit"
    assert "j_slide" in findings[0].refs


def test_trajectory_flags_below_lower(arm):
    qs = [{"j_slide": -0.1}]
    findings = check_trajectory(arm, qs)
    assert findings and findings[0].detail.startswith("joint 'j_slide'")


def test_trajectory_unknown_joint(arm):
    findings = check_trajectory(arm, [{"j_ghost": 0.0}])
    assert findings and "unknown joint" in findings[0].detail


def test_trajectory_continuous_no_bounds():
    """continuous joints accept any value (URDF convention)."""
    cont = from_xml("""<robot name="r">
      <link name="a"/><link name="b"/>
      <joint name="j" type="continuous">
        <parent link="a"/><child link="b"/><axis xyz="0 0 1"/>
      </joint>
    </robot>""")
    assert check_trajectory(cont, [{"j": 1e6}, {"j": -1e6}]) == []


def test_trajectory_summary_shape(arm):
    qs = [{"j_slide": 0.2}, {"j_slide": 0.6}, {"j_slide": 0.7}]
    findings = check_trajectory(arm, qs)
    s = trajectory_summary(findings, n_steps=len(qs))
    assert s["n_steps"] == 3
    assert s["steps_with_issues"] == 2
    assert s["limit_violations"] == 2
    assert s["collisions"] == 0
    assert s["first_bad_step"] == 1


def test_trajectory_summary_empty():
    s = trajectory_summary([], n_steps=5)
    assert s["steps_with_issues"] == 0
    assert s["first_bad_step"] is None


# --- sample_workspace ------------------------------------------------------

def test_workspace_target_link_unknown(arm):
    with pytest.raises(KeyError):
        sample_workspace(arm, target_link="ghost", n_samples=10, seed=0)


def test_workspace_bbox_covers_expected_reach(arm):
    """tool_tip at offset (0.3, 0, 0) from gripper which rotates with j_yaw
    ∈ [-π, π], lifted by j_slide ∈ [0, 0.5]. Reachable set:
      x, y ∈ [-0.3, 0.3]
      z    ∈ [0, 0.5]
    """
    res = sample_workspace(arm, target_link="tool_tip", n_samples=500, seed=1,
                           check_collisions=False)
    assert res.reachable_count == 500
    assert -0.31 <= res.bbox_min[0] <= -0.20
    assert 0.20 <= res.bbox_max[0] <= 0.31
    assert res.bbox_max[2] <= 0.51
    assert res.bbox_min[2] >= -0.01
    assert res.points == []  # default excluded


def test_workspace_points_included(arm):
    res = sample_workspace(arm, target_link="tool_tip", n_samples=20, seed=2,
                           include_points=True, check_collisions=False)
    assert len(res.points) == 20


def test_workspace_seed_reproducible(arm):
    a = sample_workspace(arm, target_link="tool_tip", n_samples=50, seed=42,
                         check_collisions=False)
    b = sample_workspace(arm, target_link="tool_tip", n_samples=50, seed=42,
                         check_collisions=False)
    assert a.bbox_min == b.bbox_min and a.bbox_max == b.bbox_max
    assert a.centroid == b.centroid


def test_workspace_centroid_within_bbox(arm):
    res = sample_workspace(arm, target_link="tool_tip", n_samples=100, seed=3,
                           check_collisions=False)
    for k in range(3):
        assert res.bbox_min[k] - 1e-9 <= res.centroid[k] <= res.bbox_max[k] + 1e-9


def test_workspace_collisions_counted():
    """Two box-equipped links overlapping at neutral but separating with j_slide.
    Half the random samples should self-collide on average — we just assert
    the field is populated.
    """
    overlap = from_xml("""<robot name="r">
      <link name="base">
        <collision><geometry><box size="0.4 0.4 0.4"/></geometry></collision>
      </link>
      <link name="end">
        <collision><geometry><box size="0.4 0.4 0.4"/></geometry></collision>
      </link>
      <link name="middle"/>
      <joint name="j_middle" type="fixed"><parent link="base"/><child link="middle"/></joint>
      <joint name="j_slide" type="prismatic">
        <parent link="middle"/><child link="end"/>
        <axis xyz="0 0 1"/>
        <limit lower="0" upper="0.5" effort="5" velocity="1"/>
      </joint>
    </robot>""")
    res = sample_workspace(overlap, target_link="end", n_samples=30, seed=7,
                           check_collisions=True)
    # base vs end is non-adjacent (separated by middle); they overlap at low j_slide.
    assert res.collision_count > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
