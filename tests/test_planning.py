"""Tests for src/fieldpilot_urdf/planning.py (RRT-Connect motion planning).

Run: python3 -m pytest tests/test_planning.py -q
"""
from __future__ import annotations

import math

import pytest

from fieldpilot_urdf import from_xml
from fieldpilot_urdf.planning import (
    path_length, plan_path, shorten_path,
)
from fieldpilot_urdf.trajectory import check_trajectory


# Two-DOF arm, no collision geometry — pure C-space, so any straight line is
# free. Good for the trivial-connect and bounds paths.
FREE_ARM = """\
<robot name="free_arm">
  <link name="base"/>
  <link name="l1"/>
  <link name="l2"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <axis xyz="0 0 1"/><origin xyz="0 0 0"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="1"/>
  </joint>
  <joint name="j2" type="prismatic">
    <parent link="l1"/><child link="l2"/>
    <axis xyz="1 0 0"/><origin xyz="0.2 0 0"/>
    <limit lower="0" upper="0.5" effort="10" velocity="1"/>
  </joint>
</robot>
"""


# A 2-DOF Cartesian slider: two prismatic joints move `end` (a 0.2 box) to
# position (jx, 0, jz). A fixed 0.2 box on `base` sits at (0.5, 0, 0.5).
# AABBs overlap iff jx ∈ [0.3, 0.7] AND jz ∈ [0.3, 0.7] — a square obstacle in
# the middle of the [0,1]² configuration space. So (0,0)→(1,1) is blocked on the
# diagonal but a detour around a corner (e.g. via (0,1)) is collision-free.
OBSTACLE = """\
<robot name="obstacle">
  <link name="base">
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry>
      <origin xyz="0.5 0 0.5"/></collision>
  </link>
  <link name="linkx"/>
  <link name="end">
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <joint name="jx" type="prismatic">
    <parent link="base"/><child link="linkx"/>
    <axis xyz="1 0 0"/><origin xyz="0 0 0"/>
    <limit lower="0" upper="1.0" effort="10" velocity="1"/>
  </joint>
  <joint name="jz" type="prismatic">
    <parent link="linkx"/><child link="end"/>
    <axis xyz="0 0 1"/><origin xyz="0 0 0"/>
    <limit lower="0" upper="1.0" effort="10" velocity="1"/>
  </joint>
</robot>
"""


@pytest.fixture
def free_arm():
    return from_xml(FREE_ARM)


@pytest.fixture
def obstacle():
    return from_xml(OBSTACLE)


# --- basic contract --------------------------------------------------------

def test_no_movable_joints_raises():
    static = from_xml("""<robot name="r">
      <link name="a"/><link name="b"/>
      <joint name="j" type="fixed"><parent link="a"/><child link="b"/></joint>
    </robot>""")
    with pytest.raises(ValueError):
        plan_path(static, {}, {})


def test_trivial_direct_connect(free_arm):
    """No obstacles → start and goal connect with a single free edge."""
    res = plan_path(free_arm, {"j1": -1.0, "j2": 0.0}, {"j1": 1.0, "j2": 0.4},
                    seed=0)
    assert res.success
    assert res.n_iter == 0
    assert res.n_waypoints == 2
    assert res.path[0] == {"j1": -1.0, "j2": 0.0}
    assert res.path[-1] == {"j1": 1.0, "j2": 0.4}


def test_path_endpoints_match_request(obstacle):
    res = plan_path(obstacle, {"jx": 0.0, "jz": 0.0}, {"jx": 1.0, "jz": 1.0}, seed=1)
    assert res.success
    assert res.path[0] == pytest.approx({"jx": 0.0, "jz": 0.0})
    assert res.path[-1] == pytest.approx({"jx": 1.0, "jz": 1.0})


def test_missing_joint_defaults_to_zero(free_arm):
    """Unspecified joints default to 0.0 (package-wide convention)."""
    res = plan_path(free_arm, {"j1": 0.5}, {"j1": -0.5}, seed=2)
    assert res.success
    assert res.path[0]["j2"] == 0.0 and res.path[-1]["j2"] == 0.0


# --- infeasible endpoints --------------------------------------------------

def test_start_out_of_limits(free_arm):
    res = plan_path(free_arm, {"j1": 5.0}, {"j1": 0.0}, seed=0)
    assert not res.success
    assert "start out of joint limits" in res.message
    assert "j1" in res.message
    assert res.path == []


def test_goal_out_of_limits(free_arm):
    res = plan_path(free_arm, {"j1": 0.0}, {"j2": 9.0}, seed=0)
    assert not res.success
    assert "goal out of joint limits" in res.message


def test_start_self_collides(obstacle):
    """(0.5, 0.5) is inside the central obstacle square → start infeasible."""
    res = plan_path(obstacle, {"jx": 0.5, "jz": 0.5}, {"jx": 1.0, "jz": 1.0}, seed=0)
    assert not res.success
    assert "start configuration self-collides" in res.message


def test_goal_self_collides(obstacle):
    res = plan_path(obstacle, {"jx": 1.0, "jz": 1.0}, {"jx": 0.5, "jz": 0.5}, seed=0)
    assert not res.success
    assert "goal configuration self-collides" in res.message


# --- collision avoidance ---------------------------------------------------

def test_planned_path_routes_around_obstacle(obstacle):
    """(0,0)→(1,1): the diagonal crosses the central obstacle, so the planner
    must detour around a corner. The result must clear check_trajectory and
    actually deviate from the straight diagonal."""
    res = plan_path(obstacle, {"jx": 0.0, "jz": 0.0}, {"jx": 1.0, "jz": 1.0},
                    step_size=0.05, seed=3)
    assert res.success
    # The returned waypoints themselves must be clean per check_trajectory.
    findings = check_trajectory(obstacle, res.path)
    assert [f for f in findings if f.code == "collision"] == []
    assert [f for f in findings if f.code == "limit"] == []
    # A detour means more than the trivial 2-point direct edge.
    assert res.n_waypoints > 2


def test_zero_iters_on_blocked_diagonal_fails(obstacle):
    """The direct edge is blocked and max_iters=0 means no tree growth → clean
    failure with an empty path, not a crash or a bogus route."""
    res = plan_path(obstacle, {"jx": 0.0, "jz": 0.0}, {"jx": 1.0, "jz": 1.0},
                    max_iters=0, step_size=0.05, seed=0)
    assert not res.success
    assert res.n_iter == 0
    assert res.path == []
    assert "no path found" in res.message


# --- continuous joints (wrap-around) ---------------------------------------

def test_continuous_joint_wraps_shortest_way():
    cont = from_xml("""<robot name="r">
      <link name="a"/><link name="b"/>
      <joint name="j" type="continuous">
        <parent link="a"/><child link="b"/><axis xyz="0 0 1"/>
      </joint>
    </robot>""")
    # 3.0 -> -3.0 is 6.0 the naive way but only ~0.28 across the wrap.
    res = plan_path(cont, {"j": 3.0}, {"j": -3.0}, seed=0)
    assert res.success
    assert path_length(cont, res.path) < 0.5  # took the short arc, not 6.0


# --- shortening / length ---------------------------------------------------

def test_shorten_path_not_longer(obstacle):
    raw = plan_path(obstacle, {"jx": 0.0, "jz": 0.0}, {"jx": 1.0, "jz": 1.0},
                    step_size=0.05, seed=4, smooth=False)
    assert raw.success
    short = shorten_path(obstacle, raw.path, step_size=0.05, seed=4)
    assert path_length(obstacle, short) <= path_length(obstacle, raw.path) + 1e-9
    # Endpoints preserved.
    assert short[0] == raw.path[0] and short[-1] == raw.path[-1]
    # Shortcuts must stay collision-free.
    assert [f for f in check_trajectory(obstacle, short) if f.code == "collision"] == []


def test_smooth_default_no_longer_than_raw(free_arm):
    # Free space: both are the direct edge, lengths equal.
    smooth = plan_path(free_arm, {"j1": -1.0}, {"j1": 1.0}, seed=5, smooth=True)
    raw = plan_path(free_arm, {"j1": -1.0}, {"j1": 1.0}, seed=5, smooth=False)
    assert path_length(free_arm, smooth.path) <= path_length(free_arm, raw.path) + 1e-9


def test_path_length_trivial():
    arm = from_xml(FREE_ARM)
    assert path_length(arm, []) == 0.0
    assert path_length(arm, [{"j1": 0.0, "j2": 0.0}]) == 0.0


def test_path_length_straight_line(free_arm):
    p = [{"j1": 0.0, "j2": 0.0}, {"j1": 1.0, "j2": 0.0}]
    assert path_length(free_arm, p) == pytest.approx(1.0)


# --- reproducibility -------------------------------------------------------

def test_seed_reproducible(obstacle):
    a = plan_path(obstacle, {"jx": 0.0, "jz": 0.0}, {"jx": 1.0, "jz": 1.0},
                  seed=11, step_size=0.05)
    b = plan_path(obstacle, {"jx": 0.0, "jz": 0.0}, {"jx": 1.0, "jz": 1.0},
                  seed=11, step_size=0.05)
    assert a.path == b.path
    assert a.n_iter == b.n_iter


# --- check_collisions=False escape hatch -----------------------------------

def test_collisions_disabled_connects_through_obstacle(obstacle):
    """With checking off, the diagonal through the obstacle connects directly."""
    res = plan_path(obstacle, {"jx": 0.0, "jz": 0.0}, {"jx": 1.0, "jz": 1.0},
                    check_collisions=False, seed=0)
    assert res.success
    assert res.n_waypoints == 2  # straight line, no detour


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
