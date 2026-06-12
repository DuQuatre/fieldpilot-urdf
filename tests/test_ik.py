"""Tests for app/urdf/ik.py.

Run: python3 -m pytest app/urdf/test_ik.py -q  (from pydexpi-server/)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from fieldpilot_urdf import from_xml
from fieldpilot_urdf.fk import forward_kinematics
from fieldpilot_urdf.ik import solve_ik


# Same 2-DOF + tool tip arm we used for trajectory tests.
ARM = """\
<robot name="arm">
  <link name="base"/>
  <link name="lift"/>
  <link name="gripper"/>
  <link name="tip"/>
  <joint name="j_slide" type="prismatic">
    <parent link="base"/><child link="lift"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="0.5" effort="20" velocity="0.1"/>
  </joint>
  <joint name="j_yaw" type="revolute">
    <parent link="lift"/><child link="gripper"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  </joint>
  <joint name="j_tip" type="fixed">
    <parent link="gripper"/><child link="tip"/>
    <origin xyz="0.3 0 0"/>
  </joint>
</robot>
"""


@pytest.fixture
def arm():
    return from_xml(ARM)


def _tip_position(arm, q):
    return forward_kinematics(arm, q=q)["tip"][:3, 3]


# --- position-only IK ------------------------------------------------------

def test_ik_converges_reachable(arm):
    target = (0.0, 0.3, 0.2)
    res = solve_ik(arm, target_link="tip", target_xyz=target)
    assert res.converged
    assert res.position_error < 1e-6
    pos = _tip_position(arm, res.q)
    assert np.allclose(pos, target, atol=1e-6)


def test_ik_unreachable_reports_residual(arm):
    """Target far outside the workspace: solver doesn't converge but the
    returned q is still feasible (within bounds) and the residual is finite."""
    res = solve_ik(arm, target_link="tip", target_xyz=(10.0, 0.0, 0.0))
    assert not res.converged
    assert res.position_error > 1.0
    # Returned q is feasible: j_slide ∈ [0, 0.5], j_yaw ∈ [-π, π]
    assert 0 <= res.q["j_slide"] <= 0.5
    assert -math.pi - 1e-6 <= res.q["j_yaw"] <= math.pi + 1e-6


def test_ik_respects_lower_bound(arm):
    """Target requires negative j_slide (z=-0.1); solver must clamp to lower
    bound and report a finite residual."""
    res = solve_ik(arm, target_link="tip", target_xyz=(0.3, 0.0, -0.1))
    assert not res.converged
    assert res.q["j_slide"] >= 0.0 - 1e-9
    assert res.position_error >= 0.1 - 1e-6


def test_ik_unknown_target_link(arm):
    with pytest.raises(KeyError):
        solve_ik(arm, target_link="ghost", target_xyz=(0, 0, 0))


def test_ik_qinit_steers_solution(arm):
    """Two valid IK solutions (q_yaw=+π/2 and q_yaw=-π/2 give the same point
    when target.y has matching sign — but for asymmetric targets the closer
    initial guess wins).
    """
    target = (0.3 * math.cos(0.6), 0.3 * math.sin(0.6), 0.1)
    a = solve_ik(arm, target_link="tip", target_xyz=target,
                 q_init={"j_yaw": 0.5})
    b = solve_ik(arm, target_link="tip", target_xyz=target,
                 q_init={"j_yaw": -2.5})
    # both reach the target (pos_err small)
    assert a.position_error < 1e-6 and b.position_error < 1e-6
    # a started near +0.6, b near -2.5: a's solution should be closer to 0.6
    assert abs(a.q["j_yaw"] - 0.6) < abs(b.q["j_yaw"] - 0.6)


# --- position + orientation ------------------------------------------------

def test_ik_position_and_orientation(arm):
    target = (0.0, 0.3, 0.2)
    target_rpy = (0.0, 0.0, math.pi / 2)
    res = solve_ik(arm, target_link="tip", target_xyz=target,
                   target_rpy=target_rpy)
    assert res.converged
    assert res.position_error < 1e-6
    assert res.orientation_error < 1e-6


def test_ik_inconsistent_position_orientation(arm):
    """For a 2-DoF arm the position and orientation aren't independent — only
    one solution simultaneously matches both. If we ask for an orientation
    inconsistent with the position, the solver compromises and the
    `converged` flag is False (high orientation residual)."""
    res = solve_ik(arm, target_link="tip", target_xyz=(0.3, 0.0, 0.0),
                   target_rpy=(0.0, 0.0, math.pi / 2))
    assert not res.converged
    # Either position or orientation residual remains substantial (we don't
    # care which — the point is the solver doesn't claim a clean solution).
    assert res.position_error > 0.05 or res.orientation_error > 0.05


# --- robot without optimisable joints --------------------------------------

def test_ik_no_joints_raises():
    r = from_xml('<robot name="r"><link name="a"/></robot>')
    with pytest.raises(ValueError):
        solve_ik(r, target_link="a", target_xyz=(0, 0, 0))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
