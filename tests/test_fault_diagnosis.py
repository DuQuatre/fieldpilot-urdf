"""Regression test for symptom-side fault injection (the `faults` module).

Guards the property the diagnosis loop relies on: a clean robot is fault-free,
and injecting a motor fault on any joint trips R003 on *exactly* that joint — so
diagnosis tracks the actual fault, not a fixed joint.

Deterministic, no network. (The LLM integration variant lives in the gated
FieldPilot SaaS, not in this open package.)
"""
from __future__ import annotations

import pytest

from fieldpilot_urdf import from_xml, run_all
from fieldpilot_urdf.faults import inject_motor_fault

# Clean 3-link arm: two revolute joints with valid limits, positive masses,
# PSD inertia, unique names — so run_all() finds nothing until we fault it.
SAMPLE = """\
<robot name="testarm">
  <link name="base"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial></link>
  <link name="upper"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial></link>
  <link name="fore"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial></link>
  <joint name="shoulder_joint" type="revolute">
    <parent link="base"/><child link="upper"/><axis xyz="0 0 1"/>
    <origin xyz="0 0 0.1"/>
    <limit lower="-3.14" upper="3.14" effort="150" velocity="3.15"/>
  </joint>
  <joint name="elbow_joint" type="revolute">
    <parent link="upper"/><child link="fore"/><axis xyz="0 1 0"/>
    <origin xyz="0 0 0.2"/>
    <limit lower="-3.14" upper="3.14" effort="150" velocity="3.15"/>
  </joint>
</robot>
"""

JOINTS = ["shoulder_joint", "elbow_joint"]


@pytest.fixture
def clean():
    return from_xml(SAMPLE)


def _r003_joints(findings) -> set[str]:
    """Set of joint names flagged by rule R003."""
    return {f.refs[0] for f in findings if f.code == "R003" and f.refs}


def test_clean_arm_has_no_r003(clean):
    assert _r003_joints(run_all(clean)) == set()


@pytest.mark.parametrize("target", JOINTS)
def test_fault_localizes_to_target_joint(clean, target):
    """Faulting a joint trips R003 on that joint and no other — so diagnosis
    tracks the actual fault, not a fixed joint."""
    other = next(j for j in JOINTS if j != target)
    inject_motor_fault(clean, target)
    flagged = _r003_joints(run_all(clean))
    assert target in flagged
    assert other not in flagged


def test_fault_zeroes_effort_and_velocity(clean):
    inject_motor_fault(clean, "elbow_joint")
    lim = clean.joint("elbow_joint").limit
    assert lim.effort == 0.0 and lim.velocity == 0.0


def test_inject_unknown_joint_raises(clean):
    with pytest.raises(KeyError):
        inject_motor_fault(clean, "no_such_joint")
