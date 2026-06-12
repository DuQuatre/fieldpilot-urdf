"""Tests for the pure two-tier diagnosis core (Tier-0 + motor_dead → cant_reach).

Hermetic and deterministic — no network, no API key. Uses a 3-link arm whose
end-effector ("tool") position depends on BOTH revolute joints, so freezing one
genuinely changes what is reachable. The LLM front-end (``diagnose_nl``) is
covered separately in ``test_diagnose_nl.py``.

Run: python3 -m pytest app/urdf/test_diagnose_core.py -q  (from pydexpi-server/)
"""
from __future__ import annotations

import pytest

from fieldpilot_urdf import from_xml, forward_kinematics
from fieldpilot_urdf.diagnose_core import Hypothesis, Symptom, Verdict, diagnose
from fieldpilot_urdf.faults import inject_motor_fault

# 3 links + 2 revolute joints + a wrist offset (fixed) so `tool` moves with BOTH
# joints: shoulder rotates everything about z; elbow rotates the wrist offset.
SAMPLE = """\
<robot name="testarm">
  <link name="base"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial></link>
  <link name="upper"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial></link>
  <link name="fore"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial></link>
  <link name="tool"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial></link>
  <joint name="shoulder_joint" type="revolute">
    <parent link="base"/><child link="upper"/><axis xyz="0 0 1"/>
    <origin xyz="0 0 0.1"/>
    <limit lower="-3.14" upper="3.14" effort="150" velocity="3.15"/>
  </joint>
  <joint name="elbow_joint" type="revolute">
    <parent link="upper"/><child link="fore"/><axis xyz="0 1 0"/>
    <origin xyz="0.3 0 0.2"/>
    <limit lower="-3.14" upper="3.14" effort="150" velocity="3.15"/>
  </joint>
  <joint name="wrist_fixed" type="fixed">
    <parent link="fore"/><child link="tool"/>
    <origin xyz="0.3 0 0"/>
  </joint>
</robot>
"""


@pytest.fixture
def clean():
    return from_xml(SAMPLE)


def _tool_xyz(robot, q):
    """Position of `tool` at joint configuration q (guaranteed reachable)."""
    return tuple(float(v) for v in forward_kinematics(robot, q=q)["tool"][:3, 3])


HYP = [Hypothesis(suspect_joint="shoulder_joint", fault_mode="motor_dead")]


def test_clean_arm_is_fault_free(clean):
    # Sanity: the healthy model has no static findings (so Tier 0 stays empty).
    from fieldpilot_urdf.diagnostics import run_all
    assert run_all(clean) == []


def test_tier0_static_motor_dead(clean):
    """effort/velocity zeroed → run_all flags R003 on the suspect → Tier 0 CONFIRMED."""
    target = _tool_xyz(clean, {"shoulder_joint": 0.5, "elbow_joint": 0.4})
    inject_motor_fault(clean, "shoulder_joint")  # the R003 static signature
    sym = Symptom(kind="cant_reach", target_link="tool", target_xyz=target)
    rep = diagnose(clean, sym, HYP)
    assert rep.verdict is Verdict.CONFIRMED
    assert rep.tier == 0
    assert rep.suspect_joint == "shoulder_joint"
    assert "R003" in str(rep.evidence)


def test_tier1_confirmed(clean):
    """Healthy model (Tier 0 empty); a target that needs the shoulder rotated
    becomes unreachable once the shoulder is frozen → Tier 1 CONFIRMED."""
    target = _tool_xyz(clean, {"shoulder_joint": 0.9, "elbow_joint": 0.5})
    sym = Symptom(kind="cant_reach", target_link="tool", target_xyz=target)
    rep = diagnose(from_xml(SAMPLE), sym, HYP)
    assert rep.verdict is Verdict.CONFIRMED
    assert rep.tier == 1
    assert rep.evidence["faulted_reachable"] is False


def test_tier1_refuted(clean):
    """A target reachable with the shoulder AT its frozen value (0) stays
    reachable when the shoulder is frozen → the hypothesis is REFUTED."""
    target = _tool_xyz(clean, {"shoulder_joint": 0.0, "elbow_joint": 0.5})
    sym = Symptom(kind="cant_reach", target_link="tool", target_xyz=target)
    rep = diagnose(from_xml(SAMPLE), sym, HYP)
    assert rep.verdict is Verdict.REFUTED
    assert rep.tier == 1
    assert rep.evidence["faulted_reachable"] is True


def test_tier1_inconclusive_when_target_never_reachable(clean):
    """If the target is unreachable even on the healthy robot, the loop must not
    blame the fault."""
    sym = Symptom(kind="cant_reach", target_link="tool", target_xyz=(5.0, 5.0, 5.0))
    rep = diagnose(from_xml(SAMPLE), sym, HYP)
    assert rep.verdict is Verdict.INCONCLUSIVE


def test_tier1_picks_correct_joint(clean):
    """Given several hypotheses (wrong one ranked first), the loop returns the
    one whose simulated fault actually reproduces the symptom. The target needs
    the shoulder rotated but the elbow at rest (0), so only a dead shoulder
    explains it."""
    target = _tool_xyz(clean, {"shoulder_joint": 0.9, "elbow_joint": 0.0})
    sym = Symptom(kind="cant_reach", target_link="tool", target_xyz=target)
    hyps = [
        Hypothesis(suspect_joint="elbow_joint", fault_mode="motor_dead"),     # wrong, first
        Hypothesis(suspect_joint="shoulder_joint", fault_mode="motor_dead"),  # right
    ]
    rep = diagnose(from_xml(SAMPLE), sym, hyps)
    assert rep.verdict is Verdict.CONFIRMED
    assert rep.suspect_joint == "shoulder_joint"


def test_unknown_target_link_raises(clean):
    sym = Symptom(kind="cant_reach", target_link="nope", target_xyz=(0.1, 0.0, 0.3))
    with pytest.raises(KeyError):
        diagnose(clean, sym, HYP)
