"""Tests for the pure two-tier diagnosis core (Tier-0 + motor_dead → cant_reach).

Hermetic and deterministic — no network, no API key. Uses a 3-link arm whose
end-effector ("tool") position depends on BOTH revolute joints, so freezing one
genuinely changes what is reachable. The LLM front-end (``diagnose_nl``) is
covered separately in ``test_diagnose_nl.py``.

Run: python3 -m pytest app/urdf/test_diagnose_core.py -q  (from pydexpi-server/)
"""
from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from fieldpilot_urdf import from_xml, forward_kinematics
from fieldpilot_urdf.diagnose_core import Hypothesis, Symptom, Verdict, diagnose
from fieldpilot_urdf.faults import inject_motor_fault
from fieldpilot_urdf.models import Box, Collision, Joint, JointLimit, Link, Origin, Robot

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


# --- joint_stuck fault mode (jammed at a reported angle) -------------------

def test_joint_stuck_confirmed_at_wrong_angle(clean):
    """A target the healthy arm reaches (shoulder rotated to 0.9) becomes
    unreachable if the shoulder is jammed at a *different* angle → CONFIRMED,
    and the evidence records the stuck angle."""
    target = _tool_xyz(clean, {"shoulder_joint": 0.9, "elbow_joint": 0.5})
    sym = Symptom(kind="cant_reach", target_link="tool", target_xyz=target)
    hyp = [Hypothesis(suspect_joint="shoulder_joint", fault_mode="joint_stuck", stuck_at=-1.2)]
    rep = diagnose(from_xml(SAMPLE), sym, hyp)
    assert rep.verdict is Verdict.CONFIRMED
    assert rep.tier == 1
    assert rep.fault_mode == "joint_stuck"
    assert rep.evidence["stuck_at"] == -1.2
    assert rep.evidence["faulted_reachable"] is False


def test_joint_stuck_refuted_when_jammed_at_solution(clean):
    """If the joint is jammed at the very angle the target needs, the arm can
    still reach it (the other joint takes up the slack) → REFUTED."""
    target = _tool_xyz(clean, {"shoulder_joint": 0.9, "elbow_joint": 0.5})
    sym = Symptom(kind="cant_reach", target_link="tool", target_xyz=target)
    hyp = [Hypothesis(suspect_joint="shoulder_joint", fault_mode="joint_stuck", stuck_at=0.9)]
    rep = diagnose(from_xml(SAMPLE), sym, hyp)
    assert rep.verdict is Verdict.REFUTED
    assert rep.evidence["faulted_reachable"] is True


def test_joint_stuck_default_angle_matches_motor_dead(clean):
    """stuck_at defaults to 0.0, i.e. jammed at the neutral pose — kinematically
    identical to a dead motor. Both verdicts must agree."""
    target = _tool_xyz(clean, {"shoulder_joint": 0.9, "elbow_joint": 0.5})
    sym = Symptom(kind="cant_reach", target_link="tool", target_xyz=target)
    dead = diagnose(from_xml(SAMPLE), sym,
                    [Hypothesis(suspect_joint="shoulder_joint", fault_mode="motor_dead")])
    stuck0 = diagnose(from_xml(SAMPLE), sym,
                      [Hypothesis(suspect_joint="shoulder_joint", fault_mode="joint_stuck")])
    assert dead.verdict is stuck0.verdict is Verdict.CONFIRMED
    assert dead.evidence["faulted_pos_err"] == pytest.approx(stuck0.evidence["faulted_pos_err"])


def test_mixed_fault_modes_picks_the_reproducing_one(clean):
    """Both modes offered for both joints; only the one whose simulated lock
    reproduces the symptom wins (best-first, first CONFIRMED)."""
    target = _tool_xyz(clean, {"shoulder_joint": 0.9, "elbow_joint": 0.0})
    sym = Symptom(kind="cant_reach", target_link="tool", target_xyz=target)
    # Elbow jammed at 0.0 = its value in the target pose, so the arm still
    # reaches (shoulder free) → that hypothesis is REFUTED; only the dead
    # shoulder reproduces the symptom.
    hyps = [
        Hypothesis(suspect_joint="elbow_joint", fault_mode="joint_stuck", stuck_at=0.0),
        Hypothesis(suspect_joint="shoulder_joint", fault_mode="motor_dead"),
    ]
    rep = diagnose(from_xml(SAMPLE), sym, hyps)
    assert rep.verdict is Verdict.CONFIRMED
    assert rep.suspect_joint == "shoulder_joint"


# --- self_collision symptom -------------------------------------------------
# A 2-joint folding arm with box collision geometry: base, upper and fore each
# carry a 0.2 cube. Extended (j2=0) the fore sits at (1,0,0), clear of the base;
# jamming the elbow at pi folds the fore back onto the base (0,0,0) — a
# non-adjacent pair, so it registers as a real self-collision.

def _box(off):
    return [Collision(origin=Origin(xyz=off), geometry=Box(size=(0.2, 0.2, 0.2)))]


def _lim():
    return JointLimit(lower=-4.0, upper=4.0, effort=1.0, velocity=1.0)


@pytest.fixture
def fold():
    return Robot(
        name="fold",
        links=[Link(name="base", collisions=_box((0, 0, 0))),
               Link(name="upper", collisions=_box((0.5, 0, 0))),
               Link(name="fore", collisions=_box((0.5, 0, 0)))],
        joints=[Joint(name="j1", type="revolute", parent="base", child="upper",
                      origin=Origin(xyz=(0, 0, 0)), axis=(0, 0, 1), limit=_lim()),
                Joint(name="j2", type="revolute", parent="upper", child="fore",
                      origin=Origin(xyz=(0.5, 0, 0)), axis=(0, 0, 1), limit=_lim())],
    )


CMD = {"j1": 0.0, "j2": 0.0}   # extended pose: collision-free on a healthy arm


def test_self_collision_healthy_pose_is_clear(fold):
    from fieldpilot_urdf import detect_self_collisions
    assert detect_self_collisions(fold, q=CMD) == []


def test_self_collision_joint_stuck_confirmed(fold):
    """Elbow jammed at pi folds the fore onto the base at the commanded pose."""
    sym = Symptom(kind="self_collision", at_config=CMD, colliding_links=("base", "fore"))
    rep = diagnose(fold, sym, [Hypothesis(suspect_joint="j2", fault_mode="joint_stuck", stuck_at=math.pi)])
    assert rep.verdict is Verdict.CONFIRMED
    assert rep.tier == 1
    assert ["base", "fore"] in rep.evidence["faulted_collisions"]
    assert rep.evidence["baseline_collisions"] == []


def test_self_collision_motor_dead_refuted(fold):
    """A dead elbow locks at 0 = the commanded value, so nothing changes."""
    sym = Symptom(kind="self_collision", at_config=CMD, colliding_links=("base", "fore"))
    rep = diagnose(fold, sym, [Hypothesis(suspect_joint="j2", fault_mode="motor_dead")])
    assert rep.verdict is Verdict.REFUTED


def test_self_collision_inconclusive_when_commanded_pose_collides(fold):
    """If the commanded pose itself self-collides on a healthy arm, the clash is
    not attributable to a joint fault."""
    sym = Symptom(kind="self_collision", at_config={"j1": 0.0, "j2": math.pi})
    rep = diagnose(fold, sym, [Hypothesis(suspect_joint="j2", fault_mode="joint_stuck", stuck_at=0.5)])
    assert rep.verdict is Verdict.INCONCLUSIVE


def test_self_collision_reported_pair_filters(fold):
    """A clash is produced, but the tech reported a different (non-colliding)
    pair — the specific hypothesis must be REFUTED, not CONFIRMED."""
    sym = Symptom(kind="self_collision", at_config=CMD, colliding_links=("base", "upper"))
    rep = diagnose(fold, sym, [Hypothesis(suspect_joint="j2", fault_mode="joint_stuck", stuck_at=math.pi)])
    assert rep.verdict is Verdict.REFUTED


def test_self_collision_unknown_joint_raises(fold):
    sym = Symptom(kind="self_collision", at_config={"nope": 0.0})
    with pytest.raises(KeyError):
        diagnose(fold, sym, [Hypothesis(suspect_joint="j2", fault_mode="motor_dead")])


# --- Symptom validation -----------------------------------------------------

def test_cant_reach_requires_target():
    with pytest.raises(ValidationError):
        Symptom(kind="cant_reach")


def test_self_collision_requires_at_config():
    with pytest.raises(ValidationError):
        Symptom(kind="self_collision")
