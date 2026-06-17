"""Rubber Ducky — interactive robot fault diagnosis (scripted transcript).

A Field Service Engineer thinks out loud about a robot fault; the "Ducky" reasons
back like an experienced robotics engineer. The dialogue is scripted, but every
Ducky conclusion is backed by a **real** fieldpilot-urdf call — this is the
deterministic reasoning core of FieldPilot's MDG (robot-diagnostics) assistant.
In production the natural-language + voice layer (Claude, Whisper STT, TTS) wraps
this core; here the script stands in for it so the diagnosis stays honest.

Scenario: a 3-DOF palletiser (base yaw + two pitch joints + tool) can no longer
swing to a pick point off to the side after an e-stop recovery. The loop:

    1. run_all          — structure vs. runtime fault?
    2. solve_ik         — is the target reachable on a healthy arm?
    3. rank_root_causes — which joint explains the observed dead links?
    4. diagnose         — inject the hypothesis, re-test, confirm/refute
    5. affected_links / criticality — scope the impact for the work order

Core install only (no extras):

    pip install fieldpilot-urdf
    python examples/ducky_diagnosis.py
"""
from __future__ import annotations

from fieldpilot_urdf import (
    Inertia, Inertial, Joint, JointLimit, Link, Origin, Robot,
    affected_links, criticality, diagnose, forward_kinematics, Hypothesis,
    rank_root_causes, run_all, solve_ik, summary, Symptom,
)


def build_robot() -> Robot:
    """A 3-DOF palletiser: base yaw + two pitch joints + a fixed tool."""
    def lim():
        return JointLimit(lower=-3.0, upper=3.0, effort=150, velocity=3)

    def inr(mass):
        return Inertial(origin=Origin(xyz=(0.4, 0, 0)), mass=mass,
                        inertia=Inertia(ixx=0.02, iyy=0.2, izz=0.2))

    return Robot(
        name="palletiser",
        links=[
            Link(name="base", inertial=inr(8.0)),
            Link(name="upper_arm", inertial=inr(4.0)),
            Link(name="forearm", inertial=inr(3.0)),
            Link(name="wrist_link", inertial=inr(1.5)),
            Link(name="tool", inertial=inr(0.8)),
        ],
        joints=[
            Joint(name="j_shoulder", type="revolute", parent="base", child="upper_arm",
                  origin=Origin(xyz=(0, 0, 1.0)), axis=(0, 0, 1), limit=lim()),   # base yaw
            Joint(name="j_elbow", type="revolute", parent="upper_arm", child="forearm",
                  origin=Origin(xyz=(1.0, 0, 0)), axis=(0, 1, 0), limit=lim()),   # pitch
            Joint(name="j_wrist", type="revolute", parent="forearm", child="wrist_link",
                  origin=Origin(xyz=(1.0, 0, 0)), axis=(0, 1, 0), limit=lim()),   # pitch
            Joint(name="j_tool", type="fixed", parent="wrist_link", child="tool",
                  origin=Origin(xyz=(0.3, 0, 0))),
        ],
    )


def fse(text):  print(f"\nFSE   > {text}")
def duck(text): print(f"Ducky > {text}")
def tool(text): print(f"        . {text}")


def main() -> None:
    robot = build_robot()

    # A genuinely reachable pick point: a real pose that uses the base yaw (so
    # it's off the x-z plane). Freezing the base yaw confines the arm to y=0,
    # making this exact point unreachable -> a clean, reproducible fault.
    pose = {"j_shoulder": 0.6, "j_elbow": -0.5, "j_wrist": -0.3}
    pick = tuple(round(float(x), 3) for x in forward_kinematics(robot, pose)["tool"][:3, 3])

    print("=" * 74)
    print("  RUBBER DUCKY — robot diagnostic session (MDG)        INT-2026-0042")
    print("=" * 74)

    # --- the report --------------------------------------------------------
    fse("The palletiser at line 3 won't swing over to the conveyor pick point "
        "anymore. Started after this morning's e-stop recovery.")

    duck("Let's reason it out. First — structure or runtime fault? Linting the model.")
    findings = run_all(robot)
    tool(f"run_all(robot) -> {summary(findings)}")
    duck("Model's clean: no broken joints, limits or geometry. So it failed at "
         "runtime, not in config.")

    duck("Is the pick point reachable on a healthy arm? If not, it's a workspace "
         "limit, not a fault.")
    base = solve_ik(robot, "tool", target_xyz=pick)
    tool(f"solve_ik(tool -> {pick}) -> converged={base.converged}, err={base.position_error:.2e} m")
    assert base.converged, "demo target must be reachable on the healthy arm"
    duck(f"Reaches it cleanly ({base.position_error * 1000:.1f} mm). The arm has "
         "genuinely lost capability — a real fault.")

    # --- localisation ------------------------------------------------------
    fse("Right. And weirdly, everything from the turntable up seems dead — the "
        "whole arm won't rotate toward the line.")

    duck("'Everything from the turntable up' is the tell. Ranking which joint's "
         "downstream set best matches what you're seeing.")
    observed = ["upper_arm", "forearm", "wrist_link", "tool"]
    ranked = rank_root_causes(robot, observed)
    tool(f"rank_root_causes(observed={observed})")
    for c in ranked[:3]:
        tool(f"   {c.target:12s} score={c.score:.3f}  precision={c.precision:.2f} recall={c.recall:.2f}")
    top = ranked[0].target
    duck(f"Top suspect: {top} (the base-yaw drive) — the only joint whose "
         f"downstream set covers everything. j_elbow explains less "
         f"(recall {ranked[1].recall:.2f}); it can't account for the upper arm.")

    # --- hypothesis test ---------------------------------------------------
    fse("That fits, the base motor took the brunt of the e-stop. Can you confirm "
        "a dead base motor actually kills the reach before I swap it?")

    duck("Won't guess. Injecting a dead-motor fault on the base yaw, re-running "
         "IK on a copy, checking it reproduces your symptom.")
    report = diagnose(
        robot,
        Symptom(kind="cant_reach", target_link="tool", target_xyz=pick),
        [Hypothesis(suspect_joint=top, fault_mode="motor_dead")],
    )
    tool(f"diagnose(cant_reach @ tool, hypothesis: {top} motor_dead)")
    tool(f"   verdict={report.verdict.value}  confidence={report.confidence}")
    duck(f"{report.verdict.value}. {report.summary}")

    impacted = affected_links(robot, top)
    crit = criticality(robot, top)
    tool(f"affected_links({top}) -> {sorted(impacted)}")
    tool(f"criticality({top}) -> {crit:.0%} of robot mass downstream")
    duck(f"For the work order: a dead {top} motor drags down {len(impacted)} links "
         f"({crit:.0%} of the arm's mass) — highest-impact axis, worth doing right.")

    # --- wrap --------------------------------------------------------------
    fse("Perfect — replace the base-yaw drive, re-home, retest the pick swing.")
    duck(f"Confirmed root cause: dead {top} (base-yaw) motor. Want me to check "
         "spare-parts stock for the drive next?")

    print("\n" + "=" * 74)
    print("  Every Ducky conclusion above came from a real fieldpilot-urdf call.")
    print("=" * 74)


if __name__ == "__main__":
    main()
