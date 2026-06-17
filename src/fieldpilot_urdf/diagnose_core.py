"""Two-tier fault diagnosis loop over URDF robots — the pure, deterministic core.

Coarse-to-fine, hypothesis-and-test — the robot analogue of the FPGA
`diagnostic_loop` (LLM → sub-block → SPICE → verdict), with the URDF kinematics
engine standing in for SPICE:

  Tier 0 (coarse, free):  one static `run_all()` rule scan. If a malformation
                          lands on a hypothesised suspect, that explains it — done.
  Tier 1 (abductive):     for each hypothesis, inject the fault into a *copy* of
                          the model, re-run the relevant kinematic primitive, and
                          check it reproduces the observed symptom.

This module covers the ``cant_reach`` symptom against two fault modes —
``motor_dead`` (a dead actuator, locked at the zero pose) and ``joint_stuck`` (a
joint jammed at a reported angle). Hypotheses are supplied by the caller. The
verdict half is fully symbolic and deterministic: **no network, no API key.** The
natural-language front-end that *generates* hypotheses from a technician's report
via an LLM lives in the companion ``diagnose_nl`` module (gated); this core is the
open showcase.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .diagnostics import Finding, run_all
from .faults import freeze_joint, freeze_joint_at
from .ik import solve_ik
from .models import Robot


class Verdict(str, Enum):
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class Symptom(BaseModel):
    """What the technician reports. Slice scope: a link can no longer reach a pose."""
    kind: Literal["cant_reach"]
    target_link: str
    target_xyz: tuple[float, float, float]
    observed_reachable: bool = False  # reported state: target is NOT reachable


class Hypothesis(BaseModel):
    suspect_joint: str
    fault_mode: Literal["motor_dead", "joint_stuck"]
    # For ``joint_stuck``: the angle (rad) / displacement (m) the joint is jammed
    # at. Ignored by ``motor_dead`` (which is a dead actuator at the zero pose).
    stuck_at: float = 0.0


class DiagnoseReport(BaseModel):
    verdict: Verdict
    tier: int                       # 0 = static scan, 1 = abductive simulation
    suspect_joint: Optional[str] = None
    fault_mode: Optional[str] = None
    confidence: float = 0.0
    evidence: dict = Field(default_factory=dict)
    summary: str = ""


# --- Tier 0 ----------------------------------------------------------------

def _static_findings_on(findings: list[Finding], joint: str) -> list[Finding]:
    return [f for f in findings if joint in f.refs]


# --- Tier 1: a locked joint → cant_reach -----------------------------------
# Both supported fault modes reduce to the same kinematic consequence — an axis
# that can no longer be actuated — differing only in *where* it locks: a dead
# motor locks at the zero pose, a jammed joint at a reported angle. One simulator
# covers both; `_apply_lock` encodes the per-mode difference.

def _apply_lock(robot: Robot, h: Hypothesis) -> tuple[Robot, str]:
    """Apply ``h``'s fault to a copy of ``robot``; return ``(faulted, phrase)``
    where ``phrase`` is a capitalised noun phrase naming the fault."""
    r = robot.model_copy(deep=True)
    if h.fault_mode == "joint_stuck":
        return (freeze_joint_at(r, h.suspect_joint, h.stuck_at),
                f"A jammed {h.suspect_joint} (stuck at {h.stuck_at:.3g})")
    return freeze_joint(r, h.suspect_joint), f"A dead {h.suspect_joint} motor"


def _simulate_locked_cant_reach(robot: Robot, h: Hypothesis, s: Symptom) -> DiagnoseReport:
    # Was the target reachable on the healthy robot? If not, losing it is not
    # evidence of THIS fault — don't attribute.
    base = solve_ik(robot, s.target_link, s.target_xyz)
    if not base.converged:
        return DiagnoseReport(
            verdict=Verdict.INCONCLUSIVE, tier=1,
            suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=0.0,
            evidence={"baseline_reachable": False,
                      "baseline_pos_err": base.position_error},
            summary=(f"Target {s.target_xyz} is unreachable even on the healthy "
                     f"robot (err {base.position_error:.3g} m); cannot attribute "
                     f"the symptom to a {h.suspect_joint} fault."),
        )

    faulted, phrase = _apply_lock(robot, h)
    try:
        sim = solve_ik(faulted, s.target_link, s.target_xyz)
        faulted_reachable = sim.converged
        faulted_err = sim.position_error
    except ValueError:
        # Freezing removed the last actuable joint — the chain is rigid and
        # cannot reach an arbitrary target.
        faulted_reachable = False
        faulted_err = float("inf")

    evidence = {
        "fault_mode": h.fault_mode,
        "baseline_pos_err": base.position_error,
        "faulted_pos_err": faulted_err,
        "faulted_reachable": faulted_reachable,
    }
    if h.fault_mode == "joint_stuck":
        evidence["stuck_at"] = h.stuck_at
    if not faulted_reachable:  # symptom = "can't reach" → reproduced
        return DiagnoseReport(
            verdict=Verdict.CONFIRMED, tier=1,
            suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=1.0,
            evidence=evidence,
            summary=(f"{phrase} freezes the chain, so {s.target_link} can no longer "
                     f"reach {s.target_xyz} (err {faulted_err:.3g} m vs healthy "
                     f"{base.position_error:.3g} m). Reproduces the reported symptom."),
        )
    return DiagnoseReport(
        verdict=Verdict.REFUTED, tier=1,
        suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=0.0,
        evidence=evidence,
        summary=(f"{phrase} still lets {s.target_link} reach {s.target_xyz} "
                 f"(err {faulted_err:.3g} m); it does not explain the symptom."),
    )


_SIMULATORS = {
    ("motor_dead", "cant_reach"): _simulate_locked_cant_reach,
    ("joint_stuck", "cant_reach"): _simulate_locked_cant_reach,
}


def diagnose(robot: Robot, symptom: Symptom, hypotheses: list[Hypothesis]) -> DiagnoseReport:
    """Run the two-tier loop. Hypotheses are caller-supplied."""
    if symptom.target_link not in {l.name for l in robot.links}:
        raise KeyError(f"unknown target_link: {symptom.target_link!r}")

    # Tier 0 — one coarse static scan; a malformation on a suspect explains it.
    findings = run_all(robot)
    for h in hypotheses:
        hits = _static_findings_on(findings, h.suspect_joint)
        if hits:
            return DiagnoseReport(
                verdict=Verdict.CONFIRMED, tier=0,
                suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=1.0,
                evidence={"static_findings": [f.model_dump() for f in hits]},
                summary=(f"Static scan flags {h.suspect_joint} — "
                         + "; ".join(f.message for f in hits)),
            )

    # Tier 1 — abductive simulation, best-first; first CONFIRMED wins.
    reports: list[DiagnoseReport] = []
    for h in hypotheses:
        sim = _SIMULATORS.get((h.fault_mode, symptom.kind))
        if sim is None:
            continue
        rep = sim(robot, h, symptom)
        if rep.verdict == Verdict.CONFIRMED:
            return rep
        reports.append(rep)

    if reports:
        # A REFUTED result is more informative than an INCONCLUSIVE one.
        reports.sort(key=lambda r: r.verdict != Verdict.REFUTED)
        return reports[0]
    return DiagnoseReport(
        verdict=Verdict.INCONCLUSIVE, tier=1, confidence=0.0,
        summary="No applicable hypothesis simulator for this symptom kind.",
    )
