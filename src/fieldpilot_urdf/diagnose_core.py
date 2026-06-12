"""Two-tier fault diagnosis loop over URDF robots — the pure, deterministic core.

Coarse-to-fine, hypothesis-and-test — the robot analogue of the FPGA
`diagnostic_loop` (LLM → sub-block → SPICE → verdict), with the URDF kinematics
engine standing in for SPICE:

  Tier 0 (coarse, free):  one static `run_all()` rule scan. If a malformation
                          lands on a hypothesised suspect, that explains it — done.
  Tier 1 (abductive):     for each hypothesis, inject the fault into a *copy* of
                          the model, re-run the relevant kinematic primitive, and
                          check it reproduces the observed symptom.

This module ships the first vertical slice: fault mode ``motor_dead`` against the
``cant_reach`` symptom. Hypotheses are supplied by the caller. The verdict half is
fully symbolic and deterministic: **no network, no API key.** The natural-language
front-end that *generates* hypotheses from a technician's report via an LLM lives
in the companion ``diagnose_nl`` module (gated); this core is the open showcase.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .diagnostics import Finding, run_all
from .faults import freeze_joint
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
    fault_mode: Literal["motor_dead"]


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


# --- Tier 1: motor_dead → cant_reach ---------------------------------------

def _simulate_motor_dead_cant_reach(robot: Robot, h: Hypothesis, s: Symptom) -> DiagnoseReport:
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

    # Kinematic consequence of a dead motor: the joint can't be actuated.
    faulted = freeze_joint(robot.model_copy(deep=True), h.suspect_joint)
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
        "baseline_pos_err": base.position_error,
        "faulted_pos_err": faulted_err,
        "faulted_reachable": faulted_reachable,
    }
    if not faulted_reachable:  # symptom = "can't reach" → reproduced
        return DiagnoseReport(
            verdict=Verdict.CONFIRMED, tier=1,
            suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=1.0,
            evidence=evidence,
            summary=(f"A dead {h.suspect_joint} motor freezes that axis; "
                     f"{s.target_link} can no longer reach {s.target_xyz} "
                     f"(err {faulted_err:.3g} m vs healthy {base.position_error:.3g} m). "
                     f"Reproduces the reported symptom."),
        )
    return DiagnoseReport(
        verdict=Verdict.REFUTED, tier=1,
        suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=0.0,
        evidence=evidence,
        summary=(f"With {h.suspect_joint} frozen, {s.target_link} still reaches "
                 f"{s.target_xyz} (err {faulted_err:.3g} m); a dead "
                 f"{h.suspect_joint} motor does not explain the symptom."),
    )


_SIMULATORS = {
    ("motor_dead", "cant_reach"): _simulate_motor_dead_cant_reach,
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
