"""Two-tier fault diagnosis loop over URDF robots — the pure, deterministic core.

Coarse-to-fine, hypothesis-and-test — the robot analogue of the FPGA
`diagnostic_loop` (LLM → sub-block → SPICE → verdict), with the URDF kinematics
engine standing in for SPICE:

  Tier 0 (coarse, free):  one static `run_all()` rule scan. If a malformation
                          lands on a hypothesised suspect, that explains it — done.
  Tier 1 (abductive):     for each hypothesis, inject the fault into a *copy* of
                          the model, re-run the relevant kinematic primitive, and
                          check it reproduces the observed symptom.

This module covers three symptoms — ``cant_reach`` (a link can no longer reach a
pose), ``self_collision`` (the robot self-collides at a commanded pose), and
``reduced_workspace`` (a link's reachable envelope shrank) — against three fault
modes: ``motor_dead`` (a dead actuator, locked at the zero pose), ``joint_stuck``
(a joint jammed at a reported angle), and ``limit_misconfig`` (a mis-set travel
``<limit>`` that clips the joint's range without freezing it). Not every pair has
a sound mapping (e.g. ``limit_misconfig`` × ``self_collision`` does not); the
``(fault_mode, symptom)`` pairs that do are registered in ``_SIMULATORS``, the
single extension point. Hypotheses may be supplied by the caller, or — when none
are given — generated from the symptom by ranking suspect joints with
``rank_root_causes`` and proposing a ``motor_dead`` for each. The verdict half is
fully symbolic and deterministic: **no network, no API key.** The richer
natural-language front-end that *generates* hypotheses from a technician's report
via an LLM lives in the companion ``diagnose_nl`` module (gated); this core is the
open showcase.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .collisions import detect_self_collisions
from .diagnostics import Finding, run_all
from .fault_propagation import rank_root_causes
from .faults import freeze_joint_at, misconfigure_limit
from .ik import solve_ik
from .models import Robot
from .trajectory import sample_workspace

# Workspace sampling is fixed (count + seed) so a baseline and its faulted twin
# are measured identically — the comparison, not the absolute value, is what counts.
_WS_SAMPLES = 400
_WS_SEED = 0


class Verdict(str, Enum):
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class Symptom(BaseModel):
    """What the technician reports. Two kinds are supported:

    - ``cant_reach``: ``target_link`` can no longer reach ``target_xyz``.
    - ``self_collision``: at the commanded pose ``at_config`` the robot
      self-collides (optionally a specific ``colliding_links`` pair) when a
      healthy robot would not.
    """
    kind: Literal["cant_reach", "self_collision", "reduced_workspace"]

    # cant_reach
    target_link: Optional[str] = None
    target_xyz: Optional[tuple[float, float, float]] = None
    observed_reachable: bool = False  # reported state: target is NOT reachable

    # self_collision
    at_config: dict[str, float] = Field(default_factory=dict)  # commanded joint pose
    colliding_links: Optional[tuple[str, str]] = None          # reported pair, if known

    # reduced_workspace (target_link is the end-effector whose envelope shrank)
    min_shrinkage: float = 0.1   # fraction the reach must drop for a fault to count

    @model_validator(mode="after")
    def _require_fields_for_kind(self) -> "Symptom":
        if self.kind == "cant_reach":
            if self.target_link is None or self.target_xyz is None:
                raise ValueError("cant_reach requires target_link and target_xyz")
        elif self.kind == "self_collision":
            if not self.at_config:
                raise ValueError("self_collision requires a non-empty at_config")
        elif self.kind == "reduced_workspace":
            if self.target_link is None:
                raise ValueError("reduced_workspace requires target_link")
        return self


class Hypothesis(BaseModel):
    suspect_joint: str
    fault_mode: Literal["motor_dead", "joint_stuck", "limit_misconfig"]
    # For ``joint_stuck``: the angle (rad) / displacement (m) the joint is jammed
    # at. Ignored by ``motor_dead`` (which is a dead actuator at the zero pose).
    stuck_at: float = 0.0
    # For ``limit_misconfig``: the erroneously-set travel bound(s). At least one
    # is required; an omitted bound keeps the model's existing value.
    bad_lower: Optional[float] = None
    bad_upper: Optional[float] = None

    @model_validator(mode="after")
    def _require_fields_for_mode(self) -> "Hypothesis":
        if self.fault_mode == "limit_misconfig" and self.bad_lower is None and self.bad_upper is None:
            raise ValueError("limit_misconfig requires bad_lower and/or bad_upper")
        return self


class DiagnoseReport(BaseModel):
    verdict: Verdict
    tier: int                       # 0 = static scan, 1 = abductive simulation
    suspect_joint: Optional[str] = None
    fault_mode: Optional[str] = None
    confidence: float = 0.0
    evidence: dict = Field(default_factory=dict)
    summary: str = ""
    auto_generated: bool = False    # hypotheses were ranked by diagnose, not supplied


# --- Tier 0 ----------------------------------------------------------------

def _static_findings_on(findings: list[Finding], joint: str) -> list[Finding]:
    return [f for f in findings if joint in f.refs]


# --- Tier 1 ----------------------------------------------------------------
# The lock fault modes (motor_dead, joint_stuck) reduce to one kinematic
# consequence — an axis held at a fixed value — differing only in *where*:
# `_lock_value_and_phrase` encodes that. limit_misconfig is a *non-lock* fault:
# the axis still moves but its travel <limit> is wrong. `_inject_fault` applies
# whichever, returning the faulted model + a noun phrase (or None if the fault
# can't apply, e.g. misconfiguring a joint that has no <limit>).

def _lock_value_and_phrase(h: Hypothesis) -> tuple[float, str]:
    """The value the suspect joint is held at, and a capitalised noun phrase
    naming the fault. Lock modes only (motor_dead / joint_stuck)."""
    if h.fault_mode == "joint_stuck":
        return h.stuck_at, f"A jammed {h.suspect_joint} (stuck at {h.stuck_at:.3g})"
    return 0.0, f"A dead {h.suspect_joint} motor"


def _inject_fault(robot: Robot, h: Hypothesis) -> tuple[Optional[Robot], str]:
    """Apply ``h``'s fault to a copy of ``robot``. Returns ``(faulted, phrase)``,
    or ``(None, reason)`` when the fault cannot be applied to this joint."""
    r = robot.model_copy(deep=True)
    if h.fault_mode == "limit_misconfig":
        if r.joint(h.suspect_joint).limit is None:   # KeyError here = unknown joint
            return None, f"{h.suspect_joint} has no <limit> to misconfigure"
        misconfigure_limit(r, h.suspect_joint, lower=h.bad_lower, upper=h.bad_upper)
        j = r.joint(h.suspect_joint)
        return r, f"A mis-set {h.suspect_joint} limit ([{j.limit.lower:.3g}, {j.limit.upper:.3g}])"
    value, phrase = _lock_value_and_phrase(h)
    return freeze_joint_at(r, h.suspect_joint, value), phrase


def _simulate_cant_reach(robot: Robot, h: Hypothesis, s: Symptom) -> DiagnoseReport:
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

    faulted, phrase = _inject_fault(robot, h)
    if faulted is None:
        return DiagnoseReport(
            verdict=Verdict.INCONCLUSIVE, tier=1,
            suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=0.0,
            evidence={"inapplicable": phrase},
            summary=f"Cannot apply {h.fault_mode} to {h.suspect_joint}: {phrase}.",
        )
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
    elif h.fault_mode == "limit_misconfig":
        evidence["bad_lower"] = h.bad_lower
        evidence["bad_upper"] = h.bad_upper
    if not faulted_reachable:  # symptom = "can't reach" → reproduced
        return DiagnoseReport(
            verdict=Verdict.CONFIRMED, tier=1,
            suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=1.0,
            evidence=evidence,
            summary=(f"{phrase} leaves {s.target_link} unable to reach {s.target_xyz} "
                     f"(err {faulted_err:.3g} m vs healthy {base.position_error:.3g} m). "
                     f"Reproduces the reported symptom."),
        )
    return DiagnoseReport(
        verdict=Verdict.REFUTED, tier=1,
        suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=0.0,
        evidence=evidence,
        summary=(f"{phrase} still lets {s.target_link} reach {s.target_xyz} "
                 f"(err {faulted_err:.3g} m); it does not explain the symptom."),
    )


# --- Tier 1: a locked joint → self_collision -------------------------------
# Config-dependent, not a workspace question: the symptom carries the commanded
# pose. A stuck joint means the suspect axis actually sits at its lock value
# instead of the commanded one — recompute collisions with that override and see
# if the reported clash appears (when the healthy commanded pose was clear).

def _norm_pair(pair: tuple[str, str]) -> frozenset:
    return frozenset(pair)


def _simulate_locked_self_collision(robot: Robot, h: Hypothesis, s: Symptom) -> DiagnoseReport:
    want = _norm_pair(s.colliding_links) if s.colliding_links else None

    def matching(hits: list[tuple[str, str]]) -> list[tuple[str, str]]:
        return [p for p in hits if want is None or _norm_pair(p) == want]

    # If the commanded pose already self-collides on a healthy robot, the clash
    # is the commanded pose's fault — not attributable to this joint.
    try:
        base_hits = detect_self_collisions(robot, q=s.at_config)
    except ValueError:
        base_hits = []
    if matching(base_hits):
        return DiagnoseReport(
            verdict=Verdict.INCONCLUSIVE, tier=1,
            suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=0.0,
            evidence={"baseline_collisions": [list(p) for p in base_hits]},
            summary=(f"The commanded pose already self-collides on the healthy "
                     f"robot {sorted(matching(base_hits)[0])}; cannot attribute it "
                     f"to a {h.suspect_joint} fault."),
        )

    # A stuck joint holds the suspect axis at its lock value, not the command.
    value, phrase = _lock_value_and_phrase(h)
    faulted_cfg = {**s.at_config, h.suspect_joint: value}
    try:
        hits = detect_self_collisions(robot, q=faulted_cfg)
    except ValueError:
        hits = []
    relevant = matching(hits)

    evidence = {
        "fault_mode": h.fault_mode,
        "baseline_collisions": [list(p) for p in base_hits],
        "faulted_collisions": [list(p) for p in hits],
    }
    if h.fault_mode == "joint_stuck":
        evidence["stuck_at"] = h.stuck_at
    if relevant:
        return DiagnoseReport(
            verdict=Verdict.CONFIRMED, tier=1,
            suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=1.0,
            evidence=evidence,
            summary=(f"{phrase} holds the axis off its command, driving "
                     f"{sorted(relevant[0])} into self-collision at the commanded "
                     f"pose (clear on the healthy robot). Reproduces the symptom."),
        )
    return DiagnoseReport(
        verdict=Verdict.REFUTED, tier=1,
        suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=0.0,
        evidence=evidence,
        summary=(f"{phrase} introduces no self-collision at the commanded pose; "
                 f"it does not explain the symptom."),
    )


# --- Tier 1: a fault → reduced_workspace -----------------------------------
# Measure the end-effector's reachable envelope (bounding-box diagonal) on the
# healthy robot vs the faulted one. Every fault mode shrinks it — a frozen axis
# removes a DOF, a clipped limit narrows one — so all three apply here.

def _workspace_reach(robot: Robot, target_link: str) -> float:
    """Scalar 'reach': the diagonal of the reachable-points bounding box."""
    ws = sample_workspace(robot, target_link, n_samples=_WS_SAMPLES, seed=_WS_SEED,
                          check_collisions=False)
    ext = [hi - lo for lo, hi in zip(ws.bbox_min, ws.bbox_max)]
    return math.sqrt(sum(e * e for e in ext))


def _simulate_reduced_workspace(robot: Robot, h: Hypothesis, s: Symptom) -> DiagnoseReport:
    base_reach = _workspace_reach(robot, s.target_link)
    if base_reach <= 1e-9:
        return DiagnoseReport(
            verdict=Verdict.INCONCLUSIVE, tier=1,
            suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=0.0,
            evidence={"baseline_reach": base_reach},
            summary=(f"{s.target_link}'s healthy workspace is already degenerate "
                     f"(reach {base_reach:.3g} m); cannot measure a reduction."),
        )

    faulted, phrase = _inject_fault(robot, h)
    if faulted is None:
        return DiagnoseReport(
            verdict=Verdict.INCONCLUSIVE, tier=1,
            suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=0.0,
            evidence={"inapplicable": phrase},
            summary=f"Cannot apply {h.fault_mode} to {h.suspect_joint}: {phrase}.",
        )
    faulted_reach = _workspace_reach(faulted, s.target_link)
    shrinkage = 1.0 - faulted_reach / base_reach

    evidence = {
        "fault_mode": h.fault_mode,
        "baseline_reach": base_reach,
        "faulted_reach": faulted_reach,
        "shrinkage": shrinkage,
        "min_shrinkage": s.min_shrinkage,
    }
    if shrinkage >= s.min_shrinkage:
        return DiagnoseReport(
            verdict=Verdict.CONFIRMED, tier=1,
            suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=1.0,
            evidence=evidence,
            summary=(f"{phrase} shrinks {s.target_link}'s reachable workspace by "
                     f"{shrinkage:.0%} (reach {base_reach:.3g} → {faulted_reach:.3g} m). "
                     f"Reproduces the reported reduction."),
        )
    return DiagnoseReport(
        verdict=Verdict.REFUTED, tier=1,
        suspect_joint=h.suspect_joint, fault_mode=h.fault_mode, confidence=0.0,
        evidence=evidence,
        summary=(f"{phrase} barely changes {s.target_link}'s workspace "
                 f"({shrinkage:.0%} < {s.min_shrinkage:.0%} threshold); does not "
                 f"explain the reported reduction."),
    )


# limit_misconfig is registered for cant_reach + reduced_workspace, NOT
# self_collision: it changes a joint's travel range, which a fixed commanded pose
# never exercises, so it can't explain a collision there. An unregistered
# (mode, symptom) pair is simply skipped in the loop below.
_SIMULATORS = {
    ("motor_dead", "cant_reach"): _simulate_cant_reach,
    ("joint_stuck", "cant_reach"): _simulate_cant_reach,
    ("limit_misconfig", "cant_reach"): _simulate_cant_reach,
    ("motor_dead", "self_collision"): _simulate_locked_self_collision,
    ("joint_stuck", "self_collision"): _simulate_locked_self_collision,
    ("motor_dead", "reduced_workspace"): _simulate_reduced_workspace,
    ("joint_stuck", "reduced_workspace"): _simulate_reduced_workspace,
    ("limit_misconfig", "reduced_workspace"): _simulate_reduced_workspace,
}


# --- auto-hypothesis generation --------------------------------------------
# When the caller supplies no hypotheses, derive candidates from the symptom:
# rank the suspect joints with `rank_root_causes` (graph precision×recall over
# the affected links) and propose a parameter-free `motor_dead` for each. The
# parametric modes (joint_stuck angle, limit_misconfig bounds) can't be guessed,
# so auto-generation stays with the one mode that needs no parameters; the
# best-first loop then finds whichever ranked joint actually reproduces it.

def _observed_links(symptom: Symptom) -> list[str]:
    if symptom.kind in ("cant_reach", "reduced_workspace") and symptom.target_link:
        return [symptom.target_link]
    if symptom.kind == "self_collision" and symptom.colliding_links:
        return list(symptom.colliding_links)
    return []


def _auto_hypotheses(robot: Robot, symptom: Symptom, max_auto: int) -> list[Hypothesis]:
    # A dead motor only makes sense on an actuated joint — drop fixed ones (and
    # any link names rank_root_causes may surface).
    movable = {j.name for j in robot.joints if j.type != "fixed"}
    observed = _observed_links(symptom)
    if observed:
        ranked = [c.target for c in rank_root_causes(robot, observed)]
    else:
        # self_collision with no reported pair: the commanded joints are the
        # natural suspects (no ranking signal to order them).
        ranked = list(symptom.at_config)
    joints = [j for j in ranked if j in movable][:max_auto]
    return [Hypothesis(suspect_joint=j, fault_mode="motor_dead") for j in joints]


def diagnose(
    robot: Robot, symptom: Symptom,
    hypotheses: Optional[list[Hypothesis]] = None, *, max_auto: int = 5,
) -> DiagnoseReport:
    """Run the two-tier loop. If ``hypotheses`` is omitted (or empty), diagnose
    ranks suspect joints from the symptom and tests a ``motor_dead`` on each (up
    to ``max_auto``); the returned report has ``auto_generated=True``."""
    link_names = {l.name for l in robot.links}
    joint_names = {j.name for j in robot.joints}
    if symptom.kind in ("cant_reach", "reduced_workspace"):
        if symptom.target_link not in link_names:
            raise KeyError(f"unknown target_link: {symptom.target_link!r}")
    elif symptom.kind == "self_collision":
        for jn in symptom.at_config:
            if jn not in joint_names:
                raise KeyError(f"unknown joint in at_config: {jn!r}")
        for ln in symptom.colliding_links or ():
            if ln not in link_names:
                raise KeyError(f"unknown link in colliding_links: {ln!r}")

    auto_generated = not hypotheses
    if auto_generated:
        hypotheses = _auto_hypotheses(robot, symptom, max_auto)
        if not hypotheses:
            return DiagnoseReport(
                verdict=Verdict.INCONCLUSIVE, tier=1, confidence=0.0, auto_generated=True,
                evidence={"auto_candidates": []},
                summary="No hypotheses supplied and none could be ranked for this symptom.",
            )

    report = _run_tiers(robot, symptom, hypotheses)
    if auto_generated:
        report.auto_generated = True
        report.evidence.setdefault("auto_candidates", [h.suspect_joint for h in hypotheses])
    return report


def _run_tiers(robot: Robot, symptom: Symptom, hypotheses: list[Hypothesis]) -> DiagnoseReport:
    # Tier 0 — one coarse static scan; a malformation on a suspect explains it.
    # Only meaningful for cant_reach: a zeroed-effort joint (R003) is a dead
    # motor. A static rule says nothing about a configuration-dependent collision.
    if symptom.kind == "cant_reach":
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
