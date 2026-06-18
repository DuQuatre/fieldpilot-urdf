"""End-to-end diagnostics workflow — localise → narrow → calibrate → record → show.

The runnable companion to the Layer-4 diagnostics story. One miscalibrated joint
sends a robot's tool off where the model says it should be; this walks the whole
loop that finds it, fixes it, remembers it, and draws it — every number from a
real call, fully offline and deterministic:

    1. Symptom      the tool is measured off the commanded pose
    2. Localise     localize_joint_fault — which joint best explains the deviation     (1.14)
    3. Prior        fault_priors from past cases sharpen the candidate set             (1.18)
    4. Dialog       rank_questions / update_beliefs — ask, narrow, resolve             (1.17)
    5. Calibrate    calibrate_joint_offsets — recover the exact offset                 (1.15)
    6. Recommend    recommend_solution — the best-proven fix for this fault            (1.18)
    7. Record       save_case — so the next diagnosis starts smarter                   (1.18)
    8. Show         3D fault-motion + oscilloscope traces (optional [viz] extra)        (1.19/1.20)
    9. Report       photos + illustrations + spare parts -> French HTML report          (1.22/1.23)
   10. Sync         report -> Gotenberg PDF request + Odoo project.task payload          (1.24)
   11. Notify       report -> Telegram bot replies to the technician                     (1.25)
   12. Order        spare parts -> Odoo SPA sale.order payload                            (1.26)
   13. Dashboard    case base -> KPI summary for the admin/MRR dashboard                  (1.27)
   14. Digest       KPIs -> weekly French email digest                                    (1.28)

    pip install fieldpilot-urdf            # steps 1–7, 9–14 (core)
    pip install "fieldpilot-urdf[viz]"     # + step 8 visuals & report illustrations
    python examples/diagnostics_workflow.py
"""
from __future__ import annotations

import base64
import math
from pathlib import Path
from typing import Optional

from fieldpilot_urdf import (
    DiagnosticCase, DiagnosticReport, Joint, JointLimit, Link, Origin, PoseObservation,
    Question, Robot, SparePart, build_simulation_illustrations, calibrate_joint_offsets,
    candidates_from_scores, case_stats_summary, fault_priors, forward_kinematics, gotenberg_request,
    intervention_task_vals, list_cases, load_cases, localize_joint_fault,
    next_question, photo_requests, recommend_solution, render_report_html,
    report_summary_text, save_case, spare_parts_order_vals, telegram_messages,
    unresolved_part_refs, update_beliefs, weekly_digest,
)
from fieldpilot_urdf.fk import R_to_rpy

# Which spare parts each fix needs (in production: the SPA spare-parts catalogue).
SPARES: dict[str, list[SparePart]] = {
    "recalibrate_encoder": [
        SparePart(reference="ENC-1024", name="Codeur incrémental 1024 ppr"),
        SparePart(reference="CAL-KIT", name="Kit d'étalonnage articulaire"),
    ],
}

TRUE_FAULT = "j_shoulder"      # the joint we secretly miscalibrate
TRUE_OFFSET = 0.05            # rad it is off by (the robot's reality vs the model)


def banner(title: str) -> None:
    print("\n" + "─" * 74)
    print(f"  {title}")
    print("─" * 74)


def build_arm() -> Robot:
    """A 3-DOF serial arm: base yaw + shoulder & elbow pitch."""
    def lim() -> JointLimit:
        return JointLimit(lower=-2.9, upper=2.9, effort=150, velocity=3.0)

    return Robot(
        name="diag_arm",
        links=[Link(name="base"), Link(name="link1"), Link(name="link2"), Link(name="tool")],
        joints=[
            Joint(name="j_base", type="revolute", parent="base", child="link1",
                  origin=Origin(xyz=(0, 0, 0.5)), axis=(0, 0, 1), limit=lim()),
            Joint(name="j_shoulder", type="revolute", parent="link1", child="link2",
                  origin=Origin(xyz=(0.9, 0, 0)), axis=(0, 1, 0), limit=lim()),
            Joint(name="j_elbow", type="revolute", parent="link2", child="tool",
                  origin=Origin(xyz=(0.9, 0, 0)), axis=(0, 1, 0), limit=lim()),
        ],
    )


def _observed_pose(robot: Robot, commanded: dict[str, float]):
    """Where the tool actually lands: the model command plus the true offset."""
    actual = dict(commanded)
    actual[TRUE_FAULT] = actual.get(TRUE_FAULT, 0.0) + TRUE_OFFSET
    T = forward_kinematics(robot, actual)["tool"]
    return tuple(T[:3, 3]), tuple(R_to_rpy(T[:3, :3]))


def _seed_history(root: Path) -> None:
    """A few past cases so priors + solution stats have something to learn from."""
    history = [
        DiagnosticCase(id="INT-2026-0007", machine="diag_arm", confirmed_fault="j_shoulder",
                       solution="recalibrate_encoder", resolved=True),
        DiagnosticCase(id="INT-2026-0019", machine="diag_arm", confirmed_fault="j_shoulder",
                       solution="recalibrate_encoder", resolved=True),
        DiagnosticCase(id="INT-2026-0023", machine="diag_arm", confirmed_fault="j_shoulder",
                       solution="replace_gearbox", resolved=False),
        DiagnosticCase(id="INT-2026-0031", machine="diag_arm", confirmed_fault="j_elbow",
                       solution="reseat_connector", resolved=True),
    ]
    for c in history:
        save_case(c, root=root)


# The technician-facing questions and what each candidate predicts. (In production
# the LLM front-end generates these; here they're fixed so the run is deterministic.)
def _questions() -> list[Question]:
    def yn(qid, text, p_yes):
        return Question(id=qid, text=text, outcomes=["yes", "no"],
                        likelihoods={c: {"yes": p, "no": 1 - p} for c, p in p_yes.items()})
    return [
        yn("vertical", "Is the tool error mostly in the vertical plane?",
           {"j_base": 0.1, "j_shoulder": 0.85, "j_elbow": 0.85}),
        yn("grows_reach", "Does the error grow as the arm extends from the shoulder?",
           {"j_base": 0.3, "j_shoulder": 0.9, "j_elbow": 0.4}),
        yn("wrist_play", "Is there noticeable play out at the wrist/elbow?",
           {"j_base": 0.15, "j_shoulder": 0.15, "j_elbow": 0.85}),
    ]


def main(out_dir: Optional[Path] = None) -> dict:
    print("=" * 74)
    print("  fieldpilot-urdf — end-to-end diagnostics workflow")
    print("=" * 74)
    robot = build_arm()

    # 1. Symptom: command a pose, but the tool is measured somewhere else.
    banner("1. SYMPTOM — the tool isn't where the model says")
    commanded = {"j_base": 0.4, "j_shoulder": -0.6, "j_elbow": 0.8}
    obs_xyz, obs_rpy = _observed_pose(robot, commanded)
    expected = forward_kinematics(robot, commanded)["tool"][:3, 3]
    drift = math.dist(expected, obs_xyz)
    print(f"  commanded  {commanded}")
    print(f"  expected tool   ({expected[0]:.3f}, {expected[1]:.3f}, {expected[2]:.3f})")
    print(f"  measured tool   ({obs_xyz[0]:.3f}, {obs_xyz[1]:.3f}, {obs_xyz[2]:.3f})")
    print(f"  -> drift {drift * 1000:.1f} mm: something is miscalibrated.")

    # 2. Localise kinematically: which joint best explains the deviation?
    banner("2. LOCALISE — rank joints by the geometric Jacobian")
    cands = localize_joint_fault(robot, "tool", commanded, obs_xyz, obs_rpy)
    for c in cands:
        print(f"  {c.joint:12s} offset≈{c.estimated_offset:+.3f}  "
              f"explains {c.explained_fraction:.0%}")
    localized = {c.joint: c.explained_fraction for c in cands if c.explained_fraction > 0}

    # 3. Prior from case history: weight candidates by how often each fault occurs.
    banner("3. PRIOR — sharpen with fault frequencies from past cases")
    root = (out_dir / "cases") if out_dir is not None else Path("diagnostics_output/cases")
    _seed_history(root)
    priors = fault_priors(load_cases(root=root),
                          machine="diag_arm", candidates=list(localized))
    combined = {j: localized.get(j, 0.0) * priors.get(j, 0.0) for j in localized}
    print(f"  fault_priors(history)  -> { {k: round(v, 2) for k, v in priors.items()} }")
    print(f"  localise × prior       -> { {k: round(v, 3) for k, v in combined.items()} }")

    # 4. Differential dialog: ask the most discriminating questions until resolved.
    banner("4. DIALOG — ask, narrow, resolve")
    candidates = candidates_from_scores(combined)
    questions = _questions()
    answers: dict[str, str] = {}
    while True:
        state = update_beliefs(candidates, questions, answers)
        print(f"  beliefs { {k: round(v, 2) for k, v in state.posteriors.items()} }  "
              f"H={state.entropy_bits:.2f} bits")
        if state.resolved:
            break
        nq = next_question(candidates, questions, answers)
        if nq is None:
            break
        # Oracle: the real robot answers per the true fault's likelihoods.
        q = next(q for q in questions if q.id == nq.id)
        answers[nq.id] = "yes" if q.likelihoods[TRUE_FAULT]["yes"] >= 0.5 else "no"
        print(f"   ask “{nq.text}” (gain {nq.info_gain:.2f} bits) -> {answers[nq.id]}")
    resolved_fault = state.leading
    print(f"  => resolved: {resolved_fault}  (p={state.leading_prob:.0%})")

    # 5. Calibrate: recover the exact offset from a handful of measurements.
    banner("5. CALIBRATE — recover the offset across several poses")
    poses = [{"j_base": 0.2 * k, "j_shoulder": -0.6 + 0.2 * k, "j_elbow": 0.3 * k} for k in range(5)]
    observations = [PoseObservation(commanded_q=p, observed_xyz=_observed_pose(robot, p)[0],
                                    observed_rpy=_observed_pose(robot, p)[1]) for p in poses]
    cal = calibrate_joint_offsets(robot, "tool", observations)
    print(f"  calibrate -> offsets { {k: round(v, 3) for k, v in cal.offsets.items()} }")
    print(f"  position RMS {cal.position_rms_before * 1000:.1f} mm -> "
          f"{cal.position_rms_after * 1000:.2f} mm")

    # 6. Recommend the fix with the best track record for this fault.
    banner("6. RECOMMEND — the best-proven fix")
    fix = recommend_solution(load_cases(root=root), resolved_fault)
    print(f"  recommend_solution({resolved_fault}) -> {fix}")

    # 7. Record this case so the next diagnosis starts smarter.
    banner("7. RECORD — remember the outcome")
    save_case(DiagnosticCase(id="INT-2026-0042", machine="diag_arm",
                             symptom=f"tool drift {drift * 1000:.0f} mm",
                             candidates=list(localized), answers=answers,
                             confirmed_fault=resolved_fault, solution=fix, resolved=True),
              root=root)
    print(f"  saved case INT-2026-0042; case base now holds {len(list_cases(root=root))} cases")

    # 8. Visuals (optional): 3D fault-motion + oscilloscope traces.
    banner("8. SHOW — 3D motion + scope traces (optional [viz] extra)")
    illustrations = _build_illustrations(robot, out_dir)

    # 9. Report: ask the tech for photos, attach the illustrations (confirmed),
    #    list the spare parts, and render the French HTML report for HTML->PDF.
    banner("9. REPORT — assemble the field report")
    symptom_fr = f"dérive de l'outil de {drift * 1000:.0f} mm"
    print("  ask the technician for:")
    for req in photo_requests(joints=[resolved_fault], symptom=symptom_fr):
        print(f"    • {req}")
    report = DiagnosticReport(
        reference="INT-2026-0042", machine="diag_arm",
        symptom=symptom_fr,
        confirmed=True, fault=resolved_fault, confidence=state.leading_prob,
        solution=fix, spare_parts=SPARES.get(fix, []),
        calibration={resolved_fault: round(cal.offsets.get(resolved_fault, 0.0), 4)},
        illustrations=illustrations,
    )
    html = render_report_html(report)
    report_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "rapport.html"
        report_path.write_text(html, encoding="utf-8")
        print(f"  wrote {report_path} ({len(html)} chars, "
              f"{len(report.spare_parts)} spare part(s), {len(illustrations)} illustration(s))")
    else:
        print(f"  rendered report HTML ({len(html)} chars) — pass out_dir to save")

    # 10. Sync: the requests/payloads the SaaS sends to make the Odoo PDF.
    banner("10. SYNC — report -> Gotenberg PDF -> Odoo intervention")
    gb = gotenberg_request(html, filename=f"rapport_{report.reference}.pdf")
    task_vals = intervention_task_vals(report, pdf_url="<set after upload>")
    print(f"  Gotenberg: POST {gb.endpoint} -> {gb.output_filename}")
    print(f"  Odoo project.task update: {task_vals}")
    print("  (the SaaS/n8n executes these; the library only builds the payloads)")

    # 11. Notify: the Telegram replies the bot sends back to the technician.
    banner("11. NOTIFY — reply to the technician on Telegram")
    tg = telegram_messages(report, chat_id="<tech_chat_id>")
    for m in tg:
        extra = f" + {m.attachment.field}" if m.attachment else ""
        print(f"  {m.method}{extra}")
    print(f"  summary:\n    " + report_summary_text(report, parse_mode="").replace("\n", "\n    "))

    # 12. Order: the spare parts become an Odoo SPA sale.order.
    banner("12. ORDER — spare parts -> Odoo SPA sale.order")
    product_map = {"ENC-1024": 1001}            # what the SPA catalogue already knows
    order = spare_parts_order_vals(report, partner_id=42, product_map=product_map)
    missing = unresolved_part_refs(report.spare_parts, product_map)
    print(f"  Odoo sale.order: partner {order['partner_id']}, origin {order['origin']}, "
          f"{len(order['order_line'])} line(s)")
    print(f"  parts needing a product in Odoo first: {missing}")

    # 13. Dashboard: roll the case base up into KPIs for the admin/MRR dashboard.
    banner("13. DASHBOARD — case stats for the MRR dashboard")
    stats = case_stats_summary(load_cases(root=root))
    print(f"  cases: {stats.total_cases}, resolved: {stats.resolved_cases} "
          f"({stats.resolution_rate:.0%}), distinct faults: {stats.distinct_faults}")
    if stats.top_faults:
        tf = stats.top_faults[0]
        print(f"  top fault: {tf.fault} ({tf.count}× — {tf.share:.0%} of confirmed)")
    if stats.top_solutions:
        ts = stats.top_solutions[0]
        print(f"  best fix: {ts.solution} ({ts.successes}/{ts.attempts} = {ts.success_rate:.0%})")

    # 14. Digest: the weekly email that pushes those KPIs to the managers.
    banner("14. DIGEST — weekly email of the KPIs")
    digest = weekly_digest(stats, period_label="semaine en cours")
    print(f"  subject: {digest.subject}")
    print(f"  html {len(digest.html_body)} chars, text {len(digest.text_body)} chars")

    print("\n" + "=" * 74)
    print(f"  Diagnosed {resolved_fault} (offset {cal.offsets.get(resolved_fault, 0):+.3f} rad), "
          f"fix: {fix}.")
    print("=" * 74)
    return {
        "localized_top": cands[0].joint if cands else None,
        "resolved_fault": resolved_fault,
        "calibrated_offset": cal.offsets.get(resolved_fault),
        "recommended": fix,
        "report_html": html,
        "report_path": str(report_path) if report_path else None,
        "odoo_task_vals": task_vals,
        "gotenberg_filename": gb.output_filename,
        "telegram_methods": [m.method for m in tg],
        "spa_order_lines": len(order["order_line"]),
        "spa_unresolved": missing,
        "dashboard_total_cases": stats.total_cases,
        "dashboard_top_fault": stats.top_faults[0].fault if stats.top_faults else None,
        "digest_subject": digest.subject,
    }


def _build_illustrations(robot: Robot, out_dir: Optional[Path]) -> list:
    """Render the simulation illustrations (3D motion + scope) as ReportImages,
    and write them to out_dir. Returns [] without the [viz] extra."""
    from fieldpilot_urdf.retime import TimedTrajectory

    sweep = [{"j_base": 0.4, "j_shoulder": -0.6 + 0.25 * k, "j_elbow": 0.8} for k in range(6)]
    faulted = [{**q, "j_shoulder": q["j_shoulder"] + TRUE_OFFSET} for q in sweep]
    times = [0.3 * k for k in range(len(sweep))]
    expected = TimedTrajectory(joint_ids=["j_shoulder"], times=times,
                               q=[[q["j_shoulder"]] for q in sweep], u=[[0.0]] * len(sweep))
    measured = TimedTrajectory(joint_ids=["j_shoulder"], times=times,
                               q=[[q["j_shoulder"]] for q in faulted], u=[[0.0]] * len(faulted))
    try:
        imgs = build_simulation_illustrations(robot, sweep, faulted, expected=expected,
                                              observed=measured, track_link="tool")
    except RuntimeError:
        print("  [skipped] visuals need:  pip install 'fieldpilot-urdf[viz]'")
        return []
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            (out_dir / img.name).write_bytes(base64.b64decode(img.data_b64))
        print(f"  wrote { [str(out_dir / i.name) for i in imgs] }")
    else:
        print(f"  rendered { [i.name for i in imgs] }")
    return imgs


if __name__ == "__main__":
    main(out_dir=Path("diagnostics_output"))
