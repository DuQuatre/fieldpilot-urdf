"""Generate the illustrations used by docs/diagnostics-guide.md.

Renders three figures into docs/images/ from the same shoulder-miscalibration
scenario as examples/diagnostics_workflow.py:

    fault_motion.gif   3D nominal vs faulted motion (render_motion_comparison)
    oscilloscope.png   expected vs observed joint position  (render_trajectory_scope)
    dialog_beliefs.png belief narrowing over the dialog      (render_scope)

Run after changing the visuals so the committed images stay in sync:

    pip install "fieldpilot-urdf[viz]"
    python docs/generate_illustrations.py
"""
from __future__ import annotations

from pathlib import Path

from fieldpilot_urdf import (
    Candidate, Joint, JointLimit, Link, Origin, Question, Robot,
    next_question, update_beliefs,
)
from fieldpilot_urdf.retime import TimedTrajectory
from fieldpilot_urdf.viz import (
    ScopePanel, ScopeSeries, render_motion_comparison, render_scope,
    render_trajectory_scope,
)

TRUE_FAULT = "j_shoulder"
TRUE_OFFSET = 0.05
IMAGES = Path(__file__).resolve().parent / "images"


def build_arm() -> Robot:
    def lim():
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


def _questions():
    def yn(qid, text, p_yes):
        return Question(id=qid, text=text, outcomes=["yes", "no"],
                        likelihoods={c: {"yes": p, "no": 1 - p} for c, p in p_yes.items()})
    return [
        yn("vertical", "Erreur dans le plan vertical ?",
           {"j_base": 0.1, "j_shoulder": 0.85, "j_elbow": 0.85}),
        yn("grows_reach", "L'erreur croît avec l'allonge ?",
           {"j_base": 0.3, "j_shoulder": 0.9, "j_elbow": 0.4}),
        yn("wrist_play", "Jeu au poignet/coude ?",
           {"j_base": 0.15, "j_shoulder": 0.15, "j_elbow": 0.85}),
    ]


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    robot = build_arm()

    # --- 3D fault motion + oscilloscope -----------------------------------
    sweep = [{"j_base": 0.4, "j_shoulder": -0.6 + 0.25 * k, "j_elbow": 0.8} for k in range(6)]
    faulted = [{**q, "j_shoulder": q["j_shoulder"] + TRUE_OFFSET} for q in sweep]
    (IMAGES / "fault_motion.gif").write_bytes(
        render_motion_comparison(robot, sweep, faulted, track_link="tool",
                                 labels=("nominal", "défaut"), layout="overlay"))

    times = [0.3 * k for k in range(len(sweep))]
    expected = TimedTrajectory(joint_ids=["j_shoulder"], times=times,
                               q=[[q["j_shoulder"]] for q in sweep], u=[[0.0]] * len(sweep))
    measured = TimedTrajectory(joint_ids=["j_shoulder"], times=times,
                               q=[[q["j_shoulder"]] for q in faulted], u=[[0.0]] * len(faulted))
    (IMAGES / "oscilloscope.png").write_bytes(
        render_trajectory_scope(expected, measured, signals=("position",),
                                labels=("attendu", "observé"), title="Position j_shoulder"))

    # --- belief narrowing over the dialog (reuse render_scope) ------------
    candidates = [Candidate(name="j_shoulder", prior=0.5),
                  Candidate(name="j_elbow", prior=0.3),
                  Candidate(name="j_base", prior=0.2)]
    questions = _questions()
    answers: dict[str, str] = {}
    history = {c.name: [] for c in candidates}
    steps = []
    for step in range(len(questions) + 1):
        state = update_beliefs(candidates, questions, answers)
        steps.append(float(step))
        for name, p in state.posteriors.items():
            history[name].append(p)
        nq = next_question(candidates, questions, answers)
        if nq is None or state.resolved:
            break
        q = next(q for q in questions if q.id == nq.id)
        answers[nq.id] = "yes" if q.likelihoods[TRUE_FAULT]["yes"] >= 0.5 else "no"
    panel = ScopePanel(ylabel="posterior P(fault)", series=[
        ScopeSeries(label=name, times=steps, values=vals) for name, vals in history.items()])
    (IMAGES / "dialog_beliefs.png").write_bytes(
        render_scope([panel], title="Differential dialog — belief narrowing",
                     xlabel="questions answered"))

    print(f"wrote {sorted(p.name for p in IMAGES.glob('*.*'))} into {IMAGES}")


if __name__ == "__main__":
    main()
