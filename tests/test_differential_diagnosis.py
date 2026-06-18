"""Differential-diagnosis engine: information-gain question ranking + Bayesian
belief updates. Anchors: a perfectly discriminating yes/no question scores ~1
bit and an uninformative one ~0; updates flip/eliminate candidates correctly;
the dialog converges to a single fault.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from fieldpilot_urdf.differential_diagnosis import (
    BeliefState, Candidate, Question, QuestionScore, candidates_from_scores,
    next_question, rank_questions, update_beliefs,
)


def _two_faults():
    return [Candidate(name="motor_dead", prior=1.0), Candidate(name="backlash", prior=1.0)]


def _yesno(qid, text, p_yes_per_cand, cost=1.0):
    """A yes/no question; p_yes_per_cand maps candidate -> P(yes | candidate)."""
    return Question(
        id=qid, text=text, outcomes=["yes", "no"], cost=cost,
        likelihoods={c: {"yes": p, "no": 1.0 - p} for c, p in p_yes_per_cand.items()})


# --- information gain on a 50/50 pair ---------------------------------------

def test_perfectly_discriminating_question_is_one_bit():
    cands = _two_faults()
    q = _yesno("hold", "Does the joint hold position under load?",
               {"motor_dead": 0.0, "backlash": 1.0})    # answer splits them perfectly
    ranked = rank_questions(cands, [q])
    assert math.isclose(ranked[0].info_gain, 1.0, abs_tol=1e-9)   # full bit on a 50/50 prior


def test_uninformative_question_is_zero_bits():
    cands = _two_faults()
    q = _yesno("warm", "Worse when warm?", {"motor_dead": 0.6, "backlash": 0.6})  # same for both
    ranked = rank_questions(cands, [q])
    assert math.isclose(ranked[0].info_gain, 0.0, abs_tol=1e-12)


def test_partial_question_between_zero_and_one():
    cands = _two_faults()
    q = _yesno("noise", "Any grinding noise?", {"motor_dead": 0.5, "backlash": 0.9})
    g = rank_questions(cands, [q])[0].info_gain
    assert 0.0 < g < 1.0


def test_ranking_orders_by_gain():
    cands = _two_faults()
    qs = [
        _yesno("weak", "Weakly informative?", {"motor_dead": 0.4, "backlash": 0.6}),
        _yesno("strong", "Strongly informative?", {"motor_dead": 0.02, "backlash": 0.98}),
        _yesno("flat", "Uninformative?", {"motor_dead": 0.5, "backlash": 0.5}),
    ]
    ranked = rank_questions(cands, qs)
    assert [r.id for r in ranked] == ["strong", "weak", "flat"]
    assert all(isinstance(r, QuestionScore) for r in ranked)


# --- Bayesian update --------------------------------------------------------

def test_update_flips_belief_toward_the_answer():
    cands = _two_faults()
    q = _yesno("hold", "Holds under load?", {"motor_dead": 0.1, "backlash": 0.9})
    # "no" -> evidence for motor_dead
    st = update_beliefs(cands, [q], {"hold": "no"})
    assert st.leading == "motor_dead"
    assert st.posteriors["motor_dead"] > st.posteriors["backlash"]
    assert math.isclose(sum(st.posteriors.values()), 1.0, abs_tol=1e-9)


def test_zero_likelihood_outcome_eliminates_candidate():
    cands = _two_faults()
    # motor_dead NEVER produces "yes"; observing "yes" must rule it out entirely
    q = _yesno("spin", "Does it spin at all?", {"motor_dead": 0.0, "backlash": 1.0})
    st = update_beliefs(cands, [q], {"spin": "yes"})
    assert st.posteriors["motor_dead"] == 0.0
    assert math.isclose(st.posteriors["backlash"], 1.0, abs_tol=1e-12)
    assert st.resolved


def test_priors_bias_the_posterior():
    cands = [Candidate(name="common", prior=9.0), Candidate(name="rare", prior=1.0)]
    q = _yesno("t", "test", {"common": 0.5, "rare": 0.5})   # uninformative
    st = update_beliefs(cands, [q], {})
    assert math.isclose(st.posteriors["common"], 0.9, abs_tol=1e-9)   # prior carries through


def test_contradictory_answer_leaves_beliefs_unchanged():
    cands = _two_faults()
    # both candidates deem "maybe" impossible -> total likelihood 0 -> ignore it
    q = Question(id="q", text="?", outcomes=["yes", "no", "maybe"],
                 likelihoods={"motor_dead": {"yes": 0.5, "no": 0.5, "maybe": 0.0},
                              "backlash": {"yes": 0.4, "no": 0.6, "maybe": 0.0}})
    st = update_beliefs(cands, [q], {"q": "maybe"})
    assert math.isclose(st.posteriors["motor_dead"], 0.5, abs_tol=1e-9)
    assert math.isclose(st.posteriors["backlash"], 0.5, abs_tol=1e-9)


# --- the dialog loop converges ----------------------------------------------

def test_dialog_narrows_to_one_fault():
    cands = [Candidate(name="A"), Candidate(name="B"), Candidate(name="C")]
    # an oracle: the true fault is B. Each question reports whether a feature
    # matching B is present (B->yes, others->no, with a little noise).
    truth = "B"
    questions = [
        _yesno("q1", "feature 1?", {"A": 0.1, "B": 0.9, "C": 0.1}),
        _yesno("q2", "feature 2?", {"A": 0.1, "B": 0.85, "C": 0.15}),
        _yesno("q3", "feature 3?", {"A": 0.2, "B": 0.9, "C": 0.1}),
        _yesno("q4", "feature 4?", {"A": 0.15, "B": 0.88, "C": 0.12}),
    ]
    answers: dict[str, str] = {}
    for _ in range(len(questions)):
        st = update_beliefs(cands, questions, answers)
        if st.resolved:
            break
        nq = next_question(cands, questions, answers)
        assert nq is not None
        # the oracle answers "yes" for the true fault's features
        q = next(q for q in questions if q.id == nq.id)
        answers[nq.id] = "yes" if q.likelihoods[truth]["yes"] >= 0.5 else "no"
    final = update_beliefs(cands, questions, answers)
    assert final.leading == "B" and final.resolved
    assert final.entropy_bits < 0.6                       # collapsed from log2(3)≈1.58


def test_next_question_none_when_resolved_or_flat():
    cands = _two_faults()
    flat = _yesno("flat", "?", {"motor_dead": 0.5, "backlash": 0.5})
    assert next_question(cands, [flat]) is None           # no gain available
    # already answered -> excluded
    q = _yesno("hold", "?", {"motor_dead": 0.0, "backlash": 1.0})
    assert next_question(cands, [q], {"hold": "no"}) is None


# --- bridge + validation ----------------------------------------------------

def test_candidates_from_scores_normalizes():
    cands = candidates_from_scores({"j1": 0.6, "j2": 0.3, "j3": 0.0})
    st = update_beliefs(cands, [], {})
    assert math.isclose(st.posteriors["j1"], 0.6 / 0.9, abs_tol=1e-9)
    assert st.posteriors["j3"] == 0.0
    with pytest.raises(ValueError):
        candidates_from_scores({"a": 0.0, "b": -1.0})


def test_single_candidate_is_resolved():
    st = update_beliefs([Candidate(name="only")], [], {})
    assert st.resolved and st.leading == "only" and st.entropy_bits == 0.0


def test_validation_errors():
    with pytest.raises(ValueError):                       # row doesn't sum to 1
        Question(id="q", text="?", outcomes=["yes", "no"],
                 likelihoods={"a": {"yes": 0.5, "no": 0.4}})
    with pytest.raises(ValueError):                       # unknown outcome
        Question(id="q", text="?", outcomes=["yes", "no"],
                 likelihoods={"a": {"maybe": 1.0}})
    good = _yesno("q", "?", {"a": 0.5})
    with pytest.raises(KeyError):                          # unknown question id
        update_beliefs([Candidate(name="a")], [good], {"nope": "yes"})
    with pytest.raises(ValueError):                        # unknown outcome answered
        update_beliefs([Candidate(name="a")], [good], {"q": "perhaps"})


def test_unmodeled_candidate_is_uniform():
    # candidate 'C' not mentioned in the question -> uniform -> stays uninformed
    cands = [Candidate(name="A"), Candidate(name="C")]
    q = _yesno("q", "?", {"A": 1.0})            # only A modeled
    assert math.isclose(q.likelihood("C", "yes"), 0.5, abs_tol=1e-12)
    st = update_beliefs(cands, [q], {"q": "yes"})
    assert st.leading == "A"                    # A predicted 'yes' (1.0) > C uniform (0.5)
