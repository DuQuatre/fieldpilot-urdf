"""Differential diagnosis: ask the question that best narrows the candidates.

When several faults could explain a symptom, the way forward is to ask the
technician the *most discriminating* question — the test whose answer best
splits the remaining candidates — then fold the answer back in and repeat until
one fault dominates. This module is the information-theoretic core of that loop.

Given:

* **candidates** — competing fault hypotheses, each with a prior
  (:class:`Candidate`; seed them from `rank_root_causes` / `localize_joint_fault`
  via :func:`candidates_from_scores`), and
* **questions** — observations/tests the tech can report, each with the
  predicted outcome likelihood per candidate (:class:`Question`),

it computes, for every unanswered question, the **expected information gain**
(mutual information, in bits) over the candidate distribution
(:func:`rank_questions`), and Bayesian-updates the posterior beliefs as answers
arrive (:func:`update_beliefs`). It is **stateless** — every call is a pure
function of ``(candidates, questions, answers-so-far)`` — so an n8n / LLM
front-end can own the dialog and persistence while this owns the maths.

Naive-Bayes independence is assumed across questions (answers are conditionally
independent given the fault) — the standard, transparent model for this loop.
Pure Python.
"""
from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

_PROB_TOL = 1e-6


class Candidate(BaseModel):
    """A competing fault hypothesis with a prior weight."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique candidate identifier")
    prior: float = Field(1.0, ge=0.0, description="Prior weight (need not be normalized; >0)")


class Question(BaseModel):
    """A test/observation the technician can report, and what each candidate
    predicts its outcome will be.

    ``likelihoods[candidate_name][outcome]`` is ``P(outcome | candidate)`` and
    each candidate's row must sum to 1 over ``outcomes``. A candidate absent from
    ``likelihoods`` is treated as uninformative (uniform over outcomes) — the
    question says nothing about it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique question identifier")
    text: str = Field(..., description="Human-readable question to ask the technician")
    outcomes: list[str] = Field(..., min_length=1, description="Possible answers")
    likelihoods: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="candidate name -> {outcome -> P(outcome | candidate)}")
    cost: float = Field(1.0, gt=0.0, description="Relative effort to obtain this answer")

    @model_validator(mode="after")
    def _check(self) -> "Question":
        if len(set(self.outcomes)) != len(self.outcomes):
            raise ValueError(f"question {self.id!r}: duplicate outcomes")
        oset = set(self.outcomes)
        for cand, row in self.likelihoods.items():
            extra = set(row) - oset
            if extra:
                raise ValueError(f"question {self.id!r}, candidate {cand!r}: "
                                 f"likelihood for unknown outcome(s) {sorted(extra)}")
            total = sum(row.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"question {self.id!r}, candidate {cand!r}: "
                                 f"likelihoods sum to {total:.6g}, must sum to 1")
        return self

    def likelihood(self, candidate_name: str, outcome: str) -> float:
        """``P(outcome | candidate)``; uniform for an unmodeled candidate."""
        row = self.likelihoods.get(candidate_name)
        if row is None:
            return 1.0 / len(self.outcomes)
        return row.get(outcome, 0.0)


class QuestionScore(BaseModel):
    """One question's discriminating power under the current beliefs."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Question identifier")
    text: str = Field(..., description="The question to ask")
    info_gain: float = Field(..., description="Expected entropy reduction (mutual information), bits")
    info_gain_per_cost: float = Field(..., description="info_gain / question cost")


class BeliefState(BaseModel):
    """Posterior beliefs over the candidates after the answers so far."""

    model_config = ConfigDict(extra="forbid")

    posteriors: dict[str, float] = Field(..., description="Normalized posterior per candidate")
    entropy_bits: float = Field(..., description="Shannon entropy of the posterior (bits)")
    leading: str = Field(..., description="Most probable candidate")
    leading_prob: float = Field(..., description="Its posterior probability")
    resolved: bool = Field(..., description="leading_prob >= the resolve threshold")


def _entropy_bits(probs) -> float:
    h = 0.0
    for p in probs:
        if p > 0.0:
            h -= p * math.log2(p)
    return h


def _normalized_priors(candidates: list[Candidate]) -> dict[str, float]:
    if not candidates:
        raise ValueError("need at least one candidate")
    names = [c.name for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    weights = {c.name: max(0.0, c.prior) for c in candidates}
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("candidate priors sum to 0")
    return {k: v / total for k, v in weights.items()}


def _apply_answers(
    p: dict[str, float], by_id: dict[str, Question], answers: dict[str, str],
) -> dict[str, float]:
    """Bayesian-update beliefs ``p`` with each answered ``{question_id: outcome}``."""
    for qid, outcome in answers.items():
        q = by_id.get(qid)
        if q is None:
            raise KeyError(f"answer references unknown question id {qid!r}")
        if outcome not in q.outcomes:
            raise ValueError(f"question {qid!r}: unknown outcome {outcome!r}")
        joint = {name: p[name] * q.likelihood(name, outcome) for name in p}
        total = sum(joint.values())
        if total <= 0.0:
            # Contradictory evidence (every candidate ruled this outcome out):
            # a modeling inconsistency — leave beliefs unchanged rather than
            # divide by zero, so the dialog can continue.
            continue
        p = {k: v / total for k, v in joint.items()}
    return p


def update_beliefs(
    candidates: list[Candidate],
    questions: list[Question],
    answers: Optional[dict[str, str]] = None,
    *,
    resolve_threshold: float = 0.9,
) -> BeliefState:
    """Posterior beliefs over ``candidates`` given the ``answers`` so far
    (``{question_id: outcome}``).

    Each answer multiplies in its likelihood (naive-Bayes) and renormalizes; an
    outcome a candidate deems impossible drives it to zero. ``resolved`` is True
    once the leading candidate's posterior reaches ``resolve_threshold``. Raises
    ``KeyError`` for an unknown question id and ``ValueError`` for an unknown
    outcome.
    """
    answers = answers or {}
    by_id = {q.id: q for q in questions}
    p = _apply_answers(_normalized_priors(candidates), by_id, answers)
    leading = max(p, key=p.get)
    return BeliefState(
        posteriors=p,
        entropy_bits=_entropy_bits(p.values()),
        leading=leading,
        leading_prob=p[leading],
        resolved=p[leading] >= resolve_threshold,
    )


def rank_questions(
    candidates: list[Candidate],
    questions: list[Question],
    answers: Optional[dict[str, str]] = None,
) -> list[QuestionScore]:
    """Rank the *unanswered* questions by expected information gain (bits) under
    the beliefs implied by ``answers`` so far, best-first.

    A question's gain is the mutual information between the fault and its outcome:
    current entropy minus the expected posterior entropy over the question's
    predicted outcomes. A perfectly discriminating yes/no question on a 50/50
    pair scores ~1 bit; a question whose outcome distribution is identical across
    candidates scores ~0. Ties broken by ``info_gain_per_cost`` then id.
    """
    answers = answers or {}
    by_id = {q.id: q for q in questions}
    p = _apply_answers(_normalized_priors(candidates), by_id, answers)
    h_now = _entropy_bits(p.values())

    scores: list[QuestionScore] = []
    for q in questions:
        if q.id in answers:
            continue
        expected_post_h = 0.0
        for outcome in q.outcomes:
            joint = {name: p[name] * q.likelihood(name, outcome) for name in p}
            p_outcome = sum(joint.values())
            if p_outcome <= 0.0:
                continue
            post = {k: v / p_outcome for k, v in joint.items()}
            expected_post_h += p_outcome * _entropy_bits(post.values())
        gain = max(0.0, h_now - expected_post_h)     # MI >= 0 (clamp fp noise)
        scores.append(QuestionScore(
            id=q.id, text=q.text, info_gain=gain, info_gain_per_cost=gain / q.cost))
    scores.sort(key=lambda s: (s.info_gain, s.info_gain_per_cost, s.id), reverse=True)
    return scores


def next_question(
    candidates: list[Candidate],
    questions: list[Question],
    answers: Optional[dict[str, str]] = None,
    *,
    min_gain: float = _PROB_TOL,
) -> Optional[QuestionScore]:
    """The single most discriminating unanswered question, or ``None`` when none
    remain or none would narrow the candidates (top gain below ``min_gain``)."""
    ranked = rank_questions(candidates, questions, answers)
    if not ranked or ranked[0].info_gain < min_gain:
        return None
    return ranked[0]


def candidates_from_scores(scores: dict[str, float]) -> list[Candidate]:
    """Build :class:`Candidate`s from a ``{name: score}`` mapping (e.g. the
    scores from `rank_root_causes` or the explained fractions from
    `localize_joint_fault`), using the scores as priors. Non-positive scores are
    floored to 0; raises if nothing positive remains."""
    cands = [Candidate(name=n, prior=max(0.0, float(s))) for n, s in scores.items()]
    if not cands or sum(c.prior for c in cands) <= 0.0:
        raise ValueError("scores must contain at least one positive value")
    return cands
