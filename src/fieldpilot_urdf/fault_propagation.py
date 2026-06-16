"""Graph-based fault propagation and root-cause localisation over a URDF Robot.

Pure, deterministic reasoning on the kinematic graph — no sim, no network, no
API key. Ported from the MecAI project (MIT) and re-targeted onto the URDF
``Robot``: MecAI scored candidates by the *sensors* mounted downstream, but URDF
has no sensor model, so here the evidence and predictions are expressed directly
in **links**.

What's here
-----------
* :func:`affected_links` — links downstream of a faulty joint or link
  (the child link plus all its descendants in the kinematic tree).
* :func:`criticality` — mass-weighted impact of a fault, normalised by total
  robot mass.
* :func:`rank_root_causes` — given a set of links observed to be affected, score
  every joint (optionally every link) by *how well* its downstream set explains
  the observation (precision × recall, with a small specificity tie-breaker so
  the most specific explanation wins a tie).

This pairs with :mod:`fieldpilot_urdf.diagnose_core`: ``rank_root_causes`` turns
a set of symptomatic links into ranked suspect joints, which can then be fed as
:class:`~fieldpilot_urdf.diagnose_core.Hypothesis` candidates to ``diagnose``.

Closed-loop note: for parallel manipulators (Delta, Stewart) the end platform is
reachable from *any* actuator, so single-fault localisation by downstream impact
alone is ambiguous — every actuator predicts the same downstream set. The ranker
returns all such ties; disambiguating needs richer evidence.
"""
from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from .graph import build_graph
from .models import Robot

__all__ = ["RootCauseCandidate", "affected_links", "criticality", "rank_root_causes"]


def _downstream_of_link(g: nx.DiGraph, link: str) -> set[str]:
    return nx.descendants(g, link) | {link}


def _downstream_of(robot: Robot, g: nx.DiGraph, faulty_id: str) -> set[str]:
    """Downstream link set of a joint (its child + descendants) or a link
    (itself + descendants). Raises ``KeyError`` if ``faulty_id`` is neither."""
    joint = next((j for j in robot.joints if j.name == faulty_id), None)
    if joint is not None:
        return _downstream_of_link(g, joint.child)
    if faulty_id in {l.name for l in robot.links}:
        return _downstream_of_link(g, faulty_id)
    raise KeyError(f"'{faulty_id}' is not a joint or link of robot '{robot.name}'")


def affected_links(robot: Robot, faulty_id: str) -> set[str]:
    """Return the set of link names downstream of a faulty joint or link.

    For a joint, that's its child link plus every descendant; for a link, the
    link itself plus its descendants. A fault at ``faulty_id`` mechanically
    propagates to exactly these links.
    """
    return _downstream_of(robot, build_graph(robot), faulty_id)


def criticality(robot: Robot, faulty_id: str) -> float:
    """Fraction of total robot mass that lies in the downstream impact set.

    Returns ``0.0`` if the robot has zero total mass (under-specified inertia);
    callers can fall back to a count-based score in that case.
    """
    mass = {l.name: (l.inertial.mass if l.inertial else 0.0) for l in robot.links}
    total = sum(mass.values())
    if total == 0:
        return 0.0
    impacted = affected_links(robot, faulty_id)
    return sum(mass.get(lid, 0.0) for lid in impacted) / total


class RootCauseCandidate(BaseModel):
    """One ranked suspect from :func:`rank_root_causes`."""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(..., description="Joint (or link) name that could explain the observation")
    score: float = Field(..., description="precision × recall + specificity tie-breaker; higher is better")
    precision: float = Field(..., description="|observed ∩ predicted| / |predicted| — false-positive resistance")
    recall: float = Field(..., description="|observed ∩ predicted| / |observed| — coverage")
    predicted_links: list[str] = Field(..., description="The candidate's downstream link set")


def rank_root_causes(
    robot: Robot,
    observed_links: Iterable[str],
    *,
    consider_links: bool = False,
    top_k: int | None = None,
) -> list[RootCauseCandidate]:
    """Rank joints (and optionally links) by how well their downstream set
    explains the ``observed_links``.

    Scoring is ``precision × recall`` plus a small additive bonus (≤ 0.1)
    favouring smaller downstream sets, so when two candidates predict the same
    observed links the *more specific* one wins. Candidates with zero overlap
    are dropped; the result is sorted best-first.

    Raises ``KeyError`` if any observed link is unknown.
    """
    observed = set(observed_links)
    if not observed:
        return []
    known = {l.name for l in robot.links}
    unknown = observed - known
    if unknown:
        raise KeyError(f"unknown link names: {sorted(unknown)}")

    g = build_graph(robot)
    n_links = max(1, len(known))

    targets = [j.name for j in robot.joints]
    if consider_links:
        targets += [l.name for l in robot.links]

    candidates: list[RootCauseCandidate] = []
    for tid in targets:
        predicted = _downstream_of(robot, g, tid)
        overlap = predicted & observed
        if not overlap:
            continue
        precision = len(overlap) / len(predicted)
        recall = len(overlap) / len(observed)
        specificity_bonus = 0.1 * (1.0 - len(predicted) / n_links)
        candidates.append(
            RootCauseCandidate(
                target=tid,
                score=precision * recall + specificity_bonus,
                precision=precision,
                recall=recall,
                predicted_links=sorted(predicted),
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k] if top_k is not None else candidates
