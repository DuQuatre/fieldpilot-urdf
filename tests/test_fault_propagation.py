"""Graph-based fault-propagation / root-cause tests."""
from __future__ import annotations

import pytest

from fieldpilot_urdf.fault_propagation import (
    affected_links, criticality, rank_root_causes,
)
from fieldpilot_urdf.models import Inertial, Joint, JointLimit, Link, Origin, Robot


def _lim():
    return JointLimit(lower=-1, upper=1, effort=5, velocity=1)


def _chain(masses=(1.0, 1.0, 1.0)):
    """base -[j1]-> l1 -[j2]-> l2, with per-link masses (base, l1, l2)."""
    mb, m1, m2 = masses
    return Robot(
        name="chain",
        links=[
            Link(name="base", inertial=Inertial(mass=mb)),
            Link(name="l1", inertial=Inertial(mass=m1)),
            Link(name="l2", inertial=Inertial(mass=m2)),
        ],
        joints=[
            Joint(name="j1", type="revolute", parent="base", child="l1",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 0, 1), limit=_lim()),
            Joint(name="j2", type="revolute", parent="l1", child="l2",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 0, 1), limit=_lim()),
        ],
    )


def _fork():
    """base -[j1]-> l1 ; base -[j2]-> l2 (two independent branches)."""
    return Robot(
        name="fork",
        links=[Link(name="base", inertial=Inertial(mass=1.0)),
               Link(name="l1", inertial=Inertial(mass=1.0)),
               Link(name="l2", inertial=Inertial(mass=1.0))],
        joints=[
            Joint(name="j1", type="revolute", parent="base", child="l1",
                  axis=(0, 0, 1), limit=_lim()),
            Joint(name="j2", type="revolute", parent="base", child="l2",
                  axis=(0, 0, 1), limit=_lim()),
        ],
    )


# --- affected_links --------------------------------------------------------

def test_affected_links_of_joint():
    assert affected_links(_chain(), "j1") == {"l1", "l2"}
    assert affected_links(_chain(), "j2") == {"l2"}


def test_affected_links_of_link():
    assert affected_links(_chain(), "l1") == {"l1", "l2"}
    assert affected_links(_chain(), "l2") == {"l2"}


def test_affected_links_unknown_id():
    with pytest.raises(KeyError):
        affected_links(_chain(), "nope")


# --- criticality -----------------------------------------------------------

def test_criticality_mass_weighted():
    # base=2, l1=1, l2=1 ; total=4. Downstream of j1 = {l1,l2} = 2/4 = 0.5
    r = _chain(masses=(2.0, 1.0, 1.0))
    assert criticality(r, "j1") == pytest.approx(0.5)
    assert criticality(r, "j2") == pytest.approx(0.25)  # {l2} = 1/4


def test_criticality_zero_mass():
    r = Robot(name="m", links=[Link(name="base"), Link(name="l1")],
              joints=[Joint(name="j", type="fixed", parent="base", child="l1")])
    assert criticality(r, "j") == 0.0


# --- rank_root_causes ------------------------------------------------------

def test_rank_specificity_tiebreak():
    # Observing only l2: both j1 (downstream {l1,l2}) and j2 (downstream {l2})
    # cover it, but j2 is more specific (precision 1.0) and must rank first.
    ranked = rank_root_causes(_chain(), {"l2"})
    assert [c.target for c in ranked] == ["j2", "j1"]
    assert ranked[0].precision == pytest.approx(1.0)
    assert ranked[0].recall == pytest.approx(1.0)
    assert ranked[1].precision == pytest.approx(0.5)


def test_rank_localises_to_branch():
    # Fork: observing l1 should pin j1, and j2 (disjoint branch) drops out.
    ranked = rank_root_causes(_fork(), {"l1"})
    assert [c.target for c in ranked] == ["j1"]


def test_rank_empty_observation():
    assert rank_root_causes(_chain(), set()) == []


def test_rank_unknown_link():
    with pytest.raises(KeyError):
        rank_root_causes(_chain(), {"ghost"})


def test_rank_top_k_and_consider_links():
    ranked = rank_root_causes(_chain(), {"l2"}, consider_links=True, top_k=2)
    assert len(ranked) == 2
    # links l1/l2 and joints j1/j2 are all candidates; the most specific
    # (link l2 or joint j2, downstream {l2}) leads.
    assert ranked[0].predicted_links == ["l2"]


def test_candidate_is_pydantic_and_strict():
    c = rank_root_causes(_chain(), {"l2"})[0]
    dumped = c.model_dump()
    assert set(dumped) == {"target", "score", "precision", "recall", "predicted_links"}
