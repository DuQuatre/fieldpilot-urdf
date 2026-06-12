"""Tests for app/urdf/repair.py.

Each repairable rule: build a robot that violates it, run the fix, confirm
the rule no longer fires and the patch records what changed.

Run: python3 -m pytest app/urdf/test_repair.py -q  (from pydexpi-server/)
"""
from __future__ import annotations

import numpy as np
import pytest

from fieldpilot_urdf import from_xml, run_all
from fieldpilot_urdf.models import Inertia, Inertial, Joint, JointLimit, Link, Robot
from fieldpilot_urdf.repair import (
    DEFAULT_AXIS, DEFAULT_EFFORT, DEFAULT_MASS, DEFAULT_VELOCITY,
    UNFIXABLE_CODES, _project_psd, fix_r003, fix_r004, fix_r005, fix_r006,
    fix_r007, repair,
)


def _codes(findings):
    return {f.code for f in findings}


# --- R003 ------------------------------------------------------------------

def test_fix_r003_swaps_inverted_limits():
    r = Robot(
        name="x",
        links=[Link(name="a"), Link(name="b")],
        joints=[Joint(name="j", type="revolute", parent="a", child="b",
                      limit=JointLimit(lower=1.0, upper=0.5,
                                       effort=5, velocity=1))],
    )
    fixed, patches = fix_r003(r)
    assert fixed.joints[0].limit.lower == 0.5
    assert fixed.joints[0].limit.upper == 1.0
    assert any(p.field == "lower<->upper" for p in patches)
    assert "R003" not in _codes(run_all(fixed))


def test_fix_r003_defaults_nonpositive_effort_velocity():
    r = Robot(
        name="x",
        links=[Link(name="a"), Link(name="b")],
        joints=[Joint(name="j", type="revolute", parent="a", child="b",
                      limit=JointLimit(lower=-1, upper=1,
                                       effort=0, velocity=-2))],
    )
    fixed, patches = fix_r003(r)
    assert fixed.joints[0].limit.effort == DEFAULT_EFFORT
    assert fixed.joints[0].limit.velocity == DEFAULT_VELOCITY
    fields = {p.field for p in patches}
    assert {"effort", "velocity"} <= fields
    assert "R003" not in _codes(run_all(fixed))


def test_fix_r003_clean_robot_noop():
    r = from_xml(
        '<robot name="x"><link name="a"/><link name="b"/>'
        '<joint name="j" type="revolute"><parent link="a"/><child link="b"/>'
        '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="5" velocity="1"/>'
        '</joint></robot>'
    )
    fixed, patches = fix_r003(r)
    assert patches == []


# --- R004 ------------------------------------------------------------------

def test_fix_r004_positive_default_mass():
    r = Robot(
        name="x",
        links=[Link(name="a", inertial=Inertial(mass=-1.0, inertia=Inertia()))],
        joints=[],
    )
    fixed, patches = fix_r004(r)
    assert fixed.links[0].inertial.mass == DEFAULT_MASS
    assert patches[0].code == "R004"
    assert "R004" not in _codes(run_all(fixed))


# --- R005 ------------------------------------------------------------------

def test_project_psd_clips_negatives():
    M = np.array([[-1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    P = _project_psd(M, floor=0.0)
    eig = np.linalg.eigvalsh(P)
    assert eig.min() >= -1e-12


def test_fix_r005_makes_inertia_psd():
    bad = Inertial(mass=1.0, inertia=Inertia(ixx=-1.0, iyy=1.0, izz=1.0))
    r = Robot(name="x", links=[Link(name="a", inertial=bad)], joints=[])
    fixed, patches = fix_r005(r)
    assert patches[0].code == "R005"
    # Eigenvalues of repaired tensor are all non-negative.
    i = fixed.links[0].inertial.inertia
    M = np.array([[i.ixx, i.ixy, i.ixz],
                  [i.ixy, i.iyy, i.iyz],
                  [i.ixz, i.iyz, i.izz]])
    assert np.linalg.eigvalsh(M).min() >= -1e-12
    assert "R005" not in _codes(run_all(fixed))


def test_fix_r005_clean_inertia_noop():
    good = Inertial(mass=1.0, inertia=Inertia(ixx=1, iyy=2, izz=3))
    r = Robot(name="x", links=[Link(name="a", inertial=good)], joints=[])
    _, patches = fix_r005(r)
    assert patches == []


# --- R006 ------------------------------------------------------------------

def test_fix_r006_dedup_link_names():
    # Pydantic Robot won't validate duplicates via constructor, build by hand.
    r = Robot.model_construct(
        name="x",
        links=[Link(name="a"), Link(name="a"), Link(name="a")],
        joints=[],
    )
    fixed, patches = fix_r006(r)
    names = [l.name for l in fixed.links]
    assert len(set(names)) == len(names) == 3
    assert names[0] == "a"
    # patch records both duplicates
    assert len([p for p in patches if p.field == "link.name"]) == 2


def test_fix_r006_dedup_joint_names():
    r = Robot.model_construct(
        name="x",
        links=[Link(name="a"), Link(name="b"), Link(name="c")],
        joints=[
            Joint(name="j", type="fixed", parent="a", child="b"),
            Joint(name="j", type="fixed", parent="b", child="c"),
        ],
    )
    fixed, patches = fix_r006(r)
    names = [j.name for j in fixed.joints]
    assert len(set(names)) == 2
    assert any(p.field == "joint.name" for p in patches)


# --- R007 ------------------------------------------------------------------

def test_fix_r007_zero_axis_defaults():
    r = Robot(
        name="x",
        links=[Link(name="a"), Link(name="b")],
        joints=[Joint(name="j", type="revolute", parent="a", child="b",
                      axis=(0.0, 0.0, 0.0),
                      limit=JointLimit(lower=-1, upper=1, effort=5, velocity=1))],
    )
    fixed, patches = fix_r007(r)
    assert tuple(fixed.joints[0].axis) == DEFAULT_AXIS
    assert patches[0].code == "R007"
    assert "R007" not in _codes(run_all(fixed))


def test_fix_r007_skips_fixed_joints():
    """A zero axis on a fixed joint is fine; don't touch it."""
    r = Robot(
        name="x",
        links=[Link(name="a"), Link(name="b")],
        joints=[Joint(name="j", type="fixed", parent="a", child="b",
                      axis=(0.0, 0.0, 0.0))],
    )
    _, patches = fix_r007(r)
    assert patches == []


# --- orchestration ---------------------------------------------------------

def test_repair_all_clears_fixable_rules():
    r = Robot(
        name="x",
        links=[Link(name="a", inertial=Inertial(mass=-1.0, inertia=Inertia()))],
        joints=[],
    )
    new, patches, unfixable = repair(r)
    assert unfixable == []  # caller didn't ask for any specific codes
    assert patches  # something changed
    # R004 is gone; R001 (multiple roots? no, single link → no R001) is N/A here
    assert "R004" not in _codes(run_all(new))


def test_repair_reports_requested_unfixable_codes():
    r = Robot(name="x", links=[Link(name="a")], joints=[])
    _, _, unfixable = repair(r, codes=["R001", "R005", "R008"])
    assert set(unfixable) == {"R001", "R008"}


def test_repair_subset_only_runs_requested():
    r = Robot(
        name="x",
        links=[Link(name="a", inertial=Inertial(mass=-1.0, inertia=Inertia()))],
        joints=[Joint(name="j", type="revolute", parent="a", child="a",
                      axis=(0.0, 0.0, 0.0),
                      limit=JointLimit(lower=-1, upper=1, effort=5, velocity=1))],
    )
    # Self-loop (a→a) won't be valid for graph rules, but R004/R007 are
    # independent of the topology — that's what we're checking.
    _, patches, _ = repair(r, codes=["R004"])
    assert all(p.code == "R004" for p in patches)
    assert any(p.code == "R004" for p in patches)


def test_repair_clean_robot_no_patches():
    r = from_xml(
        '<robot name="x"><link name="a"/><link name="b"/>'
        '<joint name="j" type="revolute"><parent link="a"/><child link="b"/>'
        '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="5" velocity="1"/>'
        '</joint></robot>'
    )
    _, patches, _ = repair(r)
    assert patches == []


def test_repair_idempotent():
    r = Robot(
        name="x",
        links=[Link(name="a", inertial=Inertial(mass=-1.0, inertia=Inertia()))],
        joints=[],
    )
    fixed, _, _ = repair(r)
    twice, patches, _ = repair(fixed)
    assert patches == []


def test_unfixable_codes_documented():
    assert set(UNFIXABLE_CODES) == {"R001", "R002", "R008"}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
