"""Diagnostic case knowledge base: persistence + statistics. Anchors: cases
round-trip to disk; fault_priors reflect observed frequencies and feed the
differential engine; solution_stats / recommend_solution rank fixes by their
success record.
"""
from __future__ import annotations

import math

import pytest

from fieldpilot_urdf.case_base import (
    DiagnosticCase, SolutionStat, delete_case, fault_priors, list_cases,
    load_case, load_cases, recommend_solution, save_case, solution_stats,
)
from fieldpilot_urdf.differential_diagnosis import candidates_from_scores, update_beliefs


def _case(cid, fault=None, solution=None, resolved=False, machine=None):
    return DiagnosticCase(id=cid, confirmed_fault=fault, solution=solution,
                          resolved=resolved, machine=machine, symptom=f"sym-{cid}")


# --- persistence ------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    c = DiagnosticCase(id="INT-2026-0001", symptom="j3 drifts", machine="robotA",
                       candidates=["motor_dead", "backlash"],
                       answers={"hold": "no"}, confirmed_fault="motor_dead",
                       solution="replace_motor", resolved=True, notes="warranty")
    saved = save_case(c, root=tmp_path)
    assert saved.created_at is not None                  # stamped on save
    back = load_case("INT-2026-0001", root=tmp_path)
    assert back == saved
    assert back.confirmed_fault == "motor_dead" and back.resolved


def test_list_and_load_all_and_delete(tmp_path):
    for i in range(3):
        save_case(_case(f"c{i}", fault="f"), root=tmp_path)
    assert list_cases(root=tmp_path) == ["c0", "c1", "c2"]
    assert len(load_cases(root=tmp_path)) == 3
    assert delete_case("c1", root=tmp_path) is True
    assert list_cases(root=tmp_path) == ["c0", "c2"]
    assert delete_case("c1", root=tmp_path) is False     # already gone
    assert load_case("missing", root=tmp_path) is None


def test_missing_dir_is_empty(tmp_path):
    assert list_cases(root=tmp_path / "nope") == []
    assert load_cases(root=tmp_path / "nope") == []


# --- fault priors -----------------------------------------------------------

def test_fault_priors_reflect_frequencies():
    cases = ([_case(f"a{i}", fault="motor_dead", resolved=True) for i in range(7)]
             + [_case(f"b{i}", fault="backlash", resolved=True) for i in range(3)])
    pri = fault_priors(cases, smoothing=0.0)              # no smoothing -> raw frequency
    assert math.isclose(pri["motor_dead"], 0.7, abs_tol=1e-9)
    assert math.isclose(pri["backlash"], 0.3, abs_tol=1e-9)
    assert math.isclose(sum(pri.values()), 1.0, abs_tol=1e-9)


def test_fault_priors_smoothing_and_unseen_candidates():
    cases = [_case("a", fault="motor_dead")]
    # force a prior over a candidate set including a never-seen fault
    pri = fault_priors(cases, candidates=["motor_dead", "encoder"], smoothing=1.0)
    # motor_dead: 1+1=2, encoder: 0+1=1 -> 2/3, 1/3
    assert math.isclose(pri["motor_dead"], 2 / 3, abs_tol=1e-9)
    assert math.isclose(pri["encoder"], 1 / 3, abs_tol=1e-9)
    # pure uniform when there's no data at all
    uni = fault_priors([], candidates=["x", "y"], smoothing=1.0)
    assert math.isclose(uni["x"], 0.5, abs_tol=1e-9) and math.isclose(uni["y"], 0.5, abs_tol=1e-9)


def test_fault_priors_machine_filter():
    cases = [_case("a", fault="motor_dead", machine="R1"),
             _case("b", fault="backlash", machine="R2"),
             _case("c", fault="backlash", machine="R2")]
    pri = fault_priors(cases, machine="R2", smoothing=0.0)
    assert set(pri) == {"backlash"} and math.isclose(pri["backlash"], 1.0, abs_tol=1e-9)


def test_fault_priors_empty():
    assert fault_priors([]) == {}
    assert fault_priors([_case("a")]) == {}               # case has no confirmed_fault


# --- the payoff: learned priors feed the differential engine ----------------

def test_priors_feed_differential_engine():
    # history says motor_dead is far more common than encoder
    cases = ([_case(f"m{i}", fault="motor_dead") for i in range(8)]
             + [_case("e0", fault="encoder")])
    pri = fault_priors(cases, candidates=["motor_dead", "encoder"], smoothing=1.0)
    candidates = candidates_from_scores(pri)
    state = update_beliefs(candidates, [], {})            # no questions yet -> just the prior
    assert state.leading == "motor_dead"
    assert state.posteriors["motor_dead"] > state.posteriors["encoder"]


# --- solution statistics ----------------------------------------------------

def test_solution_stats_track_success():
    cases = [
        _case("1", fault="motor_dead", solution="replace_motor", resolved=True),
        _case("2", fault="motor_dead", solution="replace_motor", resolved=True),
        _case("3", fault="motor_dead", solution="replace_motor", resolved=False),
        _case("4", fault="motor_dead", solution="reseat_connector", resolved=False),
    ]
    stats = solution_stats(cases, fault="motor_dead")
    assert all(isinstance(s, SolutionStat) for s in stats)
    top = stats[0]
    assert top.solution == "replace_motor"
    assert top.attempts == 3 and top.successes == 2
    assert math.isclose(top.success_rate, 2 / 3, abs_tol=1e-9)
    # the 0%-success fix ranks last
    assert stats[-1].solution == "reseat_connector" and stats[-1].success_rate == 0.0


def test_recommend_solution_picks_best_with_min_attempts():
    cases = [
        _case("1", fault="backlash", solution="tighten", resolved=True),     # 1/1 = 100% but 1 try
        _case("2", fault="backlash", solution="regrease", resolved=True),
        _case("3", fault="backlash", solution="regrease", resolved=True),
        _case("4", fault="backlash", solution="regrease", resolved=False),   # 2/3 ~ 67%, 3 tries
    ]
    # with min_attempts=1, the 100% single-shot wins
    assert recommend_solution(cases, "backlash", min_attempts=1) == "tighten"
    # requiring more evidence, the better-tested fix wins
    assert recommend_solution(cases, "backlash", min_attempts=3) == "regrease"
    # unknown fault -> nothing
    assert recommend_solution(cases, "nope") is None


def test_solution_stats_ignores_unresolved_or_solutionless():
    cases = [_case("1", fault="f"),                       # no solution
             _case("2", solution="x", resolved=True)]     # no fault
    assert solution_stats(cases) == []
