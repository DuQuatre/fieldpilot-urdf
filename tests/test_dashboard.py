"""Rolling the case base into dashboard KPIs. Pure data, so we assert the
aggregated summary the admin/MRR dashboard serves.
"""
from __future__ import annotations

import json
import math

from fieldpilot_urdf.case_base import DiagnosticCase
from fieldpilot_urdf.dashboard import CaseStatsSummary, FaultStat, case_stats_summary


def _case(cid, fault=None, solution=None, resolved=False, machine=None):
    return DiagnosticCase(id=cid, confirmed_fault=fault, solution=solution,
                          resolved=resolved, machine=machine)


def _fleet():
    return (
        [_case(f"m{i}", fault="motor_dead", solution="replace_motor", resolved=True)
         for i in range(6)]
        + [_case("g0", fault="motor_dead", solution="replace_motor", resolved=False)]   # 6/7
        + [_case(f"b{i}", fault="backlash", solution="regrease", resolved=True)
           for i in range(3)]
        + [_case("o0")]                                            # open, no fault
    )


# --- the summary ------------------------------------------------------------

def test_summary_counts_and_resolution_rate():
    s = case_stats_summary(_fleet())
    assert s.total_cases == 11
    assert s.resolved_cases == 9                              # 6 + 3
    assert math.isclose(s.resolution_rate, 9 / 11, abs_tol=1e-9)
    assert s.distinct_faults == 2


def test_top_faults_ranked_with_share():
    s = case_stats_summary(_fleet())
    assert isinstance(s.top_faults[0], FaultStat)
    assert s.top_faults[0].fault == "motor_dead" and s.top_faults[0].count == 7
    # 10 cases carry a confirmed fault (7 motor + 3 backlash)
    assert math.isclose(s.top_faults[0].share, 7 / 10, abs_tol=1e-9)
    assert s.top_faults[1].fault == "backlash" and s.top_faults[1].count == 3


def test_top_solutions_best_first():
    s = case_stats_summary(_fleet())
    # regrease 3/3 = 100% beats replace_motor 6/7
    assert s.top_solutions[0].solution == "regrease"
    assert math.isclose(s.top_solutions[0].success_rate, 1.0, abs_tol=1e-9)
    assert s.top_solutions[1].solution == "replace_motor"


def test_machine_filter():
    cases = [_case("a", fault="f1", resolved=True, machine="R1"),
             _case("b", fault="f2", resolved=False, machine="R2"),
             _case("c", fault="f2", resolved=True, machine="R2")]
    s = case_stats_summary(cases, machine="R2")
    assert s.total_cases == 2 and s.resolved_cases == 1
    assert s.machine == "R2" and s.distinct_faults == 1
    assert s.top_faults[0].fault == "f2"


def test_top_n_caps_lists():
    cases = [_case(f"c{i}", fault=f"f{i}", solution=f"s{i}", resolved=True) for i in range(8)]
    s = case_stats_summary(cases, top_n=3)
    assert len(s.top_faults) == 3 and len(s.top_solutions) == 3
    assert s.distinct_faults == 8                             # count is not capped


def test_empty_is_all_zero():
    s = case_stats_summary([])
    assert s.total_cases == 0 and s.resolved_cases == 0
    assert s.resolution_rate == 0.0 and s.distinct_faults == 0
    assert s.top_faults == [] and s.top_solutions == []


def test_summary_is_json_serializable_for_dashboard():
    s = case_stats_summary(_fleet())
    payload = json.loads(s.model_dump_json())                # the dashboard serves this
    assert payload["total_cases"] == 11
    assert payload["top_faults"][0]["fault"] == "motor_dead"
    assert isinstance(CaseStatsSummary.model_validate(payload), CaseStatsSummary)
