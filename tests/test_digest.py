"""Weekly email digest of the dashboard KPIs. Pure content — no sending — so we
assert the rendered French subject + HTML/text bodies.
"""
from __future__ import annotations

import json

from fieldpilot_urdf.case_base import DiagnosticCase
from fieldpilot_urdf.dashboard import case_stats_summary
from fieldpilot_urdf.digest import EmailDigest, weekly_digest


def _case(cid, fault=None, solution=None, resolved=False):
    return DiagnosticCase(id=cid, confirmed_fault=fault, solution=solution, resolved=resolved)


def _summary(resolved_motor=6):
    cases = (
        [_case(f"m{i}", fault="motor_dead", solution="replace_motor", resolved=True)
         for i in range(resolved_motor)]
        + [_case(f"b{i}", fault="backlash", solution="regrease", resolved=True) for i in range(3)]
    )
    return case_stats_summary(cases)


# --- content ----------------------------------------------------------------

def test_digest_subject_and_bodies_are_french():
    d = weekly_digest(_summary(), period_label="semaine du 16/06/2026")
    assert isinstance(d, EmailDigest)
    assert "Bilan diagnostic" in d.subject and "semaine du 16/06/2026" in d.subject
    assert "9 intervention(s)" in d.subject and "100 %" in d.subject
    assert d.html_body.startswith("<!DOCTYPE html>") and "lang='fr'" in d.html_body
    assert "Défauts les plus fréquents" in d.html_body
    assert "motor_dead" in d.html_body and "Solutions les plus efficaces" in d.html_body
    # plain-text mirror
    assert "motor_dead : 6×" in d.text_body
    assert "regrease" in d.text_body


def test_digest_shows_deltas_vs_previous():
    cur = _summary(resolved_motor=6)     # 9 cases, 100%
    prev = case_stats_summary([_case("x", fault="motor_dead", solution="replace_motor",
                                     resolved=True),
                               _case("y", fault="motor_dead", resolved=False)])  # 2 cases, 50%
    d = weekly_digest(cur, previous=prev)
    assert "(+7)" in d.html_body            # case-count delta 9 - 2
    assert "+50 pts" in d.html_body         # resolution 100% vs 50%
    assert "+50 pts" in d.text_body


def test_digest_empty_period():
    d = weekly_digest(case_stats_summary([]))
    assert "0 intervention(s)" in d.subject
    assert "Aucune intervention" in d.html_body and "Aucune intervention" in d.text_body
    assert "Défauts les plus fréquents" not in d.html_body   # no tables when empty


def test_digest_escapes_html():
    s = case_stats_summary([_case("a", fault="<x>", solution="y&z", resolved=True)])
    d = weekly_digest(s)
    assert "<x>" not in d.html_body and "&lt;x&gt;" in d.html_body
    assert "y&amp;z" in d.html_body


def test_digest_json_serializable():
    d = weekly_digest(_summary())
    payload = json.loads(d.model_dump_json())
    assert set(payload) == {"subject", "html_body", "text_body"}
    assert EmailDigest.model_validate(payload).subject == d.subject
