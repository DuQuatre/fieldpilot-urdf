"""Smoke test for examples/diagnostics_workflow.py — the end-to-end diagnostics
example must keep running and reaching the right conclusions (it exercises
localize → dialog → calibrate → recommend → record, plus the visuals when [viz]
is present). Guards the example against API drift in CI.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "diagnostics_workflow.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("diagnostics_workflow", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_diagnostics_workflow_runs_end_to_end(tmp_path, capsys):
    mod = _load_example()
    result = mod.main(out_dir=tmp_path)

    # the workflow identifies, confirms and calibrates the injected fault
    assert result["localized_top"] == mod.TRUE_FAULT
    assert result["resolved_fault"] == mod.TRUE_FAULT
    assert abs(result["calibrated_offset"] - mod.TRUE_OFFSET) < 1e-3
    assert result["recommended"] == "recalibrate_encoder"   # best-proven fix from history

    # the new case was recorded into the case base
    from fieldpilot_urdf import list_cases
    assert "INT-2026-0042" in list_cases(root=tmp_path / "cases")

    # it assembled a French report carrying the diagnosis + spare parts
    htmls = result["report_html"]
    assert "Rapport de diagnostic" in htmls and mod.TRUE_FAULT in htmls
    assert "Pièces de rechange" in htmls and "recalibrate_encoder" in result["recommended"]
    assert (tmp_path / "rapport.html").exists()

    # it built the Odoo intervention wiring (task field mapping + PDF filename)
    assert result["odoo_task_vals"]["x_intervention_ref"] == "INT-2026-0042"
    assert result["odoo_task_vals"]["x_cause_probable"] == mod.TRUE_FAULT
    assert result["gotenberg_filename"] == "rapport_INT-2026-0042.pdf"

    # it built the Telegram replies (summary first, then the illustrations)
    assert result["telegram_methods"][0] == "sendMessage"

    # it built the SPA spare-parts order (2 lines; CAL-KIT has no product yet)
    assert result["spa_order_lines"] == 2
    assert result["spa_unresolved"] == ["CAL-KIT"]

    # it rolled the case base into dashboard KPIs (5 cases; shoulder dominates)
    assert result["dashboard_total_cases"] == 5
    assert result["dashboard_top_fault"] == mod.TRUE_FAULT

    # it rendered the weekly French email digest
    assert "Bilan diagnostic" in result["digest_subject"]
    assert "5 intervention(s)" in result["digest_subject"]

    # it printed the full narrative
    out = capsys.readouterr().out
    assert "LOCALISE" in out and "DIALOG" in out and "REPORT" in out
    assert "SYNC" in out and "NOTIFY" in out and "ORDER" in out
    assert "DASHBOARD" in out and "DIGEST" in out


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                    reason="[viz] extra not installed")
def test_diagnostics_workflow_writes_visuals_and_report(tmp_path):
    mod = _load_example()
    mod.main(out_dir=tmp_path)
    gif = tmp_path / "mouvement_3d.gif"
    png = tmp_path / "oscilloscope.png"
    assert gif.exists() and gif.read_bytes()[:4] == b"GIF8"
    assert png.exists() and png.read_bytes()[:4] == b"\x89PNG"
    # the report embeds the illustrations (base64 data URIs)
    htmls = (tmp_path / "rapport.html").read_text(encoding="utf-8")
    assert "data:image/gif;base64," in htmls and "Illustrations de la simulation" in htmls
