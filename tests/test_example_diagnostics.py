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

    # it printed the full narrative
    out = capsys.readouterr().out
    assert "LOCALISE" in out and "DIALOG" in out and "CALIBRATE" in out


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                    reason="[viz] extra not installed")
def test_diagnostics_workflow_writes_visuals(tmp_path):
    mod = _load_example()
    result = mod.main(out_dir=tmp_path)
    gif = Path(result["visuals"]["gif"])
    png = Path(result["visuals"]["png"])
    assert gif.exists() and gif.read_bytes()[:4] == b"GIF8"
    assert png.exists() and png.read_bytes()[:4] == b"\x89PNG"
