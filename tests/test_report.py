"""Diagnostic report assembly: collect tech photos, generate simulation
illustrations (only when confirmed), render a self-contained French HTML report.
"""
from __future__ import annotations

import base64
import importlib.util

import pytest

from fieldpilot_urdf.models import Joint, JointLimit, Link, Origin, Robot
from fieldpilot_urdf.report import (
    DiagnosticReport, ReportImage, attach_simulation_illustrations,
    build_simulation_illustrations, photo_requests, render_report_html,
)

mpl_missing = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="[viz] extra (matplotlib) not installed",
)

PNG = b"\x89PNG\r\n\x1a\n_fake"


def _arm():
    return Robot(
        name="arm",
        links=[Link(name="base"), Link(name="l1"), Link(name="tool")],
        joints=[
            Joint(name="j1", type="revolute", parent="base", child="l1",
                  origin=Origin(xyz=(0, 0, 0)), axis=(0, 0, 1),
                  limit=JointLimit(lower=-3, upper=3, effort=1, velocity=1)),
            Joint(name="j2", type="revolute", parent="l1", child="tool",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 0, 1),
                  limit=JointLimit(lower=-3, upper=3, effort=1, velocity=1)),
        ],
    )


# --- ReportImage ------------------------------------------------------------

def test_report_image_round_trip():
    img = ReportImage.from_bytes("p.png", PNG, caption="essai")
    assert base64.b64decode(img.data_b64) == PNG
    assert img.data_uri().startswith("data:image/png;base64,")
    # JSON-safe (bytes are encoded as a string)
    assert "data_b64" in img.model_dump_json()


# --- ask the tech -----------------------------------------------------------

def test_photo_requests_are_french_and_joint_tailored():
    reqs = photo_requests(joints=["j_shoulder"], links=["link2"], symptom="le bras tremble")
    assert any("Vue d'ensemble" in r for r in reqs)
    assert any("j_shoulder" in r for r in reqs)              # per-joint request
    assert any("link2" in r for r in reqs)                   # per-link request
    assert any("symptôme" in r.lower() for r in reqs)        # symptom shot requested
    assert all(isinstance(r, str) for r in reqs)


def test_photo_requests_minimal():
    reqs = photo_requests()
    assert len(reqs) >= 2 and any("robot" in r.lower() for r in reqs)


# --- HTML assembly ----------------------------------------------------------

def test_render_html_is_french_and_embeds_photos():
    report = DiagnosticReport(
        reference="INT-2026-0042", machine="robotA", symptom="dérive de l'outil",
        confirmed=True, fault="j_shoulder", confidence=0.97, solution="recalibrer_codeur",
        calibration={"j_shoulder": 0.05},
        photos=[ReportImage.from_bytes("photo1.jpg", b"JPEGDATA",
                                       content_type="image/jpeg", caption="vue moteur")])
    htmls = render_report_html(report)
    assert htmls.startswith("<!DOCTYPE html>") and "lang='fr'" in htmls
    assert "Rapport de diagnostic" in htmls and "INT-2026-0042" in htmls
    assert "Défaut diagnostiqué" in htmls and "j_shoulder" in htmls
    assert "97 %" in htmls and "Confirmé" in htmls
    assert "Étalonnage mesuré" in htmls and "+0.0500" in htmls
    assert "Photos du technicien" in htmls
    assert "data:image/jpeg;base64," in htmls                # photo inlined
    assert base64.b64encode(b"JPEGDATA").decode() in htmls


def test_render_html_escapes_user_text():
    report = DiagnosticReport(reference="R<1>", symptom="<script>x</script>")
    htmls = render_report_html(report)
    assert "<script>x</script>" not in htmls
    assert "&lt;script&gt;" in htmls


def test_unconfirmed_report_renders_without_illustrations():
    report = DiagnosticReport(reference="R1", confirmed=False, fault="j2")
    htmls = render_report_html(report)
    assert "Non confirmé" in htmls
    assert "Illustrations de la simulation" not in htmls     # none, and not confirmed


# --- illustrations only when confirmed --------------------------------------

def _frames():
    nominal = [{"j1": 0.0, "j2": 0.2 * k} for k in range(4)]
    faulted = [{"j1": 0.0, "j2": 0.0} for _ in range(4)]    # j2 stuck
    return nominal, faulted


def test_attach_skips_when_not_confirmed():
    report = DiagnosticReport(reference="R1", confirmed=False)
    nominal, faulted = _frames()
    out = attach_simulation_illustrations(report, _arm(), nominal, faulted, track_link="tool")
    assert out.illustrations == []                           # nothing generated


@mpl_missing
def test_attach_generates_when_confirmed():
    report = DiagnosticReport(reference="R1", confirmed=True, fault="j2")
    nominal, faulted = _frames()
    out = attach_simulation_illustrations(report, _arm(), nominal, faulted, track_link="tool")
    assert len(out.illustrations) == 1                       # the 3D GIF
    gif = out.illustrations[0]
    assert gif.content_type == "image/gif"
    assert base64.b64decode(gif.data_b64)[:4] == b"GIF8"
    # it lands in the rendered report
    assert "Illustrations de la simulation" in render_report_html(out)


@mpl_missing
def test_build_illustrations_with_scope():
    from fieldpilot_urdf.retime import TimedTrajectory
    nominal, faulted = _frames()
    times = [0.0, 0.5, 1.0, 1.5]
    exp = TimedTrajectory(joint_ids=["j2"], times=times, q=[[r["j2"]] for r in nominal],
                          u=[[0.0]] * 4)
    obs = TimedTrajectory(joint_ids=["j2"], times=times, q=[[r["j2"]] for r in faulted],
                          u=[[0.0]] * 4)
    imgs = build_simulation_illustrations(_arm(), nominal, faulted, expected=exp,
                                          observed=obs, track_link="tool")
    assert len(imgs) == 2                                    # GIF + scope PNG
    assert {i.content_type for i in imgs} == {"image/gif", "image/png"}
