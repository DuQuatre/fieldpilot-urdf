"""Wiring the report into the Odoo intervention PDF pipeline: the Gotenberg
HTML→PDF request and the Odoo ir.attachment / project.task payloads. Pure data —
no network — so we assert the shapes the SaaS sends.
"""
from __future__ import annotations

import base64

import pytest

from fieldpilot_urdf.intervention import (
    GotenbergRequest, gotenberg_request, intervention_attachment_vals,
    intervention_task_vals,
)
from fieldpilot_urdf.report import DiagnosticReport, render_report_html


# --- Gotenberg request ------------------------------------------------------

def test_gotenberg_request_carries_html_and_options():
    html = "<html><body>rapport</body></html>"
    req = gotenberg_request(html, filename="rapport_INT-1.pdf", margin_mm=12)
    assert isinstance(req, GotenbergRequest)
    assert req.endpoint == "/forms/chromium/convert/html"
    assert req.output_filename == "rapport_INT-1.pdf"
    assert req.index_html == html
    assert req.form_data["paperWidth"] == "8.27"           # A4
    assert req.form_data["printBackground"] == "true"
    assert req.form_data["landscape"] == "false"
    assert req.form_data["marginTop"] == f"{12 / 25.4:.4f}"


def test_gotenberg_requests_kwargs_are_post_ready():
    req = gotenberg_request("<html></html>", filename="r.pdf")
    kw = req.requests_kwargs()
    name, content, mime = kw["files"]["index.html"]
    assert name == "index.html" and mime == "text/html"
    assert content == b"<html></html>"
    assert kw["headers"]["Gotenberg-Output-Filename"] == "r.pdf"
    assert kw["data"]["printBackground"] == "true"


def test_gotenberg_landscape_and_letter():
    req = gotenberg_request("<html></html>", paper="Letter", landscape=True)
    assert req.form_data["paperWidth"] == "8.5"
    assert req.form_data["landscape"] == "true"


def test_gotenberg_unknown_paper_raises():
    with pytest.raises(ValueError):
        gotenberg_request("<html></html>", paper="A3")


def test_real_report_html_flows_through():
    # the actual report HTML converts cleanly (self-contained, no external assets)
    report = DiagnosticReport(reference="INT-2026-0042", confirmed=True, fault="j_shoulder")
    req = gotenberg_request(render_report_html(report), filename="rapport_INT-2026-0042.pdf")
    assert "Rapport de diagnostic" in req.index_html
    assert req.requests_kwargs()["files"]["index.html"][1].startswith(b"<!DOCTYPE html>")


# --- Odoo ir.attachment -----------------------------------------------------

def test_attachment_vals_shape():
    pdf = b"%PDF-1.4 fake pdf bytes"
    vals = intervention_attachment_vals("INT-2026-0042", pdf, res_id=314)
    assert vals["name"] == "rapport_INT-2026-0042.pdf"
    assert vals["mimetype"] == "application/pdf"
    assert vals["res_model"] == "project.task" and vals["res_id"] == 314
    assert vals["type"] == "binary"
    assert base64.b64decode(vals["datas"]) == pdf          # round-trips


def test_attachment_vals_without_res_id():
    vals = intervention_attachment_vals("INT-1", b"%PDF")
    assert "res_id" not in vals                             # detached, linked later


# --- Odoo project.task ------------------------------------------------------

def test_task_vals_map_diagnosis_to_custom_fields():
    report = DiagnosticReport(reference="INT-2026-0042", confirmed=True, fault="j_shoulder")
    vals = intervention_task_vals(report, pdf_url="/web/content/999")
    assert vals == {
        "x_intervention_ref": "INT-2026-0042",
        "x_cause_probable": "j_shoulder",
        "x_rapport_pdf_url": "/web/content/999",
    }


def test_task_vals_omit_empty_and_unconfirmed_cause():
    # not confirmed -> no probable cause written; no url -> no pdf field
    report = DiagnosticReport(reference="INT-9", confirmed=False, fault="j2")
    vals = intervention_task_vals(report)
    assert vals == {"x_intervention_ref": "INT-9"}
    assert "x_cause_probable" not in vals and "x_rapport_pdf_url" not in vals
