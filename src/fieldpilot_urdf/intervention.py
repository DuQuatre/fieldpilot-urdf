"""Wire a diagnostic report into the Odoo intervention PDF pipeline.

:func:`render_report_html` produces the report as HTML; the field workflow then
turns it into the intervention PDF and files it on the Odoo task. That pipeline
runs external services — **Gotenberg** (HTML→PDF) and **Odoo** (JSON-RPC) — which
the open-core package deliberately does not depend on or call. Instead, this
module builds the *request and payload data* those steps need, so the n8n / SaaS
side just sends them (the same "return data, the caller does the I/O" contract as
the renderers returning bytes):

1. :func:`gotenberg_request` — the multipart request that converts the
   self-contained report HTML into a PDF via Gotenberg's Chromium route.
2. :func:`intervention_attachment_vals` — the Odoo ``ir.attachment`` create
   values that file the PDF on the intervention task (``rapport_{ref}.pdf``).
3. :func:`intervention_task_vals` — the ``project.task`` write values mapping the
   diagnosis onto the documented intervention custom fields
   (``x_intervention_ref``, ``x_cause_probable``, ``x_rapport_pdf_url``).

Pure Python, no new dependencies, no network. Pairs with
:mod:`fieldpilot_urdf.report`.
"""
from __future__ import annotations

import base64
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .report import DiagnosticReport

# A4 in inches (Gotenberg measures paper/margins in inches).
_A4_IN = (8.27, 11.69)
_LETTER_IN = (8.5, 11.0)
_PAPERS = {"A4": _A4_IN, "Letter": _LETTER_IN}


class GotenbergRequest(BaseModel):
    """A ready-to-send Gotenberg Chromium HTML→PDF conversion request. The report
    HTML is self-contained (images inlined as data URIs), so it ships as a single
    ``index.html`` form file."""

    model_config = ConfigDict(extra="forbid")

    endpoint: str = Field("/forms/chromium/convert/html",
                          description="Gotenberg route, appended to the base URL")
    output_filename: str = Field(..., description="Desired PDF filename")
    index_html: str = Field(..., description="The report HTML (sent as index.html)")
    form_data: dict[str, str] = Field(default_factory=dict,
                                      description="Gotenberg form fields (paper size, margins, …)")

    def requests_kwargs(self) -> dict:
        """Keyword args for ``requests.post(base_url + endpoint, **kwargs)`` (also
        accepted by httpx): the ``index.html`` file part, the form fields, and the
        output-filename header."""
        return {
            "files": {"index.html": ("index.html", self.index_html.encode("utf-8"), "text/html")},
            "data": dict(self.form_data),
            "headers": {"Gotenberg-Output-Filename": self.output_filename},
        }


def gotenberg_request(
    html: str,
    *,
    filename: str = "rapport.pdf",
    paper: str = "A4",
    margin_mm: float = 10.0,
    landscape: bool = False,
) -> GotenbergRequest:
    """Build the Gotenberg request that converts a report ``html`` to a PDF.

    ``paper`` is ``"A4"`` or ``"Letter"``; ``margin_mm`` sets all four margins.
    ``printBackground`` is on so the report's CSS colours survive. Raises
    ``ValueError`` for an unknown paper size."""
    if paper not in _PAPERS:
        raise ValueError(f"unknown paper {paper!r}; choose from {sorted(_PAPERS)}")
    w, h = _PAPERS[paper]
    margin_in = f"{margin_mm / 25.4:.4f}"
    form = {
        "paperWidth": f"{w}", "paperHeight": f"{h}",
        "marginTop": margin_in, "marginBottom": margin_in,
        "marginLeft": margin_in, "marginRight": margin_in,
        "landscape": "true" if landscape else "false",
        "printBackground": "true",
    }
    return GotenbergRequest(output_filename=filename, index_html=html, form_data=form)


def intervention_attachment_vals(
    reference: str,
    pdf: bytes,
    *,
    res_id: Optional[int] = None,
) -> dict:
    """Odoo ``ir.attachment`` create-values that file the intervention ``pdf`` on
    the task. ``res_id`` is the ``project.task`` id to attach to (omit to create a
    detached attachment and link it later). Named ``rapport_{reference}.pdf``."""
    vals = {
        "name": f"rapport_{reference}.pdf",
        "datas": base64.b64encode(pdf).decode("ascii"),
        "mimetype": "application/pdf",
        "res_model": "project.task",
        "type": "binary",
    }
    if res_id is not None:
        vals["res_id"] = res_id
    return vals


def intervention_task_vals(
    report: DiagnosticReport,
    *,
    pdf_url: Optional[str] = None,
) -> dict:
    """Odoo ``project.task`` write-values mapping ``report`` onto the documented
    intervention custom fields. Only fields with a value are included:

    * ``x_intervention_ref`` ← ``report.reference``
    * ``x_cause_probable``   ← ``report.fault`` (when confirmed)
    * ``x_rapport_pdf_url``  ← ``pdf_url`` (the attachment URL, once uploaded)
    """
    vals: dict[str, object] = {}
    if report.reference:
        vals["x_intervention_ref"] = report.reference
    if report.confirmed and report.fault:
        vals["x_cause_probable"] = report.fault
    if pdf_url:
        vals["x_rapport_pdf_url"] = pdf_url
    return vals
