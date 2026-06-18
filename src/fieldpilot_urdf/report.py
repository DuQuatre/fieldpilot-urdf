"""Diagnostic report assembly: the technician's photos + the simulation's
illustrations, bundled into a self-contained report.

A field diagnosis ends in a report. This module assembles one from three things:

* **the diagnosis** — the confirmed fault, confidence, recommended fix, and any
  calibration result (a :class:`DiagnosticReport`),
* **the technician's photos** — :func:`photo_requests` says *which* pictures to
  ask for (in French, for the bot to relay); each returned shot is attached as a
  :class:`ReportImage`, and
* **the simulation's illustrations** — once the diagnosis is *confirmed*,
  :func:`attach_simulation_illustrations` renders the 3D fault-motion video and
  the oscilloscope traces (the 1.19 / 1.20 visuals) and attaches them, so the
  report can set the model's prediction beside the tech's photos.

:func:`render_report_html` then emits a self-contained HTML document (every image
inlined as a base64 data URI) — **in French**, per the FieldPilot reports
convention — ready for the SaaS to turn into a PDF (e.g. via Gotenberg).

The models and HTML assembly are pure Python (core install); only the
illustration *generation* touches the ``[viz]`` extra, and lazily.
"""
from __future__ import annotations

import base64
import html
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .models import Robot


class ReportImage(BaseModel):
    """An image carried in a report — a tech photo or a generated illustration —
    held base64-encoded so the report is JSON-safe and self-contained."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Short identifier / filename")
    caption: str = Field("", description="Caption shown under the image (French)")
    content_type: str = Field("image/png", description="MIME type, e.g. image/png, image/gif")
    data_b64: str = Field(..., description="Base64-encoded image bytes")

    @classmethod
    def from_bytes(cls, name: str, data: bytes, *, content_type: str = "image/png",
                   caption: str = "") -> "ReportImage":
        return cls(name=name, caption=caption, content_type=content_type,
                   data_b64=base64.b64encode(data).decode("ascii"))

    def data_uri(self) -> str:
        return f"data:{self.content_type};base64,{self.data_b64}"


class SparePart(BaseModel):
    """A spare part the technician should bring for the recommended fix."""

    model_config = ConfigDict(extra="forbid")

    reference: str = Field(..., description="Part reference / SKU")
    name: str = Field(..., description="Part designation (French)")
    quantity: int = Field(1, ge=1, description="Quantity required")


class DiagnosticReport(BaseModel):
    """Everything a diagnostic report carries."""

    model_config = ConfigDict(extra="forbid")

    reference: str = Field(..., description="Report / intervention reference")
    machine: str = Field("", description="Machine or robot identifier")
    symptom: str = Field("", description="Reported problem (free text)")
    confirmed: bool = Field(False, description="Whether the diagnosis is confirmed")
    fault: Optional[str] = Field(None, description="The diagnosed fault")
    confidence: Optional[float] = Field(None, description="Diagnosis confidence in [0, 1]")
    solution: Optional[str] = Field(None, description="Recommended fix")
    spare_parts: list[SparePart] = Field(default_factory=list,
                                         description="Spare parts for the recommended fix")
    calibration: dict[str, float] = Field(default_factory=dict,
                                          description="Per-joint calibration offsets, if measured")
    photos: list[ReportImage] = Field(default_factory=list, description="Technician-supplied photos")
    illustrations: list[ReportImage] = Field(default_factory=list,
                                             description="Simulation-generated illustrations")
    created_at: Optional[str] = Field(None, description="ISO timestamp (caller-supplied)")
    notes: str = Field("", description="Free-text notes")


# --- ask the tech: which photos to collect ---------------------------------

def photo_requests(
    *,
    joints: Optional[list[str]] = None,
    links: Optional[list[str]] = None,
    symptom: str = "",
) -> list[str]:
    """The list of photos to ask the technician for (French prompts), tailored to
    the suspected ``joints`` / ``links``. The bot relays these; each returned shot
    is attached to the report as a :class:`ReportImage`."""
    reqs = [
        "Vue d'ensemble du robot dans son environnement",
        "Gros plan sur la zone du problème signalé",
    ]
    for j in joints or []:
        reqs.append(f"Gros plan sur l'articulation « {j} » (moteur, réducteur, câblage)")
        reqs.append(f"Plaque signalétique / étiquette du moteur de « {j} »")
    for ln in links or []:
        reqs.append(f"Photo du segment « {ln} » (déformation, jeu, usure)")
    if symptom:
        reqs.append("Photo illustrant le symptôme décrit lorsqu'il se produit")
    reqs.append("Connecteurs et faisceau de câbles à proximité du défaut")
    return reqs


# --- generate illustrations from the simulation (when confirmed) -----------

def build_simulation_illustrations(
    robot: Robot,
    nominal: list[dict],
    faulted: list[dict],
    *,
    expected=None,
    observed=None,
    track_link: Optional[str] = None,
    labels: tuple[str, str] = ("nominal", "défaut simulé"),
) -> list[ReportImage]:
    """Render the simulation illustrations for the report: a 3D fault-motion GIF
    (``nominal`` vs ``faulted`` joint-config frames) and, when ``expected`` /
    ``observed`` trajectories are given, an oscilloscope PNG. Requires the
    ``[viz]`` extra; raises ``RuntimeError`` with an install hint if it's absent."""
    try:
        from .viz import render_motion_comparison, render_trajectory_scope
    except Exception as e:  # pragma: no cover - exercised only without [viz]
        raise RuntimeError(
            "generating report illustrations needs the [viz] extra: "
            "pip install 'fieldpilot-urdf[viz]'") from e

    images: list[ReportImage] = []
    gif = render_motion_comparison(robot, nominal, faulted, labels=labels,
                                   layout="overlay", track_link=track_link)
    images.append(ReportImage.from_bytes(
        "mouvement_3d.gif", gif, content_type="image/gif",
        caption="Mouvement simulé : nominal (trait plein) vs défaut (pointillés)"))
    if expected is not None:
        png = render_trajectory_scope(expected, observed,
                                      labels=("attendu", "observé"))
        images.append(ReportImage.from_bytes(
            "oscilloscope.png", png, content_type="image/png",
            caption="Paramètres articulaires : attendu vs observé"))
    return images


def attach_simulation_illustrations(
    report: DiagnosticReport,
    robot: Robot,
    nominal: list[dict],
    faulted: list[dict],
    **kwargs,
) -> DiagnosticReport:
    """Append the simulation illustrations to ``report`` — **only when the
    diagnosis is confirmed**. A non-confirmed report is returned unchanged (no
    illustrations are generated), so the model's prediction is shown only once
    the fault is established."""
    if not report.confirmed:
        return report
    imgs = build_simulation_illustrations(robot, nominal, faulted, **kwargs)
    return report.model_copy(update={"illustrations": list(report.illustrations) + imgs})


# --- assemble the report ----------------------------------------------------

def _img_block(img: ReportImage) -> str:
    cap = f"<figcaption>{html.escape(img.caption)}</figcaption>" if img.caption else ""
    return (f'<figure><img alt="{html.escape(img.name)}" src="{img.data_uri()}"/>'
            f"{cap}</figure>")


def render_report_html(report: DiagnosticReport) -> str:
    """Render ``report`` as a self-contained HTML document (images inlined),
    **in French**, ready for HTML→PDF conversion."""
    e = html.escape
    statut = "Confirmé" if report.confirmed else "Non confirmé"
    rows = [("Machine", report.machine), ("Référence", report.reference),
            ("Statut du diagnostic", statut)]
    if report.created_at:
        rows.append(("Date", report.created_at))
    if report.fault:
        rows.append(("Défaut diagnostiqué", report.fault))
    if report.confidence is not None:
        rows.append(("Confiance", f"{report.confidence * 100:.0f} %"))
    if report.solution:
        rows.append(("Solution recommandée", report.solution))
    info = "".join(f"<tr><th>{e(k)}</th><td>{e(str(v))}</td></tr>" for k, v in rows)

    parts = ""
    if report.spare_parts:
        prows = "".join(
            f"<tr><td>{e(p.reference)}</td><td>{e(p.name)}</td><td>{p.quantity}</td></tr>"
            for p in report.spare_parts)
        parts = ("<h2>Pièces de rechange</h2><table>"
                 "<tr><th>Référence</th><th>Désignation</th><th>Qté</th></tr>"
                 f"{prows}</table>")

    cal = ""
    if report.calibration:
        items = "".join(f"<li>{e(j)} : {off:+.4f} rad/m</li>"
                        for j, off in report.calibration.items())
        cal = f"<h2>Étalonnage mesuré</h2><ul>{items}</ul>"

    symptom = f"<h2>Symptôme signalé</h2><p>{e(report.symptom)}</p>" if report.symptom else ""

    photos = ""
    if report.photos:
        photos = ("<h2>Photos du technicien</h2><div class='gallery'>"
                  + "".join(_img_block(p) for p in report.photos) + "</div>")

    illus = ""
    if report.illustrations:
        illus = ("<h2>Illustrations de la simulation</h2><div class='gallery'>"
                 + "".join(_img_block(p) for p in report.illustrations) + "</div>")
    elif report.confirmed:
        illus = ("<h2>Illustrations de la simulation</h2>"
                 "<p><em>Aucune illustration générée.</em></p>")

    notes = f"<h2>Notes</h2><p>{e(report.notes)}</p>" if report.notes else ""

    style = (
        "body{font-family:Arial,Helvetica,sans-serif;color:#2C3E50;margin:2em;}"
        "h1{border-bottom:3px solid #2E86DE;padding-bottom:.3em;}"
        "h2{color:#1B4F72;margin-top:1.4em;}"
        "table{border-collapse:collapse;margin:.5em 0;}"
        "th,td{text-align:left;padding:.3em .8em;border-bottom:1px solid #ECF0F1;}"
        "th{color:#7F8C8D;font-weight:600;}"
        ".gallery{display:flex;flex-wrap:wrap;gap:1em;}"
        "figure{margin:0;max-width:360px;}"
        "img{max-width:100%;border:1px solid #D5DBDB;border-radius:4px;}"
        "figcaption{font-size:.85em;color:#566573;margin-top:.3em;}"
    )
    return (
        "<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'/>"
        f"<title>Rapport de diagnostic — {e(report.reference)}</title>"
        f"<style>{style}</style></head><body>"
        f"<h1>Rapport de diagnostic — {e(report.reference)}</h1>"
        f"<table>{info}</table>"
        f"{symptom}{parts}{cal}{photos}{illus}{notes}"
        "</body></html>"
    )
