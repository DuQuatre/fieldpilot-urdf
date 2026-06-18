"""Wire a diagnostic report into the Telegram bot's reply to the technician.

The field loop ends back in the tech's chat: the shared bot replies with the
diagnosis summary and the simulation visuals (and the PDF, if one was made). As
with the Odoo wiring, the open-core package neither talks to Telegram nor depends
on it — it builds the **Bot API request data**, and the n8n / SaaS side POSTs it
to ``https://api.telegram.org/bot<token>/<method>``:

* :func:`report_summary_text` — the French message body (a confirmed-diagnosis
  recap: fault, confidence, fix, spare parts), ready for ``parse_mode=HTML``.
* :func:`telegram_messages` — the sequence of :class:`TelegramRequest`s to send:
  the summary (``sendMessage``), each report illustration (the 3D GIF as
  ``sendAnimation``, the oscilloscope PNG as ``sendPhoto``), and the report PDF
  (``sendDocument``) when supplied.

Each :class:`TelegramRequest` exposes ``requests_kwargs()`` (the ``data`` /
``files`` for ``requests``/``httpx``). Pure Python, no new dependencies, no
network. Pairs with :mod:`fieldpilot_urdf.report`.
"""
from __future__ import annotations

import base64
import html
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .report import DiagnosticReport


class TelegramAttachment(BaseModel):
    """A binary part of a Telegram request (a photo / animation / document)."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(..., description="Bot API field name, e.g. animation/photo/document")
    filename: str = Field(..., description="Upload filename")
    content_type: str = Field(..., description="MIME type")
    data_b64: str = Field(..., description="Base64-encoded bytes")


class TelegramRequest(BaseModel):
    """One Telegram Bot API call: the ``method``, its form ``params``, and an
    optional binary ``attachment``."""

    model_config = ConfigDict(extra="forbid")

    method: str = Field(..., description="Bot API method, e.g. sendMessage / sendAnimation")
    params: dict[str, str] = Field(default_factory=dict, description="Form fields")
    attachment: Optional[TelegramAttachment] = Field(None, description="Binary upload, if any")

    def requests_kwargs(self) -> dict:
        """Keyword args for ``requests.post(api_url + method, **kwargs)`` (also
        accepted by httpx): the form ``data`` plus the uploaded ``files``."""
        kw: dict = {"data": dict(self.params)}
        if self.attachment is not None:
            a = self.attachment
            kw["files"] = {a.field: (a.filename, base64.b64decode(a.data_b64), a.content_type)}
        return kw


def report_summary_text(report: DiagnosticReport, *, parse_mode: str = "HTML") -> str:
    """The French message body recapping ``report`` for the technician. With
    ``parse_mode="HTML"`` the heading is bold and dynamic values are escaped;
    pass ``parse_mode=""`` for plain text."""
    esc = html.escape if parse_mode == "HTML" else (lambda s: s)
    bold = (lambda s: f"<b>{s}</b>") if parse_mode == "HTML" else (lambda s: s)

    lines = [bold("✅ Diagnostic confirmé" if report.confirmed else "🔍 Diagnostic en cours")]
    if report.machine:
        lines.append(f"Machine : {esc(report.machine)}")
    if report.fault:
        lines.append(f"Défaut : {esc(report.fault)}")
    if report.confidence is not None:
        lines.append(f"Confiance : {report.confidence * 100:.0f} %")
    if report.solution:
        lines.append(f"Solution : {esc(report.solution)}")
    if report.spare_parts:
        parts = ", ".join(f"{esc(p.name)} ×{p.quantity}" for p in report.spare_parts)
        lines.append(f"Pièces : {parts}")
    lines.append(f"Réf. : {esc(report.reference)}")
    return "\n".join(lines)


def _media_method(content_type: str) -> tuple[str, str]:
    if content_type == "image/gif":
        return "sendAnimation", "animation"
    if content_type.startswith("image/"):
        return "sendPhoto", "photo"
    return "sendDocument", "document"


def telegram_messages(
    report: DiagnosticReport,
    chat_id: Union[int, str],
    *,
    pdf: Optional[bytes] = None,
    parse_mode: str = "HTML",
) -> list[TelegramRequest]:
    """Build the Telegram replies for ``report`` to ``chat_id``: the summary
    message, one media message per report illustration (the 3D GIF →
    ``sendAnimation``, the oscilloscope PNG → ``sendPhoto``), and a
    ``sendDocument`` for the report ``pdf`` when given. Each illustration's
    caption is its :class:`~fieldpilot_urdf.report.ReportImage` caption."""
    cid = str(chat_id)
    out = [TelegramRequest(
        method="sendMessage",
        params={"chat_id": cid, "text": report_summary_text(report, parse_mode=parse_mode),
                "parse_mode": parse_mode} if parse_mode else
               {"chat_id": cid, "text": report_summary_text(report, parse_mode=parse_mode)},
    )]
    for img in report.illustrations:
        method, field = _media_method(img.content_type)
        params = {"chat_id": cid}
        if img.caption:
            params["caption"] = img.caption
        out.append(TelegramRequest(
            method=method, params=params,
            attachment=TelegramAttachment(field=field, filename=img.name,
                                          content_type=img.content_type, data_b64=img.data_b64)))
    if pdf is not None:
        out.append(TelegramRequest(
            method="sendDocument",
            params={"chat_id": cid, "caption": f"Rapport {report.reference}"},
            attachment=TelegramAttachment(
                field="document", filename=f"rapport_{report.reference}.pdf",
                content_type="application/pdf", data_b64=base64.b64encode(pdf).decode("ascii"))))
    return out
