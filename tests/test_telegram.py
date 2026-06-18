"""Wiring the report into the Telegram bot reply: the French summary text and
the Bot API request sequence (summary + illustrations + optional PDF). Pure data
— no network — so we assert the shapes n8n POSTs.
"""
from __future__ import annotations

import base64

from fieldpilot_urdf.report import DiagnosticReport, ReportImage, SparePart
from fieldpilot_urdf.telegram import (
    TelegramRequest, report_summary_text, telegram_messages,
)


def _report(confirmed=True, illustrations=None):
    return DiagnosticReport(
        reference="INT-2026-0042", machine="robotA", confirmed=confirmed,
        fault="j_shoulder", confidence=0.97, solution="recalibrer_codeur",
        spare_parts=[SparePart(reference="ENC-1024", name="Codeur 1024 ppr")],
        illustrations=illustrations or [])


# --- summary text -----------------------------------------------------------

def test_summary_text_is_french_recap():
    txt = report_summary_text(_report())
    assert "Diagnostic confirmé" in txt and "<b>" in txt          # HTML heading
    assert "Défaut : j_shoulder" in txt
    assert "Confiance : 97 %" in txt
    assert "Solution : recalibrer_codeur" in txt
    assert "Codeur 1024 ppr ×1" in txt                            # spare part
    assert "Réf. : INT-2026-0042" in txt


def test_summary_text_plain_mode_and_unconfirmed():
    txt = report_summary_text(_report(confirmed=False), parse_mode="")
    assert "<b>" not in txt and "Diagnostic en cours" in txt


def test_summary_text_escapes_html():
    rep = DiagnosticReport(reference="R1", confirmed=True, fault="<x>", machine="a&b")
    txt = report_summary_text(rep)
    assert "&lt;x&gt;" in txt and "a&amp;b" in txt


# --- message sequence -------------------------------------------------------

def test_messages_start_with_summary():
    msgs = telegram_messages(_report(), chat_id=12345)
    assert all(isinstance(m, TelegramRequest) for m in msgs)
    assert msgs[0].method == "sendMessage"
    assert msgs[0].params["chat_id"] == "12345"                  # stringified
    assert msgs[0].params["parse_mode"] == "HTML"
    assert "j_shoulder" in msgs[0].params["text"]
    assert msgs[0].attachment is None


def test_illustrations_map_to_media_methods():
    gif = ReportImage.from_bytes("motion.gif", b"GIF89afake", content_type="image/gif",
                                 caption="mouvement")
    png = ReportImage.from_bytes("scope.png", b"\x89PNGfake", content_type="image/png",
                                 caption="oscilloscope")
    msgs = telegram_messages(_report(illustrations=[gif, png]), chat_id="777")
    assert [m.method for m in msgs] == ["sendMessage", "sendAnimation", "sendPhoto"]
    anim = msgs[1]
    assert anim.attachment.field == "animation"
    assert anim.params["caption"] == "mouvement"
    assert base64.b64decode(anim.attachment.data_b64) == b"GIF89afake"
    assert msgs[2].attachment.field == "photo"


def test_pdf_becomes_send_document():
    msgs = telegram_messages(_report(), chat_id=1, pdf=b"%PDF-1.4 data")
    doc = msgs[-1]
    assert doc.method == "sendDocument"
    assert doc.attachment.field == "document"
    assert doc.attachment.filename == "rapport_INT-2026-0042.pdf"
    assert base64.b64decode(doc.attachment.data_b64) == b"%PDF-1.4 data"
    assert doc.params["caption"] == "Rapport INT-2026-0042"


def test_requests_kwargs_are_post_ready():
    gif = ReportImage.from_bytes("m.gif", b"GIF89a", content_type="image/gif", caption="c")
    msgs = telegram_messages(_report(illustrations=[gif]), chat_id=42)
    # text message: data only, no files
    assert "files" not in msgs[0].requests_kwargs()
    assert msgs[0].requests_kwargs()["data"]["chat_id"] == "42"
    # media message: files carries the decoded bytes under the right field
    kw = msgs[1].requests_kwargs()
    name, content, mime = kw["files"]["animation"]
    assert name == "m.gif" and content == b"GIF89a" and mime == "image/gif"


def test_no_illustrations_just_summary():
    msgs = telegram_messages(_report(), chat_id=9)
    assert len(msgs) == 1 and msgs[0].method == "sendMessage"
