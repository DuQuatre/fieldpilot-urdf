"""Wiring the report's spare parts into the Odoo SPA sale.order. Pure data — no
network — so we assert the JSON-RPC create-values the SaaS sends.
"""
from __future__ import annotations

from fieldpilot_urdf.report import DiagnosticReport, SparePart
from fieldpilot_urdf.spare_parts import (
    spare_parts_order_lines, spare_parts_order_vals, unresolved_part_refs,
)


def _parts():
    return [SparePart(reference="ENC-1024", name="Codeur 1024 ppr"),
            SparePart(reference="CAL-KIT", name="Kit d'étalonnage", quantity=2)]


def _report():
    return DiagnosticReport(reference="INT-2026-0042", confirmed=True,
                            fault="j_shoulder", spare_parts=_parts())


# --- order lines ------------------------------------------------------------

def test_order_lines_shape_and_product_mapping():
    lines = spare_parts_order_lines(_parts(), product_map={"ENC-1024": 1001})
    assert len(lines) == 2
    op, zero, vals = lines[0]
    assert op == 0 and zero == 0                               # Odoo create command
    assert vals["product_id"] == 1001                          # resolved
    assert vals["product_uom_qty"] == 1
    assert vals["name"] == "ENC-1024 — Codeur 1024 ppr"
    # unmapped part -> description-only line (no product_id)
    assert "product_id" not in lines[1][2]
    assert lines[1][2]["product_uom_qty"] == 2


def test_order_lines_empty():
    assert spare_parts_order_lines([]) == []


# --- order vals -------------------------------------------------------------

def test_order_vals_carry_partner_and_origin():
    vals = spare_parts_order_vals(_report(), partner_id=42, product_map={"ENC-1024": 1001})
    assert vals["partner_id"] == 42
    assert vals["origin"] == "INT-2026-0042"                   # links to the intervention
    assert len(vals["order_line"]) == 2
    assert vals["order_line"][0][2]["product_id"] == 1001


def test_order_vals_empty_parts():
    report = DiagnosticReport(reference="R1", spare_parts=[])
    vals = spare_parts_order_vals(report, partner_id=1)
    assert vals["order_line"] == [] and vals["partner_id"] == 1


# --- unresolved refs --------------------------------------------------------

def test_unresolved_part_refs():
    parts = _parts()
    assert unresolved_part_refs(parts, {"ENC-1024": 1001}) == ["CAL-KIT"]
    assert unresolved_part_refs(parts, None) == ["ENC-1024", "CAL-KIT"]
    assert unresolved_part_refs(parts, {"ENC-1024": 1, "CAL-KIT": 2}) == []


def test_unresolved_part_refs_dedups():
    parts = [SparePart(reference="X", name="a"), SparePart(reference="X", name="a")]
    assert unresolved_part_refs(parts, None) == ["X"]          # order-preserving, de-duped
