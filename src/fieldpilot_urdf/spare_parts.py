"""Wire a report's spare parts into the Odoo SPA spare-parts order.

When a confirmed diagnosis recommends a fix, the parts it needs become an order
in the SPA (spare-parts) module — an Odoo ``sale.order`` with a line per part. As
with the intervention and Telegram wiring, the open-core package builds the
JSON-RPC payload *data*; the n8n / SaaS side creates the record:

* :func:`spare_parts_order_lines` — the ``sale.order.line`` command tuples for a
  list of :class:`~fieldpilot_urdf.report.SparePart`.
* :func:`spare_parts_order_vals` — the full ``sale.order`` create-values for a
  report (customer, ``origin`` = the intervention reference, the order lines).
* :func:`unresolved_part_refs` — the part references with no product mapping, so
  the caller knows which products still need creating in Odoo.

A part is matched to an Odoo product through a caller-supplied
``product_map`` (``{part reference: product_id}``); unmapped parts fall back to a
description-only order line (Odoo accepts a line with a ``name`` and no
``product_id``). Pure Python, no new dependencies, no network. Pairs with
:mod:`fieldpilot_urdf.report`.
"""
from __future__ import annotations

from typing import Optional

from .report import DiagnosticReport, SparePart


def spare_parts_order_lines(
    parts: list[SparePart],
    *,
    product_map: Optional[dict[str, int]] = None,
) -> list[tuple]:
    """Odoo ``sale.order.line`` command tuples ``(0, 0, vals)`` for ``parts``.

    Each line carries ``product_uom_qty`` and a ``name`` (``"{ref} — {desig}"``);
    when ``product_map`` resolves the part's reference, the line also carries the
    matching ``product_id``. Unmapped parts become description-only lines."""
    pm = product_map or {}
    lines: list[tuple] = []
    for p in parts:
        vals: dict[str, object] = {
            "name": f"{p.reference} — {p.name}",
            "product_uom_qty": p.quantity,
        }
        if p.reference in pm:
            vals["product_id"] = pm[p.reference]
        lines.append((0, 0, vals))
    return lines


def spare_parts_order_vals(
    report: DiagnosticReport,
    *,
    partner_id: int,
    product_map: Optional[dict[str, int]] = None,
) -> dict:
    """Odoo ``sale.order`` create-values for ``report``'s spare parts.

    ``partner_id`` is the customer the order is raised for; ``origin`` links the
    order back to the intervention (``report.reference``). The ``order_line`` is
    built by :func:`spare_parts_order_lines`. An empty parts list yields an order
    with no lines."""
    return {
        "partner_id": partner_id,
        "origin": report.reference,
        "order_line": spare_parts_order_lines(report.spare_parts, product_map=product_map),
    }


def unresolved_part_refs(
    parts: list[SparePart],
    product_map: Optional[dict[str, int]],
) -> list[str]:
    """The references of ``parts`` not present in ``product_map`` — the products
    that still need to exist in Odoo before the order can carry real product
    lines. Order-preserving and de-duplicated."""
    pm = product_map or {}
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p.reference not in pm and p.reference not in seen:
            seen.add(p.reference)
            out.append(p.reference)
    return out
