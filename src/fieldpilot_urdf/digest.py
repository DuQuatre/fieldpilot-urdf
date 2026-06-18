"""Turn the dashboard KPIs into a weekly email digest.

The admin dashboard shows the case statistics live (:mod:`fieldpilot_urdf.dashboard`);
managers also want them pushed — a weekly email recapping how many interventions
ran, how many resolved, which faults dominated, and which fixes worked.
:func:`weekly_digest` renders a :class:`CaseStatsSummary` into an
:class:`EmailDigest` (subject + HTML + plain-text body, **in French**, per the
FieldPilot reports convention). Pass last week's summary too and it shows the
deltas. The package builds the email *content*; the n8n / SaaS side sends it —
same contract as the other integrations. Pure Python, no I/O.
"""
from __future__ import annotations

import html
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .dashboard import CaseStatsSummary


class EmailDigest(BaseModel):
    """A rendered email: subject plus HTML and plain-text bodies."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(..., description="Email subject (French)")
    html_body: str = Field(..., description="Self-contained HTML body")
    text_body: str = Field(..., description="Plain-text body")


def _pts_delta(cur: float, prev: float) -> str:
    d = round((cur - prev) * 100)
    return f" ({'+' if d >= 0 else ''}{d} pts)" if d else ""


def _count_delta(cur: int, prev: int) -> str:
    d = cur - prev
    return f" ({'+' if d >= 0 else ''}{d})" if d else ""


def weekly_digest(
    summary: CaseStatsSummary,
    *,
    period_label: str = "cette semaine",
    previous: Optional[CaseStatsSummary] = None,
    title: str = "Bilan diagnostic",
) -> EmailDigest:
    """Render ``summary`` as a weekly :class:`EmailDigest` (French).

    ``period_label`` names the period in the subject and heading (e.g.
    ``"semaine du 16/06/2026"``). When ``previous`` (last period's summary) is
    given, the resolution rate and case count show week-over-week deltas. An
    empty summary produces a valid digest noting there were no interventions.
    """
    e = html.escape
    n, res = summary.total_cases, summary.resolved_cases
    rate_pct = f"{summary.resolution_rate * 100:.0f}"
    n_delta = _count_delta(n, previous.total_cases) if previous else ""
    rate_delta = _pts_delta(summary.resolution_rate, previous.resolution_rate) if previous else ""

    subject = f"{title} — {period_label} : {n} intervention(s), {rate_pct} % résolues"

    # --- HTML tables ---
    if summary.top_faults:
        fault_rows = "".join(
            f"<tr><td>{e(f.fault)}</td><td>{f.count}</td><td>{f.share * 100:.0f} %</td></tr>"
            for f in summary.top_faults)
        faults_html = ("<h2>Défauts les plus fréquents</h2><table>"
                       "<tr><th>Défaut</th><th>Occurrences</th><th>Part</th></tr>"
                       f"{fault_rows}</table>")
    else:
        faults_html = ""
    if summary.top_solutions:
        sol_rows = "".join(
            f"<tr><td>{e(s.fault)}</td><td>{e(s.solution)}</td>"
            f"<td>{s.successes}/{s.attempts}</td><td>{s.success_rate * 100:.0f} %</td></tr>"
            for s in summary.top_solutions)
        sols_html = ("<h2>Solutions les plus efficaces</h2><table>"
                     "<tr><th>Défaut</th><th>Solution</th><th>Réussite</th><th>Taux</th></tr>"
                     f"{sol_rows}</table>")
    else:
        sols_html = ""

    intro = (f"<p><b>{n}</b> intervention(s){n_delta}, <b>{res}</b> résolues "
             f"(<b>{rate_pct} %</b>{rate_delta}). {summary.distinct_faults} défaut(s) distinct(s).</p>"
             if n else "<p>Aucune intervention diagnostiquée sur la période.</p>")
    style = (
        "body{font-family:Arial,Helvetica,sans-serif;color:#2C3E50;}"
        "h1{border-bottom:3px solid #2E86DE;padding-bottom:.3em;}"
        "h2{color:#1B4F72;margin-top:1.2em;}"
        "table{border-collapse:collapse;margin:.4em 0;}"
        "th,td{text-align:left;padding:.3em .8em;border-bottom:1px solid #ECF0F1;}"
        "th{color:#7F8C8D;}"
    )
    html_body = (
        "<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'/>"
        f"<style>{style}</style></head><body>"
        f"<h1>{e(title)} — {e(period_label)}</h1>{intro}{faults_html}{sols_html}"
        "</body></html>"
    )

    # --- plain text ---
    lines = [f"{title} — {period_label}", ""]
    if n:
        lines.append(f"{n} intervention(s){n_delta}, {res} résolues ({rate_pct} %{rate_delta}).")
        lines.append(f"{summary.distinct_faults} défaut(s) distinct(s).")
        if summary.top_faults:
            lines.append("\nDéfauts les plus fréquents :")
            lines += [f"  - {f.fault} : {f.count}× ({f.share * 100:.0f} %)" for f in summary.top_faults]
        if summary.top_solutions:
            lines.append("\nSolutions les plus efficaces :")
            lines += [f"  - {s.solution} ({s.fault}) : {s.successes}/{s.attempts} "
                      f"({s.success_rate * 100:.0f} %)" for s in summary.top_solutions]
    else:
        lines.append("Aucune intervention diagnostiquée sur la période.")
    text_body = "\n".join(lines)

    return EmailDigest(subject=subject, html_body=html_body, text_body=text_body)
