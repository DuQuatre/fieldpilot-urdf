"""Roll the diagnostic case base up into KPIs for the admin / MRR dashboard.

The case knowledge base (:mod:`fieldpilot_urdf.case_base`) accumulates every
resolved diagnosis; the dashboard wants the headline numbers beside the revenue
KPIs — how many cases, how many resolved, which faults dominate, which fixes
work. :func:`case_stats_summary` aggregates a ``list[DiagnosticCase]`` into a
JSON-serializable :class:`CaseStatsSummary` the dashboard endpoint serves (the
package builds the data; the FastAPI dashboard renders it — same contract as the
other integrations). Pure Python, no I/O.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .case_base import DiagnosticCase, SolutionStat, solution_stats


class FaultStat(BaseModel):
    """How often one fault has been confirmed."""

    model_config = ConfigDict(extra="forbid")

    fault: str = Field(..., description="The confirmed fault")
    count: int = Field(..., description="Number of cases confirming it")
    share: float = Field(..., description="Fraction of confirmed-fault cases")


class CaseStatsSummary(BaseModel):
    """Dashboard KPI block aggregated from the diagnostic case base."""

    model_config = ConfigDict(extra="forbid")

    total_cases: int = Field(..., description="Cases considered")
    resolved_cases: int = Field(..., description="Cases marked resolved")
    resolution_rate: float = Field(..., description="resolved / total (0 if no cases)")
    distinct_faults: int = Field(..., description="Number of distinct confirmed faults")
    top_faults: list[FaultStat] = Field(..., description="Most frequent faults, best-first")
    top_solutions: list[SolutionStat] = Field(..., description="Best-performing fixes, best-first")
    machine: Optional[str] = Field(None, description="Machine filter, if any")


def case_stats_summary(
    cases: list[DiagnosticCase],
    *,
    machine: Optional[str] = None,
    top_n: int = 5,
) -> CaseStatsSummary:
    """Aggregate ``cases`` into a :class:`CaseStatsSummary` for the dashboard.

    Counts cases (optionally only those for ``machine``), the resolved share, the
    ``top_n`` most-confirmed faults (with their share of confirmed-fault cases),
    and the ``top_n`` best-performing fixes (via :func:`...case_base.solution_stats`,
    success-rate first). Empty input yields an all-zero summary."""
    sel = [c for c in cases if machine is None or c.machine == machine]
    total = len(sel)
    resolved = sum(1 for c in sel if c.resolved)

    with_fault = [c.confirmed_fault for c in sel if c.confirmed_fault]
    counts = Counter(with_fault)
    n_fault = len(with_fault)
    top_faults = [
        FaultStat(fault=f, count=n, share=(n / n_fault if n_fault else 0.0))
        for f, n in counts.most_common(top_n)
    ]

    return CaseStatsSummary(
        total_cases=total,
        resolved_cases=resolved,
        resolution_rate=(resolved / total if total else 0.0),
        distinct_faults=len(counts),
        top_faults=top_faults,
        top_solutions=solution_stats(sel)[:top_n],
        machine=machine,
    )
