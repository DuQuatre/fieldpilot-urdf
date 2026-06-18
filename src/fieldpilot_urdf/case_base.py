"""Diagnostic case knowledge base: remember failures and what fixed them.

Every resolved diagnosis is worth keeping: the symptom, the questions asked and
answered, the fault it turned out to be, and the fix that worked. Stored up,
those cases turn into *statistics* — how often each fault actually occurs
(:func:`fault_priors`) and how well each fix resolves it
(:func:`solution_stats`) — which feed straight back into the differential-
diagnosis engine as **learned priors**, so the dialog gets sharper the more the
fleet is serviced.

Two halves, kept separate:

* **Persistence** (`save_case` / `load_case` / `load_cases` / `list_cases` /
  `delete_case`) — one JSON file per case under a directory (the
  ``FIELDPILOT_URDF_CASE_DIR`` env var, default ``/data/diagnostic-cases``, or an
  explicit ``root=``).
* **Aggregation** (`fault_priors` / `solution_stats` / `recommend_solution`) —
  pure functions over a ``list[DiagnosticCase]``, no I/O, so they are trivial to
  test and to run over a caller-filtered slice.

Pure Python, core install. Pairs with
:mod:`fieldpilot_urdf.differential_diagnosis` (`fault_priors` →
`candidates_from_scores` → `rank_questions`).
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

CASE_DIR_ENV = "FIELDPILOT_URDF_CASE_DIR"
_DEFAULT_CASE_DIR = Path("/data/diagnostic-cases")


class DiagnosticCase(BaseModel):
    """One serviced fault: what was reported, what was asked, what it was, and
    what fixed it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique case identifier (e.g. an intervention ref)")
    symptom: str = Field("", description="Free-text description of the reported problem")
    machine: Optional[str] = Field(None, description="Machine / robot identifier, for per-machine stats")
    candidates: list[str] = Field(default_factory=list, description="Fault candidates considered")
    answers: dict[str, str] = Field(default_factory=dict, description="question_id -> outcome given")
    confirmed_fault: Optional[str] = Field(None, description="The fault it turned out to be")
    solution: Optional[str] = Field(None, description="The fix that was applied")
    resolved: bool = Field(False, description="Whether the solution resolved the issue")
    created_at: Optional[str] = Field(None, description="ISO timestamp; set on save if absent")
    notes: str = Field("", description="Free-text notes")


# --- persistence ------------------------------------------------------------

def _case_dir(root: Optional[Path]) -> Path:
    if root is not None:
        return Path(root)
    return Path(os.environ.get(CASE_DIR_ENV, _DEFAULT_CASE_DIR))


def _case_path(case_id: str, root: Optional[Path]) -> Path:
    return _case_dir(root) / f"{case_id}.json"


def save_case(case: DiagnosticCase, *, root: Optional[Path] = None) -> DiagnosticCase:
    """Persist a case as ``{id}.json``. Fills ``created_at`` with the current UTC
    time if unset. Returns the (possibly timestamp-filled) case; overwrites an
    existing case with the same id."""
    if case.created_at is None:
        case = case.model_copy(update={"created_at": datetime.now(timezone.utc).isoformat()})
    d = _case_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    _case_path(case.id, root).write_text(case.model_dump_json(indent=2))
    return case


def load_case(case_id: str, *, root: Optional[Path] = None) -> Optional[DiagnosticCase]:
    """Load one case by id, or ``None`` if it doesn't exist."""
    p = _case_path(case_id, root)
    if not p.exists():
        return None
    return DiagnosticCase.model_validate_json(p.read_text())


def list_cases(*, root: Optional[Path] = None) -> list[str]:
    """All stored case ids (sorted)."""
    d = _case_dir(root)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def load_cases(*, root: Optional[Path] = None) -> list[DiagnosticCase]:
    """Load every stored case (skipping any unreadable file), newest-first by
    ``created_at``."""
    cases: list[DiagnosticCase] = []
    for cid in list_cases(root=root):
        c = load_case(cid, root=root)
        if c is not None:
            cases.append(c)
    cases.sort(key=lambda c: c.created_at or "", reverse=True)
    return cases


def delete_case(case_id: str, *, root: Optional[Path] = None) -> bool:
    """Delete a case file. Returns ``False`` if it wasn't there."""
    p = _case_path(case_id, root)
    if not p.exists():
        return False
    p.unlink()
    return True


# --- aggregation (pure) -----------------------------------------------------

class SolutionStat(BaseModel):
    """Track record of one (fault, solution) pair across cases."""

    model_config = ConfigDict(extra="forbid")

    fault: str = Field(..., description="The confirmed fault")
    solution: str = Field(..., description="The fix that was tried")
    attempts: int = Field(..., description="How many cases tried this fix for this fault")
    successes: int = Field(..., description="How many of those resolved the issue")
    success_rate: float = Field(..., description="successes / attempts")


def fault_priors(
    cases: list[DiagnosticCase],
    *,
    machine: Optional[str] = None,
    candidates: Optional[list[str]] = None,
    smoothing: float = 1.0,
) -> dict[str, float]:
    """Empirical fault frequencies as a normalized prior — feed straight into
    `differential_diagnosis.candidates_from_scores`.

    Counts the ``confirmed_fault`` of each resolved case (optionally only those
    for ``machine``), Laplace-smoothed by ``smoothing`` so unseen faults keep a
    small mass. Pass ``candidates`` to force a prior over a specific fault set
    (faults never seen still get the smoothing share — with no data and a
    candidate list this is just a uniform prior). Returns ``{}`` when there is
    nothing to score.
    """
    relevant = [c for c in cases
                if c.confirmed_fault and (machine is None or c.machine == machine)]
    counts = Counter(c.confirmed_fault for c in relevant)
    names = set(counts)
    if candidates:
        names |= set(candidates)
    if not names:
        return {}
    scored = {n: counts.get(n, 0) + smoothing for n in names}
    total = sum(scored.values())
    return {n: v / total for n, v in scored.items()}


def solution_stats(
    cases: list[DiagnosticCase],
    *,
    fault: Optional[str] = None,
) -> list[SolutionStat]:
    """Per-(fault, solution) success statistics, best-first.

    Considers cases that have both a ``confirmed_fault`` and a ``solution``
    (optionally only those for ``fault``); a case counts as a success when
    ``resolved`` is True. Sorted by success rate, then attempts (confidence),
    then solution name.
    """
    pairs: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])  # [attempts, successes]
    for c in cases:
        if not c.confirmed_fault or not c.solution:
            continue
        if fault is not None and c.confirmed_fault != fault:
            continue
        rec = pairs[(c.confirmed_fault, c.solution)]
        rec[0] += 1
        rec[1] += 1 if c.resolved else 0
    stats = [
        SolutionStat(fault=f, solution=s, attempts=a, successes=su, success_rate=su / a)
        for (f, s), (a, su) in pairs.items()
    ]
    stats.sort(key=lambda st: (st.success_rate, st.attempts, st.solution), reverse=True)
    return stats


def recommend_solution(
    cases: list[DiagnosticCase],
    fault: str,
    *,
    min_attempts: int = 1,
) -> Optional[str]:
    """The fix with the best track record for ``fault`` (at least ``min_attempts``
    tries), or ``None`` if there's no qualifying history."""
    for st in solution_stats(cases, fault=fault):
        if st.attempts >= min_attempts:
            return st.solution
    return None
