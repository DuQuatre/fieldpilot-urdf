"""Symbolic diagnostic rules over a URDF Robot.

Each rule is a pure function `(robot, G) -> list[Finding]`. Add rules by
appending to RULES. Findings use a compact format: {code, severity, message, refs}.
"""
from __future__ import annotations

import inspect
from typing import Callable, Literal, Optional

import networkx as nx
import numpy as np
from pydantic import BaseModel

from .collisions import MeshResolver, detect_self_collisions
from .graph import build_graph, is_tree, root_links
from .models import Robot

Severity = Literal["info", "warning", "error"]


class Finding(BaseModel):
    code: str
    severity: Severity
    message: str
    refs: list[str] = []


Rule = Callable[[Robot, nx.DiGraph], list[Finding]]


# --- rules -----------------------------------------------------------------

def r001_single_root(robot: Robot, G: nx.DiGraph) -> list[Finding]:
    roots = root_links(G)
    if len(roots) == 1:
        return []
    if len(roots) == 0:
        return [Finding(code="R001", severity="error",
                        message="no root link — every link has a parent joint (cycle?)",
                        refs=[])]
    return [Finding(code="R001", severity="error",
                    message=f"multiple root links ({len(roots)}): URDF must have exactly one",
                    refs=roots)]


def r002_is_tree(robot: Robot, G: nx.DiGraph) -> list[Finding]:
    if is_tree(G):
        return []
    return [Finding(code="R002", severity="error",
                    message=f"kinematic graph is not a tree "
                            f"(nodes={G.number_of_nodes()}, edges={G.number_of_edges()})",
                    refs=[])]


def r003_joint_limits(robot: Robot, G: nx.DiGraph) -> list[Finding]:
    out: list[Finding] = []
    for j in robot.joints:
        if j.limit is None:
            continue
        if j.limit.lower >= j.limit.upper:
            out.append(Finding(code="R003", severity="error",
                               message=f"joint '{j.name}': lower ({j.limit.lower}) "
                                       f">= upper ({j.limit.upper})",
                               refs=[j.name]))
        if j.limit.effort <= 0:
            out.append(Finding(code="R003", severity="warning",
                               message=f"joint '{j.name}': effort <= 0 ({j.limit.effort})",
                               refs=[j.name]))
        if j.limit.velocity <= 0:
            out.append(Finding(code="R003", severity="warning",
                               message=f"joint '{j.name}': velocity <= 0 ({j.limit.velocity})",
                               refs=[j.name]))
    return out


def r004_mass_positive(robot: Robot, G: nx.DiGraph) -> list[Finding]:
    out: list[Finding] = []
    for l in robot.links:
        if l.inertial is None:
            continue
        if l.inertial.mass <= 0:
            out.append(Finding(code="R004", severity="error",
                               message=f"link '{l.name}': mass <= 0 ({l.inertial.mass})",
                               refs=[l.name]))
    return out


def r005_inertia_psd(robot: Robot, G: nx.DiGraph) -> list[Finding]:
    """Inertia tensor must be symmetric positive semi-definite."""
    out: list[Finding] = []
    tol = 1e-9
    for l in robot.links:
        if l.inertial is None:
            continue
        i = l.inertial.inertia
        M = np.array([[i.ixx, i.ixy, i.ixz],
                      [i.ixy, i.iyy, i.iyz],
                      [i.ixz, i.iyz, i.izz]], dtype=float)
        eig = np.linalg.eigvalsh(M)
        if (eig < -tol).any():
            out.append(Finding(code="R005", severity="error",
                               message=f"link '{l.name}': inertia not PSD "
                                       f"(min eigenvalue = {eig.min():.3e})",
                               refs=[l.name]))
    return out


def r006_unique_names(robot: Robot, G: nx.DiGraph) -> list[Finding]:
    out: list[Finding] = []
    seen: dict[str, int] = {}
    for l in robot.links:
        seen[l.name] = seen.get(l.name, 0) + 1
    for name, n in seen.items():
        if n > 1:
            out.append(Finding(code="R006", severity="error",
                               message=f"link name '{name}' appears {n} times",
                               refs=[name]))
    seen = {}
    for j in robot.joints:
        seen[j.name] = seen.get(j.name, 0) + 1
    for name, n in seen.items():
        if n > 1:
            out.append(Finding(code="R006", severity="error",
                               message=f"joint name '{name}' appears {n} times",
                               refs=[name]))
    return out


def r007_joint_axis(robot: Robot, G: nx.DiGraph) -> list[Finding]:
    """Movable joints must have a non-zero axis vector."""
    out: list[Finding] = []
    for j in robot.joints:
        if j.type not in {"revolute", "continuous", "prismatic"}:
            continue
        n = float(np.linalg.norm(np.array(j.axis)))
        if n < 1e-9:
            out.append(Finding(code="R007", severity="error",
                               message=f"joint '{j.name}' ({j.type}): "
                                       f"axis is zero vector {j.axis}",
                               refs=[j.name]))
    return out


def r008_self_collisions(
    robot: Robot, G: nx.DiGraph, *,
    mesh_resolver: Optional[MeshResolver] = None,
) -> list[Finding]:
    """AABB self-collisions at neutral pose, excluding adjacent links.

    Skipped if FK can't be computed (R001/R002 cover that case). Mesh shapes
    contribute when `mesh_resolver` can locate the file; otherwise they're
    silently ignored (the pre-mesh-phase contract).
    """
    try:
        hits = detect_self_collisions(robot, mesh_resolver=mesh_resolver)
    except ValueError:
        return []
    return [Finding(code="R008", severity="warning",
                    message=f"AABB collision at neutral pose: '{a}' <-> '{b}'",
                    refs=[a, b])
            for a, b in hits]


RULES: list[Rule] = [
    r001_single_root, r002_is_tree, r003_joint_limits,
    r004_mass_positive, r005_inertia_psd, r006_unique_names,
    r007_joint_axis, r008_self_collisions,
]


def run_all(
    robot: Robot, *,
    mesh_resolver: Optional[MeshResolver] = None,
) -> list[Finding]:
    """Run every rule. Rules that accept `mesh_resolver` get it threaded
    through; others are invoked with the original (robot, G) signature."""
    G = build_graph(robot)
    findings: list[Finding] = []
    for rule in RULES:
        if "mesh_resolver" in inspect.signature(rule).parameters:
            findings.extend(rule(robot, G, mesh_resolver=mesh_resolver))
        else:
            findings.extend(rule(robot, G))
    return findings


def summary(findings: list[Finding]) -> dict:
    by_severity: dict[str, int] = {"info": 0, "warning": 0, "error": 0}
    for f in findings:
        by_severity[f.severity] += 1
    return {"total": len(findings), **by_severity}
