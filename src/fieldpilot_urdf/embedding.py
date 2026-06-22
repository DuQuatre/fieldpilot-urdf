"""Structural embedding of a URDF ``Robot`` (topology → fixed-size vector).

Ported from the MecAI project (MIT) and re-targeted onto the URDF ``Robot``.
MecAI embedded an abstract ``MechanicalSystem`` (which carries a sensor model);
URDF has no sensors, so the sensor feature is dropped and the joint-type
histogram follows the URDF ``JointType`` set. Everything else — the deterministic,
interpretable feature vector capturing a robot's *structural role* (joint-type
mix, branching, degree distribution, mass profile, DOF) — is unchanged in spirit,
so two kinematically similar robots land near each other under cosine similarity.

This is the lightweight, deterministic stand-in for a learned GNN graph-encoder:
a trained encoder can later replace :func:`robot_embedding` behind the same
signature without touching the storage or GraphRAG layers
(:mod:`fieldpilot_urdf.graphrag`).

Pure functions only (numpy + networkx, both core deps); no DB, no torch.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .graph import build_graph, is_tree
from .models import Robot

# Fixed joint-type order (URDF JointType) → stable histogram dimensions.
_JOINT_TYPES = ("revolute", "continuous", "prismatic", "fixed", "floating", "planar")

# Degrees of freedom contributed by each URDF joint type.
_DOF_BY_TYPE = {
    "revolute": 1, "continuous": 1, "prismatic": 1,
    "planar": 2, "floating": 6, "fixed": 0,
}

# Ordered feature names — the embedding vector follows this exact order.
FEATURE_NAMES: tuple[str, ...] = (
    "n_links", "n_joints", "dof", "is_tree",
    *(f"jt_frac_{t}" for t in _JOINT_TYPES),
    "max_in_degree", "max_out_degree", "mean_out_degree",
    "n_leaf_links", "n_branch_links", "depth",
    "total_mass", "mean_link_mass", "max_link_mass", "std_link_mass",
)
EMBEDDING_DIM = len(FEATURE_NAMES)


def robot_dof(robot: Robot) -> int:
    """Total mechanical DOF: sum of each joint's degrees of freedom (fixed = 0)."""
    return sum(_DOF_BY_TYPE.get(j.type, 0) for j in robot.joints)


def _link_mass(robot: Robot) -> list[float]:
    return [link.inertial.mass if link.inertial else 0.0 for link in robot.links]


def _depth(g: nx.DiGraph) -> float:
    """Longest directed path length (DAG); 0 for empty / falls back on cycles."""
    if g.number_of_nodes() == 0:
        return 0.0
    if nx.is_directed_acyclic_graph(g):
        return float(nx.dag_longest_path_length(g))
    # Defensive: build_graph yields a DAG for valid URDF, but stay finite anyway.
    ug = g.to_undirected()
    comp = max(nx.connected_components(ug), key=len)
    return float(nx.diameter(ug.subgraph(comp)))


def embedding_features(robot: Robot) -> dict[str, float]:
    """Return the labelled structural features (same keys as :data:`FEATURE_NAMES`)."""
    g = build_graph(robot)
    n_joints = len(robot.joints)

    jt_counts = {t: 0 for t in _JOINT_TYPES}
    for j in robot.joints:
        jt_counts[j.type] = jt_counts.get(j.type, 0) + 1
    jt_frac = {t: (jt_counts[t] / n_joints if n_joints else 0.0) for t in _JOINT_TYPES}

    out_deg = [d for _, d in g.out_degree()]
    in_deg = [d for _, d in g.in_degree()]
    masses = _link_mass(robot)

    feats: dict[str, float] = {
        "n_links": float(len(robot.links)),
        "n_joints": float(n_joints),
        "dof": float(robot_dof(robot)),
        "is_tree": 1.0 if (g.number_of_nodes() and is_tree(g)) else 0.0,
        **{f"jt_frac_{t}": jt_frac[t] for t in _JOINT_TYPES},
        "max_in_degree": float(max(in_deg) if in_deg else 0),
        "max_out_degree": float(max(out_deg) if out_deg else 0),
        "mean_out_degree": float(np.mean(out_deg)) if out_deg else 0.0,
        "n_leaf_links": float(sum(1 for _, d in g.out_degree() if d == 0)),
        "n_branch_links": float(sum(1 for _, d in g.out_degree() if d > 1)),
        "depth": _depth(g),
        "total_mass": float(sum(masses)),
        "mean_link_mass": float(np.mean(masses)) if masses else 0.0,
        "max_link_mass": float(max(masses)) if masses else 0.0,
        "std_link_mass": float(np.std(masses)) if masses else 0.0,
    }
    return feats


def robot_embedding(robot: Robot) -> np.ndarray:
    """Return the structural embedding as a ``(EMBEDDING_DIM,)`` float vector."""
    feats = embedding_features(robot)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in ``[-1, 1]``; 0 if either vector is all-zero."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rank_by_similarity(
    query: np.ndarray,
    candidates: dict[str, np.ndarray],
    *,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Rank ``{id: vector}`` candidates by cosine similarity to ``query`` (desc).

    Vectorized: stacks the candidate vectors into one matrix and computes every
    cosine in a single BLAS call, instead of a Python loop per candidate.
    """
    if not candidates:
        return []
    ids = list(candidates)
    matrix = np.vstack([np.asarray(candidates[i], dtype=float).ravel() for i in ids])
    q = np.asarray(query, dtype=float).ravel()
    q_norm = np.linalg.norm(q)
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * q_norm
    sims = np.divide(matrix @ q, denom, out=np.zeros(len(ids)), where=denom != 0)
    order = np.argsort(-sims, kind="stable")
    ranked = [(ids[i], float(sims[i])) for i in order]
    return ranked[:top_k] if top_k is not None else ranked
