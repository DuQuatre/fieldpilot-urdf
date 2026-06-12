"""URDF Robot -> NetworkX DiGraph.

`build_graph(robot)` returns an `nx.DiGraph`. Nodes = links (attrs: mass,
has_inertial). Edges = joints, directed parent->child (attrs: joint_name,
joint_type, axis, limit). A valid URDF is a tree.
"""
from __future__ import annotations

from typing import Optional

import networkx as nx

from .models import Robot


def build_graph(robot: Robot) -> nx.DiGraph:
    G = nx.DiGraph(name=robot.name)
    for link in robot.links:
        G.add_node(
            link.name,
            kind="link",
            mass=link.inertial.mass if link.inertial else None,
            has_inertial=link.inertial is not None,
            n_visuals=len(link.visuals),
            n_collisions=len(link.collisions),
        )
    for joint in robot.joints:
        G.add_edge(
            joint.parent,
            joint.child,
            joint_name=joint.name,
            joint_type=joint.type,
            axis=joint.axis,
            limit=(joint.limit.model_dump() if joint.limit else None),
        )
    return G


def root_links(G: nx.DiGraph) -> list[str]:
    """Links that are no joint's child. URDF requires exactly one."""
    return [n for n in G.nodes if G.in_degree(n) == 0]


def leaf_links(G: nx.DiGraph) -> list[str]:
    return [n for n in G.nodes if G.out_degree(n) == 0]


def chain(G: nx.DiGraph, root_link: str, leaf_link: str) -> Optional[list[str]]:
    """Return link names from root to leaf, or None if no path."""
    try:
        return nx.shortest_path(G, root_link, leaf_link)
    except nx.NetworkXNoPath:
        return None


def joints_on_path(G: nx.DiGraph, path: list[str]) -> list[str]:
    return [G.edges[path[i], path[i + 1]]["joint_name"] for i in range(len(path) - 1)]


def subtree(G: nx.DiGraph, root_link: str) -> set[str]:
    """All descendant links (incl. root)."""
    return {root_link} | nx.descendants(G, root_link)


def is_tree(G: nx.DiGraph) -> bool:
    """True iff G is a single connected tree (URDF invariant)."""
    if G.number_of_nodes() == 0:
        return True
    if G.number_of_edges() != G.number_of_nodes() - 1:
        return False
    return nx.is_weakly_connected(G)
