"""Graph backends for the fieldpilot-urdf GraphRAG server.

Ported from MecAI (MIT) and re-targeted onto the URDF ``Robot``. A
:class:`GraphBackend` is the union of a :class:`~fieldpilot_urdf.graphrag.store.RobotStore`
(put/get/list/delete/clear) and a small graph-analytics surface (embeddings,
joint-type / motif / neighbourhood queries, fault history).
:class:`~fieldpilot_urdf.graphrag.rag.GraphRAG` runs over *any* backend, so the
retrieval logic is written once and tested for real — there is no hollow "fake DB".

Two implementations satisfy the contract:

* :class:`MemoryGraphBackend` (this module) — pure NetworkX, in-process, the
  default. All queries run against the same :class:`~fieldpilot_urdf.models.Robot`
  objects the store already holds; embeddings are cached. Optionally persists to
  disk via an internal :class:`~fieldpilot_urdf.graphrag.store.FileStore`.
* :class:`~fieldpilot_urdf.graphrag.neo4j_backend.Neo4jStore` — Cypher over
  Neo4j/Memgraph, for durable, cross-process, larger-than-memory storage.

A parity test suite holds both to the same behaviour.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import networkx as nx
import numpy as np

from ..embedding import robot_embedding
from ..graph import build_graph
from ..models import Robot
from .store import model_id


@runtime_checkable
class GraphBackend(Protocol):
    """Storage + graph-analytics contract shared by every backend."""

    available: bool

    # storage (RobotStore-compatible)
    def put(self, robot: Robot) -> None: ...
    def get(self, robot_id: str) -> Robot | None: ...
    def list(self) -> list[str]: ...
    def delete(self, robot_id: str) -> bool: ...
    def clear(self) -> None: ...

    # analytics
    def all_embeddings(self) -> dict[str, np.ndarray]: ...
    def models_with_joint_type(self, joint_type: str) -> list[str]: ...
    def joint_chain_motif(self, type_a: str, type_b: str) -> list[dict]: ...
    def subgraph_around(self, model_id: str, link_id: str, hops: int) -> list[dict]: ...

    # fault-history feedback loop
    def write_fault_event(self, model_id: str, fault: dict) -> None: ...
    def get_fault_events(self, model_id: str) -> list[dict]: ...


class MemoryGraphBackend:
    """In-process NetworkX backend; the default store + GraphRAG engine."""

    available = True

    def __init__(self, persist_dir=None) -> None:
        self._robots: dict[str, Robot] = {}
        self._emb: dict[str, np.ndarray] = {}      # cached embeddings
        self._faults: dict[str, list[dict]] = {}
        self._file = None
        if persist_dir is not None:
            from .store import FileStore

            self._file = FileStore(persist_dir)
            for rid in self._file.list():
                robot = self._file.get(rid)
                if robot is not None:
                    self._robots[rid] = robot

    # ------------------------------------------------------------------
    # storage
    # ------------------------------------------------------------------

    def put(self, robot: Robot) -> None:
        rid = model_id(robot)
        self._robots[rid] = robot
        self._emb.pop(rid, None)  # invalidate cached embedding
        if self._file is not None:
            self._file.put(robot)

    def get(self, robot_id: str) -> Robot | None:
        return self._robots.get(robot_id)

    def list(self) -> list[str]:
        return sorted(self._robots)

    def delete(self, robot_id: str) -> bool:
        existed = self._robots.pop(robot_id, None) is not None
        self._emb.pop(robot_id, None)
        self._faults.pop(robot_id, None)
        if self._file is not None and existed:
            self._file.delete(robot_id)
        return existed

    def clear(self) -> None:
        self._robots.clear()
        self._emb.clear()
        self._faults.clear()
        if self._file is not None:
            self._file.clear()

    # ------------------------------------------------------------------
    # analytics
    # ------------------------------------------------------------------

    def _embedding(self, robot_id: str) -> np.ndarray:
        if robot_id not in self._emb:
            self._emb[robot_id] = robot_embedding(self._robots[robot_id])
        return self._emb[robot_id]

    def all_embeddings(self) -> dict[str, np.ndarray]:
        return {rid: self._embedding(rid) for rid in self._robots}

    def models_with_joint_type(self, joint_type: str) -> list[str]:
        return sorted(
            rid
            for rid, r in self._robots.items()
            if any(j.type == joint_type for j in r.joints)
        )

    def joint_chain_motif(self, type_a: str, type_b: str) -> list[dict]:
        results: list[dict] = []
        for rid in sorted(self._robots):
            r = self._robots[rid]
            firsts = [j for j in r.joints if j.type == type_a]
            for ja in firsts:
                for jb in r.joints:
                    if jb.type == type_b and jb.parent == ja.child:
                        results.append({"id": rid, "first": ja.name, "second": jb.name})
        return results

    def subgraph_around(self, model_id: str, link_id: str, hops: int = 2) -> list[dict]:
        robot = self._robots.get(model_id)
        if robot is None or link_id not in {l.name for l in robot.links}:
            return []
        hops = max(1, min(int(hops), 5))
        g = build_graph(robot).to_undirected()
        reach = nx.single_source_shortest_path_length(g, link_id, cutoff=hops)
        masses = {l.name: (l.inertial.mass if l.inertial else 0.0) for l in robot.links}
        out = [{"link": n, "mass": masses.get(n, 0.0)} for n in reach if n != link_id]
        out.sort(key=lambda r: r["link"])
        return out

    # ------------------------------------------------------------------
    # fault-history feedback loop
    # ------------------------------------------------------------------

    def write_fault_event(self, model_id: str, fault: dict) -> None:
        self._faults.setdefault(model_id, []).append(
            {
                "type": fault.get("type", ""),
                "target": fault.get("target", ""),
                "severity": float(fault.get("severity", 0.0)),
                "ts": fault.get("ts", ""),
                "note": fault.get("note", ""),
            }
        )

    def get_fault_events(self, model_id: str) -> list[dict]:
        return sorted(self._faults.get(model_id, []), key=lambda f: f["ts"])
