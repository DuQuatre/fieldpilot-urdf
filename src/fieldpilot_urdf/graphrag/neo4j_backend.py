"""Neo4j/Memgraph graph backend for fieldpilot-urdf GraphRAG.

Ported from MecAI (MIT) and re-targeted onto the URDF ``Robot`` (no sensor model;
joint types are URDF strings; the stable id is the robot ``name``). The durable,
cross-process counterpart to
:class:`~fieldpilot_urdf.graphrag.backend.MemoryGraphBackend`. Both satisfy the
same :class:`~fieldpilot_urdf.graphrag.backend.GraphBackend` contract, so
:class:`~fieldpilot_urdf.graphrag.rag.GraphRAG` and the server endpoints work
identically over either — proven by the parity tests.

A :class:`Robot` is persisted two ways at once:

* a canonical ``model_json`` property on the ``:Model`` node → faithful
  round-trip for :meth:`GraphDB.get`.
* a decomposed ``:Link`` / ``[:JOINT]`` subgraph → Cypher pattern queries and
  structural similarity.

Writes are batched: one ``UNWIND`` query each for links and joints, instead of a
round-trip per element.

Data model
----------
    (:Model {id, name, root, model_json, embedding, n_links, n_joints, dof, is_tree})
      -[:HAS_LINK]->   (:Link {uid, model_id, id, mass})
    (:Link)-[:JOINT {joint_id, model_id, type, axis}]->(:Link)   # parent → child
    (:Model)-[:HAD_FAULT]->  (:FaultEvent {model_id, type, target, severity, ts, note})

Connection is configured by env vars and is **optional**: if the ``neo4j`` driver
isn't installed (``[graphrag]`` extra) or ``NEO4J_BOLT_URL`` is unset,
:attr:`GraphDB.available` is ``False`` and methods raise ``RuntimeError``.
:class:`Neo4jStore` raises on construction in that case — a persistent store must
not silently drop writes.

This pairs the structural fleet graph with :mod:`fieldpilot_urdf.case_base`: a
recorded ``:FaultEvent`` is the durable, queryable shadow of a ``DiagnosticCase``.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from ..embedding import robot_dof, robot_embedding
from ..graph import build_graph, is_tree, root_links
from ..models import Robot

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase

    _DRIVER_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _DRIVER_AVAILABLE = False
    logger.info("neo4j driver not installed — Neo4j backend disabled (pip install 'fieldpilot-urdf[graphrag]')")


def _env(*names: str, default: str = "") -> str:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    return default


def _robot_root(robot: Robot) -> str:
    roots = root_links(build_graph(robot))
    return roots[0] if len(roots) == 1 else ""


class GraphDB:
    """Bolt driver wrapper implementing the GraphBackend contract."""

    def __init__(self) -> None:
        self._driver = None
        self.available = False
        self._connect()

    def _connect(self) -> None:
        if not _DRIVER_AVAILABLE:
            return
        url = _env("NEO4J_BOLT_URL", "MEMGRAPH_BOLT_URL")
        if not url:
            return
        user = _env("NEO4J_USER", default="neo4j")
        pwd = _env("NEO4J_PASSWORD", "MEMGRAPH_PASSWORD")
        try:
            self._driver = GraphDatabase.driver(url, auth=(user, pwd))
            self._driver.verify_connectivity()
            self.available = True
            logger.info("graph DB connected: %s", url)
        except Exception as exc:  # pragma: no cover - needs a live server
            logger.warning("graph DB connection failed (%s) — features disabled", exc)
            self._driver = None

    # ------------------------------------------------------------------
    # low-level
    # ------------------------------------------------------------------

    def run(self, cypher: str, **params) -> list[dict]:
        """Run a Cypher statement, returning rows as dicts. Raises if unavailable."""
        if not self.available:
            raise RuntimeError("graph DB is not available")
        with self._driver.session() as s:
            return [r.data() for r in s.run(cypher, **params)]

    def ensure_schema(self) -> None:
        if not self.available:
            return
        for stmt in (
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Model) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Link) REQUIRE l.uid IS UNIQUE",
        ):
            try:
                self.run(stmt)
            except Exception as exc:  # pragma: no cover - syntax varies by engine
                logger.warning("schema stmt failed (%s): %s", stmt, exc)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    # ------------------------------------------------------------------
    # storage
    # ------------------------------------------------------------------

    def put(self, robot: Robot) -> None:
        """Upsert a robot: canonical JSON + decomposed subgraph (batched writes)."""
        if not self.available:
            raise RuntimeError("graph DB is not available")
        emb = robot_embedding(robot).tolist()
        g = build_graph(robot)
        tree = bool(robot.links) and is_tree(g)
        mid = robot.name
        links = [
            {"uid": f"{mid}:{link.name}", "id": link.name,
             "mass": (link.inertial.mass if link.inertial else 0.0)}
            for link in robot.links
        ]
        joints = [
            {"puid": f"{mid}:{j.parent}", "cuid": f"{mid}:{j.child}", "jid": j.name,
             "type": j.type, "axis": list(j.axis)}
            for j in robot.joints
        ]
        with self._driver.session() as session:
            session.execute_write(self._tx_put, robot, emb, tree, links, joints)

    @staticmethod
    def _tx_put(tx, robot, emb, tree, links, joints) -> None:
        mid = robot.name
        tx.run(
            "MATCH (m:Model {id:$mid}) OPTIONAL MATCH (m)-[:HAS_LINK|HAD_FAULT]->(n) "
            "DETACH DELETE m, n",
            mid=mid,
        )
        tx.run(
            "CREATE (m:Model {id:$mid, name:$name, root:$root, model_json:$json, "
            "embedding:$emb, n_links:$nl, n_joints:$nj, dof:$dof, is_tree:$tree})",
            mid=mid, name=robot.name, root=_robot_root(robot),
            json=robot.model_dump_json(), emb=emb,
            nl=len(robot.links), nj=len(robot.joints), dof=robot_dof(robot), tree=tree,
        )
        tx.run(
            "MATCH (m:Model {id:$mid}) UNWIND $links AS l "
            "CREATE (m)-[:HAS_LINK]->(:Link {uid:l.uid, model_id:$mid, id:l.id, mass:l.mass})",
            mid=mid, links=links,
        )
        tx.run(
            "UNWIND $joints AS j MATCH (p:Link {uid:j.puid}), (c:Link {uid:j.cuid}) "
            "CREATE (p)-[:JOINT {joint_id:j.jid, model_id:$mid, type:j.type, axis:j.axis}]->(c)",
            mid=mid, joints=joints,
        )

    def get(self, model_id: str) -> Robot | None:
        rows = self.run("MATCH (m:Model {id:$id}) RETURN m.model_json AS json", id=model_id)
        if not rows or not rows[0].get("json"):
            return None
        return Robot.model_validate_json(rows[0]["json"])

    def list(self) -> list[str]:
        return [r["id"] for r in self.run("MATCH (m:Model) RETURN m.id AS id ORDER BY id")]

    def delete(self, model_id: str) -> bool:
        rows = self.run("MATCH (m:Model {id:$id}) RETURN count(m) AS c", id=model_id)
        existed = bool(rows and rows[0].get("c"))
        if existed:
            self.run(
                "MATCH (m:Model {id:$id}) "
                "OPTIONAL MATCH (m)-[:HAS_LINK|HAD_FAULT]->(n) "
                "DETACH DELETE m, n",
                id=model_id,
            )
        return existed

    def clear(self) -> None:
        self.run("MATCH (n) WHERE n:Model OR n:Link OR n:FaultEvent DETACH DELETE n")

    # ------------------------------------------------------------------
    # analytics
    # ------------------------------------------------------------------

    def all_embeddings(self) -> dict[str, np.ndarray]:
        rows = self.run("MATCH (m:Model) RETURN m.id AS id, m.embedding AS emb")
        return {r["id"]: np.asarray(r["emb"], dtype=float) for r in rows if r.get("emb")}

    def models_with_joint_type(self, joint_type: str) -> list[str]:
        rows = self.run(
            "MATCH (:Link)-[j:JOINT {type:$t}]->(:Link) "
            "RETURN DISTINCT j.model_id AS id ORDER BY id",
            t=joint_type,
        )
        return [r["id"] for r in rows]

    def joint_chain_motif(self, type_a: str, type_b: str) -> list[dict]:
        return self.run(
            "MATCH (:Link)-[a:JOINT {type:$ta}]->(:Link)-[b:JOINT {type:$tb}]->(:Link) "
            "WHERE a.model_id = b.model_id "
            "RETURN a.model_id AS id, a.joint_id AS first, b.joint_id AS second ORDER BY id",
            ta=type_a, tb=type_b,
        )

    def subgraph_around(self, model_id: str, link_id: str, hops: int = 2) -> list[dict]:
        hops = max(1, min(int(hops), 5))
        return self.run(
            f"MATCH (l:Link {{uid:$uid}}) MATCH (l)-[:JOINT*1..{hops}]-(n:Link) "
            "WHERE n.uid <> l.uid "
            "RETURN DISTINCT n.id AS link, n.mass AS mass ORDER BY link",
            uid=f"{model_id}:{link_id}",
        )

    # ------------------------------------------------------------------
    # fault-history feedback loop
    # ------------------------------------------------------------------

    def write_fault_event(self, model_id: str, fault: dict) -> None:
        self.run(
            "MATCH (m:Model {id:$mid}) "
            "CREATE (m)-[:HAD_FAULT]->(:FaultEvent {model_id:$mid, type:$type, "
            "target:$target, severity:$sev, ts:$ts, note:$note})",
            mid=model_id, type=fault.get("type", ""), target=fault.get("target", ""),
            sev=float(fault.get("severity", 0.0)), ts=fault.get("ts", ""), note=fault.get("note", ""),
        )

    def get_fault_events(self, model_id: str) -> list[dict]:
        return self.run(
            "MATCH (:Model {id:$id})-[:HAD_FAULT]->(f:FaultEvent) "
            "RETURN f.type AS type, f.target AS target, f.severity AS severity, "
            "f.ts AS ts, f.note AS note ORDER BY f.ts",
            id=model_id,
        )


class Neo4jStore:
    """:class:`GraphBackend` wrapper that fails loud if the DB is unreachable."""

    def __init__(self, db: GraphDB | None = None) -> None:
        self.db = db or GraphDB()
        if not self.db.available:
            raise RuntimeError(
                "Neo4jStore selected (NEO4J_BOLT_URL set) but the graph DB is "
                "unreachable. Check the server / driver, or unset the env var."
            )
        self.db.ensure_schema()

    @property
    def available(self) -> bool:
        return self.db.available

    # storage
    def put(self, robot: Robot) -> None:
        self.db.put(robot)

    def get(self, robot_id: str) -> Robot | None:
        return self.db.get(robot_id)

    def list(self) -> list[str]:
        return self.db.list()

    def delete(self, robot_id: str) -> bool:
        return self.db.delete(robot_id)

    def clear(self) -> None:
        self.db.clear()

    # analytics
    def all_embeddings(self) -> dict[str, np.ndarray]:
        return self.db.all_embeddings()

    def models_with_joint_type(self, joint_type: str) -> list[str]:
        return self.db.models_with_joint_type(joint_type)

    def joint_chain_motif(self, type_a: str, type_b: str) -> list[dict]:
        return self.db.joint_chain_motif(type_a, type_b)

    def subgraph_around(self, model_id: str, link_id: str, hops: int = 2) -> list[dict]:
        return self.db.subgraph_around(model_id, link_id, hops)

    # faults
    def write_fault_event(self, model_id: str, fault: dict) -> None:
        self.db.write_fault_event(model_id, fault)

    def get_fault_events(self, model_id: str) -> list[dict]:
        return self.db.get_fault_events(model_id)

    # raw cypher escape hatch (Neo4j-only capability)
    def run(self, cypher: str, **params) -> list[dict]:
        return self.db.run(cypher, **params)
