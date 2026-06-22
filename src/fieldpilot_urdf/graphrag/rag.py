"""GraphRAG retrieval — backend-agnostic.

Ported from MecAI (MIT) and re-targeted onto the URDF ``Robot``. Runs over any
:class:`~fieldpilot_urdf.graphrag.backend.GraphBackend`
(:class:`~fieldpilot_urdf.graphrag.backend.MemoryGraphBackend` by default, or a
Neo4j-backed store), so the retrieval logic is written once and tested for real
against the in-memory backend in the default suite.

Two retrieval modes:

* **structural similarity** — embed a query robot (or an ad-hoc one) and rank
  stored robots by cosine similarity of their structural embeddings. This is the
  fleet-level companion to the per-robot diagnostics chain: when a robot starts
  misbehaving, the most structurally similar robots in the fleet are the ones
  whose past cases (:mod:`fieldpilot_urdf.case_base`) and fault history transfer.
* **pattern match** — joint-type / motif / neighbourhood queries delegated to the
  backend, plus a guarded :meth:`GraphRAG.cypher` escape hatch for backends that
  support raw Cypher (Neo4j only).
"""
from __future__ import annotations

import re

import numpy as np

from ..embedding import rank_by_similarity, robot_embedding
from ..models import Robot

# Reject any mutating operation — RAG queries must be read-only.
_WRITE = re.compile(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|DETACH)\b", re.IGNORECASE)


def is_read_only(cypher: str) -> bool:
    """True iff the statement contains no mutating Cypher keyword."""
    return _WRITE.search(cypher) is None


class CypherWriteError(ValueError):
    """Raised when a write statement is submitted to a read-only entry point."""


class CypherUnsupported(NotImplementedError):
    """Raised when raw Cypher is requested from a backend that has no Cypher engine."""


class GraphRAG:
    """Retrieval helpers over any :class:`GraphBackend`."""

    def __init__(self, backend) -> None:
        self.backend = backend

    @property
    def available(self) -> bool:
        return bool(getattr(self.backend, "available", False))

    # ------------------------------------------------------------------
    # structural similarity
    # ------------------------------------------------------------------

    def similar_to_id(self, model_id: str, *, top_k: int = 5) -> list[dict]:
        """Robots structurally most similar to a stored robot (excludes itself)."""
        robot = self.backend.get(model_id)
        if robot is None:
            raise KeyError(model_id)
        return self._rank(robot_embedding(robot), exclude=model_id, top_k=top_k)

    def similar_to_robot(self, robot: Robot, *, top_k: int = 5) -> list[dict]:
        """Robots most similar to an ad-hoc (possibly unstored) robot."""
        return self._rank(robot_embedding(robot), exclude=None, top_k=top_k)

    def _rank(self, query_vec: np.ndarray, *, exclude: str | None, top_k: int) -> list[dict]:
        embeddings = {
            mid: vec for mid, vec in self.backend.all_embeddings().items() if mid != exclude
        }
        ranked = rank_by_similarity(query_vec, embeddings, top_k=top_k)
        return [{"id": mid, "similarity": score} for mid, score in ranked]

    # ------------------------------------------------------------------
    # pattern queries (delegated to the backend)
    # ------------------------------------------------------------------

    def models_with_joint_type(self, joint_type: str) -> list[str]:
        return self.backend.models_with_joint_type(joint_type)

    def joint_chain_motif(self, type_a: str, type_b: str) -> list[dict]:
        return self.backend.joint_chain_motif(type_a, type_b)

    def subgraph_around(self, model_id: str, link_id: str, hops: int = 2) -> list[dict]:
        return self.backend.subgraph_around(model_id, link_id, hops)

    def cypher(self, statement: str, **params) -> list[dict]:
        """Run a **read-only** Cypher statement (Neo4j backends only)."""
        run = getattr(self.backend, "run", None)
        if run is None:
            raise CypherUnsupported("the active graph backend has no Cypher engine")
        if not is_read_only(statement):
            raise CypherWriteError("only read-only Cypher is permitted here")
        return run(statement, **params)

    # ------------------------------------------------------------------
    # context serialization (for LLM tool_result injection)
    # ------------------------------------------------------------------

    def similarity_context(self, model_id: str, *, top_k: int = 5) -> str:
        """Plain-text context block of similar robots, ready for a prompt (French)."""
        hits = self.similar_to_id(model_id, top_k=top_k)
        if not hits:
            return f"[GraphRAG] aucun robot similaire à '{model_id}'."
        lines = [f"[GraphRAG — robots similaires à '{model_id}']"]
        for h in hits:
            lines.append(f"  {h['id']} — similarité {h['similarity']:.3f}")
        return "\n".join(lines)
