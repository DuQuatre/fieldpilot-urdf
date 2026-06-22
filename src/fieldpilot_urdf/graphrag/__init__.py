"""GraphRAG layer for fieldpilot-urdf — fleet-level structural retrieval.

Ported from the MecAI project (MIT) and re-targeted onto the URDF ``Robot``. The
per-robot diagnostics chain (:mod:`fieldpilot_urdf.diagnose_core`,
:mod:`fieldpilot_urdf.case_base`) answers "what's wrong with *this* robot"; this
layer answers "which other robots in the fleet are structurally like it" — so a
new robot's diagnosis can borrow the cases and fault history of its nearest
structural neighbours.

The retrieval engine (:class:`GraphRAG`) runs over any
:class:`~fieldpilot_urdf.graphrag.backend.GraphBackend`. The default
:class:`MemoryGraphBackend` is pure NetworkX + numpy (core deps), so GraphRAG
works out of the box with no database. A durable Neo4j backend
(:class:`~fieldpilot_urdf.graphrag.neo4j_backend.Neo4jStore`, ``[graphrag]`` extra)
and a FastAPI server (:mod:`fieldpilot_urdf.graphrag.server`, ``[server]`` extra)
are optional. The server module is intentionally not imported here, so the import
surface stays light.
"""
from __future__ import annotations

from .backend import GraphBackend, MemoryGraphBackend
from .rag import CypherUnsupported, CypherWriteError, GraphRAG, is_read_only
from .store import FileStore, MemoryStore, RobotStore, get_store, model_id

__all__ = [
    "GraphBackend", "MemoryGraphBackend",
    "GraphRAG", "is_read_only", "CypherWriteError", "CypherUnsupported",
    "RobotStore", "MemoryStore", "FileStore", "get_store", "model_id",
]
