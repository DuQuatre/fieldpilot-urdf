"""GraphRAG layer tests — store, in-memory backend, and retrieval engine.

Ported from MecAI (MIT) and re-targeted onto Robot. The in-memory backend *is*
the GraphRAG engine, so these run on the core install (no DB).
"""
from __future__ import annotations

import numpy as np
import pytest

from fieldpilot_urdf.graphrag.backend import GraphBackend, MemoryGraphBackend
from fieldpilot_urdf.graphrag.rag import (
    CypherUnsupported, CypherWriteError, GraphRAG, is_read_only,
)
from fieldpilot_urdf.graphrag.store import FileStore, MemoryStore, get_store, model_id
from fieldpilot_urdf.models import Inertial, Joint, JointLimit, Link, Origin, Robot


def _lim():
    return JointLimit(lower=-1, upper=1, effort=5, velocity=1)


def _arm(name="arm", masses=(1.0, 2.0, 3.0)):
    mb, m1, m2 = masses
    return Robot(
        name=name,
        links=[
            Link(name="base", inertial=Inertial(mass=mb)),
            Link(name="l1", inertial=Inertial(mass=m1)),
            Link(name="tool", inertial=Inertial(mass=m2)),
        ],
        joints=[
            Joint(name="j1", type="revolute", parent="base", child="l1",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 0, 1), limit=_lim()),
            Joint(name="j2", type="prismatic", parent="l1", child="tool",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 0, 1), limit=_lim()),
        ],
    )


def _gantry(name="gantry"):
    return Robot(
        name=name,
        links=[Link(name=f"l{i}", inertial=Inertial(mass=5.0)) for i in range(4)],
        joints=[
            Joint(name="x", type="prismatic", parent="l0", child="l1", axis=(1, 0, 0), limit=_lim()),
            Joint(name="y", type="prismatic", parent="l1", child="l2", axis=(0, 1, 0), limit=_lim()),
            Joint(name="z", type="prismatic", parent="l2", child="l3", axis=(0, 0, 1), limit=_lim()),
        ],
    )


# --- store -----------------------------------------------------------------

def test_memory_store_crud():
    s = MemoryStore()
    s.put(_arm())
    assert s.list() == ["arm"]
    assert s.get("arm").name == "arm"
    assert s.get("missing") is None
    assert s.delete("arm") is True
    assert s.delete("arm") is False
    assert s.list() == []


def test_file_store_roundtrip(tmp_path):
    s = FileStore(tmp_path)
    s.put(_arm(masses=(1.5, 2.5, 3.5)))
    assert (tmp_path / "arm.json").exists()
    again = s.get("arm")
    assert again.name == "arm"
    assert again.link("base").inertial.mass == 1.5
    assert s.delete("arm") is True
    assert s.get("arm") is None


def test_file_store_rejects_unsafe_id(tmp_path):
    s = FileStore(tmp_path)
    s.put(Robot(name="ok", links=[Link(name="a")]))
    # traversal-y id never resolves to a file
    assert s.get("../etc/passwd") is None
    assert s.delete("../etc/passwd") is False


def test_model_id_is_name():
    assert model_id(_arm("foo")) == "foo"


def test_get_store_default_is_memory_graph_backend(monkeypatch):
    monkeypatch.delenv("NEO4J_BOLT_URL", raising=False)
    monkeypatch.delenv("MEMGRAPH_BOLT_URL", raising=False)
    monkeypatch.delenv("FIELDPILOT_URDF_STORE_DIR", raising=False)
    assert isinstance(get_store(), MemoryGraphBackend)


def test_get_store_file_persistence(tmp_path, monkeypatch):
    monkeypatch.delenv("NEO4J_BOLT_URL", raising=False)
    monkeypatch.setenv("FIELDPILOT_URDF_STORE_DIR", str(tmp_path))
    b1 = get_store()
    b1.put(_arm())
    # a fresh backend on the same dir reloads it
    b2 = get_store()
    assert "arm" in b2.list()


# --- backend ---------------------------------------------------------------

def test_memory_backend_satisfies_protocol():
    assert isinstance(MemoryGraphBackend(), GraphBackend)


def test_backend_embeddings_cached_and_invalidated():
    b = MemoryGraphBackend()
    b.put(_arm())
    e1 = b.all_embeddings()["arm"]
    e2 = b.all_embeddings()["arm"]
    assert np.array_equal(e1, e2)
    b.put(_arm(masses=(9.0, 9.0, 9.0)))   # same name, new masses -> re-embedded
    assert not np.array_equal(e1, b.all_embeddings()["arm"])


def test_backend_joint_type_and_motif():
    b = MemoryGraphBackend()
    b.put(_arm())
    b.put(_gantry())
    assert b.models_with_joint_type("revolute") == ["arm"]
    assert set(b.models_with_joint_type("prismatic")) == {"arm", "gantry"}
    # arm has a revolute (j1: base->l1) immediately followed by prismatic (j2: l1->tool)
    motif = b.joint_chain_motif("revolute", "prismatic")
    assert {"id": "arm", "first": "j1", "second": "j2"} in motif


def test_backend_subgraph_around():
    b = MemoryGraphBackend()
    b.put(_arm())
    out = b.subgraph_around("arm", "l1", hops=1)
    assert sorted(r["link"] for r in out) == ["base", "tool"]
    assert b.subgraph_around("arm", "nope", hops=1) == []
    assert b.subgraph_around("missing", "l1") == []


def test_backend_fault_events_sorted():
    b = MemoryGraphBackend()
    b.put(_arm())
    b.write_fault_event("arm", {"type": "jam", "target": "j1", "ts": "2026-02"})
    b.write_fault_event("arm", {"type": "drift", "target": "j2", "ts": "2026-01"})
    evs = b.get_fault_events("arm")
    assert [e["ts"] for e in evs] == ["2026-01", "2026-02"]
    assert evs[0]["severity"] == 0.0


def test_backend_delete_clears_faults_and_embeddings():
    b = MemoryGraphBackend()
    b.put(_arm())
    b.write_fault_event("arm", {"type": "jam", "ts": "x"})
    assert b.delete("arm") is True
    assert b.get_fault_events("arm") == []
    assert b.all_embeddings() == {}


# --- GraphRAG --------------------------------------------------------------

def _rag():
    b = MemoryGraphBackend()
    b.put(_arm("arm_a"))
    b.put(_arm("arm_b", masses=(1.1, 2.1, 2.9)))
    b.put(_gantry())
    return GraphRAG(b)


def test_rag_available():
    assert _rag().available is True


def test_similar_to_id_excludes_self_and_ranks():
    rag = _rag()
    hits = rag.similar_to_id("arm_a", top_k=5)
    ids = [h["id"] for h in hits]
    assert "arm_a" not in ids               # excludes itself
    assert ids[0] == "arm_b"                 # the other arm is nearest
    assert hits[0]["similarity"] >= hits[-1]["similarity"]


def test_similar_to_id_unknown_raises():
    with pytest.raises(KeyError):
        _rag().similar_to_id("nope")


def test_similar_to_robot_adhoc():
    rag = _rag()
    hits = rag.similar_to_robot(_arm("query"), top_k=1)
    assert hits[0]["id"] in {"arm_a", "arm_b"}


def test_similarity_context_text():
    ctx = _rag().similarity_context("arm_a", top_k=2)
    assert "robots similaires à 'arm_a'" in ctx
    assert "arm_b" in ctx


def test_cypher_unsupported_on_memory_backend():
    with pytest.raises(CypherUnsupported):
        _rag().cypher("MATCH (n) RETURN n")


def test_is_read_only_guard():
    assert is_read_only("MATCH (n) RETURN n") is True
    assert is_read_only("MATCH (n) DETACH DELETE n") is False
    assert is_read_only("create (n)") is False


def test_cypher_write_rejected_before_backend(monkeypatch):
    rag = _rag()
    # give the backend a fake .run so we reach the write-guard, not CypherUnsupported
    rag.backend.run = lambda stmt, **kw: [{"ran": stmt}]
    assert rag.cypher("MATCH (n) RETURN n") == [{"ran": "MATCH (n) RETURN n"}]
    with pytest.raises(CypherWriteError):
        rag.cypher("MATCH (n) DELETE n")
