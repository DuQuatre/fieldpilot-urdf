"""Backend parity: the same contract tests run against every GraphBackend.

Ported from MecAI (MIT) and re-targeted onto Robot.

* ``memory`` — :class:`MemoryGraphBackend`, runs in the default suite (real
  NetworkX query logic, no Docker).
* ``neo4j`` — a live Neo4j (real Cypher round-trip), gated behind the
  ``integration`` marker. Source: ``NEO4J_TEST_BOLT_URL`` if set, else a
  ``testcontainers`` throwaway (``neo4j:5.26``), else skipped.

Holding both backends to one set of assertions is what guarantees the in-memory
default and the Neo4j backend behave identically.

Run everything (needs Docker)::

    pytest tests/test_graph_backend_parity.py -m 'integration or not integration'
"""
from __future__ import annotations

import os

import pytest

from fieldpilot_urdf.embedding import EMBEDDING_DIM
from fieldpilot_urdf.graphrag.backend import MemoryGraphBackend
from fieldpilot_urdf.graphrag.rag import GraphRAG
from fieldpilot_urdf.models import Inertial, Joint, JointLimit, Link, Origin, Robot


def _lim():
    return JointLimit(lower=-1, upper=1, effort=5, velocity=1)


def _arm(name="arm"):
    return Robot(
        name=name,
        links=[
            Link(name="base", inertial=Inertial(mass=1.0)),
            Link(name="link1", inertial=Inertial(mass=2.0)),
            Link(name="link2", inertial=Inertial(mass=3.0)),
        ],
        joints=[
            Joint(name="shoulder", type="revolute", parent="base", child="link1",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 0, 1), limit=_lim()),
            Joint(name="elbow", type="revolute", parent="link1", child="link2",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 0, 1), limit=_lim()),
        ],
    )


def _gantry(name="gantry"):
    return Robot(
        name=name,
        links=[Link(name=f"l{i}", inertial=Inertial(mass=5.0)) for i in range(3)],
        joints=[
            Joint(name="x", type="prismatic", parent="l0", child="l1", axis=(1, 0, 0), limit=_lim()),
            Joint(name="z", type="prismatic", parent="l1", child="l2", axis=(0, 0, 1), limit=_lim()),
        ],
    )


# --------------------------------------------------------------------------
# Neo4j provisioning (env var → testcontainers → skip)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _neo4j_conn():
    pytest.importorskip("neo4j")
    url = os.environ.get("NEO4J_TEST_BOLT_URL")
    if url:
        user = os.environ.get("NEO4J_TEST_USER", "neo4j")
        pwd = os.environ.get("NEO4J_TEST_PASSWORD", "")
        from neo4j import GraphDatabase

        try:
            drv = GraphDatabase.driver(url, auth=(user, pwd))
            drv.verify_connectivity()
            drv.close()
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"NEO4J_TEST_BOLT_URL set but unreachable: {exc}")
        yield url, user, pwd
        return

    neo4j_tc = pytest.importorskip(
        "testcontainers.neo4j",
        reason="set NEO4J_TEST_BOLT_URL or install '.[integration]' (testcontainers + Docker)",
    )
    pwd = "testpassword"
    try:
        with neo4j_tc.Neo4jContainer("neo4j:5.26", password=pwd) as neo:
            yield neo.get_connection_url(), "neo4j", pwd
    except Exception as exc:  # pragma: no cover - Docker missing/unhealthy
        pytest.skip(f"could not start Neo4j testcontainer: {exc}")


# --------------------------------------------------------------------------
# backend fixture, parametrized over the two implementations
# --------------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param("memory"),
        pytest.param("neo4j", marks=pytest.mark.integration),
    ]
)
def backend(request):
    if request.param == "memory":
        b = MemoryGraphBackend()
        b.clear()
        yield b
        return

    url, user, pwd = request.getfixturevalue("_neo4j_conn")
    mp = request.getfixturevalue("monkeypatch")
    mp.setenv("NEO4J_BOLT_URL", url)
    mp.setenv("NEO4J_USER", user)
    mp.setenv("NEO4J_PASSWORD", pwd)
    from fieldpilot_urdf.graphrag.neo4j_backend import Neo4jStore

    store = Neo4jStore()
    store.clear()
    yield store
    store.clear()
    store.db.close()


# --------------------------------------------------------------------------
# contract tests — identical assertions for both backends
# --------------------------------------------------------------------------


def test_put_get_roundtrip(backend):
    arm = _arm()
    backend.put(arm)
    got = backend.get("arm")
    assert got is not None
    assert got.model_dump() == arm.model_dump()


def test_get_missing_returns_none(backend):
    assert backend.get("ghost") is None


def test_list_and_delete(backend):
    backend.put(_arm())
    backend.put(_gantry())
    assert set(backend.list()) == {"arm", "gantry"}
    assert backend.delete("gantry") is True
    assert backend.delete("gantry") is False
    assert backend.list() == ["arm"]


def test_rewrite_replaces_not_duplicates(backend):
    backend.put(_arm())
    backend.put(_arm())
    assert backend.list() == ["arm"]
    near = backend.subgraph_around("arm", "base", hops=5)
    assert sorted(r["link"] for r in near) == ["link1", "link2"]


def test_all_embeddings(backend):
    backend.put(_arm())
    backend.put(_gantry())
    embs = backend.all_embeddings()
    assert set(embs) == {"arm", "gantry"}
    assert len(embs["arm"]) == EMBEDDING_DIM


def test_models_with_joint_type(backend):
    backend.put(_arm())
    backend.put(_gantry())
    assert "arm" in backend.models_with_joint_type("revolute")
    assert "arm" not in backend.models_with_joint_type("prismatic")
    assert "gantry" in backend.models_with_joint_type("prismatic")


def test_subgraph_around(backend):
    backend.put(_arm())
    near = backend.subgraph_around("arm", "base", hops=2)
    assert {r["link"] for r in near} == {"link1", "link2"}


def test_fault_event_feedback_loop(backend):
    backend.put(_arm())
    backend.write_fault_event(
        "arm", {"type": "joint_jam", "target": "elbow", "severity": 0.8, "ts": "2026-06-22"}
    )
    faults = backend.get_fault_events("arm")
    assert len(faults) == 1
    assert faults[0]["target"] == "elbow"
    assert faults[0]["severity"] == pytest.approx(0.8)


def test_graphrag_similarity_end_to_end(backend):
    backend.put(_arm())
    backend.put(_arm("arm_copy"))
    backend.put(_gantry())
    hits = GraphRAG(backend).similar_to_id("arm", top_k=5)
    ids = [h["id"] for h in hits]
    assert "arm" not in ids
    assert ids[0] == "arm_copy"
    assert "gantry" in ids
