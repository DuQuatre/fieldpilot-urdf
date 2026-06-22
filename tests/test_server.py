"""GraphRAG FastAPI server tests (skipped without the [server] extra).

Ported from MecAI (MIT) and re-targeted onto Robot. Exercises the default
in-memory graph backend through the HTTP surface.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fieldpilot_urdf.graphrag import server as srv  # noqa: E402

ARM_URDF = """<?xml version="1.0"?>
<robot name="arm">
  <link name="base"><inertial><mass value="1.0"/></inertial></link>
  <link name="l1"><inertial><mass value="2.0"/></inertial></link>
  <link name="tool"><inertial><mass value="3.0"/></inertial></link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="5" velocity="1"/>
  </joint>
  <joint name="j2" type="prismatic">
    <parent link="l1"/><child link="tool"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="5" velocity="1"/>
  </joint>
</robot>
"""

GANTRY_JSON = {
    "name": "gantry",
    "links": [{"name": f"l{i}", "inertial": {"mass": 5.0}} for i in range(4)],
    "joints": [
        {"name": "x", "type": "prismatic", "parent": "l0", "child": "l1",
         "axis": [1, 0, 0], "limit": {"lower": -1, "upper": 1, "effort": 5, "velocity": 1}},
        {"name": "y", "type": "prismatic", "parent": "l1", "child": "l2",
         "axis": [0, 1, 0], "limit": {"lower": -1, "upper": 1, "effort": 5, "velocity": 1}},
        {"name": "z", "type": "prismatic", "parent": "l2", "child": "l3",
         "axis": [0, 0, 1], "limit": {"lower": -1, "upper": 1, "effort": 5, "velocity": 1}},
    ],
}


@pytest.fixture()
def client():
    srv.store.clear()
    return TestClient(srv.app)


def _load_arm(client):
    return client.post("/model/load", json={"format": "urdf", "content": ARM_URDF})


def test_root_and_health(client):
    assert client.get("/").json()["name"].startswith("fieldpilot-urdf")
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["graph"] is True       # in-memory backend is always available


def test_load_and_summary(client):
    r = _load_arm(client)
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "arm" and body["name"] == "arm"
    assert body["links"] == 3 and body["joints"] == 2
    assert body["dof"] == 2 and body["root"] == "base" and body["is_tree"] is True
    assert client.get("/models").json() == ["arm"]
    assert client.get("/model/arm/summary").json()["graph_edges"] == 2


def test_create_from_json_and_get(client):
    r = client.post("/model", json=GANTRY_JSON)
    assert r.status_code == 200
    got = client.get("/model/gantry")
    assert got.status_code == 200 and got.json()["name"] == "gantry"


def test_get_unknown_404(client):
    assert client.get("/model/nope").status_code == 404
    assert client.delete("/model/nope").status_code == 404


def test_bad_format_400(client):
    r = client.post("/model/load", json={"format": "toml", "content": "x"})
    assert r.status_code == 400


def test_graph_and_chain(client):
    _load_arm(client)
    g = client.get("/model/arm/graph").json()
    assert {n["id"] for n in g["nodes"]} == {"base", "l1", "tool"}
    assert len(g["edges"]) == 2
    chain = client.get("/model/arm/chain", params={"target": "tool"}).json()
    assert chain["root"] == "base" and chain["chain"] == ["base", "l1", "tool"]
    assert client.get("/model/arm/chain", params={"target": "ghost"}).status_code == 404


def test_propagate_and_root_cause(client):
    _load_arm(client)
    p = client.post("/model/arm/propagate", json={"faulty_id": "j1"}).json()
    assert p["affected_links"] == ["l1", "tool"]
    assert 0.0 < p["criticality"] <= 1.0
    assert client.post("/model/arm/propagate", json={"faulty_id": "nope"}).status_code == 404
    rc = client.post("/model/arm/root-cause", json={"observed_links": ["tool"]}).json()
    assert rc and rc[0]["target"] in {"j1", "j2"}


def test_convert_urdf_to_json(client):
    r = client.post("/convert", json={"input_format": "urdf", "output_format": "json",
                                      "content": ARM_URDF})
    assert r.status_code == 200
    assert '"name": "arm"' in r.json()["content"]


def test_delete(client):
    _load_arm(client)
    assert client.delete("/model/arm").json() == {"deleted": "arm"}
    assert client.get("/models").json() == []


# --- GraphRAG endpoints ----------------------------------------------------

def test_graph_similar(client):
    _load_arm(client)
    client.post("/model/load", json={"format": "urdf",
                                     "content": ARM_URDF.replace('name="arm"', 'name="arm2"')})
    client.post("/model", json=GANTRY_JSON)
    hits = client.get("/graph/similar/arm", params={"k": 5}).json()
    ids = [h["id"] for h in hits]
    assert "arm" not in ids and ids[0] == "arm2"
    assert client.get("/graph/similar/ghost").status_code == 404


def test_graph_similar_adhoc(client):
    _load_arm(client)
    client.post("/model", json=GANTRY_JSON)
    hits = client.post("/graph/similar", json=GANTRY_JSON, params={"k": 1}).json()
    assert hits[0]["id"] == "gantry"


def test_graph_models_with_joint(client):
    _load_arm(client)
    client.post("/model", json=GANTRY_JSON)
    assert client.get("/graph/models-with-joint/revolute").json() == ["arm"]
    assert set(client.get("/graph/models-with-joint/prismatic").json()) == {"arm", "gantry"}


def test_graph_subgraph(client):
    _load_arm(client)
    out = client.get("/graph/model/arm/subgraph", params={"link": "l1", "hops": 1}).json()
    assert sorted(r["link"] for r in out) == ["base", "tool"]


def test_graph_fault_events(client):
    _load_arm(client)
    client.post("/graph/model/arm/fault-event",
                json={"type": "jam", "target": "j1", "severity": 0.8, "ts": "2026-03"})
    faults = client.get("/graph/model/arm/faults").json()
    assert faults[0]["type"] == "jam" and faults[0]["severity"] == 0.8


def test_graph_query_unsupported_on_memory(client):
    # the in-memory backend has no Cypher engine → 501
    r = client.post("/graph/query", json={"cypher": "MATCH (n) RETURN n"})
    assert r.status_code == 501
