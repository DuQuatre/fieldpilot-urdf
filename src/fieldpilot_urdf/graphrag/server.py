"""FastAPI app for the fieldpilot-urdf GraphRAG / robot-graph server.

Ported from MecAI's ``mecai-server`` (MIT) and re-targeted onto the URDF
``Robot``. Gated behind the ``[server]`` extra (``fastapi`` + ``uvicorn``); it is
intentionally *not* imported from the package top-level, so a plain
``pip install fieldpilot-urdf`` stays light. Endpoints fall into four groups:

* **Robot store** — load / list / fetch / delete typed :class:`Robot` objects.
* **Graph** — serialized NetworkX topology + kinematic chains.
* **Diagnostics** — downstream propagation + root-cause localisation (reusing
  :mod:`fieldpilot_urdf.fault_propagation`).
* **GraphRAG** — structural similarity, motif / neighbourhood queries, and the
  fault-event feedback loop, over the configured graph backend.

Plus a stateless ``/convert`` endpoint (URDF ⇄ JSON ⇄ YAML). The store backend is
selected by :func:`~fieldpilot_urdf.graphrag.store.get_store` (in-memory by
default; file-persistent via ``FIELDPILOT_URDF_STORE_DIR``; Neo4j via
``NEO4J_BOLT_URL``). Every backend is also a graph backend, so GraphRAG works in
the default in-memory configuration — no 503.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .. import __version__
from ..fault_propagation import affected_links, criticality, rank_root_causes
from ..graph import build_graph, chain as graph_chain, is_tree, root_links
from ..loader import from_xml, to_xml
from ..models import Robot
from .rag import CypherUnsupported, CypherWriteError, GraphRAG
from .schemas import (
    ChainResult,
    ConvertRequest,
    ConvertResult,
    GraphEdge,
    GraphNode,
    GraphOut,
    LoadRequest,
    ModelSummary,
    PropagateRequest,
    PropagateResult,
    RootCauseRequest,
    ServerInfo,
)
from .store import get_store

app = FastAPI(title="fieldpilot-urdf GraphRAG Server", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robot store: in-memory graph backend by default, file-persistent
# (FIELDPILOT_URDF_STORE_DIR) or Neo4j (NEO4J_BOLT_URL) per get_store()'s
# precedence. Every backend is also a GraphBackend, so GraphRAG runs over the
# store directly — no separate copy, no 503 in the default configuration.
store = get_store()
graph = GraphRAG(store)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _parse(fmt: str, content: str) -> Robot:
    """Parse a serialized robot. Raises HTTP 400 on bad format/content."""
    fmt = fmt.lower()
    try:
        if fmt == "urdf":
            return from_xml(content)
        if fmt == "json":
            return Robot.model_validate_json(content)
        if fmt in ("yaml", "yml"):
            import yaml

            return Robot.model_validate(yaml.safe_load(content))
        raise HTTPException(status_code=400, detail=f"unsupported format '{fmt}'")
    except HTTPException:
        raise
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"failed to parse {fmt}: {exc}") from exc
    except Exception as exc:  # e.g. yaml.YAMLError, XML parse errors
        raise HTTPException(status_code=400, detail=f"failed to parse {fmt}: {exc}") from exc


def _serialize(fmt: str, robot: Robot) -> str:
    """Serialize a robot to the requested format. Raises HTTP 400 on bad format."""
    fmt = fmt.lower()
    if fmt == "urdf":
        return to_xml(robot)
    if fmt == "json":
        return robot.model_dump_json(indent=2)
    if fmt in ("yaml", "yml"):
        import yaml

        return yaml.safe_dump(robot.model_dump(), sort_keys=False, allow_unicode=True)
    raise HTTPException(status_code=400, detail=f"unsupported format '{fmt}'")


def _root(g) -> str | None:
    roots = root_links(g)
    return roots[0] if len(roots) == 1 else None


def _summary(robot: Robot) -> ModelSummary:
    from ..embedding import robot_dof

    g = build_graph(robot)
    return ModelSummary(
        id=robot.name,
        name=robot.name,
        links=len(robot.links),
        joints=len(robot.joints),
        dof=robot_dof(robot),
        root=_root(g),
        is_tree=bool(robot.links) and is_tree(g),
        graph_edges=g.number_of_edges(),
    )


def _require(robot_id: str) -> Robot:
    robot = store.get(robot_id)
    if robot is None:
        raise HTTPException(status_code=404, detail=f"unknown robot '{robot_id}'")
    return robot


def _store_put(robot: Robot) -> None:
    """Persist a robot, mapping store-level id rejections to HTTP 400."""
    try:
        store.put(robot)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------


@app.get("/", response_model=ServerInfo)
def root() -> ServerInfo:
    return ServerInfo(
        name="fieldpilot-urdf GraphRAG Server",
        version=__version__,
        description="URDF Robot store → NetworkX topology → diagnostics + GraphRAG "
        "structural retrieval.",
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "robots": len(store.list()), "graph": graph.available}


# --------------------------------------------------------------------------
# robot store
# --------------------------------------------------------------------------


@app.post("/model", response_model=ModelSummary)
def create_model(robot: Robot) -> ModelSummary:
    """Store a fully-specified Robot (validated on the way in)."""
    _store_put(robot)
    return _summary(robot)


@app.post("/model/load", response_model=ModelSummary)
def load_model(req: LoadRequest) -> ModelSummary:
    """Parse a URDF/JSON/YAML document and store the resulting robot."""
    robot = _parse(req.format, req.content)
    _store_put(robot)
    return _summary(robot)


@app.get("/models", response_model=list[str])
def list_models() -> list[str]:
    return store.list()


@app.get("/model/{robot_id}", response_model=Robot)
def get_model(robot_id: str) -> Robot:
    return _require(robot_id)


@app.get("/model/{robot_id}/summary", response_model=ModelSummary)
def get_summary(robot_id: str) -> ModelSummary:
    return _summary(_require(robot_id))


@app.delete("/model/{robot_id}")
def delete_model(robot_id: str) -> dict:
    if not store.delete(robot_id):
        raise HTTPException(status_code=404, detail=f"unknown robot '{robot_id}'")
    return {"deleted": robot_id}


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------


@app.get("/model/{robot_id}/graph", response_model=GraphOut)
def get_graph(robot_id: str) -> GraphOut:
    robot = _require(robot_id)
    g = build_graph(robot)
    nodes = [GraphNode(id=n, attributes=dict(d)) for n, d in g.nodes(data=True)]
    edges = [GraphEdge(source=u, target=v, attributes=dict(d)) for u, v, d in g.edges(data=True)]
    return GraphOut(nodes=nodes, edges=edges)


@app.get("/model/{robot_id}/chain", response_model=ChainResult)
def get_chain(robot_id: str, target: str) -> ChainResult:
    """Ordered link names from the root to ``target``."""
    robot = _require(robot_id)
    g = build_graph(robot)
    root = _root(g)
    if root is None:
        raise HTTPException(status_code=400, detail="robot has no unique root link")
    if target not in {l.name for l in robot.links}:
        raise HTTPException(status_code=404, detail=f"unknown link '{target}'")
    links = graph_chain(g, root, target)
    if links is None:
        raise HTTPException(status_code=400, detail=f"no path from '{root}' to '{target}'")
    return ChainResult(root=root, target=target, chain=links)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


@app.post("/model/{robot_id}/propagate", response_model=PropagateResult)
def propagate(robot_id: str, req: PropagateRequest) -> PropagateResult:
    """Downstream link impact + mass-weighted criticality of a fault."""
    robot = _require(robot_id)
    try:
        links = affected_links(robot, req.faulty_id)
        crit = criticality(robot, req.faulty_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PropagateResult(
        faulty_id=req.faulty_id,
        affected_links=sorted(links),
        criticality=crit,
    )


@app.post("/model/{robot_id}/root-cause", response_model=list)
def root_cause(robot_id: str, req: RootCauseRequest):
    """Rank joints (optionally links) by how well they explain the observed links."""
    robot = _require(robot_id)
    try:
        candidates = rank_root_causes(
            robot,
            req.observed_links,
            consider_links=req.consider_links,
            top_k=req.top_k,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return candidates


# --------------------------------------------------------------------------
# convert — stateless
# --------------------------------------------------------------------------


@app.post("/convert", response_model=ConvertResult)
def convert(req: ConvertRequest) -> ConvertResult:
    """Parse ``input_format`` and re-serialize to ``output_format``."""
    robot = _parse(req.input_format, req.content)
    return ConvertResult(output_format=req.output_format, content=_serialize(req.output_format, robot))


# --------------------------------------------------------------------------
# GraphRAG
# --------------------------------------------------------------------------


def _require_graph() -> GraphRAG:
    if not graph.available:
        raise HTTPException(status_code=503, detail="graph backend unavailable")
    return graph


@app.post("/graph/ingest/{robot_id}")
def graph_ingest(robot_id: str) -> dict:
    """Ensure a stored robot is queryable in the graph (idempotent)."""
    g = _require_graph()
    g.backend.put(_require(robot_id))
    return {"ingested": robot_id}


@app.get("/graph/similar/{robot_id}", response_model=list)
def graph_similar(robot_id: str, k: int = 5):
    """Robots structurally most similar to a robot already in the graph."""
    g = _require_graph()
    try:
        return g.similar_to_id(robot_id, top_k=k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"robot '{exc.args[0]}' not in graph") from exc


@app.post("/graph/similar", response_model=list)
def graph_similar_adhoc(robot: Robot, k: int = 5):
    """Rank graph robots by similarity to an ad-hoc (possibly unstored) robot."""
    g = _require_graph()
    return g.similar_to_robot(robot, top_k=k)


@app.get("/graph/models-with-joint/{joint_type}", response_model=list)
def graph_models_with_joint(joint_type: str):
    """Names of robots containing at least one joint of the given type."""
    g = _require_graph()
    return g.models_with_joint_type(joint_type)


@app.get("/graph/model/{robot_id}/subgraph", response_model=list)
def graph_subgraph(robot_id: str, link: str, hops: int = 2):
    """Links within ``hops`` of ``link`` in the given robot."""
    g = _require_graph()
    return g.subgraph_around(robot_id, link, hops=hops)


@app.post("/graph/query", response_model=list)
def graph_query(body: dict):
    """Run a read-only Cypher statement: ``{"cypher": "...", "params": {...}}``."""
    g = _require_graph()
    statement = body.get("cypher", "")
    params = body.get("params", {}) or {}
    try:
        return g.cypher(statement, **params)
    except CypherWriteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CypherUnsupported as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@app.post("/graph/model/{robot_id}/fault-event")
def graph_add_fault(robot_id: str, fault: dict) -> dict:
    """Append a FaultEvent to a robot's history (the diagnostic feedback loop)."""
    g = _require_graph()
    g.backend.write_fault_event(robot_id, fault)
    return {"recorded": robot_id}


@app.get("/graph/model/{robot_id}/faults", response_model=list)
def graph_faults(robot_id: str):
    """A robot's recorded fault history, oldest first."""
    g = _require_graph()
    return g.backend.get_fault_events(robot_id)
