"""Request/response models for the fieldpilot-urdf GraphRAG HTTP API.

Ported from MecAI (MIT) and re-targeted onto the URDF ``Robot`` (no sensor model;
no Delta kinematics). The domain model (:class:`~fieldpilot_urdf.models.Robot`) is
reused directly as a request body; this module only adds the thin transport-layer
wrappers.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ServerInfo(BaseModel):
    """Root metadata returned by ``GET /``."""

    name: str
    version: str
    description: str


class ModelSummary(BaseModel):
    """Compact description of a stored :class:`~fieldpilot_urdf.models.Robot`."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    links: int
    joints: int
    dof: int
    root: str | None
    is_tree: bool
    graph_edges: int


class LoadRequest(BaseModel):
    """Parse a robot from a serialized string into the store."""

    model_config = ConfigDict(extra="forbid")

    format: str = Field(..., description="One of 'urdf', 'json', 'yaml'")
    content: str = Field(..., description="The serialized robot document")


class ConvertRequest(BaseModel):
    """Stateless conversion between supported serialization formats."""

    model_config = ConfigDict(extra="forbid")

    input_format: str = Field(..., description="'urdf', 'json', or 'yaml'")
    output_format: str = Field(..., description="'urdf', 'json', or 'yaml'")
    content: str


class ConvertResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_format: str
    content: str


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    attributes: dict


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    attributes: dict


class GraphOut(BaseModel):
    """Serialized NetworkX kinematic graph (links=nodes, joints=edges)."""

    model_config = ConfigDict(extra="forbid")

    directed: bool = True
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class ChainResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: str
    target: str
    chain: list[str]


class PropagateRequest(BaseModel):
    """Downstream-impact query for a faulty joint or link."""

    model_config = ConfigDict(extra="forbid")

    faulty_id: str = Field(..., description="A joint name or a link name")


class PropagateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faulty_id: str
    affected_links: list[str]
    criticality: float


class RootCauseRequest(BaseModel):
    """Inverse query: rank joints by how well they explain the observed links."""

    model_config = ConfigDict(extra="forbid")

    observed_links: list[str]
    consider_links: bool = False
    top_k: int | None = None
