"""URDF (Unified Robot Description Format) Pydantic models.

Pure Pydantic models with no I/O coupling. XML serialization lives in `loader.py`.
"""
from __future__ import annotations

import uuid
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Vec3 = tuple[float, float, float]
JointType = Literal[
    "revolute", "continuous", "prismatic", "fixed", "floating", "planar"
]


class URDFBaseModel(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), exclude=True)


# --- shared sub-elements ---------------------------------------------------

class Origin(URDFBaseModel):
    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)


class Inertia(URDFBaseModel):
    ixx: float = 0.0
    ixy: float = 0.0
    ixz: float = 0.0
    iyy: float = 0.0
    iyz: float = 0.0
    izz: float = 0.0


class Inertial(URDFBaseModel):
    origin: Optional[Origin] = None
    mass: float = 0.0
    inertia: Inertia = Field(default_factory=Inertia)


# --- geometry (tagged union) ----------------------------------------------

class Box(URDFBaseModel):
    kind: Literal["box"] = "box"
    size: Vec3


class Cylinder(URDFBaseModel):
    kind: Literal["cylinder"] = "cylinder"
    radius: float
    length: float


class Sphere(URDFBaseModel):
    kind: Literal["sphere"] = "sphere"
    radius: float


class Mesh(URDFBaseModel):
    kind: Literal["mesh"] = "mesh"
    filename: str
    scale: Vec3 = (1.0, 1.0, 1.0)


Geometry = Box | Cylinder | Sphere | Mesh


class Visual(URDFBaseModel):
    name: Optional[str] = None
    origin: Optional[Origin] = None
    geometry: Geometry = Field(discriminator="kind")
    material_name: Optional[str] = None


class Collision(URDFBaseModel):
    name: Optional[str] = None
    origin: Optional[Origin] = None
    geometry: Geometry = Field(discriminator="kind")


# --- joint -----------------------------------------------------------------

class JointLimit(URDFBaseModel):
    lower: float = 0.0
    upper: float = 0.0
    effort: float
    velocity: float


class Joint(URDFBaseModel):
    name: str
    type: JointType
    parent: str  # link name
    child: str   # link name
    origin: Optional[Origin] = None
    axis: Vec3 = (1.0, 0.0, 0.0)
    limit: Optional[JointLimit] = None

    @model_validator(mode="after")
    def _limit_required_for_bounded_joints(self) -> "Joint":
        if self.type in {"revolute", "prismatic"} and self.limit is None:
            raise ValueError(
                f"joint '{self.name}': <limit> required for type='{self.type}'"
            )
        return self


# --- link ------------------------------------------------------------------

class Link(URDFBaseModel):
    name: str
    inertial: Optional[Inertial] = None
    visuals: list[Visual] = Field(default_factory=list)
    collisions: list[Collision] = Field(default_factory=list)


# --- loop closures ---------------------------------------------------------
# Standard URDF is a strict tree. A closed-loop mechanism is represented as its
# spanning-tree URDF (the usual links/joints) plus one or more LoopClosures that
# pin two link-fixed frames together. Loops are an overlay: build_graph / FK /
# is_tree keep using only `joints`, so every tree invariant and XML round-trip
# is preserved. Loops are built programmatically (not parsed from standard URDF).

class FrameRef(URDFBaseModel):
    """A frame rigidly fixed to a tree link — the link frame, optionally offset."""
    link: str
    origin: Optional[Origin] = None   # pose of this frame in the link frame


class LoopClosure(URDFBaseModel):
    """A holonomic constraint reconnecting two tree links into a closed loop.

    `a` and `b` are two link-fixed frames required to coincide (a loop has no
    kinematic parent/child). `kind` sets how many DOF the closure removes:
    ``point`` = 3 (positions coincide), ``fixed`` = 6 (full pose).
    """
    name: str
    a: FrameRef
    b: FrameRef
    kind: Literal["point", "fixed"] = "point"


# --- root ------------------------------------------------------------------

class Robot(URDFBaseModel):
    name: str
    links: list[Link] = Field(default_factory=list)
    joints: list[Joint] = Field(default_factory=list)
    loops: list[LoopClosure] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_joint_references(self) -> "Robot":
        link_names = {l.name for l in self.links}
        for j in self.joints:
            for role, ref in (("parent", j.parent), ("child", j.child)):
                if ref not in link_names:
                    raise ValueError(
                        f"joint '{j.name}': {role} link '{ref}' not in <robot>"
                    )
        for lp in self.loops:
            for side, fr in (("a", lp.a), ("b", lp.b)):
                if fr.link not in link_names:
                    raise ValueError(
                        f"loop '{lp.name}': {side} link '{fr.link}' not in <robot>"
                    )
            if lp.a.link == lp.b.link:
                raise ValueError(
                    f"loop '{lp.name}': both sides reference the same link '{lp.a.link}'"
                )
        return self

    def link(self, name: str) -> Link:
        for l in self.links:
            if l.name == name:
                return l
        raise KeyError(name)

    def joint(self, name: str) -> Joint:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(name)
