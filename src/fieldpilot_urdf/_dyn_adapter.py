"""Re-shape a URDF :class:`~fieldpilot_urdf.models.Robot` into the duck-typed
surface the ported symbolic-dynamics core reads.

:class:`fieldpilot_urdf.dynamics.SymbolicDynamics` is graduated (near) verbatim
from the MecAI project (MIT-licensed). MecAI's engine consumes an abstract,
id-keyed ``MechanicalSystem`` with flattened inertials (``link.com``,
``link.mass``) and ``list[float]`` vectors. URDF's ``Robot`` is name-keyed,
nests inertials inside ``Inertial``/``Origin``, and uses ``Vec3`` tuples.

This module is the *only* place that knows about that mismatch: it produces
lightweight shims whose attribute names match exactly what the dynamics core
expects, so the proven SymPy math ports without edits to its body.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .graph import build_graph, is_tree, root_links
from .models import Inertia, Robot


class UnsupportedSystemError(ValueError):
    """The robot violates a v1 dynamics precondition (non-tree, multi-root,
    unsupported joint type, or a non-axis-aligned inertial frame)."""


class JointType(str, Enum):
    """Mirror of MecAI's joint-type enum so the ported core's
    ``j.type == JointType.REVOLUTE`` comparisons keep working."""

    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"
    PRISMATIC = "prismatic"
    FIXED = "fixed"
    FLOATING = "floating"
    PLANAR = "planar"


_SUPPORTED_JOINT_TYPES = {
    JointType.REVOLUTE, JointType.CONTINUOUS, JointType.PRISMATIC, JointType.FIXED,
}


@dataclass
class _LinkShim:
    id: str
    mass: float
    inertia: Inertia | None
    com: list[float]


@dataclass
class _JointShim:
    id: str
    type: JointType
    parent: str
    child: str
    origin_xyz: list[float]
    origin_rpy: list[float]
    axis: list[float]


@dataclass
class _SystemShim:
    root: str
    links: dict[str, _LinkShim] = field(default_factory=dict)
    joints: dict[str, _JointShim] = field(default_factory=dict)


def robot_to_system(robot: Robot) -> _SystemShim:
    """Validate and map a URDF ``Robot`` to the dynamics-core shim.

    Raises :class:`UnsupportedSystemError` for non-tree / multi-root robots,
    joint types beyond revolute/continuous/prismatic/fixed, or a non-zero
    ``<inertial><origin rpy=...>`` (which would require rotating the inertia
    tensor ``R·I·Rᵀ`` into the body frame — deferred to a future version).
    """
    g = build_graph(robot)
    if not is_tree(g):
        raise UnsupportedSystemError(
            "dynamics v1 supports only tree-shaped robots; closed loops need "
            "Lagrange multipliers (deferred)."
        )
    roots = root_links(g)
    if len(roots) != 1:
        raise UnsupportedSystemError(
            f"expected exactly one root link, found {len(roots)}: {roots}"
        )

    system = _SystemShim(root=roots[0])

    for link in robot.links:
        inr = link.inertial
        com = [0.0, 0.0, 0.0]
        if inr is not None and inr.origin is not None:
            if any(abs(a) > 1e-12 for a in inr.origin.rpy):
                raise UnsupportedSystemError(
                    f"link '{link.name}': non-zero <inertial> origin rpy "
                    f"{inr.origin.rpy} needs an inertia-tensor rotation R·I·Rᵀ, "
                    "not handled in dynamics v1."
                )
            com = list(inr.origin.xyz)
        system.links[link.name] = _LinkShim(
            id=link.name,
            mass=inr.mass if inr is not None else 0.0,
            inertia=inr.inertia if inr is not None else None,
            com=com,
        )

    for j in robot.joints:
        jtype = JointType(j.type)
        if jtype not in _SUPPORTED_JOINT_TYPES:
            raise UnsupportedSystemError(
                f"joint '{j.name}' has type {j.type!r}; dynamics v1 supports "
                f"{sorted(t.value for t in _SUPPORTED_JOINT_TYPES)}."
            )
        xyz = list(j.origin.xyz) if j.origin is not None else [0.0, 0.0, 0.0]
        rpy = list(j.origin.rpy) if j.origin is not None else [0.0, 0.0, 0.0]
        system.joints[j.name] = _JointShim(
            id=j.name,
            type=jtype,
            parent=j.parent,
            child=j.child,
            origin_xyz=xyz,
            origin_rpy=rpy,
            axis=list(j.axis),
        )

    return system
