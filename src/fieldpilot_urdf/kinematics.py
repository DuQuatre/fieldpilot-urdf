"""Velocity kinematics: the geometric Jacobian and what it unlocks.

Forward kinematics says *where* a link is; the Jacobian says *how it moves*.
For a target link and a configuration ``q``, :func:`geometric_jacobian` returns
the 6×n matrix ``J`` mapping joint velocities to the link's spatial velocity
(twist) in the world frame::

    [v; w] = J(q) @ qdot          # linear rows 0:3 over angular rows 3:6

Columns follow the *movable* joints on the root→link path, root-first;
:func:`jacobian_joints` gives that ordering (and is what ``qdot`` vectors must
match). Fixed joints contribute no column. From ``J`` we read off forward
velocity (:func:`joint_velocity_to_twist`), the Yoshikawa **manipulability**
measure (:func:`manipulability`), and a **singularity** report
(:func:`singularity_report`).

Pure numpy — no scipy. Revolute / continuous / prismatic joints are handled;
``fixed`` contributes nothing and floating / planar joints are out of scope
(as everywhere else in the package). The convention matches :mod:`fieldpilot_urdf.fk`:
a joint's ``axis`` lives in the joint frame ``T_parent @ T_origin``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from pydantic import BaseModel

from .fk import forward_kinematics, origin_to_T
from .graph import build_graph, chain, joints_on_path, root_links
from .models import Robot


def jacobian_joints(robot: Robot, link: str) -> list[str]:
    """Movable (non-``fixed``) joints on the root→``link`` path, root-first.

    This is the column ordering of :func:`geometric_jacobian` and the element
    ordering expected by :func:`joint_velocity_to_twist`'s ``qdot`` vector.
    Raises ``ValueError`` if the robot is not a single-rooted tree or ``link``
    is unknown / unreachable from the root.
    """
    G = build_graph(robot)
    roots = root_links(G)
    if len(roots) != 1:
        raise ValueError(f"Jacobian requires single root link, found {len(roots)}: {roots}")
    if link not in G:
        raise ValueError(f"unknown link: {link!r}")
    path = chain(G, roots[0], link)
    if path is None:
        raise ValueError(f"link {link!r} is not reachable from root {roots[0]!r}")
    by_name = {j.name: j for j in robot.joints}
    return [jn for jn in joints_on_path(G, path) if by_name[jn].type != "fixed"]


def geometric_jacobian(
    robot: Robot,
    q: Optional[dict[str, float]] = None,
    link: str = "",
    *,
    joints: Optional[list[str]] = None,
) -> np.ndarray:
    """Geometric (basic) Jacobian of ``link`` at configuration ``q``: a 6×n array.

    Rows are stacked linear-over-angular — ``J[0:3]`` maps joint velocities to
    the linear velocity of the link's origin, ``J[3:6]`` to its angular velocity,
    both in the world frame. Columns follow ``joints`` if given, else
    :func:`jacobian_joints` (root→link order). ``n`` may be 0 (a fixed chain),
    yielding a 6×0 array.

    For a revolute/continuous joint with world axis ``z`` at world point ``p``,
    the column is ``[z × (p_e − p); z]`` (``p_e`` = link origin). For a prismatic
    joint it is ``[z; 0]``.
    """
    if not link:
        raise ValueError("link is required")
    q = q or {}
    cols = jacobian_joints(robot, link) if joints is None else list(joints)

    tf = forward_kinematics(robot, q)
    if link not in tf:
        raise ValueError(f"unknown link: {link!r}")
    p_e = tf[link][:3, 3]

    by_name = {j.name: j for j in robot.joints}
    J = np.zeros((6, len(cols)))
    for i, jn in enumerate(cols):
        j = by_name[jn]
        if j.type == "fixed":
            continue  # zero column; defensive — jacobian_joints already drops these
        # Joint frame in world: T_parent @ T_origin (the frame the axis lives in).
        T_joint = tf[j.parent] @ origin_to_T(j.origin)
        R_joint = T_joint[:3, :3]
        axis = np.asarray(j.axis, dtype=float)
        n = float(np.linalg.norm(axis))
        z = R_joint @ (axis / n) if n > 1e-12 else np.zeros(3)
        if j.type == "prismatic":
            J[0:3, i] = z
            # angular part stays zero
        else:  # revolute / continuous
            p = T_joint[:3, 3]
            J[0:3, i] = np.cross(z, p_e - p)
            J[3:6, i] = z
    return J


def joint_velocity_to_twist(
    robot: Robot,
    q: Optional[dict[str, float]],
    qdot: dict[str, float],
    link: str,
) -> np.ndarray:
    """Forward velocity: the world-frame twist ``[v; w]`` (6-vector) of ``link``.

    ``qdot`` is a ``{joint_name: velocity}`` dict; joints absent from it (or not
    on the chain) contribute zero. Equivalent to ``J @ qdot`` with ``qdot``
    ordered by :func:`jacobian_joints`.
    """
    cols = jacobian_joints(robot, link)
    J = geometric_jacobian(robot, q, link, joints=cols)
    qd = np.array([float(qdot.get(jn, 0.0)) for jn in cols])
    return J @ qd if cols else np.zeros(6)


def _selected_jacobian(
    robot: Robot,
    q: Optional[dict[str, float]],
    link: str,
    rows: Optional[Sequence[int]],
) -> np.ndarray:
    """Geometric Jacobian, optionally restricted to a subset of the 6 twist
    rows. ``rows`` indexes the stacked ``[v; w]`` vector: ``(0, 1, 2)`` is the
    translational part, ``(3, 4, 5)`` the rotational part."""
    J = geometric_jacobian(robot, q, link)
    if rows is None:
        return J
    idx = list(rows)
    if any(i < 0 or i > 5 for i in idx):
        raise ValueError(f"rows must index the 6-row twist (0..5), got {idx}")
    return J[idx, :]


def manipulability(
    robot: Robot,
    q: Optional[dict[str, float]] = None,
    link: str = "",
    *,
    rows: Optional[Sequence[int]] = None,
) -> float:
    """Yoshikawa manipulability measure of ``link`` at ``q``: the volume of the
    manipulability ellipsoid, i.e. the product of the singular values of ``J``.

    Equals ``sqrt(det(J @ J.T))`` when the chain has ≥6 movable joints; for
    under-actuated chains it is the product of the ``min(rows, n)`` singular
    values (``sqrt(det(J.T @ J))``), a meaningful "distance from singularity"
    either way. Zero at a singular configuration; ``0.0`` for a fixed chain
    (``n = 0``).

    ``rows`` restricts the measure to a task subspace of the 6-row ``[v; w]``
    twist — pass ``rows=(0, 1, 2)`` for a **translational (position-only)**
    measure. This matters for sub-6-DoF arms: the *full* 6×n Jacobian of a
    planar 2R never loses rank (its angular rows keep the columns independent),
    so the classic "singular when fully extended" only shows up in the
    positional sub-Jacobian.
    """
    J = _selected_jacobian(robot, q, link, rows)
    if J.shape[1] == 0:
        return 0.0
    sv = np.linalg.svd(J, compute_uv=False)
    return float(np.prod(sv))


class SingularityReport(BaseModel):
    singular_values: list[float]   # of J, descending
    sigma_min: float               # smallest singular value
    sigma_max: float               # largest singular value
    condition_number: float        # sigma_max / sigma_min (inf if sigma_min == 0)
    manipulability: float          # product of singular values
    is_singular: bool              # sigma_min < tol


def singularity_report(
    robot: Robot,
    q: Optional[dict[str, float]] = None,
    link: str = "",
    *,
    tol: float = 1e-4,
    rows: Optional[Sequence[int]] = None,
) -> SingularityReport:
    """Singular-value health check of the Jacobian of ``link`` at ``q``.

    A configuration is *singular* when the end-effector loses an instantaneous
    degree of freedom — the Jacobian drops rank and ``sigma_min`` collapses
    toward 0 (the condition number blows up). ``tol`` is the threshold on
    ``sigma_min`` for the ``is_singular`` flag.

    ``rows`` restricts the analysis to a task subspace of the 6-row ``[v; w]``
    twist (see :func:`manipulability`) — e.g. ``rows=(0, 1, 2)`` to detect the
    positional singularity of a sub-6-DoF arm that the full Jacobian masks.

    A fixed chain (``n = 0``) reports all-zero values and ``is_singular=True``.
    """
    J = _selected_jacobian(robot, q, link, rows)
    if J.shape[1] == 0:
        return SingularityReport(
            singular_values=[], sigma_min=0.0, sigma_max=0.0,
            condition_number=float("inf"), manipulability=0.0, is_singular=True,
        )
    sv = np.linalg.svd(J, compute_uv=False)
    sigma_min = float(sv[-1])
    sigma_max = float(sv[0])
    cond = float(sigma_max / sigma_min) if sigma_min > 0.0 else float("inf")
    return SingularityReport(
        singular_values=[float(s) for s in sv],
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        condition_number=cond,
        manipulability=float(np.prod(sv)),
        is_singular=sigma_min < tol,
    )
