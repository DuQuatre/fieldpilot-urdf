"""Forward kinematics for a URDF Robot. Pure numpy.

Convention: URDF rpy is fixed-axis roll(X)-pitch(Y)-yaw(Z), composed as
R = Rz(yaw) @ Ry(pitch) @ Rx(roll). For each joint, the child link frame is
    T_child = T_parent @ T_origin @ T_motion(q)
where T_motion is identity at q=0 (the "neutral pose").
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .graph import build_graph, root_links
from .models import Joint, Origin, Robot


def rpy_to_R(rpy: tuple[float, float, float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def origin_to_T(o: Optional[Origin]) -> np.ndarray:
    T = np.eye(4)
    if o is None:
        return T
    T[:3, :3] = rpy_to_R(o.rpy)
    T[:3, 3] = o.xyz
    return T


def rotation_around_axis(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    """Rodrigues. Returns 4x4 with translation = 0."""
    ax = np.array(axis, dtype=float)
    n = float(np.linalg.norm(ax))
    if n < 1e-12 or angle == 0.0:
        return np.eye(4)
    ax = ax / n
    K = np.array([[0, -ax[2], ax[1]],
                  [ax[2], 0, -ax[0]],
                  [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    return T


def joint_motion(joint: Joint, q: float) -> np.ndarray:
    if joint.type == "fixed":
        return np.eye(4)
    if joint.type in ("revolute", "continuous"):
        return rotation_around_axis(joint.axis, q)
    if joint.type == "prismatic":
        ax = np.array(joint.axis, dtype=float)
        n = float(np.linalg.norm(ax))
        T = np.eye(4)
        if n > 1e-12:
            T[:3, 3] = (ax / n) * q
        return T
    # floating / planar: punt to identity (rare in practice)
    return np.eye(4)


def forward_kinematics(
    robot: Robot, q: Optional[dict[str, float]] = None
) -> dict[str, np.ndarray]:
    """Return {link_name: 4x4 world transform}. Empty q -> neutral pose.

    Raises ValueError if the URDF is not a single-rooted tree.
    """
    q = q or {}
    G = build_graph(robot)
    roots = root_links(G)
    if len(roots) != 1:
        raise ValueError(f"FK requires single root link, found {len(roots)}: {roots}")
    by_name = {j.name: j for j in robot.joints}
    root = roots[0]
    tf: dict[str, np.ndarray] = {root: np.eye(4)}
    queue: deque[str] = deque([root])
    while queue:
        parent = queue.popleft()
        for child in G.successors(parent):
            jname = G.edges[parent, child]["joint_name"]
            j = by_name[jname]
            tf[child] = tf[parent] @ origin_to_T(j.origin) @ joint_motion(j, q.get(jname, 0.0))
            queue.append(child)
    return tf
