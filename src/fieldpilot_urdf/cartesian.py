"""Cartesian (task-space) motion: move a link along a straight line.

Where :func:`fieldpilot_urdf.planning.plan_path` plans in *joint* space (an
RRT-Connect path between two configurations), this module plans in *task* space:
:func:`plan_cartesian_path` returns a joint-space path whose chosen link follows
a straight line in SE(3) — linear interpolation of position, geodesic (slerp)
interpolation of orientation — from its current pose to a target pose.

The engine is a **resolved-rate** servo built on the 1.10 geometric Jacobian
(:func:`fieldpilot_urdf.kinematics.geometric_jacobian`): each step solves the
damped least-squares (Levenberg) inverse

    dq = Jᵀ (J Jᵀ + λ²I)⁻¹ · e

for the world-frame pose error twist ``e = [Δposition; Δrotation]``, clamped to
joint limits. The damping ``λ`` keeps the step well-behaved where the line
passes near a kinematic singularity (cf. :func:`...singularity_report`).

The returned ``path`` (``list[dict]``) drops straight into ``check_trajectory``
and ``forward_kinematics``, exactly like ``plan_path``'s ``PlanResult``.

Pure NumPy. No scipy. Revolute / continuous / prismatic joints.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from pydantic import BaseModel

from .fk import forward_kinematics, rpy_to_R
from .kinematics import geometric_jacobian, jacobian_joints
from .models import Joint, Robot


class CartesianPlanResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    path: list[dict[str, float]] = []  # configs start..target; [] if nothing moved
    success: bool                      # target reached within tolerance
    n_waypoints: int                   # == len(path)
    reached_fraction: float            # fraction of the line actually followed [0,1]
    position_error: float              # final ‖Δp‖ to the target (m)
    orientation_error: float           # final rotation error to the target (rad)
    message: str


# --- SE(3) helpers ----------------------------------------------------------

def _rotation_log(R: np.ndarray) -> np.ndarray:
    """Axis·angle (rotation vector) of a rotation matrix — inverse of Rodrigues.

    Returns a 3-vector whose direction is the rotation axis and whose norm is the
    angle in ``[0, π]``. Robust at 0 and π."""
    R = np.asarray(R, dtype=float)
    cos = (np.trace(R) - 1.0) / 2.0
    cos = max(-1.0, min(1.0, cos))
    angle = math.acos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        # Near π: axis from the largest-diagonal column of (R + I).
        A = R + np.eye(3)
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k]
        n = float(np.linalg.norm(axis))
        axis = axis / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
        return axis * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * math.sin(angle)) * angle


def _exp_so3(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation matrix from an axis·angle 3-vector."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rotvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def interpolate_pose(T0: np.ndarray, T1: np.ndarray, s: float) -> np.ndarray:
    """Interpolate between two 4×4 poses at ``s`` ∈ [0, 1]: straight-line
    (lerp) translation and geodesic (slerp) rotation. ``s=0``→``T0``, ``s=1``→``T1``."""
    T0 = np.asarray(T0, dtype=float)
    T1 = np.asarray(T1, dtype=float)
    s = float(s)
    T = np.eye(4)
    T[:3, 3] = (1.0 - s) * T0[:3, 3] + s * T1[:3, 3]
    R_rel = T1[:3, :3] @ T0[:3, :3].T
    T[:3, :3] = _exp_so3(s * _rotation_log(R_rel)) @ T0[:3, :3]
    return T


def pose_error(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """``(position_error, orientation_error)`` between two poses: the translation
    distance (m) and the angle (rad, ``[0, π]``) of the relative rotation."""
    T_a = np.asarray(T_a, dtype=float)
    T_b = np.asarray(T_b, dtype=float)
    pos = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    rot = float(np.linalg.norm(_rotation_log(T_b[:3, :3] @ T_a[:3, :3].T)))
    return pos, rot


def _pose_error_twist(T_cur: np.ndarray, T_goal: np.ndarray) -> np.ndarray:
    """World-frame error twist ``[Δp; Δrot]`` driving ``T_cur`` toward ``T_goal``,
    matching the geometric Jacobian's linear-over-angular row order."""
    e = np.zeros(6)
    e[:3] = T_goal[:3, 3] - T_cur[:3, 3]
    e[3:] = _rotation_log(T_goal[:3, :3] @ T_cur[:3, :3].T)
    return e


# --- joints / limits --------------------------------------------------------

def _joint_bounds(joints: list[Joint]) -> tuple[np.ndarray, np.ndarray]:
    """Box bounds per joint; continuous joints are unbounded (±inf)."""
    lo, hi = [], []
    for j in joints:
        if j.type == "continuous" or j.limit is None:
            lo.append(-np.inf); hi.append(np.inf)
        else:
            lo.append(j.limit.lower); hi.append(j.limit.upper)
    return np.array(lo), np.array(hi)


def _selectively_damped_step(J: np.ndarray, e: np.ndarray, damping: float) -> np.ndarray:
    """Resolved-rate step ``dq`` toward error twist ``e``, via an SVD pseudo-inverse
    that damps **only** the singular directions.

    For each singular value ``σ`` of ``J``, the inverse gain is ``σ/(σ² + λ²)``
    with ``λ`` ramped from 0 (at ``σ ≥ damping``) up to ``damping`` (at ``σ = 0``).
    Well-conditioned directions are therefore inverted exactly (``1/σ``) — so the
    servo reaches tight tolerance where it can — while directions collapsing
    toward a singularity are smoothly damped instead of blowing up. This is the
    standard cure for the steady-state bias of constant-λ damped least squares.
    """
    U, S, Vt = np.linalg.svd(J, full_matrices=False)  # J(6×n) = U(6×k) diag(S) Vt(k×n)
    utE = U.T @ e
    dq = np.zeros(J.shape[1])
    eps = max(damping, 1e-12)
    for i, s in enumerate(S):
        if s >= eps:
            inv = 1.0 / s
        else:
            lam2 = (1.0 - (s / eps) ** 2) * (damping ** 2)  # 0 at σ=eps → λ² at σ=0
            inv = s / (s * s + lam2)
        dq += inv * utE[i] * Vt[i]
    return dq


def _servo_to_pose(
    robot: Robot,
    link: str,
    cols: list[str],
    by_name: dict[str, Joint],
    held: dict[str, float],
    q0: dict[str, float],
    T_goal: np.ndarray,
    *,
    damping: float,
    pos_tol: float,
    rot_tol: float,
    gain: float,
    max_iters: int,
    respect_limits: bool,
) -> tuple[dict[str, float], float, float]:
    """Damped-least-squares servo of ``link`` toward ``T_goal`` from ``q0``.
    Returns ``(q, pos_err, rot_err)`` at the best configuration reached."""
    q = dict(q0)
    lo, hi = _joint_bounds([by_name[c] for c in cols])
    pos_err = rot_err = float("inf")
    for _ in range(max_iters):
        tf = forward_kinematics(robot, {**held, **q})
        e = _pose_error_twist(tf[link], T_goal)
        pos_err = float(np.linalg.norm(e[:3]))
        rot_err = float(np.linalg.norm(e[3:]))
        if pos_err <= pos_tol and rot_err <= rot_tol:
            break
        J = geometric_jacobian(robot, {**held, **q}, link, joints=cols)
        dq = _selectively_damped_step(J, gain * e, damping)
        qv = np.array([q[c] for c in cols]) + dq
        if respect_limits:
            qv = np.clip(qv, lo, hi)
        q = {c: float(v) for c, v in zip(cols, qv)}
    return q, pos_err, rot_err


def plan_cartesian_path(
    robot: Robot,
    link: str,
    target_xyz: tuple[float, float, float],
    target_rpy: Optional[tuple[float, float, float]] = None,
    start_q: Optional[dict[str, float]] = None,
    *,
    n_waypoints: int = 20,
    damping: float = 0.05,
    gain: float = 1.0,
    pos_tol: float = 1e-3,
    rot_tol: float = 1e-3,
    max_step_iters: int = 100,
    respect_limits: bool = True,
) -> CartesianPlanResult:
    """Plan a joint path whose ``link`` follows a **straight line** in task space
    from its current pose (at ``start_q``) to the target pose.

    The target position is ``target_xyz``; ``target_rpy`` is the target
    orientation (URDF fixed-axis roll/pitch/yaw) — omit it to hold the link's
    current orientation (a pure translation, the common "move in a straight
    line" case). The path is generated by interpolating the pose into
    ``n_waypoints`` steps (:func:`interpolate_pose`) and driving a resolved-rate
    servo (damped least squares over the geometric Jacobian) to each.

    Returns a :class:`CartesianPlanResult`. ``success`` means the final target
    was reached within ``pos_tol`` / ``rot_tol``; ``reached_fraction`` is how far
    along the line the path got before a waypoint became unreachable (a joint
    limit, a singularity the damping couldn't push through, or a target outside
    the workspace). Each ``path`` config includes the held (non-driven) joints,
    so ``forward_kinematics`` on it reproduces the intended pose.

    Joints driven are the movable joints on the root→``link`` path
    (:func:`...jacobian_joints`); any other movable joints are held at
    ``start_q``. Raises ``ValueError`` if the robot is not a single-rooted tree
    or ``link`` is unknown.
    """
    start_q = dict(start_q or {})
    cols = jacobian_joints(robot, link)
    by_name = {j.name: j for j in robot.joints}
    # Movable joints not on the chain are held fixed at their start value.
    held = {
        j.name: float(start_q.get(j.name, 0.0))
        for j in robot.joints
        if j.type != "fixed" and j.name not in cols
    }
    q_start = {c: float(start_q.get(c, 0.0)) for c in cols}

    tf0 = forward_kinematics(robot, {**held, **q_start})
    if link not in tf0:
        raise ValueError(f"unknown link: {link!r}")
    T_start = tf0[link]
    T_goal = np.eye(4)
    T_goal[:3, 3] = np.asarray(target_xyz, dtype=float)
    T_goal[:3, :3] = rpy_to_R(target_rpy) if target_rpy is not None else T_start[:3, :3]

    def _full(q: dict[str, float]) -> dict[str, float]:
        return {**held, **q}

    path: list[dict[str, float]] = [_full(q_start)]

    # Degenerate: nothing to drive. Success only if we are already at the target.
    if not cols:
        pos_err, rot_err = pose_error(T_start, T_goal)
        ok = pos_err <= pos_tol and rot_err <= rot_tol
        return CartesianPlanResult(
            path=path, success=ok, n_waypoints=len(path),
            reached_fraction=1.0 if ok else 0.0,
            position_error=pos_err, orientation_error=rot_err,
            message="no movable joints on the chain; "
                    + ("already at target" if ok else "cannot move to target"),
        )

    n = max(1, int(n_waypoints))
    q_cur = dict(q_start)
    reached = 0
    pos_err = rot_err = float("inf")
    for i in range(1, n + 1):
        T_wp = interpolate_pose(T_start, T_goal, i / n)
        q_cur, pos_err, rot_err = _servo_to_pose(
            robot, link, cols, by_name, held, q_cur, T_wp,
            damping=damping, pos_tol=pos_tol, rot_tol=rot_tol, gain=gain,
            max_iters=max_step_iters, respect_limits=respect_limits,
        )
        if pos_err > pos_tol or rot_err > rot_tol:
            # Stalled at this waypoint — record progress and stop.
            path.append(_full(q_cur))
            return CartesianPlanResult(
                path=path, success=False, n_waypoints=len(path),
                reached_fraction=reached / n,
                position_error=pos_err, orientation_error=rot_err,
                message=f"stalled at waypoint {i}/{n} "
                        f"(pos_err={pos_err:.2e}, rot_err={rot_err:.2e}): "
                        "joint limit, singularity, or target out of reach",
            )
        path.append(_full(q_cur))
        reached = i

    return CartesianPlanResult(
        path=path, success=True, n_waypoints=len(path),
        reached_fraction=1.0, position_error=pos_err, orientation_error=rot_err,
        message=f"reached target along a straight line in {n} waypoints",
    )
