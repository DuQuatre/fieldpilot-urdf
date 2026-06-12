"""Numerical inverse kinematics.

Given a target 6-DoF (or 3-DoF position-only) pose for a chosen link, find a
joint configuration q that places the link there. Uses scipy.optimize
.least_squares (Levenberg–Marquardt with bounds) on the FK residual; the
Jacobian is estimated by finite differences. No analytical Jacobian —
keeps the implementation cave-man, costs <50 ms on a typical 6-DoF arm.

Joint limits are honored as box bounds on the LS variable. Joints not under
optimization (e.g. fixed, or no <limit>) are held at q_init.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from pydantic import BaseModel
from scipy.optimize import least_squares

from .fk import forward_kinematics, rpy_to_R
from .models import Joint, Robot


class IKResult(BaseModel):
    q: dict[str, float]
    position_error: float        # ‖p_actual - p_target‖ (m)
    orientation_error: float     # angle of relative rotation (rad), 0 if position-only
    converged: bool
    n_iter: int
    message: str


def _optimizable_joints(robot: Robot) -> list[Joint]:
    out: list[Joint] = []
    for j in robot.joints:
        if j.type == "fixed":
            continue
        if j.type in {"revolute", "prismatic"} and j.limit is not None:
            out.append(j)
        elif j.type == "continuous":
            out.append(j)
    return out


def _bounds(joints: list[Joint]) -> tuple[np.ndarray, np.ndarray]:
    """Box bounds for least_squares. continuous joints get [-π, π]."""
    lo, hi = [], []
    for j in joints:
        if j.type == "continuous":
            lo.append(-math.pi)
            hi.append(math.pi)
        else:
            lo.append(j.limit.lower)
            hi.append(j.limit.upper)
    return np.array(lo), np.array(hi)


def _rotation_angle(R: np.ndarray) -> float:
    """Magnitude of the rotation R, in radians. Robust to numerical drift."""
    cos = (np.trace(R) - 1.0) / 2.0
    return float(math.acos(max(-1.0, min(1.0, cos))))


def _residual(
    q_vec: np.ndarray,
    joint_names: list[str],
    q_init: dict[str, float],
    robot: Robot,
    target_link: str,
    target_xyz: np.ndarray,
    target_R: Optional[np.ndarray],
) -> np.ndarray:
    q = dict(q_init)
    for name, val in zip(joint_names, q_vec):
        q[name] = float(val)
    tfs = forward_kinematics(robot, q=q)
    T = tfs[target_link]
    pos_err = T[:3, 3] - target_xyz
    if target_R is None:
        return pos_err
    # Orientation residual: log-map of relative rotation as 3-vector.
    Rerr = T[:3, :3].T @ target_R
    angle = _rotation_angle(Rerr)
    if angle < 1e-9:
        ori = np.zeros(3)
    else:
        # axis from skew part
        axis = np.array([
            Rerr[2, 1] - Rerr[1, 2],
            Rerr[0, 2] - Rerr[2, 0],
            Rerr[1, 0] - Rerr[0, 1],
        ]) / (2.0 * math.sin(angle))
        ori = axis * angle
    return np.concatenate([pos_err, ori])


def solve_ik(
    robot: Robot,
    target_link: str,
    target_xyz: tuple[float, float, float],
    target_rpy: Optional[tuple[float, float, float]] = None,
    q_init: Optional[dict[str, float]] = None,
    *,
    max_nfev: int = 200,
    tol: float = 1e-4,
) -> IKResult:
    """Solve IK for `target_link` to reach `target_xyz` (and optionally `target_rpy`).

    Returns the best-effort q + convergence info. `converged=True` iff the
    position residual is below `tol` and orientation residual (when provided)
    is below 10*tol.
    """
    if target_link not in {l.name for l in robot.links}:
        raise KeyError(f"unknown target_link: {target_link!r}")
    joints = _optimizable_joints(robot)
    if not joints:
        raise ValueError("robot has no optimisable joints — IK is undefined")

    q_full = {j.name: 0.0 for j in robot.joints}
    # Default movable joints to the midpoint of their bounds — sitting an
    # initial guess flush against a bound makes scipy's TRF take zero-length
    # steps (the projected gradient vanishes on that axis).
    for j in joints:
        if j.type == "continuous":
            q_full[j.name] = 0.0
        else:
            q_full[j.name] = (j.limit.lower + j.limit.upper) / 2.0
    if q_init:
        q_full.update(q_init)

    names = [j.name for j in joints]
    lo, hi = _bounds(joints)
    # Keep x0 strictly interior; TRF's projected gradient is zero on active bounds.
    margin = np.maximum((hi - lo) * 1e-3, 1e-6)
    x0 = np.array([
        float(np.clip(q_full[name], lo[i] + margin[i], hi[i] - margin[i]))
        for i, name in enumerate(names)
    ])
    target_R = rpy_to_R(target_rpy) if target_rpy is not None else None

    # Drive scipy hard: solver tolerances are much tighter than the user-facing
    # `converged` threshold so we get the best feasible residual every time.
    solver_tol = max(tol * 1e-4, 1e-12)
    res = least_squares(
        _residual,
        x0,
        bounds=(lo, hi),
        method="trf",
        args=(names, q_full, robot, target_link,
              np.array(target_xyz, dtype=float), target_R),
        max_nfev=max_nfev,
        xtol=solver_tol, ftol=solver_tol, gtol=solver_tol,
    )

    q_out = dict(q_full)
    for name, val in zip(names, res.x):
        q_out[name] = float(val)

    final_residual = res.fun
    pos_err = float(np.linalg.norm(final_residual[:3]))
    ori_err = float(np.linalg.norm(final_residual[3:])) if target_R is not None else 0.0
    converged = pos_err < tol and ori_err < tol * 10

    return IKResult(
        q=q_out,
        position_error=pos_err,
        orientation_error=ori_err,
        converged=converged,
        n_iter=int(res.nfev),
        message=res.message,
    )
