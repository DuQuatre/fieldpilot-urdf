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


def _interior_x0(
    q_full: dict[str, float], names: list[str], lo: np.ndarray, hi: np.ndarray,
) -> np.ndarray:
    """Project a seed config to a strictly-interior LS start point. TRF's
    projected gradient is zero on active bounds, so flush-against-a-bound seeds
    take zero-length steps — keep a small margin off each edge."""
    margin = np.maximum((hi - lo) * 1e-3, 1e-6)
    return np.array([
        float(np.clip(q_full[name], lo[i] + margin[i], hi[i] - margin[i]))
        for i, name in enumerate(names)
    ])


def _run_ls(
    x0: np.ndarray, names: list[str], q_full: dict[str, float], robot: Robot,
    target_link: str, target_xyz: np.ndarray, target_R: Optional[np.ndarray],
    lo: np.ndarray, hi: np.ndarray, max_nfev: int, tol: float,
) -> IKResult:
    """One least_squares solve from `x0`; package the outcome as an IKResult."""
    # Drive scipy hard: solver tolerances are much tighter than the user-facing
    # `converged` threshold so we get the best feasible residual every time.
    solver_tol = max(tol * 1e-4, 1e-12)
    res = least_squares(
        _residual,
        x0,
        bounds=(lo, hi),
        method="trf",
        args=(names, q_full, robot, target_link, target_xyz, target_R),
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
        q=q_out, position_error=pos_err, orientation_error=ori_err,
        converged=converged, n_iter=int(res.nfev), message=res.message,
    )


def _midpoint_seed(robot: Robot, joints: list[Joint],
                   q_init: Optional[dict[str, float]]) -> dict[str, float]:
    """Full-config seed: every joint at 0, movable ones at the midpoint of
    their bounds (continuous at 0), overridden by q_init."""
    q_full = {j.name: 0.0 for j in robot.joints}
    for j in joints:
        if j.type == "continuous":
            q_full[j.name] = 0.0
        else:
            q_full[j.name] = (j.limit.lower + j.limit.upper) / 2.0
    if q_init:
        q_full.update(q_init)
    return q_full


def _random_seed(robot: Robot, joints: list[Joint], lo: np.ndarray,
                 hi: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """Full-config seed with movable joints sampled uniformly within bounds."""
    q_full = {j.name: 0.0 for j in robot.joints}
    sample = lo + rng.random(len(joints)) * (hi - lo)
    for j, v in zip(joints, sample):
        q_full[j.name] = float(v)
    return q_full


def _better(a: IKResult, b: IKResult) -> IKResult:
    """The preferred of two results: converged beats not-converged, then lower
    total (position + orientation) error wins."""
    ka = (0 if a.converged else 1, a.position_error + a.orientation_error)
    kb = (0 if b.converged else 1, b.position_error + b.orientation_error)
    return a if ka <= kb else b


def solve_ik(
    robot: Robot,
    target_link: str,
    target_xyz: tuple[float, float, float],
    target_rpy: Optional[tuple[float, float, float]] = None,
    q_init: Optional[dict[str, float]] = None,
    *,
    max_nfev: int = 200,
    tol: float = 1e-4,
    n_restarts: int = 0,
    seed: Optional[int] = None,
) -> IKResult:
    """Solve IK for `target_link` to reach `target_xyz` (and optionally `target_rpy`).

    Returns the best-effort q + convergence info. `converged=True` iff the
    position residual is below `tol` and orientation residual (when provided)
    is below 10*tol.

    With `n_restarts > 0`, if the primary solve (from `q_init`/midpoint) doesn't
    converge, the solver retries from up to `n_restarts` random in-bounds seeds
    and keeps the best result — a cheap way past local minima on hard targets.
    `seed` makes the restart sampling reproducible. `n_restarts=0` (default)
    leaves the single-shot behaviour exactly as before.
    """
    if target_link not in {l.name for l in robot.links}:
        raise KeyError(f"unknown target_link: {target_link!r}")
    joints = _optimizable_joints(robot)
    if not joints:
        raise ValueError("robot has no optimisable joints — IK is undefined")

    names = [j.name for j in joints]
    lo, hi = _bounds(joints)
    target_R = rpy_to_R(target_rpy) if target_rpy is not None else None
    target = np.array(target_xyz, dtype=float)

    q_full = _midpoint_seed(robot, joints, q_init)
    best = _run_ls(_interior_x0(q_full, names, lo, hi), names, q_full, robot,
                   target_link, target, target_R, lo, hi, max_nfev, tol)

    if n_restarts > 0 and not best.converged:
        rng = np.random.default_rng(seed)
        for _ in range(n_restarts):
            qf = _random_seed(robot, joints, lo, hi, rng)
            r = _run_ls(_interior_x0(qf, names, lo, hi), names, qf, robot,
                        target_link, target, target_R, lo, hi, max_nfev, tol)
            best = _better(best, r)
            if best.converged:
                break
    return best


def _q_distance(a: dict[str, float], b: dict[str, float],
                joints: list[Joint]) -> float:
    """Joint-space distance between two configs over the optimisable joints;
    continuous joints measured on the shortest wrapped arc."""
    total = 0.0
    for j in joints:
        d = a[j.name] - b[j.name]
        if j.type == "continuous":
            d = (d + math.pi) % (2 * math.pi) - math.pi
        total += d * d
    return math.sqrt(total)


def solve_ik_multi(
    robot: Robot,
    target_link: str,
    target_xyz: tuple[float, float, float],
    target_rpy: Optional[tuple[float, float, float]] = None,
    *,
    n_restarts: int = 24,
    seed: Optional[int] = None,
    max_nfev: int = 200,
    tol: float = 1e-4,
    dedup_tol: float = 1e-3,
    max_solutions: Optional[int] = None,
    require_converged: bool = True,
) -> list[IKResult]:
    """Find multiple *distinct* IK solutions via random restarts.

    Many arms reach a pose more than one way (elbow-up / elbow-down, joint
    flips). `solve_ik` returns one; this runs the solver from the midpoint seed
    plus `n_restarts` random in-bounds seeds, then collapses results that land
    on the same configuration (joint-space distance below `dedup_tol`,
    continuous joints compared on the wrapped arc).

    Returns the distinct solutions sorted best-first (lowest position +
    orientation error). With `require_converged=True` (default) only converged
    solutions are returned — an empty list means none were found. Set it False
    to also surface best-effort near-misses. `max_solutions` caps the count;
    `seed` makes the run reproducible.
    """
    if target_link not in {l.name for l in robot.links}:
        raise KeyError(f"unknown target_link: {target_link!r}")
    joints = _optimizable_joints(robot)
    if not joints:
        raise ValueError("robot has no optimisable joints — IK is undefined")

    names = [j.name for j in joints]
    lo, hi = _bounds(joints)
    target_R = rpy_to_R(target_rpy) if target_rpy is not None else None
    target = np.array(target_xyz, dtype=float)
    rng = np.random.default_rng(seed)

    seeds = [_midpoint_seed(robot, joints, None)]
    seeds += [_random_seed(robot, joints, lo, hi, rng) for _ in range(n_restarts)]

    results: list[IKResult] = []
    for qf in seeds:
        r = _run_ls(_interior_x0(qf, names, lo, hi), names, qf, robot,
                    target_link, target, target_R, lo, hi, max_nfev, tol)
        if require_converged and not r.converged:
            continue
        results.append(r)

    # Best-first, then keep the first representative of each distinct cluster.
    results.sort(key=lambda r: r.position_error + r.orientation_error)
    distinct: list[IKResult] = []
    for r in results:
        if any(_q_distance(r.q, d.q, joints) < dedup_tol for d in distinct):
            continue
        distinct.append(r)
        if max_solutions is not None and len(distinct) >= max_solutions:
            break
    return distinct
