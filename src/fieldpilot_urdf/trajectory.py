"""Dynamic analysis: joint trajectories + reachable workspace.

Both are thin layers over forward_kinematics + detect_self_collisions. The
trajectory checker reports per-step limit / collision findings, the workspace
sampler returns a point cloud + bounding box of a target link under random
joint values within their bounds.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
from pydantic import BaseModel

from .collisions import (
    MeshResolver, Obstacle, detect_obstacle_collisions, detect_self_collisions,
)
from .fk import forward_kinematics
from .models import Joint, Robot


# --- trajectory check ------------------------------------------------------

class StepFinding(BaseModel):
    step: int
    code: str  # "limit" | "collision" (self or obstacle — see detail)
    detail: str
    refs: list[str] = []


def _joint_index(robot: Robot) -> dict[str, Joint]:
    return {j.name: j for j in robot.joints}


def check_trajectory(
    robot: Robot,
    qs: Iterable[dict[str, float]],
    *,
    check_collisions: bool = True,
    mesh_resolver: Optional[MeshResolver] = None,
    obstacles: Optional[list[Obstacle]] = None,
) -> list[StepFinding]:
    """For each q in the sequence: flag joint-limit violations and (optionally)
    self-collisions, plus collisions against any world ``obstacles``. Returns a
    flat list of per-step findings; an empty list means the trajectory is clean.
    Obstacle hits use ``code="collision"`` (the detail text says which).
    """
    by_name = _joint_index(robot)
    out: list[StepFinding] = []
    for step, q in enumerate(qs):
        for jname, val in q.items():
            j = by_name.get(jname)
            if j is None:
                out.append(StepFinding(
                    step=step, code="limit",
                    detail=f"unknown joint '{jname}'",
                    refs=[jname],
                ))
                continue
            if j.type == "continuous":
                continue  # no bounds by definition
            if j.limit is None:
                continue
            if not (j.limit.lower <= val <= j.limit.upper):
                out.append(StepFinding(
                    step=step, code="limit",
                    detail=(f"joint '{jname}' value {val:g} outside "
                            f"[{j.limit.lower:g}, {j.limit.upper:g}]"),
                    refs=[jname],
                ))
        if check_collisions:
            try:
                hits = detect_self_collisions(robot, q=q, mesh_resolver=mesh_resolver)
            except ValueError:
                continue  # not single-rooted; let R001/R002 surface that
            for a, b in hits:
                out.append(StepFinding(
                    step=step, code="collision",
                    detail=f"AABB self-collision: '{a}' <-> '{b}'",
                    refs=[a, b],
                ))
            if obstacles:
                try:
                    obs_hits = detect_obstacle_collisions(
                        robot, obstacles, q=q, mesh_resolver=mesh_resolver)
                except ValueError:
                    obs_hits = []
                for link, obs_name in obs_hits:
                    out.append(StepFinding(
                        step=step, code="collision",
                        detail=f"AABB obstacle-collision: '{link}' <-> obstacle '{obs_name}'",
                        refs=[link, obs_name],
                    ))
    return out


def trajectory_summary(findings: list[StepFinding], n_steps: int) -> dict:
    bad_steps = {f.step for f in findings}
    return {
        "n_steps": n_steps,
        "steps_with_issues": len(bad_steps),
        "limit_violations": sum(1 for f in findings if f.code == "limit"),
        "collisions": sum(1 for f in findings if f.code == "collision"),
        "first_bad_step": min(bad_steps) if bad_steps else None,
    }


# --- workspace sampling ----------------------------------------------------

class WorkspaceResult(BaseModel):
    target_link: str
    n_samples: int
    reachable_count: int  # samples that produced a valid pose
    collision_count: int  # of the reachable poses, how many self-collide
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    centroid: tuple[float, float, float]
    points: list[tuple[float, float, float]] = []  # excluded by default


def _bounded_joints(robot: Robot) -> list[Joint]:
    """Movable joints we'll sample over. Fixed joints contribute nothing;
    continuous have no bounds (we wrap to [-pi, pi]); revolute/prismatic
    need a <limit>.
    """
    out: list[Joint] = []
    for j in robot.joints:
        if j.type in {"revolute", "prismatic"} and j.limit is not None:
            out.append(j)
        elif j.type == "continuous":
            out.append(j)
    return out


def _sample_q(joints: list[Joint], rng: np.random.Generator) -> dict[str, float]:
    q: dict[str, float] = {}
    for j in joints:
        if j.type == "continuous":
            q[j.name] = float(rng.uniform(-math.pi, math.pi))
        else:
            q[j.name] = float(rng.uniform(j.limit.lower, j.limit.upper))
    return q


def sample_workspace(
    robot: Robot,
    target_link: str,
    n_samples: int = 200,
    *,
    seed: Optional[int] = None,
    check_collisions: bool = True,
    include_points: bool = False,
    mesh_resolver: Optional[MeshResolver] = None,
) -> WorkspaceResult:
    """Uniformly sample joint values within bounds and accumulate the target
    link's world-frame origin. Reports bounding box + collision rate.
    """
    if target_link not in {l.name for l in robot.links}:
        raise KeyError(f"unknown link: {target_link!r}")
    joints = _bounded_joints(robot)
    rng = np.random.default_rng(seed)

    pts: list[tuple[float, float, float]] = []
    collisions = 0
    for _ in range(n_samples):
        q = _sample_q(joints, rng)
        try:
            tfs = forward_kinematics(robot, q=q)
        except ValueError:
            continue
        T = tfs.get(target_link)
        if T is None:
            continue
        pts.append((float(T[0, 3]), float(T[1, 3]), float(T[2, 3])))
        if check_collisions:
            try:
                if detect_self_collisions(robot, q=q, mesh_resolver=mesh_resolver):
                    collisions += 1
            except ValueError:
                pass

    if pts:
        arr = np.array(pts)
        mn = tuple(float(x) for x in arr.min(axis=0))
        mx = tuple(float(x) for x in arr.max(axis=0))
        ce = tuple(float(x) for x in arr.mean(axis=0))
    else:
        mn = mx = ce = (0.0, 0.0, 0.0)

    return WorkspaceResult(
        target_link=target_link,
        n_samples=n_samples,
        reachable_count=len(pts),
        collision_count=collisions,
        bbox_min=mn,
        bbox_max=mx,
        centroid=ce,
        points=pts if include_points else [],
    )
