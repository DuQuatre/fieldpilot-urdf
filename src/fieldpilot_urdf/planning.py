"""Collision-free motion planning: RRT-Connect in joint space.

Where :func:`fieldpilot_urdf.trajectory.check_trajectory` *validates* a path you
already have, this *generates* one: given a start and goal joint configuration,
:func:`plan_path` searches for a sequence of waypoints that moves between them
without violating joint limits or self-colliding.

The planner is a bidirectional RRT (RRT-Connect): two trees grow from the start
and the goal and try to meet. It's probabilistically complete — given enough
iterations it finds a path if one exists — but not optimal, so the raw path is
jagged. :func:`shorten_path` post-processes it into a shorter, smoother route by
greedily short-cutting waypoint pairs that connect collision-free.

Configuration space = the robot's movable joints, the same set IK optimises over:
revolute/prismatic with a ``<limit>`` (sampled within bounds) and continuous
(wrapped to ``[-π, π]``). Fixed joints are ignored. A configuration is a plain
``dict[str, float]`` of joint name → value, exactly like the rest of the package,
so a returned path feeds straight into ``check_trajectory`` or ``forward_kinematics``.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from pydantic import BaseModel

from .collisions import MeshResolver, detect_self_collisions
from .models import Joint, Robot


# --- result ----------------------------------------------------------------

class PlanResult(BaseModel):
    path: list[dict[str, float]] = []  # waypoints start..goal; [] if not found
    success: bool
    n_iter: int            # tree-extension iterations consumed
    n_waypoints: int       # == len(path)
    message: str


# --- joint set + config <-> vector -----------------------------------------

def _planning_joints(robot: Robot) -> list[Joint]:
    """Movable joints we plan over — mirrors ik._optimizable_joints: fixed
    joints are skipped, revolute/prismatic need a <limit>, continuous are kept
    (wrapped to [-π, π]).
    """
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
    """Box bounds; continuous joints get [-π, π] (same convention as ik/trajectory)."""
    lo, hi = [], []
    for j in joints:
        if j.type == "continuous":
            lo.append(-math.pi)
            hi.append(math.pi)
        else:
            lo.append(j.limit.lower)
            hi.append(j.limit.upper)
    return np.array(lo, dtype=float), np.array(hi, dtype=float)


def _to_vec(q: dict[str, float], joints: list[Joint]) -> np.ndarray:
    return np.array([float(q[j.name]) for j in joints], dtype=float)


def _to_dict(vec: np.ndarray, joints: list[Joint]) -> dict[str, float]:
    return {j.name: float(v) for j, v in zip(joints, vec)}


def _diff(a: np.ndarray, b: np.ndarray, is_continuous: np.ndarray) -> np.ndarray:
    """b - a per joint, with continuous joints taking the shortest wrapped
    angular path (result in [-π, π]). Bounded joints are plain subtraction.
    """
    d = b - a
    if is_continuous.any():
        wrapped = (d[is_continuous] + math.pi) % (2 * math.pi) - math.pi
        # (-π) % … gives +π; both endpoints are equivalent for a continuous joint.
        d = d.copy()
        d[is_continuous] = wrapped
    return d


def _dist(a: np.ndarray, b: np.ndarray, is_continuous: np.ndarray) -> float:
    return float(np.linalg.norm(_diff(a, b, is_continuous)))


# --- feasibility checks ----------------------------------------------------

def _config_free(
    robot: Robot,
    vec: np.ndarray,
    joints: list[Joint],
    check_collisions: bool,
    mesh_resolver: Optional[MeshResolver],
) -> bool:
    """A config is feasible if it doesn't self-collide. Joint limits are
    enforced by construction (samples + steering stay within bounds), so we
    only need the collision query here.
    """
    if not check_collisions:
        return True
    q = _to_dict(vec, joints)
    try:
        return not detect_self_collisions(robot, q=q, mesh_resolver=mesh_resolver)
    except ValueError:
        # Not single-rooted — FK/collisions undefined; treat as blocked so the
        # planner fails cleanly rather than emitting a bogus path.
        return False


def _edge_free(
    robot: Robot,
    a: np.ndarray,
    b: np.ndarray,
    joints: list[Joint],
    is_continuous: np.ndarray,
    step_size: float,
    check_collisions: bool,
    mesh_resolver: Optional[MeshResolver],
) -> bool:
    """Densely sample the straight segment a→b (in wrapped joint space) and
    reject if any interior config self-collides. Endpoint `a` is assumed already
    validated by the caller; `b` is checked here.
    """
    if not check_collisions:
        return True
    d = _diff(a, b, is_continuous)
    dist = float(np.linalg.norm(d))
    n = max(1, int(math.ceil(dist / step_size)))
    for i in range(1, n + 1):
        vec = a + d * (i / n)
        if not _config_free(robot, vec, joints, check_collisions, mesh_resolver):
            return False
    return True


# --- RRT-Connect -----------------------------------------------------------

class _Tree:
    """Nodes + parent pointers for path reconstruction."""

    def __init__(self, root: np.ndarray):
        self.nodes: list[np.ndarray] = [root]
        self.parents: list[int] = [-1]

    def nearest(self, target: np.ndarray, is_continuous: np.ndarray) -> int:
        best_i, best_d = 0, math.inf
        for i, n in enumerate(self.nodes):
            dd = _dist(n, target, is_continuous)
            if dd < best_d:
                best_i, best_d = i, dd
        return best_i

    def add(self, vec: np.ndarray, parent: int) -> int:
        self.nodes.append(vec)
        self.parents.append(parent)
        return len(self.nodes) - 1

    def path_to_root(self, idx: int) -> list[np.ndarray]:
        out = []
        while idx != -1:
            out.append(self.nodes[idx])
            idx = self.parents[idx]
        return out  # leaf -> root order


def _steer(
    frm: np.ndarray, to: np.ndarray, is_continuous: np.ndarray, step_size: float,
) -> np.ndarray:
    """One step of at most `step_size` from `frm` toward `to`, along the wrapped
    difference. The result is NOT re-wrapped into [-π, π]; edge checks operate on
    raw coordinates and only the *difference* is wrapped, so a continuous joint
    may legitimately drift outside [-π, π] along the path (it's periodic).
    """
    d = _diff(frm, to, is_continuous)
    dist = float(np.linalg.norm(d))
    if dist <= step_size:
        return frm + d
    return frm + d * (step_size / dist)


def _extend(
    tree: _Tree, target: np.ndarray, robot: Robot, joints: list[Joint],
    is_continuous: np.ndarray, step_size: float,
    check_collisions: bool, mesh_resolver: Optional[MeshResolver],
) -> tuple[str, int]:
    """Grow `tree` one step toward `target`. Returns ("trapped"|"advanced"|
    "reached", new_node_index). "reached" means the new node coincides with
    `target` (within step_size).
    """
    near_i = tree.nearest(target, is_continuous)
    new_vec = _steer(tree.nodes[near_i], target, is_continuous, step_size)
    if not _edge_free(robot, tree.nodes[near_i], new_vec, joints, is_continuous,
                      step_size, check_collisions, mesh_resolver):
        return "trapped", near_i
    new_i = tree.add(new_vec, near_i)
    reached = _dist(new_vec, target, is_continuous) <= 1e-9
    return ("reached" if reached else "advanced"), new_i


def _connect(
    tree: _Tree, target: np.ndarray, robot: Robot, joints: list[Joint],
    is_continuous: np.ndarray, step_size: float,
    check_collisions: bool, mesh_resolver: Optional[MeshResolver],
) -> tuple[str, int]:
    """Repeatedly extend `tree` toward `target` until it reaches it or gets
    trapped (RRT-Connect's greedy connect heuristic)."""
    status, idx = "advanced", -1
    while status == "advanced":
        status, idx = _extend(tree, target, robot, joints, is_continuous,
                              step_size, check_collisions, mesh_resolver)
    return status, idx


def plan_path(
    robot: Robot,
    start: dict[str, float],
    goal: dict[str, float],
    *,
    step_size: float = 0.1,
    max_iters: int = 5000,
    seed: Optional[int] = None,
    check_collisions: bool = True,
    mesh_resolver: Optional[MeshResolver] = None,
    smooth: bool = True,
) -> PlanResult:
    """Plan a collision-free joint-space path from `start` to `goal`.

    Uses RRT-Connect: two trees rooted at start and goal grow toward random
    samples and try to link up. Returns a :class:`PlanResult` whose ``path`` is a
    list of waypoint dicts ``[start, …, goal]`` (empty on failure). The path is
    collision-free at `step_size` resolution and, unless ``smooth=False``,
    short-cut via :func:`shorten_path`.

    `start`/`goal` must give a value for every movable joint (missing joints
    default to 0.0). Both endpoints are validated up front: if either
    self-collides the planner fails immediately with an explanatory message.
    `max_iters` bounds the tree-extension attempts before giving up.
    """
    joints = _planning_joints(robot)
    if not joints:
        raise ValueError("robot has no movable joints — planning is undefined")

    lo, hi = _bounds(joints)
    is_continuous = np.array([j.type == "continuous" for j in joints])

    # Missing joints default to 0.0, consistent with the rest of the package.
    start_full = {j.name: 0.0 for j in joints}
    start_full.update({k: v for k, v in start.items() if k in start_full})
    goal_full = {j.name: 0.0 for j in joints}
    goal_full.update({k: v for k, v in goal.items() if k in goal_full})
    s_vec = _to_vec(start_full, joints)
    g_vec = _to_vec(goal_full, joints)

    # Bounds are hard constraints — reject endpoints outside them rather than
    # silently clipping (that would plan to a different goal than asked).
    for label, vec in (("start", s_vec), ("goal", g_vec)):
        below = vec < lo - 1e-9
        above = vec > hi + 1e-9
        if below.any() or above.any():
            bad = [joints[i].name for i in range(len(joints)) if below[i] or above[i]]
            return PlanResult(success=False, n_iter=0, n_waypoints=0,
                              path=[], message=f"{label} out of joint limits: {bad}")

    if not _config_free(robot, s_vec, joints, check_collisions, mesh_resolver):
        return PlanResult(success=False, n_iter=0, n_waypoints=0, path=[],
                          message="start configuration self-collides")
    if not _config_free(robot, g_vec, joints, check_collisions, mesh_resolver):
        return PlanResult(success=False, n_iter=0, n_waypoints=0, path=[],
                          message="goal configuration self-collides")

    # Trivial case: a single collision-free edge already connects them.
    if _edge_free(robot, s_vec, g_vec, joints, is_continuous, step_size,
                  check_collisions, mesh_resolver):
        path = [_to_dict(s_vec, joints), _to_dict(g_vec, joints)]
        return PlanResult(success=True, n_iter=0, n_waypoints=len(path), path=path,
                          message="connected directly (no obstacle between endpoints)")

    rng = np.random.default_rng(seed)
    tree_a = _Tree(s_vec)   # rooted at start
    tree_b = _Tree(g_vec)   # rooted at goal
    a_is_start = True       # which physical endpoint tree_a currently is

    for it in range(1, max_iters + 1):
        rand = lo + rng.random(len(joints)) * (hi - lo)
        status, new_i = _extend(tree_a, rand, robot, joints, is_continuous,
                                step_size, check_collisions, mesh_resolver)
        if status != "trapped":
            # Try to connect the other tree to tree_a's new node.
            cstatus, other_i = _connect(
                tree_b, tree_a.nodes[new_i], robot, joints, is_continuous,
                step_size, check_collisions, mesh_resolver)
            if cstatus == "reached":
                # Stitch: start-side leaf..root + goal-side root..leaf.
                branch_a = tree_a.path_to_root(new_i)        # new -> root_a
                branch_b = tree_b.path_to_root(other_i)      # new -> root_b
                if a_is_start:
                    vecs = list(reversed(branch_a)) + branch_b[1:]
                else:
                    vecs = list(reversed(branch_b)) + branch_a[1:]
                path_vecs = vecs
                if smooth:
                    path_vecs = _shorten_vecs(
                        robot, path_vecs, joints, is_continuous, step_size,
                        check_collisions, mesh_resolver, rng)
                path = [_to_dict(v, joints) for v in path_vecs]
                return PlanResult(success=True, n_iter=it, n_waypoints=len(path),
                                  path=path, message="path found")
        # Swap roles so both trees grow greedily (RRT-Connect).
        tree_a, tree_b = tree_b, tree_a
        a_is_start = not a_is_start

    return PlanResult(success=False, n_iter=max_iters, n_waypoints=0, path=[],
                      message=f"no path found within {max_iters} iterations")


# --- path shortening / smoothing -------------------------------------------

def _shorten_vecs(
    robot: Robot, vecs: list[np.ndarray], joints: list[Joint],
    is_continuous: np.ndarray, step_size: float, check_collisions: bool,
    mesh_resolver: Optional[MeshResolver], rng: np.random.Generator,
    n_iters: int = 100,
) -> list[np.ndarray]:
    """Greedy random short-cutting: repeatedly pick two waypoints and, if the
    straight segment between them is collision-free, drop everything in between.
    Monotonically non-increasing in length; cheap and effective on RRT output.
    """
    if len(vecs) <= 2:
        return vecs
    path = list(vecs)
    for _ in range(n_iters):
        if len(path) <= 2:
            break
        i = int(rng.integers(0, len(path) - 1))
        j = int(rng.integers(i + 1, len(path)))
        if j - i <= 1:
            continue  # already adjacent
        if _edge_free(robot, path[i], path[j], joints, is_continuous, step_size,
                      check_collisions, mesh_resolver):
            path = path[: i + 1] + path[j:]
    return path


def shorten_path(
    robot: Robot,
    path: list[dict[str, float]],
    *,
    step_size: float = 0.1,
    seed: Optional[int] = None,
    check_collisions: bool = True,
    mesh_resolver: Optional[MeshResolver] = None,
    n_iters: int = 100,
) -> list[dict[str, float]]:
    """Shorten a waypoint path by greedily short-cutting collision-free pairs.

    Safe to call on any path of movable-joint configs (e.g. straight from
    :func:`plan_path` with ``smooth=False``, or a hand-built one). Returns a new
    list; the input is untouched. Endpoints are preserved.
    """
    joints = _planning_joints(robot)
    if not joints or len(path) <= 2:
        return list(path)
    is_continuous = np.array([j.type == "continuous" for j in joints])
    vecs = [_to_vec({**{j.name: 0.0 for j in joints}, **q}, joints) for q in path]
    rng = np.random.default_rng(seed)
    out = _shorten_vecs(robot, vecs, joints, is_continuous, step_size,
                        check_collisions, mesh_resolver, rng, n_iters=n_iters)
    return [_to_dict(v, joints) for v in out]


def path_length(
    robot: Robot, path: list[dict[str, float]],
) -> float:
    """Total joint-space length of a waypoint path (continuous joints measured
    on the shortest wrapped arc). 0.0 for a path of fewer than two waypoints.
    """
    joints = _planning_joints(robot)
    if not joints or len(path) < 2:
        return 0.0
    is_continuous = np.array([j.type == "continuous" for j in joints])
    base = {j.name: 0.0 for j in joints}
    vecs = [_to_vec({**base, **q}, joints) for q in path]
    return float(sum(_dist(vecs[i], vecs[i + 1], is_continuous)
                     for i in range(len(vecs) - 1)))
