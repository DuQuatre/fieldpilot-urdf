"""Environment (world-obstacle) collision and obstacle-aware planning. Anchors:
detect_obstacle_collisions flags exactly the links overlapping an obstacle; a
planner given an obstacle routes around it (and the resulting path is clean
against that same obstacle); endpoints inside an obstacle fail up front.
"""
from __future__ import annotations

import numpy as np

from fieldpilot_urdf.collisions import (
    Obstacle, box_obstacle, detect_obstacle_collisions, detect_self_collisions,
    sphere_obstacle,
)
from fieldpilot_urdf.models import (
    Box, Collision, Geometry, Joint, JointLimit, Link, Origin, Robot,
)
from fieldpilot_urdf.planning import plan_path, shorten_path
from fieldpilot_urdf.trajectory import check_trajectory


def _lim(lo=-3.0, hi=3.0):
    return JointLimit(lower=lo, upper=hi, effort=1.0, velocity=1.0)


def _densify(path, step=0.1):
    """Interpolate a waypoint path at ~`step` joint-space resolution, so a
    discrete check_trajectory covers the continuous motion between waypoints."""
    keys = list(path[0].keys())
    out = [dict(path[0])]
    for a, b in zip(path, path[1:]):
        va = np.array([a[k] for k in keys]); vb = np.array([b[k] for k in keys])
        n = max(1, int(np.ceil(np.linalg.norm(vb - va) / step)))
        for i in range(1, n + 1):
            v = va + (vb - va) * (i / n)
            out.append(dict(zip(keys, v.tolist())))
    return out


def _col(name, sx=0.4, sy=0.4, sz=0.4, xyz=(0, 0, 0)):
    return Collision(name=name, origin=Origin(xyz=xyz),
                     geometry=Box(size=(sx, sy, sz)))


def _slider_x():
    """A single prismatic joint sliding a small box along +X — easy to reason
    about where the link's world AABB sits for a given joint value."""
    return Robot(
        name="slider",
        links=[Link(name="base"),
               Link(name="tool", collisions=[_col("toolbox", 0.2, 0.2, 0.2)])],
        joints=[Joint(name="jx", type="prismatic", parent="base", child="tool",
                      origin=Origin(xyz=(0, 0, 0)), axis=(1, 0, 0), limit=_lim(-1, 3))],
    )


# --- obstacle primitives ----------------------------------------------------

def test_box_and_sphere_obstacle_aabbs():
    b = box_obstacle("b", center=(1, 2, 3), size=(2, 4, 6))
    lo, hi = b.aabb
    assert np.allclose(lo, [0, 0, 0]) and np.allclose(hi, [2, 4, 6])
    s = sphere_obstacle("s", center=(0, 0, 0), radius=0.5)
    lo, hi = s.aabb
    assert np.allclose(lo, [-0.5, -0.5, -0.5]) and np.allclose(hi, [0.5, 0.5, 0.5])
    assert isinstance(b, Obstacle) and b.name == "b"


# --- detect_obstacle_collisions (the anchor) --------------------------------

def test_detects_only_overlapping_link():
    r = _slider_x()
    # tool box is 0.2³ centred at x = jx. Obstacle slab around x≈2.0.
    obs = [box_obstacle("wall", center=(2.0, 0, 0), size=(0.4, 2, 2))]
    # jx=2.0 -> tool at the wall -> collision
    assert detect_obstacle_collisions(r, obs, q={"jx": 2.0}) == [("tool", "wall")]
    # jx=0.0 -> tool far from the wall -> clear
    assert detect_obstacle_collisions(r, obs, q={"jx": 0.0}) == []
    # an obstacle hit is independent of self-collision (none here)
    assert detect_self_collisions(r, q={"jx": 2.0}) == []


def test_empty_obstacles_is_noop():
    r = _slider_x()
    assert detect_obstacle_collisions(r, None, q={"jx": 2.0}) == []
    assert detect_obstacle_collisions(r, [], q={"jx": 2.0}) == []


def test_tol_inflates_the_check():
    r = _slider_x()
    # tool spans x∈[1.9,2.1] at jx=2.0; wall spans x∈[2.25,2.35]: a 0.15 gap.
    obs = [box_obstacle("wall", center=(2.3, 0, 0), size=(0.1, 2, 2))]
    assert detect_obstacle_collisions(r, obs, q={"jx": 2.0}) == []
    assert detect_obstacle_collisions(r, obs, q={"jx": 2.0}, tol=0.2) == [("tool", "wall")]


# --- obstacle-aware planning ------------------------------------------------

def _planar_xz():
    """2-DoF prismatic gantry in the X–Z plane carrying a small box tool — a
    clean testbed for routing a point around a 2-D obstacle."""
    return Robot(
        name="gantry",
        links=[Link(name="base"), Link(name="mid"),
               Link(name="tool", collisions=[_col("toolbox", 0.2, 0.2, 0.2)])],
        joints=[
            Joint(name="jx", type="prismatic", parent="base", child="mid",
                  origin=Origin(xyz=(0, 0, 0)), axis=(1, 0, 0), limit=_lim(-1, 3)),
            Joint(name="jz", type="prismatic", parent="mid", child="tool",
                  origin=Origin(xyz=(0, 0, 0)), axis=(0, 0, 1), limit=_lim(-1, 3)),
        ],
    )


def test_plan_routes_around_obstacle():
    r = _planar_xz()
    start, goal = {"jx": 0.0, "jz": 0.0}, {"jx": 2.0, "jz": 0.0}
    # A wall blocking the straight x-line at x≈1.0 for |z|<0.5; detour over the top.
    wall = box_obstacle("wall", center=(1.0, 0, 0), size=(0.3, 2, 1.0))

    # The straight x-line genuinely crosses the wall: a densely-sampled direct
    # path (z held at 0) collides somewhere in the middle.
    direct = [{"jx": t, "jz": 0.0} for t in np.linspace(0.0, 2.0, 41)]
    assert check_trajectory(r, direct, obstacles=[wall]) != []

    # With the obstacle, the planned path must avoid the wall entirely — checked
    # at the planner's own edge resolution (densify each segment to step_size).
    routed = plan_path(r, start, goal, obstacles=[wall], seed=1, step_size=0.1)
    assert routed.success
    dense = _densify(routed.path, step=0.1)
    assert check_trajectory(r, dense, obstacles=[wall]) == []        # clean along the motion
    # and it genuinely detours (leaves the z≈0 straight line to get around)
    assert max(abs(wp["jz"]) for wp in routed.path) > 0.5


def test_endpoint_inside_obstacle_fails_up_front():
    r = _planar_xz()
    wall = box_obstacle("wall", center=(0.0, 0, 0), size=(1.0, 2, 1.0))  # over the start
    res = plan_path(r, {"jx": 0.0, "jz": 0.0}, {"jx": 2.0, "jz": 0.0},
                    obstacles=[wall], seed=0)
    assert not res.success
    assert "start configuration collides (self or obstacle)" in res.message


def test_message_unchanged_without_obstacles():
    """Backward compat: the self-collision wording is preserved when no obstacles."""
    r = _planar_xz()
    blocker = box_obstacle("b", center=(0, 0, 0), size=(1, 2, 1))
    # feed via obstacles -> new wording; without -> old wording stays intact
    with_obs = plan_path(r, {"jx": 0.0, "jz": 0.0}, {"jx": 2.0, "jz": 0.0},
                         obstacles=[blocker], seed=0)
    assert "self or obstacle" in with_obs.message


def test_shorten_path_respects_obstacles():
    r = _planar_xz()
    wall = box_obstacle("wall", center=(1.0, 0, 0), size=(0.3, 2, 1.0))
    routed = plan_path(r, {"jx": 0.0, "jz": 0.0}, {"jx": 2.0, "jz": 0.0},
                       obstacles=[wall], seed=2, smooth=False)
    assert routed.success
    short = shorten_path(r, routed.path, obstacles=[wall], seed=2)
    assert check_trajectory(r, short, obstacles=[wall]) == []   # still obstacle-free
    assert short[0] == routed.path[0] and short[-1] == routed.path[-1]


# --- trajectory validation --------------------------------------------------

def test_check_trajectory_flags_obstacle_steps():
    r = _slider_x()
    wall = box_obstacle("wall", center=(2.0, 0, 0), size=(0.4, 2, 2))
    qs = [{"jx": 0.0}, {"jx": 1.0}, {"jx": 2.0}]   # last step drives into the wall
    findings = check_trajectory(r, qs, obstacles=[wall])
    assert [f.step for f in findings] == [2]
    assert findings[0].code == "collision"
    assert "obstacle" in findings[0].detail and "wall" in findings[0].detail
    # without the obstacle the same trajectory is clean
    assert check_trajectory(r, qs) == []
