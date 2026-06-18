"""Collision-free IK: solve to a pose but return a configuration that doesn't
self-collide or hit an obstacle. The arm links carry collision geometry (not
just the tool), so the distinct IK branches collide *differently* — which is the
whole point. Each test asserts its precondition (a branch genuinely collides) so
none can pass vacuously.
"""
from __future__ import annotations

import numpy as np

from fieldpilot_urdf.collisions import (
    box_obstacle, detect_obstacle_collisions, detect_self_collisions,
)
from fieldpilot_urdf.fk import forward_kinematics
from fieldpilot_urdf.ik import solve_ik, solve_ik_collision_free, solve_ik_multi
from fieldpilot_urdf.models import (
    Box, Collision, Joint, JointLimit, Link, Origin, Robot, Sphere,
)


def _lim(lo=-3.2, hi=3.2):
    return JointLimit(lower=lo, upper=hi, effort=1.0, velocity=1.0)


def _linkbox(name, length, w=0.15):
    return Collision(name=name, origin=Origin(xyz=(length / 2, 0, 0)),
                     geometry=Box(size=(length, w, w)))


def _planar_2r(l1=1.0, l2=1.0, tool_r=0.2):
    """Planar 2R about +Z. l1/l2 carry box geometry (the arm body) and the tool a
    sphere, so elbow-up / elbow-down postures sweep different space and collide
    differently."""
    return Robot(
        name="rr",
        links=[Link(name="base"),
               Link(name="l1", collisions=[_linkbox("l1b", l1)]),
               Link(name="l2", collisions=[_linkbox("l2b", l2)]),
               Link(name="tool", collisions=[Collision(name="tip", origin=Origin(),
                                                       geometry=Sphere(radius=tool_r))])],
        joints=[
            Joint(name="j1", type="revolute", parent="base", child="l1",
                  origin=Origin(xyz=(0, 0, 0)), axis=(0, 0, 1), limit=_lim()),
            Joint(name="j2", type="revolute", parent="l1", child="l2",
                  origin=Origin(xyz=(l1, 0, 0)), axis=(0, 0, 1), limit=_lim()),
            Joint(name="tip", type="fixed", parent="l2", child="tool",
                  origin=Origin(xyz=(l2, 0, 0))),
        ],
    )


def _tcp(robot, q):
    return forward_kinematics(robot, q)["tool"][:3, 3]


# --- happy path -------------------------------------------------------------

def test_happy_path_reaches_and_is_collision_free():
    r = _planar_2r()
    target = (1.2, 0.6, 0.0)
    res = solve_ik_collision_free(r, "tool", target, seed=0)
    assert res.converged
    assert np.allclose(_tcp(r, res.q), target, atol=1e-3)
    assert detect_self_collisions(r, q=res.q) == []
    assert "collision-free" in res.message


# --- obstacle blocks one branch; the clear branch is returned ---------------

def test_obstacle_blocks_one_branch_clear_one_chosen():
    r = _planar_2r()
    target = (1.0, 0.8, 0.0)
    branches = solve_ik_multi(r, "tool", target, seed=1)
    assert len(branches) >= 2                                    # two postures exist

    # Place an obstacle on the first branch's elbow (the l2-link origin).
    elbow0 = forward_kinematics(r, branches[0].q)["l2"][:3, 3]
    obs = [box_obstacle("blob", center=tuple(elbow0), size=(0.3, 0.3, 1.0))]

    # Precondition: that branch really collides, and some other branch is clear.
    assert detect_obstacle_collisions(r, obs, q=branches[0].q) != []
    clear = [b for b in branches if not detect_obstacle_collisions(r, obs, q=b.q)]
    assert clear, "test needs at least one obstacle-free branch"

    free = solve_ik_collision_free(r, "tool", target, obstacles=obs, seed=1)
    assert free.converged
    assert np.allclose(_tcp(r, free.q), target, atol=1e-3)       # still reaches the pose
    assert detect_obstacle_collisions(r, obs, q=free.q) == []    # but clears the obstacle


# --- self-collision (no obstacles) filters a folded branch ------------------

def test_self_colliding_branch_is_filtered_out():
    r = _planar_2r(tool_r=0.2)
    target = (0.5, 0.2, 0.0)                                     # reachable folded & extended
    branches = solve_ik_multi(r, "tool", target, seed=5)
    sc = [detect_self_collisions(r, q=b.q) != [] for b in branches]
    assert any(sc) and not all(sc), "test needs one self-colliding and one clean branch"

    free = solve_ik_collision_free(r, "tool", target, seed=5)
    assert free.converged
    assert detect_self_collisions(r, q=free.q) == []            # the clean branch
    assert np.allclose(_tcp(r, free.q), target, atol=1e-3)


# --- honest failure reporting -----------------------------------------------

def test_no_collision_free_solution_reports_all_in_collision():
    r = _planar_2r()
    target = (1.0, 0.0, 0.0)
    cage = [box_obstacle("cage", center=(1.0, 0.0, 0.0), size=(3.5, 3.5, 1.0))]
    # every posture that reaches the target tool position lands inside the cage
    res = solve_ik_collision_free(r, "tool", target, obstacles=cage, seed=2)
    assert not res.converged
    assert "all in collision" in res.message
    assert np.allclose(_tcp(r, res.q), target, atol=1e-3)        # still reaches (just colliding)


def test_unreachable_pose_returns_empty_nonconverged():
    r = _planar_2r(1.0, 1.0)                                     # max reach 2.0
    res = solve_ik_collision_free(r, "tool", (5.0, 0.0, 0.0), seed=0)
    assert not res.converged
    assert res.q == {}
    assert "no IK solution converged" in res.message


# --- parity with plain IK ---------------------------------------------------

def test_reaches_same_pose_as_plain_ik_when_unobstructed():
    r = _planar_2r(tool_r=0.05)
    target = (0.9, 0.9, 0.0)
    plain = solve_ik(r, "tool", target, seed=3)
    free = solve_ik_collision_free(r, "tool", target, seed=3)
    assert plain.converged and free.converged
    assert np.allclose(_tcp(r, plain.q), _tcp(r, free.q), atol=1e-3)   # same pose
    assert detect_self_collisions(r, q=free.q) == []
