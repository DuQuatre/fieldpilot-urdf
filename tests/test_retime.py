"""Time-parameterization: turn a geometric path into a timed trajectory under
joint velocity limits. Anchor checks: per-joint velocity limits are never
exceeded, every sample lies on the original path, the endpoints match, and the
output feeds the project's own validators.
"""
from __future__ import annotations

import numpy as np

from fieldpilot_urdf.models import Joint, JointLimit, Link, Origin, Robot
from fieldpilot_urdf.retime import TimedTrajectory, time_parameterize
from fieldpilot_urdf.trajectory import check_trajectory


def _arm(v1=1.0, v2=2.0):
    """Two revolute joints with distinct velocity limits, +X links."""
    return Robot(
        name="a",
        links=[Link(name="base"), Link(name="l1"), Link(name="tool")],
        joints=[
            Joint(name="j1", type="revolute", parent="base", child="l1",
                  origin=Origin(xyz=(0, 0, 0)), axis=(0, 0, 1),
                  limit=JointLimit(lower=-5, upper=5, effort=1.0, velocity=v1)),
            Joint(name="j2", type="revolute", parent="l1", child="tool",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 0, 1),
                  limit=JointLimit(lower=-5, upper=5, effort=1.0, velocity=v2)),
        ],
    )


def _max_abs_velocity(tt: TimedTrajectory) -> dict[str, float]:
    U = np.abs(np.array(tt.u))
    return dict(zip(tt.joint_ids, U.max(axis=0)))


def _numerical_velocity(tt: TimedTrajectory) -> np.ndarray:
    """Finite-difference dq/dt between consecutive samples (per joint)."""
    Q = np.array(tt.q); T = np.array(tt.times)
    dT = np.diff(T)[:, None]
    return np.diff(Q, axis=0) / np.where(dT == 0, 1.0, dT)


# --- velocity-limit compliance (the anchor) ---------------------------------

def test_velocity_limits_are_respected():
    r = _arm(v1=1.0, v2=2.0)
    path = [{"j1": 0.0, "j2": 0.0}, {"j1": 1.5, "j2": 0.4}, {"j1": 2.0, "j2": -1.0}]
    tt = time_parameterize(r, path, dt=0.02)
    vmax = _max_abs_velocity(tt)
    assert vmax["j1"] <= 1.0 + 1e-9
    assert vmax["j2"] <= 2.0 + 1e-9
    # the most-constraining joint should actually reach (cruise at) its limit
    assert max(vmax["j1"] / 1.0, vmax["j2"] / 2.0) > 0.99


def test_reported_velocity_matches_finite_difference():
    r = _arm()
    path = [{"j1": 0.0, "j2": 0.0}, {"j1": 1.0, "j2": 0.5}, {"j1": 1.2, "j2": 1.5}]
    tt = time_parameterize(r, path, dt=0.01)
    fd = _numerical_velocity(tt)                 # (n-1)×2
    U = np.array(tt.u)[:-1]                       # drop the pinned-to-rest endpoint
    # within each segment, reported u equals dq/dt; allow slack at the row that
    # straddles a waypoint corner (direction changes there).
    err = np.abs(fd - U).max(axis=1)
    assert np.median(err) < 1e-6
    assert np.mean(err < 1e-6) > 0.8


def test_velocity_scale_slows_proportionally():
    r = _arm()
    path = [{"j1": 0.0, "j2": 0.0}, {"j1": 2.0, "j2": 1.0}]
    full = time_parameterize(r, path)
    half = time_parameterize(r, path, velocity_scale=0.5)
    assert np.isclose(half.duration, 2.0 * full.duration, rtol=1e-6)
    assert max(_max_abs_velocity(half).values()) <= max(_max_abs_velocity(full).values()) + 1e-9


# --- geometry: samples lie on the path --------------------------------------

def test_endpoints_match_path():
    r = _arm()
    path = [{"j1": 0.1, "j2": -0.2}, {"j1": 1.0, "j2": 0.3}, {"j1": 0.5, "j2": 1.0}]
    tt = time_parameterize(r, path, dt=0.03)
    assert tt.as_dicts()[0] == {"j1": 0.1, "j2": -0.2}
    assert np.allclose(list(tt.final_q().values()), [0.5, 1.0])
    assert tt.times[0] == 0.0 and tt.duration > 0.0


def test_samples_lie_on_collinear_path():
    """A straight (single-direction) path -> every sample must be collinear with
    the endpoints, with a monotonically advancing arc position."""
    r = _arm()
    path = [{"j1": 0.0, "j2": 0.0}, {"j1": 1.0, "j2": 0.5}, {"j1": 2.0, "j2": 1.0}]
    tt = time_parameterize(r, path, dt=0.02, max_acceleration=2.0)
    P = np.array(tt.q)
    a, b = P[0], P[-1]
    line = b - a
    prev_t = -1.0
    for p in P:
        t = np.dot(p - a, line) / np.dot(line, line)
        assert np.linalg.norm((p - a) - t * line) < 1e-9   # on the line
        assert t >= prev_t - 1e-9                            # monotonic
        prev_t = t


def test_output_feeds_check_trajectory_clean():
    r = _arm()
    path = [{"j1": 0.0, "j2": 0.0}, {"j1": 1.5, "j2": 1.0}, {"j1": 2.5, "j2": 0.0}]
    tt = time_parameterize(r, path, dt=0.05)
    assert check_trajectory(r, tt.as_dicts()) == []


# --- profile shape ----------------------------------------------------------

def test_ramped_profile_starts_and_ends_at_rest():
    r = _arm()
    path = [{"j1": 0.0, "j2": 0.0}, {"j1": 2.0, "j2": 1.0}]
    tt = time_parameterize(r, path, max_acceleration=1.5, dt=0.02)
    assert np.allclose(tt.u[0], 0.0, atol=1e-9)
    assert np.allclose(tt.u[-1], 0.0, atol=1e-9)
    # velocity ramps up then down -> peak somewhere in the interior
    speed = np.linalg.norm(np.array(tt.u), axis=1)
    assert speed.argmax() not in (0, len(speed) - 1)


def test_acceleration_cap_is_respected_on_straight_path():
    r = _arm()
    path = [{"j1": 0.0, "j2": 0.0}, {"j1": 2.0, "j2": 1.0}]   # single direction, no corners
    a_max = 1.0
    tt = time_parameterize(r, path, max_acceleration=a_max, dt=0.01)
    U = np.array(tt.u); T = np.array(tt.times)
    acc = np.abs(np.diff(U, axis=0) / np.diff(T)[:, None])
    assert acc.max() <= a_max + 1e-6


def test_rectangular_profile_starts_moving_immediately():
    r = _arm()
    path = [{"j1": 0.0, "j2": 0.0}, {"j1": 1.0, "j2": 0.0}]
    tt = time_parameterize(r, path)                # no max_acceleration
    assert abs(tt.u[0][0]) > 1e-6                   # j1 already at speed at t=0
    assert np.isclose(tt.duration, 1.0, rtol=1e-6)  # 1 rad at 1 rad/s


# --- sampling + edge cases --------------------------------------------------

def test_sample_interpolates_between_stored_samples():
    r = _arm()
    path = [{"j1": 0.0, "j2": 0.0}, {"j1": 2.0, "j2": 0.0}]
    tt = time_parameterize(r, path, dt=0.1)
    mid = tt.sample(tt.duration / 2)
    assert np.isclose(mid["j1"], 1.0, atol=1e-9)   # halfway in time == halfway along (rect.)
    assert tt.sample(-5.0) == tt.as_dicts()[0]      # clamps low
    assert tt.sample(1e9) == tt.final_q()           # clamps high


def test_zero_motion_path_returns_single_sample():
    r = _arm()
    tt = time_parameterize(r, [{"j1": 0.3, "j2": -0.1}])
    assert tt.times == [0.0] and tt.duration == 0.0
    assert tt.final_q() == {"j1": 0.3, "j2": -0.1}
    assert np.allclose(tt.u[0], 0.0)


def test_continuous_joint_takes_short_way_around():
    r = Robot(name="c", links=[Link(name="base"), Link(name="l1")],
              joints=[Joint(name="c1", type="continuous", parent="base", child="l1",
                            origin=Origin(xyz=(0, 0, 0)), axis=(0, 0, 1),
                            limit=JointLimit(lower=0, upper=0, effort=1.0, velocity=1.0))])
    # 0.1 -> 6.0 rad: the short way is backwards through 0 (≈ -0.38 rad), not +5.9
    tt = time_parameterize(r, [{"c1": 0.1}, {"c1": 6.0}], dt=0.05)
    span = abs(2 * np.pi - (6.0 - 0.1))
    assert np.isclose(tt.duration, span / 1.0, rtol=1e-6)   # short arc at 1 rad/s


def test_invalid_inputs_raise():
    r = _arm()
    path = [{"j1": 0.0, "j2": 0.0}, {"j1": 1.0, "j2": 0.0}]
    try:
        time_parameterize(r, [])
        assert False, "empty path should raise"
    except ValueError:
        pass
    try:
        time_parameterize(r, path, dt=0)
        assert False, "dt<=0 should raise"
    except ValueError:
        pass
    # a robot whose moving joint has no velocity limit -> nothing to time against
    r2 = Robot(name="n", links=[Link(name="base"), Link(name="l1")],
               joints=[Joint(name="c1", type="continuous", parent="base", child="l1",
                             origin=Origin(xyz=(0, 0, 0)), axis=(0, 0, 1))])
    try:
        time_parameterize(r2, [{"c1": 0.0}, {"c1": 1.0}])
        assert False, "no velocity limit should raise"
    except ValueError:
        pass
