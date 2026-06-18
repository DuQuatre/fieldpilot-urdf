"""Time-parameterization: turn a geometric path into a timed trajectory.

The planners produce *geometric* paths — :func:`...planning.plan_path` (joint
space) and :func:`...cartesian.plan_cartesian_path` (task space) both return a
``list[dict]`` of waypoints with **no timing**. :func:`time_parameterize` assigns
that path a schedule that respects the joints' velocity limits
(``JointLimit.velocity``, until now unused for motion), producing a
:class:`TimedTrajectory` of positions *and* velocities sampled over time —
which then feeds the dynamics / simulation layer.

The method is a single **trapezoidal velocity profile over the path's
joint-space arc length** ``s``: the cruise speed ``ṡ`` is capped by the most
velocity-constraining segment (conservative, but the per-joint velocity limits
are *always* respected), and an optional ``max_acceleration`` adds smooth
accel / decel ramps. Omit it for a rectangular profile (instantaneous
start/stop). This is single-profile retiming — full per-segment time-optimal
parameterization (TOPP) is out of scope.

Pure NumPy. Continuous joints take the shortest wrapped path between waypoints
(matching the planners' convention).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from pydantic import BaseModel

from .models import Robot


class TimedTrajectory(BaseModel):
    """A time-parameterized path. ``q[k]`` / ``u[k]`` are the joint positions /
    velocities at ``times[k]``, ordered by ``joint_ids``. Mirrors the shape of
    ``simulate.Trajectory`` (so the two interoperate), and adds ``duration`` and
    arbitrary-time sampling (``sample``)."""

    joint_ids: list[str]
    times: list[float]
    q: list[list[float]]
    u: list[list[float]]

    @property
    def duration(self) -> float:
        return self.times[-1] if self.times else 0.0

    def as_dicts(self) -> list[dict[str, float]]:
        """Each sample's positions as a ``{joint_id: value}`` dict — drops
        straight into ``check_trajectory`` / ``forward_kinematics``."""
        return [dict(zip(self.joint_ids, row)) for row in self.q]

    def final_q(self) -> dict[str, float]:
        return dict(zip(self.joint_ids, self.q[-1]))

    def sample(self, t: float) -> dict[str, float]:
        """Joint positions at arbitrary time ``t`` (linearly interpolated between
        stored samples; clamped to ``[0, duration]``)."""
        ts = self.times
        t = max(ts[0], min(ts[-1], float(t)))
        # locate the bracketing sample
        hi = int(np.searchsorted(ts, t, side="left"))
        if hi <= 0:
            row = self.q[0]
            return dict(zip(self.joint_ids, row))
        if hi >= len(ts):
            return dict(zip(self.joint_ids, self.q[-1]))
        lo = hi - 1
        span = ts[hi] - ts[lo]
        f = 0.0 if span <= 0 else (t - ts[lo]) / span
        row = [a + f * (b - a) for a, b in zip(self.q[lo], self.q[hi])]
        return dict(zip(self.joint_ids, row))


def _continuous_mask(robot: Robot, joint_ids: list[str]) -> np.ndarray:
    by_name = {j.name: j for j in robot.joints}
    return np.array([by_name.get(jn) is not None and by_name[jn].type == "continuous"
                     for jn in joint_ids])


def _velocity_caps(robot: Robot, joint_ids: list[str]) -> np.ndarray:
    """Per-joint velocity limit (``inf`` where a joint has none / a non-positive
    limit, i.e. it does not constrain the path speed)."""
    by_name = {j.name: j for j in robot.joints}
    caps = []
    for jn in joint_ids:
        j = by_name.get(jn)
        v = j.limit.velocity if (j is not None and j.limit is not None) else 0.0
        caps.append(v if v and v > 0.0 else math.inf)
    return np.array(caps, dtype=float)


def _wrapped_delta(d: np.ndarray, is_continuous: np.ndarray) -> np.ndarray:
    """Per-joint step with continuous joints taking the shortest wrapped path
    (result in [-π, π]) — mirrors planning._diff."""
    if is_continuous.any():
        d = d.copy()
        d[is_continuous] = (d[is_continuous] + math.pi) % (2 * math.pi) - math.pi
    return d


def time_parameterize(
    robot: Robot,
    path: list[dict[str, float]],
    *,
    velocity_scale: float = 1.0,
    max_acceleration: Optional[float] = None,
    dt: float = 0.05,
) -> TimedTrajectory:
    """Assign timing to a geometric ``path`` (a list of joint configs, e.g. the
    output of ``plan_path`` / ``plan_cartesian_path``), respecting each joint's
    velocity limit. Returns a :class:`TimedTrajectory` sampled every ``dt``.

    A single trapezoidal velocity profile is laid over the path's joint-space
    arc length: cruise speed is set by the tightest per-joint velocity limit
    along the path, scaled by ``velocity_scale`` ∈ (0, 1]. ``max_acceleration``
    (a scalar joint accel cap, applied per joint) adds smooth ramps — omit it for
    a rectangular profile. Velocities ``u`` are the exact derivatives of the
    sampled positions under the profile (zero at the endpoints when ramped).

    Raises ``ValueError`` if the path is empty, ``dt`` ≤ 0, ``velocity_scale`` ≤ 0,
    or no joint on the path carries a velocity limit (nothing to time against).
    """
    if not path:
        raise ValueError("path is empty")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if velocity_scale <= 0:
        raise ValueError("velocity_scale must be positive")

    joint_ids = list(path[0].keys())
    is_cont = _continuous_mask(robot, joint_ids)
    vcaps = _velocity_caps(robot, joint_ids)

    Q = np.array([[float(cfg.get(jn, 0.0)) for jn in joint_ids] for cfg in path])
    M = Q.shape[0]

    # Single-waypoint or zero-motion path: one sample, no motion.
    seg = np.array([_wrapped_delta(Q[k + 1] - Q[k], is_cont) for k in range(M - 1)]) \
        if M > 1 else np.zeros((0, len(joint_ids)))
    seg_len = np.linalg.norm(seg, axis=1) if M > 1 else np.zeros(0)
    S = float(seg_len.sum())
    if S <= 1e-12:
        return TimedTrajectory(
            joint_ids=joint_ids, times=[0.0],
            q=[Q[0].tolist()], u=[[0.0] * len(joint_ids)],
        )

    s_knots = np.concatenate([[0.0], np.cumsum(seg_len)])

    # Map per-joint velocity (and accel) limits to a cap on the path speed ṡ.
    # On segment k, joint j moves at (seg[k,j]/seg_len[k])·ṡ; require ≤ vcap_j.
    sdot_cap = math.inf
    sddot_cap = math.inf
    for k in range(len(seg_len)):
        if seg_len[k] <= 1e-12:
            continue
        direction = np.abs(seg[k] / seg_len[k])
        moving = direction > 1e-12
        if np.any(moving & np.isfinite(vcaps)):
            sel = moving & np.isfinite(vcaps)
            sdot_cap = min(sdot_cap, float(np.min(vcaps[sel] / direction[sel])))
        if max_acceleration is not None and np.any(moving):
            sddot_cap = min(sddot_cap, float(max_acceleration / np.max(direction[moving])))

    if not math.isfinite(sdot_cap):
        raise ValueError("no joint on the path carries a velocity limit to time against")

    v_s = sdot_cap * float(velocity_scale)

    # Build the arc-length profile s(t): rectangular (no accel cap) or trapezoidal.
    ramped = max_acceleration is not None and math.isfinite(sddot_cap) and sddot_cap > 0
    if not ramped:
        T = S / v_s

        def s_of(t: float) -> float:
            return min(S, v_s * t)

        def sdot_of(t: float) -> float:
            return v_s if 0.0 <= t < T else 0.0
    else:
        a_s = sddot_cap
        t_acc = v_s / a_s
        d_acc = 0.5 * a_s * t_acc * t_acc          # distance to reach v_s
        if 2.0 * d_acc <= S:                        # trapezoid (reaches cruise)
            d_cruise = S - 2.0 * d_acc
            t_cruise = d_cruise / v_s
            T = 2.0 * t_acc + t_cruise
        else:                                       # triangle (never reaches v_s)
            v_peak = math.sqrt(a_s * S)
            t_acc = v_peak / a_s
            d_acc = 0.5 * S
            t_cruise = 0.0
            v_s = v_peak
            T = 2.0 * t_acc

        t_dec0 = t_acc + t_cruise

        def s_of(t: float) -> float:
            if t <= 0.0:
                return 0.0
            if t < t_acc:
                return 0.5 * a_s * t * t
            if t < t_dec0:
                return d_acc + v_s * (t - t_acc)
            if t < T:
                td = T - t
                return S - 0.5 * a_s * td * td
            return S

        def sdot_of(t: float) -> float:
            if t <= 0.0 or t >= T:
                return 0.0
            if t < t_acc:
                return a_s * t
            if t < t_dec0:
                return v_s
            return a_s * (T - t)

    # Sample the profile every dt (always include the exact end time T).
    n = max(1, int(math.ceil(T / dt)))
    sample_ts = [i * dt for i in range(n)] + [T]

    times: list[float] = []
    qs: list[list[float]] = []
    us: list[list[float]] = []
    for t in sample_ts:
        s = min(S, max(0.0, s_of(t)))
        sd = sdot_of(t)
        k = int(np.searchsorted(s_knots, s, side="right") - 1)
        k = max(0, min(k, len(seg_len) - 1))
        # skip any zero-length segments we may have landed exactly on
        while k < len(seg_len) - 1 and seg_len[k] <= 1e-12:
            k += 1
        L = seg_len[k]
        f = 0.0 if L <= 1e-12 else (s - s_knots[k]) / L
        q_t = Q[k] + f * seg[k]
        u_t = (seg[k] / L) * sd if L > 1e-12 else np.zeros(len(joint_ids))
        times.append(float(t))
        qs.append(q_t.tolist())
        us.append(u_t.tolist())

    # Pin the final sample exactly to the path end, at rest.
    qs[-1] = Q[-1].tolist()
    us[-1] = [0.0] * len(joint_ids)

    return TimedTrajectory(joint_ids=joint_ids, times=times, q=qs, u=us)
