"""Numerical simulation of the symbolic dynamics — the ``[dynamics]`` extra.

Where :class:`fieldpilot_urdf.dynamics.SymbolicDynamics` builds the *instantaneous*
equations of motion (``M(q)·q̈ = F(q, q̇, τ)``), this module rolls them forward in
time and inverts them:

* :func:`integrate_dynamics` — integrate the forward dynamics from an initial
  state under an applied-torque law, returning the state :class:`Trajectory`.
* :func:`inverse_dynamics` — the torques that produce a desired acceleration
  (``τ = M(q)·q̈ − F(q, q̇, 0)``).
* :func:`gravity_torques` — the static holding torques at a configuration
  (inverse dynamics with ``q̇ = q̈ = 0``).

It's a small, pure-NumPy layer over the callables `SymbolicDynamics` already
lambdifies — no SciPy needed (RK4 and semi-implicit Euler are inlined), though
`solve_ivp` remains a fine alternative for stiff problems. Everything works in
the actuated-joint vector space ordered by ``dyn.actuated_joint_ids``; configs
are exchanged as ``{joint_id: value}`` dicts to match the rest of the package.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from pydantic import BaseModel

from .dynamics import SymbolicDynamics

# An applied-torque law: a constant per-joint dict, a constant vector, or a
# callable (t, q_vec, u_vec) -> vector. None means zero torque (passive).
TorqueLaw = Union[None, dict, "np.ndarray", list, tuple, Callable]


class Trajectory(BaseModel):
    """A sampled state trajectory. `q[k]` / `u[k]` are the joint positions /
    velocities at `times[k]`, ordered by `joint_ids`. `len(times) == steps + 1`
    (the initial state is row 0)."""
    joint_ids: list[str]
    times: list[float]
    q: list[list[float]]
    u: list[list[float]]

    def as_dicts(self) -> list[dict[str, float]]:
        """Each sample's positions as a `{joint_id: value}` dict."""
        return [dict(zip(self.joint_ids, row)) for row in self.q]

    def final_q(self) -> dict[str, float]:
        return dict(zip(self.joint_ids, self.q[-1]))

    def final_u(self) -> dict[str, float]:
        return dict(zip(self.joint_ids, self.u[-1]))


def _vec(value: Union[None, dict, np.ndarray, list, tuple],
         joint_ids: list[str]) -> np.ndarray:
    """Coerce a dict/sequence/None into an ordered float vector over joint_ids."""
    n = len(joint_ids)
    if value is None:
        return np.zeros(n)
    if isinstance(value, dict):
        return np.array([float(value.get(jid, 0.0)) for jid in joint_ids])
    arr = np.asarray(value, dtype=float).ravel()
    if arr.shape != (n,):
        raise ValueError(f"expected {n} values for {n} actuated joints, got {arr.shape[0]}")
    return arr


def _torque_fn(tau: TorqueLaw, joint_ids: list[str]) -> Callable[[float, np.ndarray, np.ndarray], np.ndarray]:
    """Normalise any TorqueLaw into a callable (t, q, u) -> vector."""
    if callable(tau):
        return lambda t, q, u: _vec(tau(t, q, u), joint_ids)
    const = _vec(tau, joint_ids)  # None / dict / vector → fixed vector
    return lambda t, q, u: const


def inverse_dynamics(dyn: SymbolicDynamics) -> Callable[..., np.ndarray]:
    """Return a callable ``tau(q, u, qdd) -> ndarray`` giving the joint torques
    that realise acceleration ``qdd`` at state ``(q, u)``.

    Solves the equation of motion for the input torque: with
    ``M(q)·q̈ = τ + b(q, q̇)`` (the applied torque enters the generalized forces
    additively), ``τ = M(q)·q̈ − b`` where ``b = F(q, q̇, 0)`` is the bias
    (Coriolis/centrifugal + gravity) term. Each argument is a `{joint_id: value}`
    dict or an ordered vector.
    """
    joint_ids = list(dyn.actuated_joint_ids)
    M_fn = dyn.lambdify_mass_matrix()
    F_fn = dyn.lambdify_forcing()
    zero = np.zeros(len(joint_ids))

    def tau(q, u, qdd) -> np.ndarray:
        qv = _vec(q, joint_ids)
        uv = _vec(u, joint_ids)
        av = _vec(qdd, joint_ids)
        M = np.atleast_2d(np.asarray(M_fn(qv), dtype=float))
        bias = np.asarray(F_fn(qv, uv, zero), dtype=float).ravel()
        return M @ av - bias

    return tau


def gravity_torques(dyn: SymbolicDynamics, q) -> np.ndarray:
    """The static holding torques at configuration ``q`` — the torque needed to
    keep the robot at rest there (``q̇ = q̈ = 0``), i.e. gravity compensation."""
    return inverse_dynamics(dyn)(q, None, None)


def integrate_dynamics(
    dyn: SymbolicDynamics,
    q0,
    u0=None,
    *,
    dt: float,
    steps: int,
    tau: TorqueLaw = None,
    method: str = "rk4",
) -> Trajectory:
    """Integrate the forward dynamics from ``(q0, u0)`` for ``steps`` steps of
    ``dt`` under the applied-torque law ``tau``.

    `q0`/`u0` are `{joint_id: value}` dicts or ordered vectors (`u0=None` → at
    rest). `tau` is None (passive), a constant per-joint dict/vector, or a
    callable ``(t, q_vec, u_vec) -> vector``. `method` is ``"rk4"`` (default,
    4th-order) or ``"euler"`` (semi-implicit / symplectic — cheaper, better
    long-run energy behaviour at small `dt`). Returns the sampled
    :class:`Trajectory` including the initial state.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    joint_ids = list(dyn.actuated_joint_ids)
    n = len(joint_ids)

    q = _vec(q0, joint_ids)
    u = _vec(u0, joint_ids)
    tau_fn = _torque_fn(tau, joint_ids)

    times = [0.0]
    q_hist = [q.tolist()]
    u_hist = [u.tolist()]

    # Degenerate (all-fixed) robot: nothing to integrate, just hold the state.
    if n == 0:
        for k in range(1, steps + 1):
            times.append(k * dt)
            q_hist.append([])
            u_hist.append([])
        return Trajectory(joint_ids=joint_ids, times=times, q=q_hist, u=u_hist)

    fwd = dyn.lambdify_forward_dynamics()

    def accel(t: float, qv: np.ndarray, uv: np.ndarray) -> np.ndarray:
        return np.asarray(fwd(qv, uv, tau_fn(t, qv, uv)), dtype=float).ravel()

    if method not in ("rk4", "euler"):
        raise ValueError(f"unknown method {method!r} (use 'rk4' or 'euler')")

    t = 0.0
    for k in range(steps):
        if method == "euler":
            # Semi-implicit (symplectic) Euler: advance velocity, then position.
            a = accel(t, q, u)
            u = u + dt * a
            q = q + dt * u
        else:
            # Classic RK4 on the state y = (q, u), ẏ = (u, accel).
            k1q, k1u = u, accel(t, q, u)
            k2q, k2u = u + 0.5 * dt * k1u, accel(t + 0.5 * dt, q + 0.5 * dt * k1q, u + 0.5 * dt * k1u)
            k3q, k3u = u + 0.5 * dt * k2u, accel(t + 0.5 * dt, q + 0.5 * dt * k2q, u + 0.5 * dt * k2u)
            k4q, k4u = u + dt * k3u, accel(t + dt, q + dt * k3q, u + dt * k3u)
            q = q + (dt / 6.0) * (k1q + 2 * k2q + 2 * k3q + k4q)
            u = u + (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
        t = (k + 1) * dt
        times.append(t)
        q_hist.append(q.tolist())
        u_hist.append(u.tolist())

    return Trajectory(joint_ids=joint_ids, times=times, q=q_hist, u=u_hist)
