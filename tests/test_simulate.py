"""Tests for src/fieldpilot_urdf/simulate.py — numerical simulation of the
symbolic dynamics. Needs SymPy (the [dynamics] extra); skipped without it.

Run: python3 -m pytest tests/test_simulate.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from fieldpilot_urdf.models import (
    Inertia, Inertial, Joint, JointLimit, Link, Origin, Robot,
)

sympy = pytest.importorskip("sympy")  # noqa: F841 — gate the whole module

from fieldpilot_urdf.dynamics import SymbolicDynamics  # noqa: E402
from fieldpilot_urdf.simulate import (  # noqa: E402
    Trajectory, gravity_torques, integrate_dynamics, inverse_dynamics,
)


def _lim(lo=-3.14, hi=3.14):
    return JointLimit(lower=lo, upper=hi, effort=10, velocity=5)


def _pendulum(axis=(0, 1, 0), m=1.0, L=0.75):
    """Single revolute joint at the origin; point mass m at (L, 0, 0)."""
    return Robot(
        name="pend",
        links=[
            Link(name="base"),
            Link(name="bob", inertial=Inertial(
                origin=Origin(xyz=(L, 0, 0)), mass=m,
                inertia=Inertia(ixx=0, iyy=0, izz=0))),
        ],
        joints=[Joint(name="j", type="revolute", parent="base", child="bob",
                      origin=Origin(xyz=(0, 0, 0)), axis=axis, limit=_lim())],
    )


def _two_link_arm():
    def inr():
        return Inertial(origin=Origin(xyz=(0.5, 0, 0)), mass=1.0,
                        inertia=Inertia(ixx=0.0, iyy=0.1, izz=0.1))
    return Robot(
        name="arm",
        links=[Link(name="base"), Link(name="l1", inertial=inr()),
               Link(name="l2", inertial=inr())],
        joints=[
            Joint(name="j1", type="revolute", parent="base", child="l1",
                  origin=Origin(xyz=(0, 0, 0)), axis=(0, 1, 0), limit=_lim()),
            Joint(name="j2", type="revolute", parent="l1", child="l2",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 1, 0), limit=_lim()),
        ],
    )


# --- inverse dynamics ------------------------------------------------------

def test_inverse_dynamics_round_trips_forward():
    """tau that realises a desired qdd, fed back through forward dynamics, must
    recover that qdd — the defining property, independent of sign conventions."""
    dyn = SymbolicDynamics(_two_link_arm(), gravity=(0, 0, -9.81))
    inv = inverse_dynamics(dyn)
    fwd = dyn.lambdify_forward_dynamics()
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-1, 1, 2)
        u = rng.uniform(-1, 1, 2)
        qdd_des = rng.uniform(-2, 2, 2)
        tau = inv(q, u, qdd_des)
        qdd_got = np.asarray(fwd(q, u, tau), dtype=float).ravel()
        assert np.allclose(qdd_got, qdd_des, atol=1e-7)


def test_inverse_dynamics_accepts_dicts():
    dyn = SymbolicDynamics(_two_link_arm(), gravity=(0, 0, -9.81))
    inv = inverse_dynamics(dyn)
    ids = dyn.actuated_joint_ids
    tau_dict = inv({ids[0]: 0.3, ids[1]: -0.2}, None, {ids[0]: 1.0, ids[1]: 0.5})
    tau_vec = inv([0.3, -0.2], [0, 0], [1.0, 0.5])
    assert np.allclose(tau_dict, tau_vec)


# --- gravity compensation --------------------------------------------------

def test_gravity_torques_zero_acceleration():
    """Holding torques at a config must produce zero acceleration there."""
    dyn = SymbolicDynamics(_pendulum(axis=(0, 1, 0)), gravity=(0, 0, -9.81))
    fwd = dyn.lambdify_forward_dynamics()
    for q in (0.0, 0.7, -1.1):
        tau = gravity_torques(dyn, [q])
        qdd = np.asarray(fwd([q], [0.0], tau), dtype=float).ravel()
        assert abs(qdd[0]) < 1e-9


def test_gravity_torques_hold_static_under_integration():
    """Applying the (constant) holding torque keeps a pendulum near rest."""
    dyn = SymbolicDynamics(_pendulum(axis=(0, 1, 0)), gravity=(0, 0, -9.81))
    q0 = [0.6]
    tau_hold = gravity_torques(dyn, q0)
    traj = integrate_dynamics(dyn, q0, dt=1e-3, steps=200, tau=tau_hold)
    # Started at rest with gravity exactly cancelled → drifts only by O(dt²·motion).
    assert abs(traj.q[-1][0] - 0.6) < 1e-3


# --- forward integration ---------------------------------------------------

def _pendulum_energy(dyn, m, L, g, q, u):
    M = float(np.atleast_2d(dyn.lambdify_mass_matrix()([q]))[0, 0])
    ke = 0.5 * M * u * u
    R, p = dyn.link_pose("bob", q={"j": q})
    com_z = (p + R @ np.array([L, 0, 0]))[2]  # mass sits at (L,0,0) in the bob frame
    return ke + m * g * com_z


def test_passive_pendulum_conserves_energy_rk4():
    m, L, g = 1.0, 0.75, 9.81
    dyn = SymbolicDynamics(_pendulum(axis=(0, 1, 0), m=m, L=L), gravity=(0, 0, -g))
    traj = integrate_dynamics(dyn, {"j": 0.0}, dt=5e-4, steps=1200, method="rk4")
    e0 = _pendulum_energy(dyn, m, L, g, traj.q[0][0], traj.u[0][0])
    e1 = _pendulum_energy(dyn, m, L, g, traj.q[-1][0], traj.u[-1][0])
    assert abs(e1 - e0) < 1e-3 * max(1.0, abs(e0))


def test_passive_pendulum_falls():
    """From horizontal at rest under gravity, the bob swings down (|q| grows)."""
    dyn = SymbolicDynamics(_pendulum(axis=(0, 1, 0)), gravity=(0, 0, -9.81))
    traj = integrate_dynamics(dyn, {"j": 0.0}, dt=1e-3, steps=300)
    assert abs(traj.q[-1][0]) > 0.1


def test_euler_method_runs_and_conserves_reasonably():
    m, L, g = 1.0, 0.75, 9.81
    dyn = SymbolicDynamics(_pendulum(axis=(0, 1, 0), m=m, L=L), gravity=(0, 0, -g))
    traj = integrate_dynamics(dyn, {"j": 0.0}, dt=1e-4, steps=3000, method="euler")
    e0 = _pendulum_energy(dyn, m, L, g, traj.q[0][0], traj.u[0][0])
    e1 = _pendulum_energy(dyn, m, L, g, traj.q[-1][0], traj.u[-1][0])
    assert abs(e1 - e0) < 1e-2 * max(1.0, abs(e0))


def test_torque_callable_drives_motion():
    """A callable torque law (here a constant push) is honoured."""
    dyn = SymbolicDynamics(_pendulum(axis=(0, 0, 1)), gravity=(0, 0, -9.81))
    # axis=z → gravity does no work; a constant torque should accelerate it.
    traj = integrate_dynamics(dyn, {"j": 0.0}, dt=1e-3, steps=200,
                              tau=lambda t, q, u: [0.5])
    assert traj.u[-1][0] > 0.1  # spun up by the applied torque


# --- trajectory shape / API ------------------------------------------------

def test_trajectory_shape_and_helpers():
    dyn = SymbolicDynamics(_two_link_arm(), gravity=(0, 0, -9.81))
    traj = integrate_dynamics(dyn, {"j1": 0.1, "j2": -0.2}, dt=1e-3, steps=10)
    assert isinstance(traj, Trajectory)
    assert len(traj.times) == 11 and len(traj.q) == 11 and len(traj.u) == 11
    assert all(len(row) == 2 for row in traj.q)
    assert traj.joint_ids == list(dyn.actuated_joint_ids)
    assert set(traj.final_q()) == set(traj.joint_ids)
    assert len(traj.as_dicts()) == 11
    # Initial row matches the requested start (ordered by joint_ids).
    assert traj.as_dicts()[0] == pytest.approx(
        dict(zip(traj.joint_ids, traj.q[0])))


def test_steps_zero_returns_initial_only():
    dyn = SymbolicDynamics(_pendulum(), gravity=(0, 0, -9.81))
    traj = integrate_dynamics(dyn, {"j": 0.4}, dt=1e-3, steps=0)
    assert traj.times == [0.0]
    assert traj.q == [[0.4]] and traj.u == [[0.0]]


def test_degenerate_all_fixed_robot():
    r = Robot(
        name="rigid",
        links=[Link(name="base"), Link(name="l1", inertial=Inertial(mass=1.0))],
        joints=[Joint(name="j", type="fixed", parent="base", child="l1")],
    )
    dyn = SymbolicDynamics(r)
    traj = integrate_dynamics(dyn, {}, dt=1e-2, steps=3)
    assert traj.joint_ids == []
    assert len(traj.times) == 4
    assert traj.q == [[], [], [], []]


# --- error handling --------------------------------------------------------

def test_unknown_method_raises():
    dyn = SymbolicDynamics(_pendulum(), gravity=(0, 0, -9.81))
    with pytest.raises(ValueError):
        integrate_dynamics(dyn, {"j": 0.0}, dt=1e-3, steps=5, method="midpoint")


def test_negative_steps_raises():
    dyn = SymbolicDynamics(_pendulum(), gravity=(0, 0, -9.81))
    with pytest.raises(ValueError):
        integrate_dynamics(dyn, {"j": 0.0}, dt=1e-3, steps=-1)


def test_wrong_vector_length_raises():
    dyn = SymbolicDynamics(_two_link_arm(), gravity=(0, 0, -9.81))
    with pytest.raises(ValueError):
        integrate_dynamics(dyn, [0.1, 0.2, 0.3], dt=1e-3, steps=1)  # 3 != 2 DOF


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
