"""Symbolic-dynamics tests (ported from MecAI, re-targeted onto Robot).

The dynamics core needs SymPy; the whole module is skipped without it. The
adapter tests below don't strictly need SymPy but live here for cohesion.
"""
from __future__ import annotations

import numpy as np
import pytest

from fieldpilot_urdf.fk import forward_kinematics
from fieldpilot_urdf.models import (
    Inertia, Inertial, Joint, JointLimit, Link, Origin, Robot,
)
from fieldpilot_urdf._dyn_adapter import UnsupportedSystemError, robot_to_system

sympy = pytest.importorskip("sympy")  # noqa: F841 — gate the whole module

from fieldpilot_urdf.dynamics import SymbolicDynamics  # noqa: E402


def _lim(lo=-3.14, hi=3.14):
    return JointLimit(lower=lo, upper=hi, effort=10, velocity=5)


# --- adapter mapping -------------------------------------------------------

def test_adapter_maps_fields():
    r = Robot(
        name="m",
        links=[
            Link(name="base"),
            Link(name="l1", inertial=Inertial(
                origin=Origin(xyz=(0.1, 0.2, 0.3)),
                mass=2.5,
                inertia=Inertia(ixx=1, iyy=2, izz=3),
            )),
        ],
        joints=[Joint(name="j", type="revolute", parent="base", child="l1",
                      origin=Origin(xyz=(1, 0, 0), rpy=(0, 0, 0.5)),
                      axis=(0, 0, 1), limit=_lim())],
    )
    sysm = robot_to_system(r)
    assert sysm.root == "base"
    assert sysm.links["l1"].mass == 2.5
    assert sysm.links["l1"].com == [0.1, 0.2, 0.3]
    assert sysm.links["l1"].inertia.izz == 3
    assert sysm.links["base"].mass == 0.0  # no inertial -> 0
    j = sysm.joints["j"]
    assert j.origin_xyz == [1.0, 0.0, 0.0]
    assert j.origin_rpy == [0.0, 0.0, 0.5]
    assert j.axis == [0.0, 0.0, 1.0]


def test_adapter_rejects_multi_root():
    r = Robot(name="x", links=[Link(name="a"), Link(name="b")], joints=[])
    with pytest.raises(UnsupportedSystemError):
        robot_to_system(r)


def test_adapter_rejects_inertial_rpy():
    r = Robot(
        name="x",
        links=[
            Link(name="base"),
            Link(name="l1", inertial=Inertial(
                origin=Origin(xyz=(0, 0, 0), rpy=(0.1, 0, 0)), mass=1.0)),
        ],
        joints=[Joint(name="j", type="fixed", parent="base", child="l1")],
    )
    with pytest.raises(UnsupportedSystemError):
        robot_to_system(r)


def test_adapter_rejects_unsupported_joint():
    r = Robot(
        name="x",
        links=[Link(name="base"), Link(name="l1")],
        joints=[Joint(name="j", type="floating", parent="base", child="l1")],
    )
    with pytest.raises(UnsupportedSystemError):
        robot_to_system(r)


# --- FK-consistency gate (the convention check) ----------------------------

def _bent_two_link():
    """Two revolute joints with combined non-zero origin rpy — the case where
    body-fixed vs space-fixed XYZ diverge."""
    return Robot(
        name="bent",
        links=[
            Link(name="base"),
            Link(name="l1", inertial=Inertial(mass=1.0, inertia=Inertia(ixx=0.01, iyy=0.01, izz=0.01))),
            Link(name="l2", inertial=Inertial(mass=1.0, inertia=Inertia(ixx=0.01, iyy=0.01, izz=0.01))),
        ],
        joints=[
            Joint(name="j1", type="revolute", parent="base", child="l1",
                  origin=Origin(xyz=(0.1, 0.2, 0.3), rpy=(0.3, 0.4, 0.5)),
                  axis=(0, 0, 1), limit=_lim()),
            Joint(name="j2", type="revolute", parent="l1", child="l2",
                  origin=Origin(xyz=(0.5, 0.0, 0.0), rpy=(0.1, -0.2, 0.15)),
                  axis=(0, 1, 0), limit=_lim()),
        ],
    )


def test_link_pose_matches_fk():
    robot = _bent_two_link()
    dyn = SymbolicDynamics(robot)
    for q in ({}, {"j1": 0.4, "j2": -0.3}):
        tf = forward_kinematics(robot, q=q)
        for link in ("l1", "l2"):
            R, p = dyn.link_pose(link, q=q)
            assert np.allclose(R, tf[link][:3, :3], atol=1e-9), (link, q)
            assert np.allclose(p, tf[link][:3, 3], atol=1e-9), (link, q)


# --- dynamics correctness --------------------------------------------------

def _pendulum(axis, m=1.0, L=0.75):
    """Single revolute joint at the origin; point mass m at (L,0,0)."""
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


def test_mass_matrix_point_mass():
    # Spin about z: gravity (-z) is along the axis, so M = m*L^2 for all q.
    m, L = 2.0, 0.75
    dyn = SymbolicDynamics(_pendulum((0, 0, 1), m=m, L=L))
    M_fn = dyn.lambdify_mass_matrix()
    for q in (0.0, 0.6, -1.2):
        M = np.asarray(M_fn([q]), dtype=float)
        assert M.shape == (1, 1)
        assert np.isclose(M[0, 0], m * L**2, atol=1e-9)


def test_all_fixed_is_degenerate():
    r = Robot(
        name="rigid",
        links=[Link(name="base"), Link(name="l1", inertial=Inertial(mass=1.0))],
        joints=[Joint(name="j", type="fixed", parent="base", child="l1")],
    )
    dyn = SymbolicDynamics(r)
    assert dyn.n_dof == 0
    assert dyn.mass_matrix.shape == (0, 0)


def test_freefall_conserves_energy():
    """A pendulum swinging about +y under gravity must conserve total energy
    when integrated with zero applied torque."""
    scipy_integrate = pytest.importorskip("scipy.integrate")
    m, L, g = 1.0, 0.75, 9.81
    robot = _pendulum((0, 1, 0), m=m, L=L)
    dyn = SymbolicDynamics(robot, gravity=(0, 0, -g))
    fwd = dyn.lambdify_forward_dynamics()
    M_fn = dyn.lambdify_mass_matrix()

    def energy(q, u):
        M = float(np.asarray(M_fn([q]), dtype=float)[0, 0])
        ke = 0.5 * M * u * u
        _, p = dyn.link_pose("bob", q={"j": q})
        com_z = (p + dyn.link_pose("bob", q={"j": q})[0] @ np.array([L, 0, 0]))[2]
        pe = m * g * com_z
        return ke + pe

    def rhs(_t, y):
        q, u = y
        qdd = float(np.asarray(fwd([q], [u], [0.0]), dtype=float).ravel()[0])
        return [u, qdd]

    y0 = [0.0, 0.0]  # start horizontal, at rest
    e0 = energy(*y0)
    sol = scipy_integrate.solve_ivp(rhs, (0.0, 0.6), y0, max_step=1e-3, rtol=1e-9, atol=1e-12)
    e_end = energy(sol.y[0, -1], sol.y[1, -1])
    assert abs(e_end - e0) < 1e-3 * max(1.0, abs(e0))


# --- tree Lagrangian (step ②) ----------------------------------------------

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


def test_lagrangian_matches_kane():
    """The Lagrangian's equations of motion must agree with the already-validated
    Kane forward dynamics for a tree (no constraints), at sample states."""
    import sympy as sp
    from sympy.physics.mechanics import LagrangesMethod

    dyn = SymbolicDynamics(_two_link_arm(), gravity=(0, 0, -9.81))
    L = dyn.lagrangian(simplify=False)
    LM = LagrangesMethod(L, dyn.q)
    LM.form_lagranges_equations()

    qd = [sp.diff(qi, dyn.t) for qi in dyn.q]
    subs = {qd[i]: dyn.u[i] for i in range(len(qd))}     # q̇ -> Kane's speed symbols
    M_fn = sp.lambdify([dyn.q, dyn.u], LM.mass_matrix.subs(subs), "numpy")
    F_fn = sp.lambdify([dyn.q, dyn.u], LM.forcing.subs(subs), "numpy")

    kane = dyn.lambdify_forward_dynamics()
    zero_tau = [0.0] * dyn.n_dof
    for qv, uv in [([0.3, -0.4], [0.5, 0.2]),
                   ([1.0, 0.7], [-0.3, 0.6]),
                   ([0.0, 0.0], [0.0, 0.0])]:
        M = np.asarray(M_fn(qv, uv), dtype=float)
        F = np.asarray(F_fn(qv, uv), dtype=float).ravel()
        qdd_lag = np.linalg.solve(M, F)
        qdd_kane = kane(qv, uv, zero_tau)
        assert np.allclose(qdd_lag, qdd_kane, atol=1e-9), (qv, uv, qdd_lag, qdd_kane)


def test_lagrangian_zero_at_rest_neutral():
    # At q=0 the arm lies along +x (CoMs at z=0 → V=0) and at rest T=0 → L=0.
    import sympy as sp
    dyn = SymbolicDynamics(_two_link_arm())
    L = dyn.lagrangian(simplify=False)
    qd = [sp.diff(qi, dyn.t) for qi in dyn.q]
    L0 = L.subs({d: 0 for d in qd}).subs({qi: 0 for qi in dyn.q})
    assert abs(float(L0)) < 1e-9


def test_lagrangian_simplify_runs():
    dyn = SymbolicDynamics(_pendulum((0, 1, 0)))
    L = dyn.lagrangian(simplify=True)
    assert L.free_symbols          # a non-trivial expression in q, q̇
