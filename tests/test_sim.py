"""PyBullet sim wrapper tests (the [sim] extra). The mesh-path rewrite is a pure
unit test; the rest need PyBullet, and the cross-validation also needs SymPy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fieldpilot_urdf.models import (
    Box, Collision, Inertia, Inertial, Joint, JointLimit, Link, Mesh, Origin, Robot,
)
from fieldpilot_urdf.sim import rewrite_mesh_paths


def _lim():
    return JointLimit(lower=-3.0, upper=3.0, effort=50, velocity=20)


def _inr(com=(0.5, 0, 0)):
    return Inertial(origin=Origin(xyz=com), mass=1.0,
                    inertia=Inertia(ixx=0.05, iyy=0.05, izz=0.05))


# --- mesh-path bridge (no PyBullet needed) ---------------------------------

def _mesh_robot(filename):
    return Robot(
        name="m",
        links=[Link(name="base", inertial=_inr()),
               Link(name="tool", inertial=_inr(),
                    collisions=[Collision(geometry=Mesh(filename=filename))])],
        joints=[Joint(name="j", type="fixed", parent="base", child="tool")],
    )


def test_rewrite_package_uri_to_abs():
    r = rewrite_mesh_paths(_mesh_robot("package://mypkg/meshes/box.stl"),
                           mesh_dir=Path("/data/meshes"))
    fn = r.links[1].collisions[0].geometry.filename
    assert fn == str(Path("/data/meshes/mypkg/meshes/box.stl").resolve())


def test_rewrite_relative_against_mesh_dir():
    r = rewrite_mesh_paths(_mesh_robot("meshes/box.stl"), mesh_dir=Path("/data"))
    assert r.links[1].collisions[0].geometry.filename == str(Path("/data/meshes/box.stl").resolve())


def test_rewrite_absolute_left_untouched():
    r = rewrite_mesh_paths(_mesh_robot("/abs/box.stl"), mesh_dir=Path("/data"))
    assert r.links[1].collisions[0].geometry.filename == "/abs/box.stl"


def test_rewrite_noop_without_mesh_dir():
    r = rewrite_mesh_paths(_mesh_robot("package://p/m.stl"), mesh_dir=None)
    assert r.links[1].collisions[0].geometry.filename == "package://p/m.stl"


# --- PyBullet load / control (needs the [sim] extra) -----------------------

pytest.importorskip("pybullet")  # noqa
from fieldpilot_urdf.sim import PyBulletSim  # noqa: E402


def _box_robot():
    return Robot(
        name="prim",
        links=[
            Link(name="base", inertial=_inr((0, 0, 0)),
                 collisions=[Collision(geometry=Box(size=(0.2, 0.2, 0.2)))]),
            Link(name="arm", inertial=_inr((0, 0, 0)),
                 collisions=[Collision(geometry=Box(size=(0.1, 0.1, 0.4)))]),
        ],
        joints=[Joint(name="j", type="revolute", parent="base", child="arm",
                      origin=Origin(xyz=(0, 0, 0.2)), axis=(0, 0, 1), limit=_lim())],
    )


def test_loads_primitive_and_maps_joints():
    with PyBulletSim(_box_robot()) as sim:
        assert sim.body >= 0
        assert set(sim.joints) == {"j"}
        assert "arm" in sim.links
        shapes = sim._pb.getCollisionShapeData(sim.body, sim.links["arm"], physicsClientId=sim.cid)
        assert shapes and shapes[0][2] == sim._pb.GEOM_BOX


def test_loads_mesh_robot_via_bridge(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    stl = tmp_path / "mypkg" / "meshes" / "box.stl"
    stl.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(0.2, 0.2, 0.2)).export(stl)

    robot = _mesh_robot("package://mypkg/meshes/box.stl")
    with PyBulletSim(robot, mesh_dir=tmp_path) as sim:
        shapes = sim._pb.getCollisionShapeData(sim.body, sim.links["tool"], physicsClientId=sim.cid)
        assert shapes and shapes[0][2] == sim._pb.GEOM_MESH   # mesh actually loaded


def test_position_control_reaches_target():
    # Joint about +z under gravity -z feels no gravity torque -> control converges.
    with PyBulletSim(_box_robot(), timestep=1 / 240) as sim:
        sim.set_position_targets({"j": 0.6}, force=200.0)
        sim.step(600)
        assert abs(sim.joint_states()["j"][0] - 0.6) < 1e-2


# --- cross-validation: PyBullet free-fall vs Kane forward dynamics ----------

def _two_link_arm():
    return Robot(
        name="arm",
        links=[Link(name="base"), Link(name="l1", inertial=_inr()),
               Link(name="l2", inertial=_inr())],
        joints=[
            Joint(name="j1", type="revolute", parent="base", child="l1",
                  origin=Origin(xyz=(0, 0, 0)), axis=(0, 1, 0), limit=_lim()),
            Joint(name="j2", type="revolute", parent="l1", child="l2",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 1, 0), limit=_lim()),
        ],
    )


def test_freefall_matches_kane():
    pytest.importorskip("sympy")
    from fieldpilot_urdf.dynamics import SymbolicDynamics

    robot = _two_link_arm()
    robot.links[0].inertial = _inr((0, 0, 0))   # give the fixed base an inertial
    g = (0.0, 0.0, -9.81)
    q0 = {"j1": 0.2, "j2": 0.3}
    dt, steps = 1.0 / 500.0, 150          # 0.3 s

    # PyBullet free-fall
    with PyBulletSim(robot, gravity=g, timestep=dt) as sim:
        for j, v in q0.items():
            sim.reset_joint(j, v, 0.0)
        sim.free()
        sim.step(steps)
        pb = sim.joint_states()

    # Kane forward dynamics, same dt, semi-implicit Euler
    dyn = SymbolicDynamics(robot, gravity=g)
    fwd = dyn.lambdify_forward_dynamics()
    order = dyn.actuated_joint_ids
    q = np.array([q0[j] for j in order]); qd = np.zeros(len(order))
    for _ in range(steps):
        qdd = np.asarray(fwd(q, qd, [0.0] * len(order)), float).ravel()
        qd = qd + dt * qdd
        q = q + dt * qd
    kane = dict(zip(order, q))

    # With matching dt and the inertia honoured (URDF_USE_INERTIA_FROM_FILE),
    # PyBullet and the Kane forward dynamics track to ~1e-5 over 0.3 s. A tight
    # bound here is what caught the missing-inertia-flag bug during development.
    for j in order:
        assert abs(pb[j][0] - kane[j]) < 1e-3, (j, pb[j][0], kane[j])
