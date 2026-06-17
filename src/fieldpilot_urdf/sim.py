"""PyBullet numerical simulation of a URDF ``Robot`` — the optional ``[sim]`` extra.

A thin, import-fed wrapper: hand it a :class:`~fieldpilot_urdf.models.Robot` (e.g.
straight from :func:`~fieldpilot_urdf.importer.import_urdf`) and it writes a
PyBullet-loadable URDF, drops it into a simulation, and drives it. The one piece
of real glue is :func:`rewrite_mesh_paths`: PyBullet's ``loadURDF`` can't resolve
``package://`` URIs, so mesh filenames are rewritten to absolute paths inside the
``mesh_dir`` that :func:`~fieldpilot_urdf.importer.fetch_meshes` populated.

PyBullet is a compiled physics engine behind the optional ``[sim]`` extra and is
imported lazily, so ``import fieldpilot_urdf`` stays pure-Python. This wrapper is
deliberately minimal — load, step, control, read joint/link state — not a
general simulation framework; for that, use PyBullet (or MuJoCo / Drake) directly
on the URDF this package imports.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from .importer import package_uri_parts
from .loader import to_xml
from .models import Mesh, Robot

__all__ = ["PyBulletSim", "rewrite_mesh_paths"]


def rewrite_mesh_paths(robot: Robot, mesh_dir: Optional[Path] = None) -> Robot:
    """Return a copy of ``robot`` with mesh filenames made PyBullet-loadable.

    ``package://pkg/sub`` URIs are rewritten to the absolute path
    ``mesh_dir/pkg/sub`` (the layout :func:`fetch_meshes` writes); a plain
    relative path is resolved against ``mesh_dir``; absolute paths and (when
    ``mesh_dir`` is ``None``) everything else are left untouched. Robots with
    only primitive geometry need no ``mesh_dir``.
    """
    r = robot.model_copy(deep=True)
    if mesh_dir is not None:
        mesh_dir = Path(mesh_dir)
    for link in r.links:
        for holder in (*link.visuals, *link.collisions):
            g = holder.geometry
            if not isinstance(g, Mesh):
                continue
            parts = package_uri_parts(g.filename)
            if parts is not None and mesh_dir is not None:
                pkg, sub = parts
                g.filename = str((mesh_dir / pkg / sub).resolve())
            elif (mesh_dir is not None and not parts
                  and not g.filename.startswith("/")):
                g.filename = str((mesh_dir / g.filename).resolve())
    return r


class PyBulletSim:
    """A minimal PyBullet simulation of a URDF ``Robot``.

    Use as a context manager (``with PyBulletSim(robot) as sim: ...``) or call
    :meth:`close` to release the physics client and temp files.
    """

    def __init__(
        self,
        robot: Robot,
        *,
        mesh_dir: Optional[Path] = None,
        gui: bool = False,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        fixed_base: bool = True,
        timestep: float = 1.0 / 240.0,
    ):
        try:
            import pybullet as pb
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyBullet is not installed. Install the sim extra: "
                'pip install "fieldpilot-urdf[sim]"'
            ) from exc

        self._pb = pb
        self.timestep = timestep
        self.cid = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setGravity(*gravity, physicsClientId=self.cid)
        pb.setTimeStep(timestep, physicsClientId=self.cid)

        self._tmp = Path(tempfile.mkdtemp(prefix="fp_sim_"))
        loadable = rewrite_mesh_paths(robot, mesh_dir)
        urdf_path = self._tmp / f"{robot.name or 'robot'}.urdf"
        urdf_path.write_text(to_xml(loadable))
        # URDF_USE_INERTIA_FROM_FILE: honour the link <inertia> tensors. Without
        # it PyBullet *recomputes* inertia from the collision shape (or falls
        # back to a point-mass when there's none), silently diverging from the
        # robot's declared dynamics — a notorious footgun.
        self.body = pb.loadURDF(
            str(urdf_path), useFixedBase=fixed_base,
            flags=pb.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.cid,
        )

        # Map joint/link names to PyBullet indices.
        self.joints: dict[str, int] = {}     # movable joints only
        self.links: dict[str, int] = {}      # child link of each joint
        for i in range(pb.getNumJoints(self.body, physicsClientId=self.cid)):
            info = pb.getJointInfo(self.body, i, physicsClientId=self.cid)
            self.links[info[12].decode()] = i
            if info[2] in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC):
                self.joints[info[1].decode()] = i

    # ------------------------------------------------------------------
    # state / control
    # ------------------------------------------------------------------

    def reset_joint(self, name: str, position: float, velocity: float = 0.0) -> None:
        self._pb.resetJointState(self.body, self.joints[name], position, velocity,
                                 physicsClientId=self.cid)

    def free(self) -> None:
        """Disable joint motors and artificial link damping → pure gravity/inertia
        dynamics (use before a free-fall / passive simulation)."""
        for i in self.joints.values():
            self._pb.setJointMotorControl2(self.body, i, self._pb.VELOCITY_CONTROL,
                                           force=0.0, physicsClientId=self.cid)
        for i in range(-1, self._pb.getNumJoints(self.body, physicsClientId=self.cid)):
            self._pb.changeDynamics(self.body, i, linearDamping=0.0, angularDamping=0.0,
                                    jointDamping=0.0, physicsClientId=self.cid)

    def set_position_targets(self, targets: dict[str, float], *, force: float = 100.0) -> None:
        for name, q in targets.items():
            self._pb.setJointMotorControl2(
                self.body, self.joints[name], self._pb.POSITION_CONTROL,
                targetPosition=q, force=force, physicsClientId=self.cid)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._pb.stepSimulation(physicsClientId=self.cid)

    def joint_states(self) -> dict[str, tuple[float, float]]:
        """Return ``{joint_name: (position, velocity)}`` for movable joints."""
        return {
            name: tuple(self._pb.getJointState(self.body, i, physicsClientId=self.cid)[:2])
            for name, i in self.joints.items()
        }

    def link_pose(self, link_name: str) -> tuple[tuple, tuple]:
        """Return ``(world_position, world_orientation_quat)`` of a link."""
        st = self._pb.getLinkState(self.body, self.links[link_name],
                                   physicsClientId=self.cid)
        return st[0], st[1]

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if getattr(self, "cid", None) is not None:
            try:
                self._pb.disconnect(physicsClientId=self.cid)
            except Exception:  # pragma: no cover - already disconnected
                pass
            self.cid = None
        if getattr(self, "_tmp", None) and self._tmp.exists():
            shutil.rmtree(self._tmp, ignore_errors=True)

    def __enter__(self) -> "PyBulletSim":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
