"""Kane's-method symbolic dynamics for a tree-shaped URDF ``Robot``.

Ported from the MecAI project (MIT-licensed) and re-targeted onto
:class:`fieldpilot_urdf.models.Robot` via :mod:`fieldpilot_urdf._dyn_adapter`.
The SymPy frame/body construction is unchanged from MecAI apart from one
correction: joint-origin frames are built with a **space-fixed** ``XYZ``
rotation so they reproduce URDF's ``R = Rz(yaw)·Ry(pitch)·Rx(roll)`` exactly
(see :mod:`fieldpilot_urdf.fk`); MecAI's body-fixed ``XYZ`` only agrees for
single-axis origins.

SymPy lives behind the optional ``[dynamics]`` extra and is imported lazily, so
``import fieldpilot_urdf`` never pulls it in.

For each joint, in topological order from the root link, we add a generalized
coordinate ``q_i`` (and its derivative ``u_i``) and a reference frame offset
from the parent body by the joint origin, then rotated (revolute/continuous) or
translated (prismatic) along the joint axis by ``q_i``. The child link's body
frame is the post-motion joint frame; its CoM lives at ``com`` in that frame
with the link's inertia tensor expressed at the CoM in body coordinates.

The resulting ``mass_matrix`` ``M(q)`` and ``forcing`` vector
``F(q, q̇, τ)`` give ``M(q) q̈ = F = τ − C(q, q̇) q̇ − G(q)``. Gravity is the
only external load applied; ``τ`` is left symbolic so callers can lambdify with
any joint-torque profile.
"""
from __future__ import annotations

from collections.abc import Callable

from ._dyn_adapter import (
    JointType, UnsupportedSystemError, _JointShim, robot_to_system,
)
from .graph import build_graph
from .models import Inertia, Robot

__all__ = ["SymbolicDynamics", "UnsupportedSystemError"]


class SymbolicDynamics:
    """Symbolic dynamics for a serial / tree URDF ``Robot``.

    Construct, then read off ``q``, ``u``, ``mass_matrix``, ``forcing``, or use
    :meth:`lambdify_forward_dynamics` for a NumPy callable ``(q, u, tau) -> qdd``
    suitable for ``scipy.integrate.solve_ivp``. :meth:`link_pose` returns a
    link's world transform at a given configuration (handy for cross-checking
    against :func:`fieldpilot_urdf.fk.forward_kinematics`).
    """

    def __init__(
        self,
        robot: Robot,
        *,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ):
        try:
            import sympy as sp  # noqa: F401
            from sympy.physics.mechanics import KanesMethod  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SymPy is not installed. Install the dynamics extra: "
                'pip install "fieldpilot-urdf[dynamics]"'
            ) from exc

        self._robot = robot
        self.system = robot_to_system(robot)  # validates tree / root / joint types
        self.gravity = gravity

        self._order = self._topological_order()
        self._build()

    # ------------------------------------------------------------------
    # topology
    # ------------------------------------------------------------------

    def _topological_order(self) -> list[str]:
        import networkx as nx
        return list(nx.topological_sort(build_graph(self._robot)))

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        import sympy as sp
        from sympy.physics.mechanics import (
            KanesMethod, Point, ReferenceFrame, RigidBody, dynamicsymbols, inertia,
        )

        self.t = sp.Symbol("t")

        # Joints in topological order (the root link has no parent joint).
        joints_in_order: list[_JointShim] = []
        joint_of_link: dict[str, _JointShim] = {}
        for joint in self.system.joints.values():
            joint_of_link[joint.child] = joint
        for link_id in self._order:
            if link_id == self.system.root:
                continue
            if link_id in joint_of_link:
                joints_in_order.append(joint_of_link[link_id])
        self._joints_in_order = joints_in_order

        # One generalized coordinate per non-fixed joint.
        q_syms: list = []
        u_syms: list = []
        tau_syms: list = []
        actuated_joint_ids: list[str] = []
        for j in joints_in_order:
            if j.type == JointType.FIXED:
                continue
            q = dynamicsymbols(f"q_{j.id}")
            u = dynamicsymbols(f"u_{j.id}")
            tau = sp.Symbol(f"tau_{j.id}")
            q_syms.append(q)
            u_syms.append(u)
            tau_syms.append(tau)
            actuated_joint_ids.append(j.id)
        self.q = q_syms
        self.u = u_syms
        self.tau = tau_syms
        self.actuated_joint_ids = actuated_joint_ids

        # World inertial frame.
        N = ReferenceFrame("N")
        O = Point("O")
        O.set_vel(N, 0)
        self.N = N
        self.O = O

        # Per-link frame + origin point.
        link_frames: dict[str, ReferenceFrame] = {self.system.root: N}
        link_origins: dict[str, Point] = {self.system.root: O}

        q_iter = iter(q_syms)
        u_iter = iter(u_syms)
        for j in joints_in_order:
            parent_frame = link_frames[j.parent]
            parent_origin = link_origins[j.parent]

            # Joint frame: offset by origin_xyz + origin_rpy from parent.
            # SPACE-fixed XYZ reproduces URDF Rz(y)Ry(p)Rx(r) (cf. fk.rpy_to_R);
            # MecAI used "Body" here, which only matches for single-axis rpy.
            joint_frame = parent_frame.orientnew(
                f"J_{j.id}", "Space",
                [sp.Float(j.origin_rpy[0]), sp.Float(j.origin_rpy[1]), sp.Float(j.origin_rpy[2])],
                "XYZ",
            )
            joint_origin = parent_origin.locatenew(
                f"P_{j.id}",
                sp.Float(j.origin_xyz[0]) * parent_frame.x
                + sp.Float(j.origin_xyz[1]) * parent_frame.y
                + sp.Float(j.origin_xyz[2]) * parent_frame.z,
            )
            joint_origin.set_vel(N, joint_origin.pos_from(O).dt(N))

            # Apply the joint motion to obtain the child frame + origin.
            if j.type in (JointType.REVOLUTE, JointType.CONTINUOUS):
                q_i = next(q_iter)
                next(u_iter)
                axis_vec = (
                    sp.Float(j.axis[0]) * joint_frame.x
                    + sp.Float(j.axis[1]) * joint_frame.y
                    + sp.Float(j.axis[2]) * joint_frame.z
                )
                child_frame = joint_frame.orientnew(
                    f"B_{j.child}", "Axis", [q_i, axis_vec],
                )
                child_origin = joint_origin
                child_origin.set_vel(N, child_origin.pos_from(O).dt(N))
            elif j.type == JointType.PRISMATIC:
                q_i = next(q_iter)
                next(u_iter)
                axis_vec = (
                    sp.Float(j.axis[0]) * joint_frame.x
                    + sp.Float(j.axis[1]) * joint_frame.y
                    + sp.Float(j.axis[2]) * joint_frame.z
                )
                child_origin = joint_origin.locatenew(
                    f"P_{j.child}", q_i * axis_vec,
                )
                child_origin.set_vel(N, child_origin.pos_from(O).dt(N))
                child_frame = joint_frame  # prismatic doesn't rotate
            else:  # FIXED
                child_origin = joint_origin
                child_frame = joint_frame

            link_frames[j.child] = child_frame
            link_origins[j.child] = child_origin
        self._link_frames = link_frames
        self._link_origins = link_origins

        # Build a RigidBody per link.
        bodies: list[RigidBody] = []
        loads: list[tuple] = []
        gx, gy, gz = (sp.Float(g) for g in self.gravity)
        for link_id, link in self.system.links.items():
            frame = link_frames[link_id]
            link_origin = link_origins[link_id]
            com_offset = (
                sp.Float(link.com[0]) * frame.x
                + sp.Float(link.com[1]) * frame.y
                + sp.Float(link.com[2]) * frame.z
            )
            masscenter = link_origin.locatenew(f"CoM_{link_id}", com_offset)
            masscenter.set_vel(N, masscenter.pos_from(O).dt(N))
            ii = link.inertia or Inertia()
            inertia_dyadic = inertia(
                frame,
                sp.Float(ii.ixx), sp.Float(ii.iyy), sp.Float(ii.izz),
                sp.Float(ii.ixy), sp.Float(ii.ixz), sp.Float(ii.iyz),
            )
            body = RigidBody(
                f"body_{link_id}", masscenter, frame,
                sp.Float(link.mass), (inertia_dyadic, masscenter),
            )
            bodies.append(body)
            if link.mass > 0:
                weight_vec = sp.Float(link.mass) * (gx * N.x + gy * N.y + gz * N.z)
                loads.append((masscenter, weight_vec))

        # Torque loads: tau_i applied about the joint axis (parent frame).
        tau_iter = iter(tau_syms)
        for j in joints_in_order:
            if j.type == JointType.FIXED:
                continue
            tau_i = next(tau_iter)
            child_frame = link_frames[j.child]
            parent_frame = link_frames[j.parent]
            axis_vec_parent = (
                sp.Float(j.axis[0]) * parent_frame.x
                + sp.Float(j.axis[1]) * parent_frame.y
                + sp.Float(j.axis[2]) * parent_frame.z
            )
            if j.type in (JointType.REVOLUTE, JointType.CONTINUOUS):
                loads.append((child_frame, tau_i * axis_vec_parent))
                loads.append((parent_frame, -tau_i * axis_vec_parent))
            elif j.type == JointType.PRISMATIC:
                child_origin = link_origins[j.child]
                parent_origin = link_origins[j.parent]
                loads.append((child_origin, tau_i * axis_vec_parent))
                loads.append((parent_origin, -tau_i * axis_vec_parent))

        # Keep the bodies for the Lagrangian builder (CoM velocities + inertias
        # are already set on them above).
        self._bodies = bodies

        # Kinematic differential equations: u_i = qdot_i.
        kd_eqs = [u_syms[i] - sp.diff(q_syms[i], self.t) for i in range(len(q_syms))]

        if q_syms:
            self.KM = KanesMethod(N, q_ind=q_syms, u_ind=u_syms, kd_eqs=kd_eqs)
            self.KM.kanes_equations(bodies=bodies, loads=loads)
            self.mass_matrix = self.KM.mass_matrix
            self.forcing = self.KM.forcing
        else:  # all-fixed system — degenerate, no dynamics.
            self.KM = None
            self.mass_matrix = sp.zeros(0, 0)
            self.forcing = sp.zeros(0, 1)

    # ------------------------------------------------------------------
    # convenience accessors
    # ------------------------------------------------------------------

    @property
    def n_dof(self) -> int:
        return len(self.q)

    def frame_pose_symbolic(self, link_id: str):
        """Return the *symbolic* ``(R, p)`` of a link frame in the world frame,
        in terms of ``self.q`` (the un-substituted twin of :meth:`link_pose`).

        ``R`` is a 3x3 SymPy matrix whose columns are the link axes in world;
        ``p`` is a 3x1 SymPy matrix of the link origin's world position. Used by
        :mod:`fieldpilot_urdf.loops` to derive loop-closure constraints.
        """
        import sympy as sp
        frame = self._link_frames[link_id]
        R = sp.Matrix.hstack(*[v.to_matrix(self.N) for v in (frame.x, frame.y, frame.z)])
        p = self._link_origins[link_id].pos_from(self.O).to_matrix(self.N)
        return R, p

    def lagrangian(self, *, simplify: bool = True):
        """Return the tree's Lagrangian ``L = T − V`` as a SymPy expression in
        ``self.q`` and their time derivatives.

        ``T`` is the total kinetic energy of the rigid bodies (translational +
        rotational, in terms of ``q̇``); ``V`` is gravitational potential energy,
        ``V = −Σ mᵢ (g · rᵢ)`` with ``g`` = :attr:`gravity` and ``rᵢ`` the link
        CoM positions. This is the input a Lagrange-multiplier solver needs to
        consume the constraints from :mod:`fieldpilot_urdf.loops`; for a tree it
        is equivalent to the Kane-based ``mass_matrix``/``forcing`` already built
        here. Pass ``simplify=False`` to skip ``sympy.simplify`` on large robots.
        """
        import sympy as sp
        T = sum((b.kinetic_energy(self.N) for b in self._bodies), sp.Integer(0))
        gx, gy, gz = (sp.Float(g) for g in self.gravity)
        V = sp.Integer(0)
        for b in self._bodies:
            r = b.masscenter.pos_from(self.O).to_matrix(self.N)
            V -= b.mass * (gx * r[0] + gy * r[1] + gz * r[2])
        L = T - V
        return sp.simplify(L) if simplify else L

    def link_pose(self, link_id: str, q: dict[str, float] | None = None):
        """Return ``(R, p)`` of a link's body frame in the world frame at
        configuration ``q`` (a ``{joint_name: value}`` dict; unset joints are 0).

        ``R`` is 3x3 with columns = the link axes in world (matching
        :func:`fieldpilot_urdf.fk.forward_kinematics`'s rotation block); ``p``
        is the link origin's world position.
        """
        import numpy as np
        import sympy as sp

        qvals = q or {}
        subs = {
            qsym: float(qvals.get(jid, 0.0))
            for jid, qsym in zip(self.actuated_joint_ids, self.q)
        }
        frame = self._link_frames[link_id]
        cols = [v.to_matrix(self.N) for v in (frame.x, frame.y, frame.z)]
        R_sym = sp.Matrix.hstack(*cols)
        p_sym = self._link_origins[link_id].pos_from(self.O).to_matrix(self.N)
        R = np.array(R_sym.subs(subs).evalf(), dtype=float)
        p = np.array(p_sym.subs(subs).evalf(), dtype=float).ravel()
        return R, p

    # ------------------------------------------------------------------
    # lambdify
    # ------------------------------------------------------------------

    def lambdify_mass_matrix(self) -> Callable:
        """Return a NumPy callable ``M(q_vec) -> (n, n) ndarray``."""
        import sympy as sp
        return sp.lambdify([self.q], self.mass_matrix, modules="numpy")

    def lambdify_forcing(self) -> Callable:
        """Return a NumPy callable ``F(q_vec, u_vec, tau_vec) -> (n,) ndarray``."""
        import sympy as sp
        return sp.lambdify([self.q, self.u, self.tau], self.forcing, modules="numpy")

    def lambdify_forward_dynamics(self) -> Callable:
        """Return a callable ``f(q, u, tau) -> qdd`` that solves ``M·qdd = F``."""
        import numpy as np

        M_fn = self.lambdify_mass_matrix()
        F_fn = self.lambdify_forcing()

        def forward(q_vec, u_vec, tau_vec):
            q_vec = np.asarray(q_vec, dtype=float).ravel()
            u_vec = np.asarray(u_vec, dtype=float).ravel()
            tau_vec = np.asarray(tau_vec, dtype=float).ravel()
            M = np.asarray(M_fn(q_vec), dtype=float)
            F = np.asarray(F_fn(q_vec, u_vec, tau_vec), dtype=float).ravel()
            return np.linalg.solve(M, F)
        return forward
