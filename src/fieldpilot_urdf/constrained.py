"""Holonomic-constrained dynamics via Lagrange multipliers — closed-loop robots.

This is the solver that consumes the closed-loop pieces built earlier: the tree
Lagrangian (:meth:`fieldpilot_urdf.dynamics.SymbolicDynamics.lagrangian`) and the
loop-closure constraints (:func:`fieldpilot_urdf.loops.derive_loop_constraints`).

Two layers:

* :class:`ConstrainedDynamics` — model-agnostic wrapper over SymPy's
  ``LagrangesMethod`` (ported from MecAI, MIT). Give it generalized coordinates
  ``q``, a Lagrangian ``L``, and holonomic constraints ``c(q) = 0``; it builds
  the augmented system and lambdifies forward dynamics.
* :func:`constrained_dynamics` — the high-level entry: hand it a
  :class:`~fieldpilot_urdf.models.Robot` carrying ``loops`` and it wires the
  whole chain together, returning a ready :class:`ConstrainedDynamics`.

The augmented system solved at a fixed state is

.. math::

    \\begin{pmatrix} M & -A^\\top \\\\ A & 0 \\end{pmatrix}
    \\begin{pmatrix} \\ddot q \\\\ \\lambda \\end{pmatrix}
    = \\begin{pmatrix} F \\\\ -\\dot A\\,\\dot q \\end{pmatrix},
    \\qquad A = \\partial c / \\partial q.

**Caveats.** This is an index-3 DAE. The augmented matrix is singular at
kinematic singularities *and* when the constraints are redundant (e.g. a planar
``point`` closure, whose out-of-plane row is identically zero — drop the
redundant constraint first). Time integration drifts off the manifold unless
stabilized: :meth:`ConstrainedDynamics.lambdify_forward_dynamics` takes Baumgarte
gains, and :meth:`ConstrainedDynamics.project` snaps a state back onto the
manifold exactly (and tolerates redundant constraints via a pseudo-inverse).
"""
from __future__ import annotations

from typing import Callable

__all__ = ["ConstrainedDynamics", "constrained_dynamics"]


class ConstrainedDynamics:
    """Symbolic constrained dynamics via Lagrange multipliers.

    Construct from ``q`` (SymPy ``dynamicsymbols``), a ``lagrangian`` expression,
    and a list of holonomic ``constraints`` ``c(q) = 0``. Read off
    ``mass_matrix`` / ``forcing`` / ``constraint_jacobian`` or call
    :meth:`lambdify_forward_dynamics` for a ``(q, q̇) -> (q̈, λ)`` callable.
    """

    def __init__(self, *, q, lagrangian, constraints, forcelist=None, frame=None):
        try:
            import sympy as sp
            from sympy.physics.mechanics import (
                LagrangesMethod, ReferenceFrame, dynamicsymbols,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SymPy is not installed. Install the dynamics extra: "
                'pip install "fieldpilot-urdf[dynamics]"'
            ) from exc

        self.q = list(q)
        self.constraints = list(constraints)
        self.t = dynamicsymbols._t
        self.qdot = [sp.diff(qi, self.t) for qi in self.q]
        self.frame = frame if frame is not None else ReferenceFrame("N")

        n = len(self.q)
        # Constraint Jacobian A = ∂c/∂q : (m × n).
        if self.constraints:
            self.constraint_jacobian = sp.Matrix(
                [[sp.diff(c, qi) for qi in self.q] for c in self.constraints]
            )
        else:
            self.constraint_jacobian = sp.zeros(0, n)

        self.LM = LagrangesMethod(
            lagrangian, self.q,
            hol_coneqs=self.constraints or None,
            forcelist=forcelist, frame=self.frame,
        )
        self.LM.form_lagranges_equations()
        # mass_matrix_full is the (2n+m)×(2n+m) DAE state form; its lower-right
        # (n+m)×(n+m) block is the augmented [M, -Aᵀ; A, 0] we solve for (q̈, λ)
        # (the top n rows are the trivial kinematic identities q̇ = q̇).
        self.mass_matrix = self.LM.mass_matrix_full[n:, n:]
        self.forcing = self.LM.forcing_full[n:, :]

        # Plain placeholder speeds so we can lambdify away the Derivative(q, t).
        self._u = sp.symbols(f"_u0:{n}") if n else ()
        self._sub = {self.qdot[i]: self._u[i] for i in range(n)}

    @property
    def n_q(self) -> int:
        return len(self.q)

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)

    def lambdify_constraint_residual(self) -> Callable:
        """``r(q_vec) -> (m,) ndarray`` — zero on assembled configurations."""
        import numpy as np
        import sympy as sp
        f = sp.lambdify([self.q], sp.Matrix(self.constraints), "numpy")
        return lambda q_vec: np.asarray(f(list(q_vec)), dtype=float).ravel()

    def lambdify_constraint_jacobian(self) -> Callable:
        """``J(q_vec) -> (m, n) ndarray`` for ``A = ∂c/∂q``."""
        import numpy as np
        import sympy as sp
        f = sp.lambdify([self.q], self.constraint_jacobian, "numpy")
        return lambda q_vec: np.asarray(f(list(q_vec)), dtype=float).reshape(self.n_constraints, self.n_q)

    def lambdify_forward_dynamics(self, *, alpha: float = 0.0, beta: float = 0.0) -> Callable:
        """Return ``f(q_vec, qdot_vec) -> (qdd, lambdas)`` solving the augmented
        system. Actuator torques, if any, must be folded into ``forcelist`` /
        ``lagrangian`` at construction.

        **Baumgarte stabilization.** With ``alpha`` (velocity gain) or ``beta``
        (position gain) non-zero, the acceleration-level constraint target is
        replaced by ``A q̈ + Ȧ q̇ = −2α (A q̇) − β² c(q)``, so position/velocity
        drift is fed back toward the manifold during integration instead of left
        to accumulate. A critically-damped choice is ``alpha = beta``; with both
        zero (the default) the behaviour is unchanged. Projection
        (:meth:`project`) is the exact complement — Baumgarte damps drift cheaply
        every step, projection removes whatever remains.
        """
        import numpy as np
        import sympy as sp

        n, m = self.n_q, self.n_constraints
        M_fn = sp.lambdify([self.q, self._u], self.mass_matrix.subs(self._sub), "numpy")
        F_fn = sp.lambdify([self.q, self._u], self.forcing.subs(self._sub), "numpy")
        stabilize = bool(m) and (alpha != 0.0 or beta != 0.0)
        if stabilize:
            A_fn = self.lambdify_constraint_jacobian()
            c_fn = self.lambdify_constraint_residual()

        def forward(q_vec, qdot_vec):
            qv = list(np.asarray(q_vec, dtype=float).ravel())
            qd = np.asarray(qdot_vec, dtype=float).ravel()
            M = np.asarray(M_fn(qv, list(qd)), dtype=float)
            F = np.asarray(F_fn(qv, list(qd)), dtype=float).ravel()
            if stabilize:
                A = A_fn(qv)
                c = c_fn(qv)
                F[n:] = F[n:] - 2.0 * alpha * (A @ qd) - (beta ** 2) * c
            sol = np.linalg.solve(M, F)
            return sol[:n], sol[n:]
        return forward

    def project(self, q_vec, qdot_vec, *, tol: float = 1e-12, max_iter: int = 100):
        """Project a state back onto the constraint manifold and return
        ``(q, q̇)`` corrected so that ``c(q) ≈ 0`` and ``A q̇ ≈ 0``.

        Position drift is removed by Newton iteration ``q ← q − A⁺ c(q)``
        (pseudo-inverse, so redundant or rank-deficient constraints — e.g. a
        planar ``point`` closure — are handled gracefully); velocity drift by the
        single projection ``q̇ ← q̇ − A⁺ (A q̇)``. With no constraints the state is
        returned unchanged. Use after each integration step (optionally with
        Baumgarte gains) to keep a closed-loop simulation on the manifold.
        """
        import numpy as np

        q = np.array(q_vec, dtype=float).ravel()
        qd = np.array(qdot_vec, dtype=float).ravel()
        if self.n_constraints == 0:
            return q, qd

        A_fn = self.lambdify_constraint_jacobian()
        c_fn = self.lambdify_constraint_residual()
        for _ in range(max_iter):
            c = c_fn(q)
            if np.linalg.norm(c) <= tol:
                break
            q = q - np.linalg.pinv(A_fn(q)) @ c
        A = A_fn(q)
        qd = qd - np.linalg.pinv(A) @ (A @ qd)
        return q, qd


def constrained_dynamics(robot, *, gravity=(0.0, 0.0, -9.81), simplify=False) -> ConstrainedDynamics:
    """Build :class:`ConstrainedDynamics` for a ``Robot`` carrying ``loops``.

    Wires the chain end-to-end: the tree Lagrangian from
    :class:`~fieldpilot_urdf.dynamics.SymbolicDynamics` plus the loop-closure
    constraints from :func:`fieldpilot_urdf.loops.derive_loop_constraints`. The
    returned object also exposes ``.dyn`` (the underlying tree dynamics) and
    ``.actuated_joint_ids`` so callers can map joint names to the ``q`` order.

    Gravity enters through the Lagrangian's potential energy (no force list
    needed). For a robot with no ``loops`` this reduces to the unconstrained
    tree dynamics.
    """
    from .dynamics import SymbolicDynamics
    from .loops import derive_loop_constraints

    dyn = SymbolicDynamics(robot, gravity=gravity)
    lagrangian = dyn.lagrangian(simplify=simplify)
    constraints = derive_loop_constraints(dyn, robot.loops, simplify=simplify)
    cd = ConstrainedDynamics(q=dyn.q, lagrangian=lagrangian, constraints=constraints)
    cd.dyn = dyn
    cd.actuated_joint_ids = dyn.actuated_joint_ids
    return cd
