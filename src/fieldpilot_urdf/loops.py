"""Derive holonomic loop-closure constraints for a closed-loop robot.

A closed-loop mechanism is modelled as a spanning-tree :class:`~fieldpilot_urdf.models.Robot`
plus :class:`~fieldpilot_urdf.models.LoopClosure` entries (see ``models.py``).
This module turns those closures into symbolic constraint equations ``c(q) = 0``
expressed in the tree's generalized coordinates — exactly the form a
Lagrange-multiplier solver needs.

It reuses the symbolic per-link frames that
:class:`fieldpilot_urdf.dynamics.SymbolicDynamics` already builds, so there is no
duplicate FK. SymPy is required (the ``[dynamics]`` extra); this module is not
imported by the package root, so the core install stays light.

What's here
-----------
* :func:`derive_loop_constraints` — symbolic ``c(q)`` for every closure.
* :func:`lambdify_loop_residual` — a NumPy ``r(q) -> ndarray`` (zero on assembled
  configurations); handy for assembly checks and projection loops.
* :func:`mobility` — closed-loop DOF = ``len(q) − rank(∂c/∂q)``.

These feed a future ``ConstrainedDynamics`` drop-in once a tree Lagrangian
builder exists; on their own they already verify assembly and count DOF.
"""
from __future__ import annotations

from .fk import rpy_to_R
from .models import FrameRef, LoopClosure


def _offset(origin):
    """Constant ``(R_off, p_off)`` for a FrameRef.origin, in fieldpilot-urdf's
    ``Rz·Ry·Rx`` rpy convention (reused from :mod:`fieldpilot_urdf.fk`)."""
    import sympy as sp
    if origin is None:
        return sp.eye(3), sp.zeros(3, 1)
    R = sp.Matrix(rpy_to_R(origin.rpy).tolist())
    p = sp.Matrix([float(c) for c in origin.xyz])
    return R, p


def _frameref_world(dyn, ref: FrameRef):
    """Symbolic ``(R, p)`` of a FrameRef in the world frame, in terms of ``dyn.q``."""
    R_link, p_link = dyn.frame_pose_symbolic(ref.link)
    R_off, p_off = _offset(ref.origin)
    return R_link @ R_off, p_link + R_link @ p_off


def derive_loop_constraints(dyn, loops: list[LoopClosure] | None = None, *, simplify: bool = True):
    """Return the holonomic constraint expressions ``c(q) = 0`` for ``loops``,
    in the tree coordinates ``dyn.q``.

    ``point`` closures contribute 3 position constraints; ``fixed`` closures add
    3 more orientation constraints (the vee of the skew part of ``Rₐᵀ R_b``,
    analytic everywhere). ``loops`` defaults to the robot's own ``loops``.
    Pass ``simplify=False`` to skip ``sympy.simplify`` on large robots.
    """
    import sympy as sp
    loops = dyn._robot.loops if loops is None else loops
    cons: list = []
    for lp in loops:
        Ra, pa = _frameref_world(dyn, lp.a)
        Rb, pb = _frameref_world(dyn, lp.b)
        cons.extend(list(pa - pb))                      # 3 position constraints
        if lp.kind == "fixed":
            E = Ra.T @ Rb                               # = I iff frames aligned
            cons.extend([E[2, 1] - E[1, 2],
                         E[0, 2] - E[2, 0],
                         E[1, 0] - E[0, 1]])            # 3 orientation constraints
    return [sp.simplify(c) for c in cons] if simplify else cons


def lambdify_loop_residual(dyn, loops: list[LoopClosure] | None = None, *, simplify: bool = True):
    """Return a NumPy callable ``r(q_vec) -> ndarray`` of the constraint residuals
    (zero on assembled / closed configurations). ``q_vec`` is ordered as
    ``dyn.actuated_joint_ids``."""
    import numpy as np
    import sympy as sp

    cons = derive_loop_constraints(dyn, loops, simplify=simplify)
    f = sp.lambdify([dyn.q], sp.Matrix(cons), modules="numpy")

    def residual(q_vec):
        return np.asarray(f(list(q_vec)), dtype=float).ravel()
    return residual


def mobility(dyn, loops: list[LoopClosure] | None = None, *, at=None) -> int:
    """Closed-loop degrees of freedom: ``len(dyn.q) − rank(∂c/∂q)``.

    The rank is evaluated at ``at`` (a ``q`` vector) or at a generic sample
    configuration if omitted, so it reflects the *generic* mobility rather than
    a singular pose.
    """
    import numpy as np
    import sympy as sp

    cons = derive_loop_constraints(dyn, loops)
    if not cons:
        return len(dyn.q)
    A = sp.Matrix([[sp.diff(c, qi) for qi in dyn.q] for c in cons])
    if at is None:
        subs = {qi: 0.1 * (i + 1) for i, qi in enumerate(dyn.q)}
    else:
        subs = {qi: float(at[i]) for i, qi in enumerate(dyn.q)}
    A0 = np.array(A.subs(subs).evalf(), dtype=float)
    return len(dyn.q) - int(np.linalg.matrix_rank(A0))
