"""Kinematic fault localization: which joint explains a pose deviation?

The field question for a misbehaving arm is geometric: the tool isn't where the
model says it should be — measured pose ≠ ``forward_kinematics(commanded q)`` —
so *which joint is miscalibrated, and by how much?* The rest of the diagnostics
layer reasons structurally (downstream-link overlap) or by simulation;
:func:`localize_joint_fault` reasons *kinematically*, using the 1.10 geometric
Jacobian.

A small offset ``δqⱼ`` on joint *j* shifts the link's pose by the twist
``J[:, j]·δqⱼ`` — a known direction in the 6-D twist space. Given the observed
deviation twist ``e`` (expected → observed), the best single-joint explanation
is the least-squares projection of ``e`` onto each Jacobian column:
``δqⱼ = (cⱼ·e)/(cⱼ·cⱼ)``. Joints are ranked by how much of the deviation that
offset removes (``explained_fraction``), best-first.

This is a **linearization** — exact in the limit of small offsets, and
degrading as the true offset grows (``explained_fraction`` then drops, which is
the signal that a single-joint story no longer fully fits). Pure NumPy.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .cartesian import _pose_error_twist
from .fk import forward_kinematics, rpy_to_R
from .kinematics import geometric_jacobian, jacobian_joints
from .models import Robot


class JointFaultCandidate(BaseModel):
    """One ranked suspect from :func:`localize_joint_fault`."""

    model_config = ConfigDict(extra="forbid")

    joint: str = Field(..., description="Joint whose offset could explain the observed deviation")
    estimated_offset: float = Field(
        ..., description="Best-fit offset δq (rad for revolute/continuous, m for prismatic) "
                         "that this joint would need to produce the observed deviation")
    explained_fraction: float = Field(
        ..., description="Fraction of the deviation's twist magnitude removed by this single "
                         "offset, in [0, 1]; 1.0 = the deviation is fully explained by this joint")
    residual_position: float = Field(
        ..., description="Remaining end-effector position error after applying the offset (m)")
    residual_orientation: float = Field(
        ..., description="Remaining end-effector orientation error after applying the offset (rad)")


def localize_joint_fault(
    robot: Robot,
    link: str,
    expected_q: dict[str, float],
    observed_xyz: tuple[float, float, float],
    observed_rpy: Optional[tuple[float, float, float]] = None,
    *,
    orientation_weight: float = 1.0,
    min_explained: float = 0.0,
    max_candidates: Optional[int] = None,
) -> list[JointFaultCandidate]:
    """Rank the movable joints on the chain to ``link`` by how well a single
    offset on each explains the observed pose deviation, best-first.

    ``expected_q`` is the commanded configuration; ``observed_xyz`` (and optional
    ``observed_rpy`` — omit it to compare position only, holding the expected
    orientation) is where ``link`` actually ended up. For each joint the routine
    computes the least-squares offset that best accounts for the deviation and
    the fraction of it thereby explained.

    Because the pose-error twist mixes metres and radians, ``orientation_weight``
    scales the rotational rows when fitting (raise it to trust orientation more);
    the reported ``residual_position`` / ``residual_orientation`` are always in
    physical units. ``min_explained`` filters out weak candidates;
    ``max_candidates`` caps the list.

    Returns ``[]`` when there is no measurable deviation (or no movable joints on
    the chain). Raises ``ValueError`` (via the kinematics layer) if the robot is
    not a single-rooted tree or ``link`` is unknown.
    """
    cols = jacobian_joints(robot, link)
    if not cols:
        return []

    tf = forward_kinematics(robot, expected_q)
    if link not in tf:
        raise ValueError(f"unknown link: {link!r}")
    T_exp = tf[link]
    T_obs = np.eye(4)
    T_obs[:3, 3] = np.asarray(observed_xyz, dtype=float)
    T_obs[:3, :3] = rpy_to_R(observed_rpy) if observed_rpy is not None else T_exp[:3, :3]

    e = _pose_error_twist(T_exp, T_obs)              # [Δp; Δrot], world frame
    # Fit only over the observed subspace: with no observed_rpy the orientation
    # rows carry no information, so they must be excluded from the fit entirely
    # (weight 0) — otherwise the columns' orientation rows pollute the projection.
    observed_orientation = observed_rpy is not None
    rot_w = orientation_weight if observed_orientation else 0.0
    w = np.array([1.0, 1.0, 1.0, rot_w, rot_w, rot_w])
    e_w = e * w
    norm_e = float(np.linalg.norm(e_w))
    if norm_e < 1e-12:
        return []                                    # no deviation -> nothing to localize

    J = geometric_jacobian(robot, expected_q, link, joints=cols)  # 6×n
    out: list[JointFaultCandidate] = []
    for i, jn in enumerate(cols):
        col = J[:, i]
        col_w = col * w
        denom = float(col_w @ col_w)
        if denom < 1e-18:
            continue                                 # this joint can't move the link here
        delta = float((col_w @ e_w) / denom)
        residual_w = e_w - delta * col_w
        explained = 1.0 - float(np.linalg.norm(residual_w)) / norm_e
        residual = e - delta * col                   # unweighted, for physical reporting
        out.append(JointFaultCandidate(
            joint=jn,
            estimated_offset=delta,
            explained_fraction=explained,
            residual_position=float(np.linalg.norm(residual[:3])),
            # orientation residual is meaningless when orientation wasn't observed
            residual_orientation=(float(np.linalg.norm(residual[3:]))
                                  if observed_orientation else 0.0),
        ))

    out.sort(key=lambda c: c.explained_fraction, reverse=True)
    out = [c for c in out if c.explained_fraction >= min_explained]
    if max_candidates is not None:
        out = out[:max_candidates]
    return out
