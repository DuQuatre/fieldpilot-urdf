"""Fault injection for URDF robots.

Single source of truth for what a simulated motor fault *is*, shared by the
fixture CLI (make_fault_fixtures.py) and the regression test
(test_fault_diagnosis.py). Keeping it here means a fixture and the test that
guards it can never drift apart.
"""
from __future__ import annotations

from .models import Origin, Robot


def inject_motor_fault(robot: Robot, joint_name: str) -> Robot:
    """Simulate a dead/failed actuator on ``joint_name`` by zeroing its effort
    and velocity limits.

    This trips diagnostic rule R003 (``effort <= 0`` and ``velocity <= 0``) on
    exactly that joint — mirroring a motor that can no longer exert torque or
    move. Mutates and returns ``robot``.

    Raises ``KeyError`` if the joint is unknown or has no ``<limit>`` to fault.
    """
    j = robot.joint(joint_name)  # raises KeyError on an unknown joint name
    if j.limit is None:
        raise KeyError(f"joint '{joint_name}' has no <limit> to fault")
    j.limit.effort = 0.0
    j.limit.velocity = 0.0
    return robot


def freeze_joint(robot: Robot, joint_name: str) -> Robot:
    """Lock a joint so it can no longer be actuated — the *kinematic* consequence
    of a dead or jammed motor.

    The joint becomes ``fixed`` (held at its zero pose), so it drops out of the
    IK degree-of-freedom set and FK treats it as rigid. Use this to *simulate*
    whether a dead motor on ``joint_name`` explains a "can't reach" symptom.
    Mutates and returns ``robot``. Raises ``KeyError`` if the joint is unknown.
    """
    j = robot.joint(joint_name)  # raises KeyError on an unknown joint name
    j.type = "fixed"
    return robot


def freeze_joint_at(robot: Robot, joint_name: str, angle: float) -> Robot:
    """Lock a joint *at a non-zero pose* ``angle`` — a joint jammed/stuck part
    way through its travel (vs :func:`freeze_joint`, which locks at the neutral
    zero pose).

    The joint's motion at ``angle`` is baked into its ``<origin>`` and the joint
    is set ``fixed``, so FK/IK see a rigid offset frame holding the stuck pose.
    ``angle`` in radians for revolute/continuous joints, metres for prismatic.
    ``angle == 0`` reduces to :func:`freeze_joint`. Mutates and returns ``robot``.
    Raises ``KeyError`` if the joint is unknown.
    """
    # Local import: faults is otherwise dependency-free; FK pulls in numpy/graph.
    from .fk import R_to_rpy, joint_motion, origin_to_T

    j = robot.joint(joint_name)  # raises KeyError on an unknown joint name
    if j.type == "fixed" or angle == 0.0:
        j.type = "fixed"
        return robot
    T = origin_to_T(j.origin) @ joint_motion(j, angle)   # parent->child at `angle`
    j.origin = Origin(xyz=tuple(float(x) for x in T[:3, 3]), rpy=R_to_rpy(T[:3, :3]))
    j.type = "fixed"
    return robot


def misconfigure_limit(
    robot: Robot, joint_name: str, *,
    lower: float | None = None, upper: float | None = None,
) -> Robot:
    """Simulate a mis-set joint travel ``<limit>`` — a commissioning/config error
    that narrows (or shifts) a joint's reachable range without freezing it.

    Overwrites the given bound(s) on ``joint_name``'s limit; an omitted bound is
    left as-is. Unlike :func:`freeze_joint`/:func:`freeze_joint_at` the joint
    still moves — it just can't travel as far — so IK that needs the clipped range
    will fail to converge. Mutates and returns ``robot``. Raises ``KeyError`` if
    the joint is unknown or has no ``<limit>`` to misconfigure.
    """
    j = robot.joint(joint_name)  # raises KeyError on an unknown joint name
    if j.limit is None:
        raise KeyError(f"joint '{joint_name}' has no <limit> to misconfigure")
    if lower is not None:
        j.limit.lower = lower
    if upper is not None:
        j.limit.upper = upper
    return robot


def fault_source_tag(joint_name: str, origin: str) -> str:
    """Canonical ``source_file`` tag for a persisted fault fixture, so the CLI
    can recognise (and skip) a fixture it already created."""
    return f"motor_fault_injected:{joint_name}_from:{origin}"
