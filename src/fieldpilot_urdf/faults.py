"""Fault injection for URDF robots.

Single source of truth for what a simulated motor fault *is*, shared by the
fixture CLI (make_fault_fixtures.py) and the regression test
(test_fault_diagnosis.py). Keeping it here means a fixture and the test that
guards it can never drift apart.
"""
from __future__ import annotations

from .models import Robot


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


def fault_source_tag(joint_name: str, origin: str) -> str:
    """Canonical ``source_file`` tag for a persisted fault fixture, so the CLI
    can recognise (and skip) a fixture it already created."""
    return f"motor_fault_injected:{joint_name}_from:{origin}"
