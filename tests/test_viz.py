"""Tests for app/urdf/viz.py.

Smoke level: confirms each renderer returns non-empty bytes with the right
magic header. Requires graphviz's `dot` binary on PATH; tests are skipped
if it's missing.

Run: python3 -m pytest app/urdf/test_viz.py -q  (from pydexpi-server/)
"""
from __future__ import annotations

import shutil

import pytest

from fieldpilot_urdf import from_xml
from fieldpilot_urdf.viz import render_kinematic_tree, render_pose_3d


SAMPLE = """\
<robot name="arm">
  <link name="base">
    <inertial><mass value="2"/><inertia ixx="1" iyy="1" izz="1"/></inertial>
  </link>
  <link name="upper"/>
  <link name="lower"/>
  <link name="gripper"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="upper"/>
    <axis xyz="0 0 1"/><origin xyz="0 0 0.2"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
  <joint name="j2" type="prismatic">
    <parent link="upper"/><child link="lower"/>
    <axis xyz="0 0 1"/><origin xyz="0 0 0.3"/>
    <limit lower="0" upper="0.4" effort="20" velocity="0.1"/>
  </joint>
  <joint name="j3" type="fixed">
    <parent link="lower"/><child link="gripper"/>
    <origin xyz="0 0 0.05"/>
  </joint>
</robot>
"""

PNG_MAGIC = b"\x89PNG"
SVG_PREFIX = b"<?xml"

dot_missing = pytest.mark.skipif(
    shutil.which("dot") is None,
    reason="graphviz `dot` binary not installed",
)


@pytest.fixture
def robot():
    return from_xml(SAMPLE)


# --- kinematic tree --------------------------------------------------------

@dot_missing
def test_render_tree_png(robot):
    out = render_kinematic_tree(robot, fmt="png")
    assert out[:4] == PNG_MAGIC
    assert len(out) > 1000  # any reasonable rendering


@dot_missing
def test_render_tree_svg_contains_link_names(robot):
    out = render_kinematic_tree(robot, fmt="svg")
    assert out[:5] == SVG_PREFIX
    text = out.decode()
    for name in ("base", "upper", "lower", "gripper", "j1", "j2", "j3"):
        assert name in text


@dot_missing
def test_render_tree_highlight_changes_output(robot):
    plain = render_kinematic_tree(robot, fmt="svg").decode().lower()
    hi = render_kinematic_tree(robot, fmt="svg", highlight=["gripper"]).decode().lower()
    # Highlight color (#F1C40F) must appear only in the highlighted render.
    # Graphviz emits hex colours in lowercase.
    assert "f1c40f" not in plain
    assert "f1c40f" in hi


@dot_missing
def test_render_tree_with_q_annotates_title(robot):
    out = render_kinematic_tree(robot, fmt="svg", q={"j1": 0.5})
    assert b"j1=0.5" in out


# --- 3D pose ---------------------------------------------------------------

def test_render_pose_png(robot):
    out = render_pose_3d(robot, fmt="png")
    assert out[:4] == PNG_MAGIC
    assert len(out) > 1000


def test_render_pose_svg(robot):
    out = render_pose_3d(robot, fmt="svg")
    assert out[:5] == SVG_PREFIX
    text = out.decode()
    for name in ("base", "upper", "lower", "gripper"):
        assert name in text


def test_render_pose_q_extends_prismatic(robot):
    """A non-default q should change link positions visibly in the title text."""
    out = render_pose_3d(robot, fmt="svg", q={"j2": 0.4})
    assert b"j2=0.4" in out


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
