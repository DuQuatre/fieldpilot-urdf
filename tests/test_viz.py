"""Tests for app/urdf/viz.py.

Smoke level: confirms each renderer returns non-empty bytes with the right
magic header. Requires graphviz's `dot` binary on PATH; tests are skipped
if it's missing.

Run: python3 -m pytest app/urdf/test_viz.py -q  (from pydexpi-server/)
"""
from __future__ import annotations

import importlib.util
import shutil

import pytest

from fieldpilot_urdf import from_xml
from fieldpilot_urdf.viz import (
    render_kinematic_tree,
    render_pose_3d,
    render_pose_mesh,
)


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


# --- mesh-accurate pose (urchin + pyrender) --------------------------------
#
# Two skip layers: the [meshviz] extra may be absent, and even with it a CI box
# may lack a working headless GL backend (EGL/osmesa). Both skip rather than
# fail so the core suite stays green without the heavy stack.

meshviz_missing = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("urchin", "pyrender")),
    reason="[meshviz] extra (urchin + pyrender) not installed",
)


# A robot with primitive *visual* geometry — renders without any mesh_dir and
# gives the scene something to draw (SAMPLE's links are bare/inertial-only).
PRIMITIVE_SAMPLE = """\
<robot name="prim">
  <link name="base">
    <visual><geometry><box size="0.3 0.3 0.1"/></geometry></visual>
  </link>
  <link name="arm">
    <visual><origin xyz="0 0 0.2"/><geometry><cylinder radius="0.05" length="0.4"/></geometry></visual>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="arm"/>
    <axis xyz="0 0 1"/><origin xyz="0 0 0.1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
</robot>
"""


@meshviz_missing
def test_render_pose_mesh_primitives_png():
    """Primitive visual geometry needs no mesh_dir; this drives the full
    urchin->pyrender->GL path and must yield a non-trivial PNG."""
    r = from_xml(PRIMITIVE_SAMPLE)
    try:
        out = render_pose_mesh(r, q={"j1": 0.6}, width=320, height=240)
    except Exception as e:  # no usable GL backend on this box
        pytest.skip(f"headless GL render unavailable: {e}")
    assert out[:4] == PNG_MAGIC
    assert len(out) > 1000  # actual geometry rendered, not a blank frame


# No skip marker: the fmt guard raises before urchin/pyrender are imported,
# so this validation runs even without the [meshviz] extra installed.
def test_render_pose_mesh_rejects_svg(robot):
    with pytest.raises(ValueError):
        render_pose_mesh(robot, fmt="svg")


def test_render_pose_mesh_missing_extra_message(robot, monkeypatch):
    """When urchin/pyrender aren't importable, the error names the [meshviz]
    extra — and does NOT blame the GL backend. Deterministic via a find_spec
    stub, so it holds whether or not the extra is actually installed."""
    import importlib.util as ilu

    real_find_spec = ilu.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in ("urchin", "pyrender"):
            return None  # simulate the extra not installed
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)

    with pytest.raises(ImportError, match="meshviz") as exc:
        render_pose_mesh(robot)
    assert "backend" not in str(exc.value).lower()  # not the GL-backend message


def test_resolve_mesh_robot_rewrites_and_drops(tmp_path):
    """_resolve_mesh_robot rewrites present meshes to absolute paths and drops
    visuals whose mesh is missing — no GL or extras needed for this unit."""
    from fieldpilot_urdf.viz import _resolve_mesh_robot

    urdf = """\
<robot name="m">
  <link name="base">
    <visual><geometry><mesh filename="package://pkg/present.stl"/></geometry></visual>
    <visual><geometry><mesh filename="package://pkg/missing.stl"/></geometry></visual>
    <visual><geometry><box size="1 1 1"/></geometry></visual>
  </link>
</robot>
"""
    r = from_xml(urdf)
    present = tmp_path / "pkg" / "present.stl"
    present.parent.mkdir(parents=True)
    present.write_bytes(b"solid\n")

    out, dropped = _resolve_mesh_robot(r, tmp_path)
    assert dropped == 1  # missing.stl
    geoms = [v.geometry for v in out.links[0].visuals]
    kinds = sorted(g.kind for g in geoms)
    assert kinds == ["box", "mesh"]  # missing mesh visual removed
    mesh_geom = next(g for g in geoms if g.kind == "mesh")
    assert mesh_geom.filename == str(present)  # absolute, resolvable by urchin


def test_resolve_mesh_robot_direct_path_needs_no_mesh_dir(tmp_path):
    """Regression test: a Mesh geometry whose filename is ALREADY a valid
    on-disk path (e.g. written by mesh_primitives, not fetched under a
    package/mesh_dir layout) must NOT be dropped just because mesh_dir is
    None -- previously it always was, silently, with no error and no
    warning: render_pose_mesh(robot) came back a blank frame, indistinguishable
    byte-for-byte from an empty scene, unless a mesh_dir happened to be passed
    too (at which point it worked only by an unrelated pathlib accident:
    Path(mesh_dir) / <absolute path> discards mesh_dir entirely)."""
    from fieldpilot_urdf.viz import _resolve_mesh_robot

    mesh_file = tmp_path / "generated" / "link1.stl"
    mesh_file.parent.mkdir(parents=True)
    mesh_file.write_bytes(b"solid\n")

    urdf = f"""\
<robot name="m">
  <link name="base">
    <visual><geometry><mesh filename="{mesh_file}"/></geometry></visual>
  </link>
</robot>
"""
    r = from_xml(urdf)

    out, dropped = _resolve_mesh_robot(r, None)  # no mesh_dir at all
    assert dropped == 0
    mesh_geom = out.links[0].visuals[0].geometry
    assert mesh_geom.filename == str(mesh_file.resolve())

    # Also holds when a mesh_dir IS given but doesn't contain this file --
    # the direct-path fallback must still catch it rather than dropping.
    unrelated_dir = tmp_path / "unrelated_mesh_dir"
    unrelated_dir.mkdir()
    out2, dropped2 = _resolve_mesh_robot(r, unrelated_dir)
    assert dropped2 == 0
    assert out2.links[0].visuals[0].geometry.filename == str(mesh_file.resolve())


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
