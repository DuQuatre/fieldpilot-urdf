"""Tests for local-filesystem import (importer.import_urdf_file and friends).

These exercise the filesystem pipeline with real temp files — no network, no
mocking. Run: python3 -m pytest tests/test_import_local.py -q
"""
from __future__ import annotations

import pytest

from fieldpilot_urdf import (
    import_urdf_file, resolve_includes_local,
)
from fieldpilot_urdf.importer import ImportError_


XACRO_HEADER = '<robot xmlns:xacro="http://ros.org/wiki/xacro" name="{name}">'

PLAIN_URDF = """\
<robot name="plain">
  <link name="base"/>
  <link name="tool"/>
  <joint name="j" type="fixed"><parent link="base"/><child link="tool"/></joint>
</robot>
"""


def _write(d, name, text):
    p = d / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


# --- plain URDF ------------------------------------------------------------

def test_import_plain_urdf(tmp_path):
    p = _write(tmp_path, "robot.urdf", PLAIN_URDF)
    robot = import_urdf_file(p)
    assert robot.name == "plain"
    assert {l.name for l in robot.links} == {"base", "tool"}


def test_import_plain_urdf_skip_macros(tmp_path):
    """expand_macros=False parses the file as-is (plain URDF)."""
    p = _write(tmp_path, "robot.urdf", PLAIN_URDF)
    robot = import_urdf_file(p, expand_macros=False)
    assert robot.name == "plain"


def test_import_accepts_str_path(tmp_path):
    p = _write(tmp_path, "robot.urdf", PLAIN_URDF)
    robot = import_urdf_file(str(p))
    assert robot.name == "plain"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_urdf_file(tmp_path / "nope.urdf")


# --- xacro property / macro expansion --------------------------------------

def test_xacro_property_and_macro(tmp_path):
    src = (XACRO_HEADER.format(name="x") + """
  <xacro:property name="L" value="0.5"/>
  <xacro:macro name="seg" params="n">
    <link name="${n}"/>
  </xacro:macro>
  <xacro:seg n="a"/>
  <xacro:seg n="b"/>
  <joint name="j" type="prismatic">
    <parent link="a"/><child link="b"/>
    <limit lower="0" upper="${L}" effort="1" velocity="1"/>
  </joint>
</robot>""")
    p = _write(tmp_path, "arm.xacro", src)
    robot = import_urdf_file(p)
    assert {l.name for l in robot.links} == {"a", "b"}
    assert robot.joints[0].limit.upper == pytest.approx(0.5)


# --- relative includes -----------------------------------------------------

def test_relative_include(tmp_path):
    _write(tmp_path, "parts/arm.xacro",
           XACRO_HEADER.format(name="part") + '<link name="arm_link"/></robot>')
    top = _write(tmp_path, "robot.xacro",
                 XACRO_HEADER.format(name="top")
                 + '<xacro:include filename="parts/arm.xacro"/>'
                 + '<link name="base"/></robot>')
    robot = import_urdf_file(top)
    assert {l.name for l in robot.links} == {"base", "arm_link"}


def test_nested_includes(tmp_path):
    _write(tmp_path, "c.xacro",
           XACRO_HEADER.format(name="c") + '<link name="c_link"/></robot>')
    _write(tmp_path, "b.xacro",
           XACRO_HEADER.format(name="b")
           + '<xacro:include filename="c.xacro"/><link name="b_link"/></robot>')
    top = _write(tmp_path, "a.xacro",
                 XACRO_HEADER.format(name="a")
                 + '<xacro:include filename="b.xacro"/><link name="a_link"/></robot>')
    robot = import_urdf_file(top)
    assert {l.name for l in robot.links} == {"a_link", "b_link", "c_link"}


def test_duplicate_include_deduped(tmp_path):
    _write(tmp_path, "shared.xacro",
           XACRO_HEADER.format(name="s") + '<link name="shared"/></robot>')
    top = _write(tmp_path, "robot.xacro",
                 XACRO_HEADER.format(name="top")
                 + '<xacro:include filename="shared.xacro"/>'
                 + '<xacro:include filename="shared.xacro"/>'
                 + '<link name="base"/></robot>')
    robot = import_urdf_file(top)
    # `shared` appears once despite two includes (include-guard semantics).
    assert [l.name for l in robot.links].count("shared") == 1


# --- package:// includes via package_roots ---------------------------------

def test_package_include(tmp_path):
    pkg_dir = tmp_path / "my_robot_description"
    _write(pkg_dir, "urdf/arm.xacro",
           XACRO_HEADER.format(name="arm") + '<link name="arm_link"/></robot>')
    top = _write(tmp_path, "robot.xacro",
                 XACRO_HEADER.format(name="top")
                 + '<xacro:include filename="package://my_robot_description/urdf/arm.xacro"/>'
                 + '<link name="base"/></robot>')
    robot = import_urdf_file(top, package_roots={"my_robot_description": str(pkg_dir)})
    assert {l.name for l in robot.links} == {"base", "arm_link"}


def test_find_substitution_in_include(tmp_path):
    """`$(find pkg)/...` is rewritten to package:// then resolved locally."""
    pkg_dir = tmp_path / "rover"
    _write(pkg_dir, "wheel.xacro",
           XACRO_HEADER.format(name="w") + '<link name="wheel"/></robot>')
    top = _write(tmp_path, "robot.xacro",
                 XACRO_HEADER.format(name="top")
                 + '<xacro:include filename="$(find rover)/wheel.xacro"/>'
                 + '<link name="chassis"/></robot>')
    robot = import_urdf_file(top, package_roots={"rover": str(pkg_dir)})
    assert {l.name for l in robot.links} == {"chassis", "wheel"}


def test_unknown_package_root_raises(tmp_path):
    top = _write(tmp_path, "robot.xacro",
                 XACRO_HEADER.format(name="top")
                 + '<xacro:include filename="package://ghost/x.xacro"/>'
                 + '<link name="base"/></robot>')
    with pytest.raises(ImportError_) as exc:
        import_urdf_file(top)
    assert "ghost" in str(exc.value)


def test_missing_include_file_raises(tmp_path):
    top = _write(tmp_path, "robot.xacro",
                 XACRO_HEADER.format(name="top")
                 + '<xacro:include filename="gone.xacro"/>'
                 + '<link name="base"/></robot>')
    with pytest.raises(ImportError_) as exc:
        import_urdf_file(top)
    assert "cannot read include" in str(exc.value)


# --- network-free guarantee ------------------------------------------------

def test_remote_include_rejected(tmp_path):
    top = _write(tmp_path, "robot.xacro",
                 XACRO_HEADER.format(name="top")
                 + '<xacro:include filename="https://example.com/x.xacro"/>'
                 + '<link name="base"/></robot>')
    with pytest.raises(ImportError_) as exc:
        import_urdf_file(top)
    assert "not allowed in local import" in str(exc.value)


# --- resolve_includes_local standalone -------------------------------------

def test_resolve_includes_local_returns_spliced_text(tmp_path):
    _write(tmp_path, "part.xacro",
           XACRO_HEADER.format(name="p") + '<link name="from_include"/></robot>')
    text = (XACRO_HEADER.format(name="top")
            + '<xacro:include filename="part.xacro"/>'
            + '<link name="base"/></robot>')
    out = resolve_includes_local(text, tmp_path)
    assert "from_include" in out
    assert "xacro:include" not in out  # placeholder removed


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
