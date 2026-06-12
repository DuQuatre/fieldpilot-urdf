"""Standalone tests for graph + diagnostics.
Run: python3 -m app.urdf.test_graph (from pydexpi-server/).
"""
from __future__ import annotations

from fieldpilot_urdf import from_xml
from fieldpilot_urdf.diagnostics import run_all, summary
from fieldpilot_urdf.graph import build_graph, chain, is_tree, joints_on_path, leaf_links, root_links
from fieldpilot_urdf.models import Inertia, Inertial, Joint, JointLimit, Link, Robot

SAMPLE_OK = """\
<robot name="arm">
  <link name="base"/>
  <link name="upper"/>
  <link name="lower"/>
  <link name="gripper"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="upper"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="upper"/><child link="lower"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="5" velocity="1"/>
  </joint>
  <joint name="j3" type="fixed">
    <parent link="lower"/><child link="gripper"/>
  </joint>
</robot>
"""


def test_graph_basics():
    r = from_xml(SAMPLE_OK)
    G = build_graph(r)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3
    assert root_links(G) == ["base"]
    assert set(leaf_links(G)) == {"gripper"}
    assert is_tree(G)
    path = chain(G, "base", "gripper")
    assert path == ["base", "upper", "lower", "gripper"]
    assert joints_on_path(G, path) == ["j1", "j2", "j3"]


def test_clean_robot_no_findings():
    r = from_xml(SAMPLE_OK)
    findings = run_all(r)
    assert findings == [], f"unexpected findings on clean URDF: {findings}"


def test_r001_multiple_roots():
    # Two floating links -> two roots
    r = Robot(name="x", links=[Link(name="a"), Link(name="b")], joints=[])
    findings = run_all(r)
    codes = {f.code for f in findings}
    assert "R001" in codes


def test_r003_bad_limit():
    r = Robot(
        name="x",
        links=[Link(name="a"), Link(name="b")],
        joints=[Joint(name="j", type="revolute", parent="a", child="b",
                      limit=JointLimit(lower=1.0, upper=0.5, effort=5, velocity=1))],
    )
    findings = run_all(r)
    assert any(f.code == "R003" and "lower" in f.message for f in findings)


def test_r004_negative_mass():
    r = Robot(
        name="x",
        links=[Link(name="a", inertial=Inertial(mass=-1.0))],
        joints=[],
    )
    findings = run_all(r)
    assert any(f.code == "R004" for f in findings)


def test_r005_non_psd_inertia():
    # Diagonal with a negative entry -> negative eigenvalue
    bad = Inertial(mass=1.0, inertia=Inertia(ixx=-1.0, iyy=1.0, izz=1.0))
    r = Robot(name="x", links=[Link(name="a", inertial=bad)], joints=[])
    findings = run_all(r)
    assert any(f.code == "R005" for f in findings)


def test_summary():
    r = Robot(
        name="x",
        links=[Link(name="a", inertial=Inertial(mass=-1.0))],
        joints=[],
    )
    s = summary(run_all(r))
    assert s["total"] >= 1 and s["error"] >= 1


if __name__ == "__main__":
    test_graph_basics()
    test_clean_robot_no_findings()
    test_r001_multiple_roots()
    test_r003_bad_limit()
    test_r004_negative_mass()
    test_r005_non_psd_inertia()
    test_summary()
    print("OK — 7 tests passed")
