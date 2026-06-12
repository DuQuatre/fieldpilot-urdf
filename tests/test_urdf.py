"""Standalone tests — run: python3 -m app.urdf.test_urdf (from pydexpi-server/)."""
from __future__ import annotations

from fieldpilot_urdf import from_xml, to_xml
from fieldpilot_urdf.models import Joint, Link, Robot

SAMPLE = """\
<robot name="r2d2">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="2.5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/>
    </inertial>
    <visual>
      <geometry><cylinder radius="0.2" length="0.6"/></geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry><box size="0.4 0.4 0.6"/></geometry>
    </collision>
  </link>
  <link name="head"/>
  <joint name="neck" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
</robot>
"""


def test_parse():
    r = from_xml(SAMPLE)
    assert r.name == "r2d2"
    assert len(r.links) == 2 and len(r.joints) == 1
    base = r.link("base_link")
    assert base.inertial.mass == 2.5
    assert base.visuals[0].geometry.kind == "cylinder"
    assert base.collisions[0].geometry.kind == "box"
    j = r.joint("neck")
    assert j.parent == "base_link" and j.child == "head"
    assert j.limit.upper == 1.57


def test_round_trip():
    r1 = from_xml(SAMPLE)
    r2 = from_xml(to_xml(r1))
    # Compare by dump (ids are excluded from serialization)
    assert r1.model_dump() == r2.model_dump()


def test_joint_limit_required():
    try:
        Joint(name="j", type="revolute", parent="a", child="b")
    except ValueError as e:
        assert "limit" in str(e).lower()
        return
    raise AssertionError("expected ValueError for revolute joint without <limit>")


def test_dangling_joint_ref():
    try:
        Robot(name="x",
              links=[Link(name="a")],
              joints=[Joint(name="j", type="fixed", parent="a", child="ghost")])
    except ValueError as e:
        assert "ghost" in str(e)
        return
    raise AssertionError("expected ValueError for joint referencing missing link")


if __name__ == "__main__":
    test_parse()
    test_round_trip()
    test_joint_limit_required()
    test_dangling_joint_ref()
    print("OK — 4 tests passed")
