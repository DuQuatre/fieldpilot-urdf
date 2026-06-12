"""Tests for the URDF model registry.

Uses tmp_path + monkeypatch on FIELDPILOT_URDF_DATA_DIR so the real /data
directory is never touched.

Run: python3 -m pytest app/urdf/test_storage.py -q  (from pydexpi-server/)
"""
from __future__ import annotations

import time

import pytest

from fieldpilot_urdf import from_xml
from fieldpilot_urdf import storage as st


SAMPLE = """\
<robot name="arm">
  <link name="base"/>
  <link name="upper"/>
  <link name="gripper"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="upper"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
  <joint name="j2" type="fixed">
    <parent link="upper"/><child link="gripper"/>
  </joint>
</robot>
"""

OTHER = """\
<robot name="gripper-only">
  <link name="hand"/>
</robot>
"""


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(st.DATA_DIR_ENV, str(tmp_path))
    yield


def test_save_returns_metadata():
    robot = from_xml(SAMPLE)
    meta = st.save_robot(robot, source_file="arm.urdf")
    assert len(meta["model_id"]) == 8
    assert meta["name"] == "arm"
    assert meta["source_file"] == "arm.urdf"
    assert meta["n_links"] == 3
    assert meta["n_joints"] == 2
    assert meta["created_at"]


def test_load_round_trip():
    saved = st.save_robot(from_xml(SAMPLE))
    loaded = st.load_robot(saved["model_id"])
    assert loaded is not None
    assert loaded.name == "arm"
    assert [l.name for l in loaded.links] == ["base", "upper", "gripper"]
    assert [j.name for j in loaded.joints] == ["j1", "j2"]


def test_load_meta():
    saved = st.save_robot(from_xml(SAMPLE), source_file="arm.urdf")
    meta = st.load_meta(saved["model_id"])
    assert meta == saved


def test_load_missing_returns_none():
    assert st.load_robot("deadbeef") is None
    assert st.load_meta("deadbeef") is None


def test_list_orders_newest_first():
    a = st.save_robot(from_xml(SAMPLE))
    time.sleep(0.01)  # ensure distinct created_at
    b = st.save_robot(from_xml(OTHER))
    listing = st.list_robots()
    assert [s["model_id"] for s in listing] == [b["model_id"], a["model_id"]]
    assert listing[0]["name"] == "gripper-only"


def test_delete_removes_files_and_index():
    saved = st.save_robot(from_xml(SAMPLE))
    mid = saved["model_id"]
    assert st._urdf_path(mid).exists()
    assert st._meta_path(mid).exists()
    assert mid in st._load_index()
    assert st.delete_robot(mid) is True
    assert not st._urdf_path(mid).exists()
    assert not st._meta_path(mid).exists()
    assert mid not in st._load_index()


def test_delete_missing_returns_false():
    assert st.delete_robot("nope1234") is False


def test_index_survives_multiple_writes():
    ids = [st.save_robot(from_xml(SAMPLE))["model_id"] for _ in range(3)]
    index = st._load_index()
    for mid in ids:
        assert mid in index
    assert len({s["model_id"] for s in st.list_robots()}) == 3


def test_saved_urdf_is_round_trippable_xml():
    """The on-disk .urdf must still be parseable by from_xml."""
    saved = st.save_robot(from_xml(SAMPLE))
    raw = st._urdf_path(saved["model_id"]).read_text()
    assert "<robot" in raw
    again = from_xml(raw)
    assert again.name == "arm"
    assert len(again.joints) == 2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
