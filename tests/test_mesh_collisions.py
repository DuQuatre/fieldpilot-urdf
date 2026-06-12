"""Tests for mesh-aware collision support.

Generates STL fixtures on disk via trimesh in tmp_path; nothing here writes to
the real data directory.

Run: python3 -m pytest app/urdf/test_mesh_collisions.py -q  (from pydexpi-server/)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from fieldpilot_urdf import from_xml
from fieldpilot_urdf.collisions import (
    MeshResolver, clear_mesh_cache, detect_self_collisions,
    link_collision_aabbs, local_aabb, unresolved_meshes,
)
from fieldpilot_urdf.models import Mesh


@pytest.fixture(autouse=True)
def _drop_cache():
    clear_mesh_cache()
    yield
    clear_mesh_cache()


@pytest.fixture
def stl_dir(tmp_path):
    """Two STL boxes: a 1×2×3 box and a 0.5-cube."""
    (tmp_path / "meshes").mkdir()
    trimesh.creation.box(extents=(1, 2, 3)).export(tmp_path / "meshes" / "big.stl")
    trimesh.creation.box(extents=(0.5, 0.5, 0.5)).export(tmp_path / "meshes" / "small.stl")
    return tmp_path


# --- resolver --------------------------------------------------------------

def test_resolver_relative_path(stl_dir):
    r = MeshResolver(mesh_dir=stl_dir)
    assert r.resolve("meshes/big.stl") == stl_dir / "meshes/big.stl"


def test_resolver_absolute_path(stl_dir):
    r = MeshResolver()
    abs_p = stl_dir / "meshes" / "big.stl"
    assert r.resolve(str(abs_p)) == abs_p


def test_resolver_file_uri(stl_dir):
    r = MeshResolver()
    assert r.resolve(f"file://{stl_dir / 'meshes' / 'big.stl'}") == \
        stl_dir / "meshes" / "big.stl"


def test_resolver_package_uri(stl_dir):
    r = MeshResolver(packages={"my_pkg": stl_dir})
    assert r.resolve("package://my_pkg/meshes/big.stl") == stl_dir / "meshes/big.stl"


def test_resolver_unknown_package(stl_dir):
    assert MeshResolver().resolve("package://nope/x.stl") is None


def test_resolver_no_mesh_dir_for_relative():
    assert MeshResolver().resolve("meshes/x.stl") is None


# --- local_aabb on Mesh ----------------------------------------------------

def test_local_aabb_loads_stl(stl_dir):
    r = MeshResolver(mesh_dir=stl_dir)
    m = Mesh(filename="meshes/big.stl", scale=(1.0, 1.0, 1.0))
    aabb = local_aabb(m, resolver=r)
    assert aabb is not None
    assert np.allclose(aabb[0], (-0.5, -1.0, -1.5), atol=1e-6)
    assert np.allclose(aabb[1], (0.5, 1.0, 1.5), atol=1e-6)


def test_local_aabb_applies_scale(stl_dir):
    r = MeshResolver(mesh_dir=stl_dir)
    m = Mesh(filename="meshes/big.stl", scale=(2.0, 0.5, 1.0))
    aabb = local_aabb(m, resolver=r)
    assert np.allclose(aabb[0], (-1.0, -0.5, -1.5), atol=1e-6)
    assert np.allclose(aabb[1], (1.0, 0.5, 1.5), atol=1e-6)


def test_local_aabb_no_resolver_returns_none():
    """Backwards compat: no resolver means mesh-blind, like before."""
    m = Mesh(filename="meshes/big.stl")
    assert local_aabb(m) is None


def test_local_aabb_missing_file(stl_dir):
    r = MeshResolver(mesh_dir=stl_dir)
    m = Mesh(filename="meshes/nope.stl")
    assert local_aabb(m, resolver=r) is None  # silent skip, current contract


# --- caching ---------------------------------------------------------------

def test_mesh_aabb_cached(stl_dir, monkeypatch):
    r = MeshResolver(mesh_dir=stl_dir)
    m = Mesh(filename="meshes/big.stl")
    a1 = local_aabb(m, resolver=r)

    # Trip a sentinel if trimesh.load is called again — the cached entry must
    # be returned without a second load.
    calls = {"n": 0}
    real_load = trimesh.load

    def counting_load(*args, **kwargs):
        calls["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(trimesh, "load", counting_load)
    a2 = local_aabb(m, resolver=r)
    assert calls["n"] == 0
    assert np.allclose(a1[0], a2[0]) and np.allclose(a1[1], a2[1])


# --- end-to-end via robot --------------------------------------------------

URDF_OVERLAP = """\
<robot name="meshbot">
  <link name="root"/>
  <link name="a">
    <collision><geometry><mesh filename="meshes/big.stl"/></geometry></collision>
  </link>
  <link name="b">
    <collision>
      <origin xyz="0.2 0 0"/>
      <geometry><mesh filename="meshes/big.stl"/></geometry>
    </collision>
  </link>
  <joint name="j1" type="fixed">
    <parent link="root"/><child link="a"/>
  </joint>
  <joint name="j2" type="fixed">
    <parent link="root"/><child link="b"/>
    <origin xyz="0.2 0 0"/>
  </joint>
</robot>
"""

URDF_NON_OVERLAP = """\
<robot name="meshbot">
  <link name="root"/>
  <link name="a">
    <collision><geometry><mesh filename="meshes/small.stl"/></geometry></collision>
  </link>
  <link name="b">
    <collision>
      <origin xyz="5 0 0"/>
      <geometry><mesh filename="meshes/small.stl"/></geometry>
    </collision>
  </link>
  <joint name="j1" type="fixed">
    <parent link="root"/><child link="a"/>
  </joint>
  <joint name="j2" type="fixed">
    <parent link="root"/><child link="b"/>
    <origin xyz="5 0 0"/>
  </joint>
</robot>
"""


def test_self_collision_mesh_overlap(stl_dir):
    robot = from_xml(URDF_OVERLAP)
    r = MeshResolver(mesh_dir=stl_dir)
    # Without resolver: silently mesh-blind → no hits.
    assert detect_self_collisions(robot) == []
    # With resolver: AABBs overlap → 'a' and 'b' reported.
    hits = detect_self_collisions(robot, mesh_resolver=r)
    assert frozenset(hits[0]) == frozenset({"a", "b"})


def test_self_collision_mesh_no_overlap(stl_dir):
    robot = from_xml(URDF_NON_OVERLAP)
    r = MeshResolver(mesh_dir=stl_dir)
    assert detect_self_collisions(robot, mesh_resolver=r) == []


def test_link_collision_aabbs_includes_meshes(stl_dir):
    robot = from_xml(URDF_OVERLAP)
    r = MeshResolver(mesh_dir=stl_dir)
    aabbs = link_collision_aabbs(robot, mesh_resolver=r)
    assert "a" in aabbs and "b" in aabbs


def test_unresolved_meshes_reports_missing(stl_dir):
    bad = """\
<robot name="b">
  <link name="root"/>
  <link name="x">
    <collision><geometry><mesh filename="meshes/missing.stl"/></geometry></collision>
  </link>
  <joint name="j" type="fixed"><parent link="root"/><child link="x"/></joint>
</robot>
"""
    robot = from_xml(bad)
    r = MeshResolver(mesh_dir=stl_dir)
    issues = unresolved_meshes(robot, mesh_resolver=r)
    assert len(issues) == 1
    link, fname, reason = issues[0]
    assert link == "x"
    assert fname == "meshes/missing.stl"
    assert "not found" in reason


def test_unresolved_meshes_empty_without_resolver():
    bad = """\
<robot name="b">
  <link name="root"/>
  <link name="x">
    <collision><geometry><mesh filename="meshes/missing.stl"/></geometry></collision>
  </link>
  <joint name="j" type="fixed"><parent link="root"/><child link="x"/></joint>
</robot>
"""
    robot = from_xml(bad)
    assert unresolved_meshes(robot) == []  # mesh-blind = nothing to report


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
