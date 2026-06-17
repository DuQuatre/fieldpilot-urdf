"""Tests for the pure-Python mesh bounds reader (src/fieldpilot_urdf/mesh.py).

All fixtures are hand-built byte/text files — NO trimesh, proving the STL/OBJ/PLY
path needs no [mesh] extra. Run: python3 -m pytest tests/test_mesh_formats.py -q
"""
from __future__ import annotations

import struct

import pytest

from fieldpilot_urdf import read_mesh_bounds, SUPPORTED_FORMATS
from fieldpilot_urdf.collisions import MeshResolver, clear_mesh_cache, local_aabb
from fieldpilot_urdf.models import Mesh

# Every fixture below spans the same box: min (-1,-2,-3), max (1, 2, 3).
EXPECTED = ((-1.0, -2.0, -3.0), (1.0, 2.0, 3.0))
TRIS = [(0, 0, 0), (1, 2, 3), (-1, -2, -3)]


@pytest.fixture(autouse=True)
def _clear():
    clear_mesh_cache()
    yield
    clear_mesh_cache()


def _approx(bounds):
    return (pytest.approx(bounds[0]), pytest.approx(bounds[1]))


# --- OBJ -------------------------------------------------------------------

def test_obj(tmp_path):
    p = tmp_path / "m.obj"
    p.write_text("vn 0 0 1\nvt 0 0\nv 0 0 0\nv 1 2 3\nv -1 -2 -3\nf 1 2 3\n")
    assert read_mesh_bounds(p) == _approx(EXPECTED)


def test_obj_ignores_non_vertex_lines(tmp_path):
    p = tmp_path / "m.obj"
    # vn/vt values are large but must NOT affect the geometric-vertex bounds.
    p.write_text("v 0 0 0\nvn 99 99 99\nvt 50 50\nv 1 2 3\nv -1 -2 -3\n")
    assert read_mesh_bounds(p) == _approx(EXPECTED)


# --- STL -------------------------------------------------------------------

def test_stl_ascii(tmp_path):
    lines = ["solid s", " facet normal 0 0 0", "  outer loop"]
    lines += [f"   vertex {x} {y} {z}" for x, y, z in TRIS]
    lines += ["  endloop", " endfacet", "endsolid s"]
    p = tmp_path / "m.stl"
    p.write_text("\n".join(lines))
    assert read_mesh_bounds(p) == _approx(EXPECTED)


def test_stl_binary(tmp_path):
    flat = [c for v in TRIS for c in v]
    rec = struct.pack("<I", 1) + struct.pack("<12fH", 0, 0, 0, *flat, 0)
    p = tmp_path / "m.stl"
    p.write_bytes(b"\x00" * 80 + rec)
    assert read_mesh_bounds(p) == _approx(EXPECTED)


def test_stl_binary_header_starting_with_solid(tmp_path):
    """A binary STL whose 80-byte header text happens to start 'solid' must
    still be detected as binary (size rule wins over the prefix heuristic)."""
    flat = [c for v in TRIS for c in v]
    header = b"solid exported_by_some_tool".ljust(80, b"\x00")
    p = tmp_path / "m.stl"
    p.write_bytes(header + struct.pack("<I", 1) + struct.pack("<12fH", 0, 0, 0, *flat, 0))
    assert read_mesh_bounds(p) == _approx(EXPECTED)


# --- PLY -------------------------------------------------------------------

_PLY_HEADER_ASCII = (
    "ply\nformat ascii 1.0\nelement vertex 3\n"
    "property float x\nproperty float y\nproperty float z\nend_header\n"
)


def test_ply_ascii(tmp_path):
    body = "\n".join(f"{x} {y} {z}" for x, y, z in TRIS) + "\n"
    p = tmp_path / "m.ply"
    p.write_text(_PLY_HEADER_ASCII + body)
    assert read_mesh_bounds(p) == _approx(EXPECTED)


def test_ply_ascii_extra_properties_pick_xyz(tmp_path):
    header = (
        "ply\nformat ascii 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    body = "\n".join(f"{x} {y} {z} 10 20 30" for x, y, z in TRIS) + "\n"
    p = tmp_path / "m.ply"
    p.write_text(header + body)
    assert read_mesh_bounds(p) == _approx(EXPECTED)


def _ply_binary(endian_word, packfmt, tmp_path):
    header = (
        f"ply\nformat {endian_word} 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    ).encode("ascii")
    flat = [c for v in TRIS for c in v]
    p = tmp_path / "m.ply"
    p.write_bytes(header + struct.pack(packfmt, *flat))
    return read_mesh_bounds(p)


def test_ply_binary_little_endian(tmp_path):
    assert _ply_binary("binary_little_endian", "<9f", tmp_path) == _approx(EXPECTED)


def test_ply_binary_big_endian(tmp_path):
    assert _ply_binary("binary_big_endian", ">9f", tmp_path) == _approx(EXPECTED)


def test_ply_binary_extra_props_offsets(tmp_path):
    """A double x/y/z plus a trailing uchar must still resolve x,y,z by name."""
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar flag\nend_header\n"
    ).encode("ascii")
    rows = b"".join(struct.pack("<3dB", x, y, z, 7) for x, y, z in TRIS)
    p = tmp_path / "m.ply"
    p.write_bytes(header + rows)
    assert read_mesh_bounds(p) == _approx(EXPECTED)


def test_ply_non_vertex_first_defers(tmp_path):
    """Vertex not the first element → return None (caller falls back to trimesh)."""
    header = (
        "ply\nformat ascii 1.0\nelement camera 1\nproperty float view\n"
        "element vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
        "end_header\n0.0\n1 2 3\n"
    )
    p = tmp_path / "m.ply"
    p.write_text(header)
    assert read_mesh_bounds(p) is None


# --- dispatch / robustness -------------------------------------------------

def test_supported_formats():
    assert SUPPORTED_FORMATS == {".stl", ".obj", ".ply"}


def test_unknown_extension_returns_none(tmp_path):
    p = tmp_path / "m.dae"
    p.write_text("<COLLADA/>")
    assert read_mesh_bounds(p) is None  # defer to trimesh


def test_extension_case_insensitive(tmp_path):
    p = tmp_path / "M.OBJ"
    p.write_text("v 0 0 0\nv 1 2 3\nv -1 -2 -3\n")
    assert read_mesh_bounds(p) == _approx(EXPECTED)


def test_empty_file_returns_none(tmp_path):
    p = tmp_path / "m.stl"
    p.write_bytes(b"")
    assert read_mesh_bounds(p) is None


def test_corrupt_binary_stl_returns_none(tmp_path):
    """Truncated binary STL (claims 100 triangles, has none) → None, not a crash."""
    p = tmp_path / "m.stl"
    p.write_bytes(b"\x00" * 80 + struct.pack("<I", 100) + b"\x00" * 10)
    assert read_mesh_bounds(p) is None


def test_missing_file_returns_none(tmp_path):
    assert read_mesh_bounds(tmp_path / "nope.stl") is None


# --- integration through the collision AABB path ---------------------------

def test_local_aabb_uses_native_obj_reader_without_trimesh(tmp_path):
    """local_aabb resolves an OBJ mesh's AABB via the native reader — OBJ here
    is plain text, no trimesh involved — and applies the URDF scale."""
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / "part.obj").write_text(
        "v 0 0 0\nv 1 2 3\nv -1 -2 -3\n")
    r = MeshResolver(mesh_dir=tmp_path)
    m = Mesh(filename="meshes/part.obj", scale=(2.0, 1.0, 0.5))
    mn, mx = local_aabb(m, resolver=r)
    assert tuple(mn) == pytest.approx((-2.0, -2.0, -1.5))
    assert tuple(mx) == pytest.approx((2.0, 2.0, 1.5))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
