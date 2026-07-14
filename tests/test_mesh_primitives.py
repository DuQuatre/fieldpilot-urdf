"""Primitive mesh generation smoke tests."""

from __future__ import annotations

import pytest

trimesh = pytest.importorskip("trimesh")

from fieldpilot_urdf.mesh_primitives import save_box_mesh, save_cylinder_mesh, save_sphere_mesh


def test_save_box_mesh(tmp_path):
    out = save_box_mesh(0.1, 0.2, 0.3, tmp_path / "box.stl")
    assert out.exists()
    mesh = trimesh.load(out)
    assert len(mesh.vertices) > 0
    extents = sorted(mesh.extents)
    assert extents == pytest.approx(sorted([0.1, 0.2, 0.3]), abs=1e-6)


def test_save_cylinder_mesh(tmp_path):
    out = save_cylinder_mesh(0.05, 0.4, tmp_path / "cyl.stl")
    assert out.exists()
    mesh = trimesh.load(out)
    assert len(mesh.vertices) > 0
    # cylinder extents: diameter x diameter x length
    assert sorted(mesh.extents)[-1] == pytest.approx(0.4, abs=1e-3)
    assert sorted(mesh.extents)[0] == pytest.approx(0.1, abs=1e-3)
    # base at local origin, extending along +Z -- NOT centered (see docstring)
    assert mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    assert mesh.bounds[1][2] == pytest.approx(0.4, abs=1e-6)


def test_save_sphere_mesh(tmp_path):
    out = save_sphere_mesh(0.02, tmp_path / "sphere.stl")
    assert out.exists()
    mesh = trimesh.load(out)
    assert len(mesh.vertices) > 0
    for extent in mesh.extents:
        assert extent == pytest.approx(0.04, abs=1e-3)


def test_missing_trimesh_raises_actionable_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "trimesh":
            raise ImportError("no trimesh")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from fieldpilot_urdf.mesh_primitives import _require_trimesh

    with pytest.raises(ImportError, match=r"pip install 'fieldpilot-urdf\[mesh\]'"):
        _require_trimesh()
