"""Primitive mesh generation (box/cylinder/sphere) via trimesh.

Ported from the MecAI project (MIT): ``mecai.cad.primitives``. No functional
retargeting was needed — unlike most ports here, this module has zero
coupling to ``Robot``/``Link``/``Joint``; it takes plain floats and a path,
and returns a path.

Note the target schema differs from MecAI's, though: MecAI's ``Link`` has a
single ``mesh_uri: str | None`` field, but this package's :class:`Link`
(``models.py``) has ``visuals: list[Visual]``, each a
``Visual(origin: Origin | None, geometry: Box | Cylinder | Sphere | Mesh)``.
So here the caller builds a ``Visual(geometry=Mesh(filename=str(path)))`` and
appends it to ``link.visuals``, rather than assigning ``mesh_uri`` directly.
``Visual.origin`` *does* support a position/orientation offset (unlike
MecAI's mesh export) — but :func:`save_cylinder_mesh` still bakes its
base-at-origin convention into the mesh's own vertices rather than relying
on that, so the two ports stay behaviourally identical and either can be
swapped in without the caller needing to know which one it's calling.

trimesh>=4.0 is already an optional dependency here (the ``mesh`` extra,
currently used only by :mod:`fieldpilot_urdf.collisions`'s AABB fallback) —
this gives it a second, more central use.
"""

from __future__ import annotations

from pathlib import Path


def _require_trimesh():
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - exercised manually
        raise ImportError(
            "trimesh is not installed. Install fieldpilot-urdf's mesh extras: "
            "pip install 'fieldpilot-urdf[mesh]'"
        ) from exc
    return trimesh


def save_box_mesh(x: float, y: float, z: float, out_path: str | Path) -> Path:
    """Write an axis-aligned box (extents x, y, z), centered at the local
    origin, to ``out_path``.

    Format is inferred from ``out_path``'s suffix (.stl, .obj, .glb, ...).
    Unlike :func:`save_cylinder_mesh`, this stays centered -- a box is more
    often a plate/block positioned via its own ``Visual.origin`` than a
    chain segment with an implied base-to-tip convention.
    """
    trimesh = _require_trimesh()
    mesh = trimesh.creation.box(extents=(x, y, z))
    out_path = Path(out_path)
    mesh.export(out_path)
    return out_path


def save_cylinder_mesh(radius: float, length: float, out_path: str | Path,
                        *, sections: int = 32) -> Path:
    """Write a cylinder to ``out_path``, base at the local origin, extending
    along +Z for ``length`` -- NOT centered at the origin (trimesh's own
    default). Matches the link convention used throughout MecAI and this
    package's own examples (a segment spans from its proximal joint, at the
    link's local origin, to its distal joint at local z=length). Baked into
    the mesh's own vertices rather than relying on ``Visual.origin`` -- see
    the module docstring for why.
    """
    trimesh = _require_trimesh()
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    mesh.apply_translation([0.0, 0.0, length / 2])
    out_path = Path(out_path)
    mesh.export(out_path)
    return out_path


def save_sphere_mesh(radius: float, out_path: str | Path, *, subdivisions: int = 2) -> Path:
    """Write a sphere to ``out_path``."""
    trimesh = _require_trimesh()
    mesh = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)
    out_path = Path(out_path)
    mesh.export(out_path)
    return out_path
