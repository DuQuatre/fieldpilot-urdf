"""AABB collision approximation in world frame.

Cheap conservative bound: each <collision> shape -> local AABB -> 8 corners
transformed by world matrix -> world AABB. Mesh shapes can be resolved by
passing a MeshResolver; without one they are silently skipped (the original
contract). Useful for catching gross self-overlap; not a substitute for real
broad/narrow-phase collision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .fk import forward_kinematics, origin_to_T
from .mesh import read_mesh_bounds
from .models import Box, Cylinder, Geometry, Mesh, Robot, Sphere

AABB = tuple[np.ndarray, np.ndarray]  # (min_xyz, max_xyz)


# --- mesh resolution -------------------------------------------------------

@dataclass
class MeshResolver:
    """Maps URDF <mesh filename="..."/> values to on-disk files.

    URDF mesh filenames commonly use three forms:
      - "package://my_pkg/meshes/arm.stl"  → packages["my_pkg"] / "meshes/arm.stl"
      - "file:///abs/path/arm.stl"         → /abs/path/arm.stl
      - "meshes/arm.stl"                   → mesh_dir / meshes/arm.stl  (or absolute)
    """
    mesh_dir: Optional[Path] = None
    packages: dict[str, Path] = field(default_factory=dict)

    def resolve(self, filename: str) -> Optional[Path]:
        if filename.startswith("package://"):
            rest = filename[len("package://"):]
            pkg, _, sub = rest.partition("/")
            root = self.packages.get(pkg)
            if root is None:
                return None
            return root / sub
        if filename.startswith("file://"):
            return Path(filename[len("file://"):])
        p = Path(filename)
        if p.is_absolute():
            return p
        if self.mesh_dir is None:
            return None
        return self.mesh_dir / p


# Cache: (resolved abs path, scale tuple) -> local AABB. Loading + bound
# computation on a real ROS mesh is ~tens of ms, so caching matters when the
# same mesh is referenced by many links (very common).
_MESH_AABB_CACHE: dict[tuple[str, tuple[float, float, float]], AABB] = {}


def _load_mesh_aabb(m: Mesh, resolver: Optional[MeshResolver]) -> Optional[AABB]:
    if resolver is None:
        return None
    path = resolver.resolve(m.filename)
    if path is None or not path.exists():
        return None
    key = (str(path.resolve()), tuple(m.scale))
    cached = _MESH_AABB_CACHE.get(key)
    if cached is not None:
        return cached

    # Try the pure-Python reader first (STL/OBJ/PLY — no [mesh] extra needed);
    # fall back to trimesh for everything else (COLLADA .dae, glTF, …).
    bounds = read_mesh_bounds(path)
    if bounds is not None:
        raw_mn, raw_mx = np.array(bounds[0], dtype=float), np.array(bounds[1], dtype=float)
    else:
        try:
            import trimesh  # lazy: only paid when the native reader can't handle it
            mesh = trimesh.load(path, force="mesh")
        except Exception:  # noqa: BLE001 — corrupt file, unsupported format, etc.
            return None
        if mesh is None or not hasattr(mesh, "bounds") or mesh.bounds is None:
            return None
        raw_mn, raw_mx = np.array(mesh.bounds[0], dtype=float), np.array(mesh.bounds[1], dtype=float)

    mn = raw_mn * np.array(m.scale, dtype=float)
    mx = raw_mx * np.array(m.scale, dtype=float)
    aabb = (np.minimum(mn, mx), np.maximum(mn, mx))
    _MESH_AABB_CACHE[key] = aabb
    return aabb


def clear_mesh_cache() -> None:
    """Drop the (path, scale) → AABB cache. Mostly for tests."""
    _MESH_AABB_CACHE.clear()


def local_aabb(g: Geometry, resolver: Optional[MeshResolver] = None) -> Optional[AABB]:
    if isinstance(g, Box):
        hs = np.array(g.size) / 2.0
        return (-hs, hs)
    if isinstance(g, Cylinder):
        r, L = g.radius, g.length / 2.0
        return (np.array([-r, -r, -L]), np.array([r, r, L]))
    if isinstance(g, Sphere):
        r = g.radius
        return (np.array([-r, -r, -r]), np.array([r, r, r]))
    if isinstance(g, Mesh):
        return _load_mesh_aabb(g, resolver)
    return None


def transform_aabb(T: np.ndarray, aabb: AABB) -> AABB:
    """Transform 8 corners of a local AABB by T, return enclosing world AABB."""
    mn, mx = aabb
    corners = np.array([[x, y, z, 1.0]
                        for x in (mn[0], mx[0])
                        for y in (mn[1], mx[1])
                        for z in (mn[2], mx[2])])
    world = (T @ corners.T).T[:, :3]
    return world.min(axis=0), world.max(axis=0)


def aabb_overlap(a: AABB, b: AABB, tol: float = 0.0) -> bool:
    amn, amx = a
    bmn, bmx = b
    return bool(np.all(amx >= bmn - tol) and np.all(bmx >= amn - tol))


def link_collision_aabbs(
    robot: Robot,
    q: Optional[dict[str, float]] = None,
    mesh_resolver: Optional[MeshResolver] = None,
) -> dict[str, list[tuple[str, AABB]]]:
    """For each link, list of (collision_name, world_aabb).

    Mesh shapes contribute only when mesh_resolver can locate the file; on
    failure they're silently skipped (matches the original mesh-blind contract).
    """
    tfs = forward_kinematics(robot, q=q)
    out: dict[str, list[tuple[str, AABB]]] = {}
    for link in robot.links:
        link_T = tfs.get(link.name)
        if link_T is None:
            continue
        boxes: list[tuple[str, AABB]] = []
        for idx, c in enumerate(link.collisions):
            local = local_aabb(c.geometry, resolver=mesh_resolver)
            if local is None:
                continue
            geom_T = link_T @ origin_to_T(c.origin)
            boxes.append((c.name or f"col_{idx}", transform_aabb(geom_T, local)))
        if boxes:
            out[link.name] = boxes
    return out


def adjacent_link_pairs(robot: Robot) -> set[frozenset[str]]:
    return {frozenset({j.parent, j.child}) for j in robot.joints}


def detect_self_collisions(
    robot: Robot,
    q: Optional[dict[str, float]] = None,
    ignore_adjacent: bool = True,
    mesh_resolver: Optional[MeshResolver] = None,
) -> list[tuple[str, str]]:
    """Return list of (link_a, link_b) link pairs whose AABBs overlap.

    With ignore_adjacent=True (default), pairs connected by a joint are
    excluded — adjacent links are expected to touch at the joint. Mesh
    shapes contribute only when mesh_resolver can resolve the file.
    """
    aabbs = link_collision_aabbs(robot, q=q, mesh_resolver=mesh_resolver)
    adj = adjacent_link_pairs(robot) if ignore_adjacent else set()
    names = list(aabbs.keys())
    hits: list[tuple[str, str]] = []
    for i in range(len(names)):
        for k in range(i + 1, len(names)):
            a, b = names[i], names[k]
            if frozenset({a, b}) in adj:
                continue
            if any(aabb_overlap(ba[1], bb[1])
                   for ba in aabbs[a] for bb in aabbs[b]):
                hits.append((a, b))
    return hits


def unresolved_meshes(
    robot: Robot, mesh_resolver: Optional[MeshResolver] = None,
) -> list[tuple[str, str, str]]:
    """Return [(link_name, mesh_filename, reason), ...] for meshes the resolver
    couldn't load. Empty list when no <mesh> shapes exist, when no resolver is
    supplied (mesh-blind = nothing to complain about), or when everything resolved.
    """
    if mesh_resolver is None:
        return []
    out: list[tuple[str, str, str]] = []
    for link in robot.links:
        for c in link.collisions:
            if not isinstance(c.geometry, Mesh):
                continue
            path = mesh_resolver.resolve(c.geometry.filename)
            if path is None:
                out.append((link.name, c.geometry.filename, "unresolvable URI"))
            elif not path.exists():
                out.append((link.name, c.geometry.filename, f"file not found: {path}"))
    return out
