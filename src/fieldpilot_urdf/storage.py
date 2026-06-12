"""File-based URDF model registry.

Each robot is stored as its serialised URDF XML alongside a JSON sidecar with
metadata; a single index.json carries the list summary for fast lookups.

Layout under FIELDPILOT_URDF_DATA_DIR (default /data/urdf-models/):
    index.json
    {model_id}.urdf      — round-trippable via loader.to_xml()
    {model_id}.json      — {model_id, name, source_file, created_at, n_links, n_joints}
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .collisions import MeshResolver
from .loader import from_xml, to_xml
from .models import Robot


DATA_DIR_ENV = "FIELDPILOT_URDF_DATA_DIR"
_LEGACY_DATA_DIR_ENV = "MECHDIAG_DATA_DIR"  # deprecated fallback (keeps SaaS deployments working)
_DEFAULT_DIR = Path("/data/urdf-models")


def _data_dir() -> Path:
    return Path(os.environ.get(DATA_DIR_ENV)
                or os.environ.get(_LEGACY_DATA_DIR_ENV, _DEFAULT_DIR))


def _ensure_dir() -> None:
    _data_dir().mkdir(parents=True, exist_ok=True)


def _index_path() -> Path:
    return _data_dir() / "index.json"


def _urdf_path(model_id: str) -> Path:
    return _data_dir() / f"{model_id}.urdf"


def _meta_path(model_id: str) -> Path:
    return _data_dir() / f"{model_id}.json"


def _mesh_dir_for(model_id: str) -> Path:
    """Conventional location for downloaded meshes of a given model."""
    return _data_dir() / f"{model_id}_meshes"


def _load_index() -> dict[str, dict]:
    _ensure_dir()
    p = _index_path()
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_index(index: dict[str, dict]) -> None:
    _ensure_dir()
    _index_path().write_text(json.dumps(index, indent=2))


def _summary(meta: dict) -> dict:
    """Trim a full metadata dict down to the listing shape."""
    return {k: meta[k] for k in (
        "model_id", "name", "source_file", "n_links", "n_joints", "created_at"
    ) if k in meta}


def save_robot(
    robot: Robot,
    source_file: str = "",
    *,
    mesh_dir: Optional[Path] = None,
    mesh_packages: Optional[list[str]] = None,
) -> dict:
    """Persist a parsed Robot. Returns the metadata dict (incl. model_id).

    Pass `mesh_dir` + `mesh_packages` when the caller has already downloaded
    referenced meshes to disk. Future operations on this model (notably
    `load_resolver` and r008 self-collision diagnostics) will then resolve
    `package://` URIs against that directory.
    """
    model_id = uuid.uuid4().hex[:8]
    meta = {
        "model_id": model_id,
        "name": robot.name,
        "source_file": source_file,
        "n_links": len(robot.links),
        "n_joints": len(robot.joints),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if mesh_dir is not None:
        meta["mesh_dir"] = str(mesh_dir)
        meta["mesh_packages"] = mesh_packages or []
    _ensure_dir()
    _urdf_path(model_id).write_text(to_xml(robot))
    _meta_path(model_id).write_text(json.dumps(meta, indent=2))
    index = _load_index()
    index[model_id] = _summary(meta)
    _save_index(index)
    return meta


def load_robot(model_id: str) -> Optional[Robot]:
    p = _urdf_path(model_id)
    if not p.exists():
        return None
    return from_xml(p.read_text())


def load_meta(model_id: str) -> Optional[dict]:
    p = _meta_path(model_id)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def list_robots() -> list[dict]:
    """All stored robots, newest first."""
    index = _load_index()
    return sorted(index.values(),
                  key=lambda s: s.get("created_at", ""), reverse=True)


def delete_robot(model_id: str) -> bool:
    """Remove the URDF + metadata + index entry. Returns False if missing."""
    urdf = _urdf_path(model_id)
    meta = _meta_path(model_id)
    if not urdf.exists() and not meta.exists():
        return False
    urdf.unlink(missing_ok=True)
    meta.unlink(missing_ok=True)
    # Best-effort: drop the mesh cache too.
    mesh_dir = _mesh_dir_for(model_id)
    if mesh_dir.exists():
        import shutil
        shutil.rmtree(mesh_dir, ignore_errors=True)
    index = _load_index()
    index.pop(model_id, None)
    _save_index(index)
    return True


def load_resolver(model_id: str) -> Optional[MeshResolver]:
    """Build a MeshResolver from a model's stored mesh directory. Returns
    None when the model has no persisted meshes — callers can fall back to
    mesh-blind behaviour."""
    meta = load_meta(model_id)
    if meta is None or "mesh_dir" not in meta:
        return None
    mesh_dir = Path(meta["mesh_dir"])
    if not mesh_dir.exists():
        return None
    # Each downloaded package lives in its own sub-directory:
    #   {mesh_dir}/{pkg_a}/{sub}/file.stl
    #   {mesh_dir}/{pkg_b}/...
    packages = {pkg: mesh_dir / pkg for pkg in meta.get("mesh_packages", [])}
    return MeshResolver(mesh_dir=mesh_dir, packages=packages)
