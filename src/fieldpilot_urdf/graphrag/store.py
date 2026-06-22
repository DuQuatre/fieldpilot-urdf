"""Pluggable robot store for the fieldpilot-urdf GraphRAG server.

Ported from MecAI (MIT) and re-targeted onto the URDF ``Robot``. A robot is keyed
by its ``name`` (URDF's stable identity — the random ``id`` field is excluded from
serialization), so the same robot round-trips through JSON by name.

Two plain key-value backends, plus :func:`get_store` which selects the graph
backend (see :mod:`fieldpilot_urdf.graphrag.backend`):

* :class:`MemoryStore` — process-local dict.
* :class:`FileStore` — one ``{name}.json`` per robot under a directory, so robots
  survive restarts and are shared across workers that mount the same path.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

from ..models import Robot

STORE_DIR_ENV = "FIELDPILOT_URDF_STORE_DIR"

# Filesystem-safe id guard: prevents path traversal and collisions on disk.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def model_id(robot: Robot) -> str:
    """The stable identity of a robot in the store: its URDF ``name``."""
    return robot.name


def _check_safe_id(robot_id: str) -> None:
    if not _SAFE_ID.fullmatch(robot_id) or robot_id in (".", ".."):
        raise ValueError(
            f"robot id {robot_id!r} is not filesystem-safe; persistent storage "
            "requires names matching [A-Za-z0-9._-]+"
        )


class RobotStore(Protocol):
    """Minimal CRUD surface used by the server."""

    def put(self, robot: Robot) -> None: ...
    def get(self, robot_id: str) -> Robot | None: ...
    def list(self) -> list[str]: ...
    def delete(self, robot_id: str) -> bool: ...
    def clear(self) -> None: ...


class MemoryStore:
    """In-memory, process-local store."""

    def __init__(self) -> None:
        self._d: dict[str, Robot] = {}

    def put(self, robot: Robot) -> None:
        self._d[model_id(robot)] = robot

    def get(self, robot_id: str) -> Robot | None:
        return self._d.get(robot_id)

    def list(self) -> list[str]:
        return sorted(self._d)

    def delete(self, robot_id: str) -> bool:
        return self._d.pop(robot_id, None) is not None

    def clear(self) -> None:
        self._d.clear()


class FileStore:
    """JSON-file-backed store: one ``{name}.json`` per robot under ``directory``."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, robot_id: str) -> Path:
        _check_safe_id(robot_id)
        return self.dir / f"{robot_id}.json"

    def put(self, robot: Robot) -> None:
        path = self._path(model_id(robot))
        # Atomic write: temp file in the same dir, then rename.
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(robot.model_dump_json(indent=2))
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def get(self, robot_id: str) -> Robot | None:
        try:
            path = self._path(robot_id)
        except ValueError:
            return None
        if not path.exists():
            return None
        return Robot.model_validate_json(path.read_text())

    def list(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.json"))

    def delete(self, robot_id: str) -> bool:
        try:
            path = self._path(robot_id)
        except ValueError:
            return False
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> None:
        for p in self.dir.glob("*.json"):
            p.unlink()


def get_store():
    """Return the configured graph backend, in precedence order:

    1. ``NEO4J_BOLT_URL`` (or ``MEMGRAPH_BOLT_URL``) → :class:`Neo4jStore`
       (durable Neo4j graph backend, needs the ``[graphrag]`` extra).
    2. ``FIELDPILOT_URDF_STORE_DIR`` → :class:`MemoryGraphBackend` with JSON-file
       persistence (in-memory graph queries + on-disk durability).
    3. otherwise → :class:`MemoryGraphBackend` (process-local).

    Every backend implements both the :class:`RobotStore` surface *and* the
    graph-analytics surface, so GraphRAG works regardless of which is selected.
    The standalone :class:`MemoryStore` / :class:`FileStore` remain available for
    callers that want a plain key-value store without the graph layer.
    """
    if os.environ.get("NEO4J_BOLT_URL") or os.environ.get("MEMGRAPH_BOLT_URL"):
        from .neo4j_backend import Neo4jStore

        return Neo4jStore()
    from .backend import MemoryGraphBackend

    data_dir = os.environ.get(STORE_DIR_ENV)
    return MemoryGraphBackend(persist_dir=data_dir) if data_dir else MemoryGraphBackend()
