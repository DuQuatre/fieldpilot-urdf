"""Pure-Python mesh bounding-box reader for the common URDF mesh formats.

URDF `<mesh>` elements point at STL / OBJ / DAE / PLY files. For collision AABBs
all we need is the vertex bounding box — not the full mesh — so this module reads
just that, in pure Python (NumPy only), for the formats that dominate real robot
descriptions:

* **STL** — binary *and* ASCII (auto-detected by the 84 + 50·n size rule).
* **OBJ** — Wavefront `v x y z` vertex lines.
* **PLY** — ASCII and binary (little/big endian), vertex element first.

:func:`read_mesh_bounds` returns ``(min_xyz, max_xyz)`` or ``None`` for an
unknown/unsupported/corrupt file — callers (see :mod:`fieldpilot_urdf.collisions`)
fall back to ``trimesh`` (the heavier ``[mesh]`` extra) for the rest, e.g. COLLADA
``.dae`` or glTF. The point: the STL/OBJ/PLY common case now needs no extra.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import numpy as np

Bounds = tuple[tuple[float, float, float], tuple[float, float, float]]


def _bounds_from_vertices(verts: np.ndarray) -> Optional[Bounds]:
    if verts.size == 0:
        return None
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    return (float(mn[0]), float(mn[1]), float(mn[2])), \
           (float(mx[0]), float(mx[1]), float(mx[2]))


# --- STL -------------------------------------------------------------------

def _stl_is_binary(data: bytes) -> bool:
    """Binary STL is 80-byte header + uint32 count + 50 bytes/triangle. The
    exact-size match is the robust discriminator (an ASCII file beginning with
    'solid' won't satisfy it); fall back to the 'solid' prefix only otherwise."""
    if len(data) >= 84:
        (count,) = struct.unpack_from("<I", data, 80)
        if len(data) == 84 + count * 50:
            return True
    return not data[:80].lstrip().startswith(b"solid")


def _stl_bounds(data: bytes) -> Optional[Bounds]:
    if _stl_is_binary(data):
        (count,) = struct.unpack_from("<I", data, 80)
        rec = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        arr = np.frombuffer(data, dtype=rec, count=count, offset=84)
        verts = arr["v"].reshape(-1, 3).astype(float)
        return _bounds_from_vertices(verts)
    # ASCII: collect every "vertex x y z" line.
    text = data.decode("utf-8", errors="replace")
    pts = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "vertex":
            pts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return _bounds_from_vertices(np.array(pts, dtype=float))


# --- OBJ -------------------------------------------------------------------

def _obj_bounds(data: bytes) -> Optional[Bounds]:
    text = data.decode("utf-8", errors="replace")
    pts = []
    for line in text.splitlines():
        parts = line.split()
        # "v x y z [w]" — geometric vertex. Skip "vt"/"vn"/"vp".
        if len(parts) >= 4 and parts[0] == "v":
            pts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return _bounds_from_vertices(np.array(pts, dtype=float))


# --- PLY -------------------------------------------------------------------

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _ply_bounds(data: bytes) -> Optional[Bounds]:
    end = data.find(b"end_header")
    if end == -1:
        return None
    header_end = data.find(b"\n", end) + 1
    header = data[:header_end].decode("ascii", errors="replace")

    fmt = None
    elements: list[tuple[str, int, list[tuple[str, str]]]] = []  # (name, count, props)
    for line in header.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            elements.append((tok[1], int(tok[2]), []))
        elif tok[0] == "property" and elements:
            # "property <type> <name>" or "property list <count_t> <val_t> <name>".
            if tok[1] == "list":
                elements[-1][2].append(("list", tok[-1]))
            else:
                elements[-1][2].append((tok[1], tok[2]))

    if not elements or elements[0][0] != "vertex":
        return None  # need a vertex-first layout; let trimesh handle the rest
    _, n_vert, props = elements[0]
    names = [name for _, name in props]
    if not {"x", "y", "z"} <= set(names):
        return None

    if fmt == "ascii":
        ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
        body = data[header_end:].decode("ascii", errors="replace").split()
        ncol = len(props)
        pts = []
        for r in range(n_vert):
            row = body[r * ncol:(r + 1) * ncol]
            if len(row) < ncol:
                return None
            pts.append((float(row[ix]), float(row[iy]), float(row[iz])))
        return _bounds_from_vertices(np.array(pts, dtype=float))

    if fmt in ("binary_little_endian", "binary_big_endian"):
        if any(t == "list" for t, _ in props):
            return None  # variable-stride vertex record — uncommon; defer
        endian = "<" if fmt.endswith("little_endian") else ">"
        dt = np.dtype([(name, endian + _PLY_TYPES[t]) for t, name in props])
        arr = np.frombuffer(data, dtype=dt, count=n_vert, offset=header_end)
        verts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(float)
        return _bounds_from_vertices(verts)

    return None


# --- dispatch --------------------------------------------------------------

_READERS = {".stl": _stl_bounds, ".obj": _obj_bounds, ".ply": _ply_bounds}

SUPPORTED_FORMATS = frozenset(_READERS)


def read_mesh_bounds(path: str | Path) -> Optional[Bounds]:
    """Return the vertex bounding box ``((minx, miny, minz), (maxx, maxy, maxz))``
    of a mesh file, or ``None`` if the format is unsupported, the file is empty,
    or it can't be parsed.

    Supports the extensions in :data:`SUPPORTED_FORMATS` (``.stl``, ``.obj``,
    ``.ply``) in pure Python. Any parse error is swallowed into ``None`` so a
    caller can fall back to a heavier loader (``trimesh``) cleanly.
    """
    path = Path(path)
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data:
        return None
    try:
        return reader(data)
    except Exception:  # noqa: BLE001 — corrupt/truncated file → defer to fallback
        return None
