"""Visualisation for URDF Robot models.

Three pure-bytes renderers:

    render_kinematic_tree(robot, q?, highlight?, format)  -> bytes (png|svg)
    render_pose_3d(robot, q?, format)                     -> bytes (png|svg)
    render_pose_mesh(robot, q?, mesh_dir?, ...)           -> bytes (png)

The first two require the optional `[viz]` extra: graphviz's `dot` binary for
the tree, matplotlib for the 3D pose. They are mesh-free — `render_pose_3d`
plots link-frame origins from forward kinematics and never touches mesh
geometry, so it works with zero downloads and no GL stack.

`render_pose_mesh` is the industrial, mesh-accurate view. It needs the heavier
`[meshviz]` extra (urchin + pyrender) AND a headless GL backend (EGL by
default; set `FIELDPILOT_URDF_RENDER_BACKEND=osmesa` for pure-software). It
also needs the link meshes already on disk (see `importer.fetch_meshes`); pass
their root as `mesh_dir`. Robots built from primitive geometry (box / cylinder
/ sphere) render without any `mesh_dir`.

No I/O — every renderer returns bytes the caller writes wherever it likes.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Iterable, Literal, Optional

import graphviz
import matplotlib

matplotlib.use("Agg")  # headless backend; safe under uvicorn / pytest
import matplotlib.pyplot as plt  # noqa: E402

from .fk import forward_kinematics
from .graph import build_graph
from .importer import package_uri_parts
from .loader import to_xml
from .models import Mesh, Robot


Format = Literal["png", "svg"]


# Joint type → edge style (color + dash)
JOINT_STYLE: dict[str, dict[str, str]] = {
    "revolute":   {"color": "#2E86DE", "style": "solid"},
    "continuous": {"color": "#27AE60", "style": "dashed"},
    "prismatic":  {"color": "#E74C3C", "style": "solid"},
    "fixed":      {"color": "#7F8C8D", "style": "solid"},
    "floating":   {"color": "#8E44AD", "style": "dotted"},
    "planar":     {"color": "#D35400", "style": "dotted"},
}

HIGHLIGHT_FILL = "#F1C40F"
ROOT_FILL = "#34495E"
ROOT_FONT = "white"
LINK_FILL_INERTIAL = "#3498DB"
LINK_FILL_INERTIAL_FONT = "white"
LINK_FILL_BARE = "#ECF0F1"
LINK_FILL_BARE_FONT = "#2C3E50"


def render_kinematic_tree(
    robot: Robot,
    q: Optional[dict[str, float]] = None,
    highlight: Optional[Iterable[str]] = None,
    fmt: Format = "png",
) -> bytes:
    """Graphviz DiGraph of links + joints. q is currently informational
    (annotated on the title); the layout itself is the topology, not the pose.
    """
    G = build_graph(robot)
    hi = set(highlight or [])
    in_deg = dict(G.in_degree())

    pose_note = ""
    if q:
        pose_note = f"\\n(q: " + ", ".join(f"{k}={v:g}" for k, v in q.items()) + ")"

    dot = graphviz.Digraph(
        f"urdf_{robot.name}",
        format=fmt,
        engine="dot",
        graph_attr={
            "rankdir": "TB",
            "label": f"URDF: {robot.name} - "
                     f"{len(robot.links)} links / {len(robot.joints)} joints"
                     + pose_note,
            "labelloc": "t",
            "fontsize": "14",
            "fontname": "Arial Bold",
            "bgcolor": "#FAFAFA",
            "pad": "0.4",
            "nodesep": "0.3",
            "ranksep": "0.5",
        },
        node_attr={"fontname": "Arial", "fontsize": "11", "style": "filled"},
        edge_attr={"fontname": "Arial", "fontsize": "9"},
    )

    for link in robot.links:
        if link.name in hi:
            fill, font = HIGHLIGHT_FILL, "#2C3E50"
        elif in_deg.get(link.name, 0) == 0:
            fill, font = ROOT_FILL, ROOT_FONT
        elif link.inertial is not None:
            fill, font = LINK_FILL_INERTIAL, LINK_FILL_INERTIAL_FONT
        else:
            fill, font = LINK_FILL_BARE, LINK_FILL_BARE_FONT
        mass_txt = (f"\\n{link.inertial.mass:g} kg"
                    if link.inertial is not None and link.inertial.mass else "")
        dot.node(link.name, label=f"{link.name}{mass_txt}",
                 fillcolor=fill, fontcolor=font, shape="box")

    for joint in robot.joints:
        style = JOINT_STYLE.get(joint.type, JOINT_STYLE["fixed"])
        dot.edge(joint.parent, joint.child,
                 label=f"{joint.name}\\n({joint.type})",
                 color=style["color"], style=style["style"], penwidth="1.6")

    return dot.pipe(format=fmt)


def render_pose_3d(
    robot: Robot,
    q: Optional[dict[str, float]] = None,
    fmt: Format = "png",
) -> bytes:
    """3D scatter of link world-frame origins + parent→child segments at pose q.

    Mesh shapes are ignored; we plot link frame origins from forward_kinematics.
    """
    tfs = forward_kinematics(robot, q=q or {})
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # links
    xs, ys, zs, names = [], [], [], []
    for name, T in tfs.items():
        xs.append(float(T[0, 3]))
        ys.append(float(T[1, 3]))
        zs.append(float(T[2, 3]))
        names.append(name)
    ax.scatter(xs, ys, zs, c="#3498DB", s=60, edgecolors="#1B4F72", depthshade=True)
    for x, y, z, n in zip(xs, ys, zs, names):
        ax.text(x, y, z, "  " + n, fontsize=8, color="#2C3E50")

    # parent→child line segments coloured by joint type
    for j in robot.joints:
        if j.parent not in tfs or j.child not in tfs:
            continue
        c = JOINT_STYLE.get(j.type, JOINT_STYLE["fixed"])["color"]
        p, ch = tfs[j.parent], tfs[j.child]
        ax.plot([p[0, 3], ch[0, 3]], [p[1, 3], ch[1, 3]], [p[2, 3], ch[2, 3]],
                color=c, linewidth=2)

    pose_note = ""
    if q:
        pose_note = "  (q: " + ", ".join(f"{k}={v:g}" for k, v in q.items()) + ")"
    ax.set_title(f"{robot.name}{pose_note}", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # equal aspect (best-effort: matplotlib 3D doesn't honour set_aspect well)
    if xs:
        span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-3)
        cx, cy, cz = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2
        ax.set_xlim(cx - span / 2, cx + span / 2)
        ax.set_ylim(cy - span / 2, cy + span / 2)
        ax.set_zlim(cz - span / 2, cz + span / 2)

    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return buf.getvalue()


# --- mesh-accurate pose render ---------------------------------------------
#
# Unlike render_pose_3d (mesh-free stick figure), this delegates the actual
# geometry to `urchin` (URDF -> trimesh) and renders offscreen with `pyrender`.
# We use urchin purely as a loader: round-trip the model through to_xml(), let
# urchin attach the link meshes, ask it for the visual trimeshes + their FK
# poses, then composite them in a pyrender scene. No urchin model mapping leaks
# out of this function.


def _resolve_mesh_robot(
    robot: Robot, mesh_dir: Optional[Path]
) -> tuple[Robot, int]:
    """Deep-copy `robot` for handing to urchin's loader:

    * Rewrite every <visual> <mesh> filename to an absolute on-disk path under
      `mesh_dir` so urchin resolves it regardless of where the URDF file lives.
      Visuals whose mesh can't be located on disk are dropped (the resolvable
      parts still render).
    * Drop all <collision> geometry. We render visuals only, and urchin loaded
      with lazy_load_meshes=False would otherwise try (and fail) to load
      collision meshes too.

    Returns (robot_copy, dropped_visual_count).

    package://pkg/sub  ->  {mesh_dir}/pkg/sub      (matches fetch_meshes layout)
    other/relative     ->  {mesh_dir}/{filename}   (best-effort)
    """
    out = robot.model_copy(deep=True)
    dropped = 0
    for link in out.links:
        link.collisions = []  # not rendered; avoids urchin loading their meshes
        kept = []
        for v in link.visuals:
            geom = v.geometry
            if not isinstance(geom, Mesh):
                kept.append(v)
                continue
            on_disk: Optional[Path] = None
            if mesh_dir is not None:
                parts = package_uri_parts(geom.filename)
                rel = f"{parts[0]}/{parts[1]}" if parts else geom.filename
                cand = mesh_dir / rel
                if cand.is_file():
                    on_disk = cand
            if on_disk is None:
                dropped += 1
                continue
            v.geometry = geom.model_copy(update={"filename": str(on_disk)})
            kept.append(v)
        link.visuals = kept
    return out, dropped


def _look_at(eye, target, up):
    """4x4 camera pose for a pyrender camera (looks down -z, +y up) placed at
    `eye` aimed at `target`. Falls back to a sane basis on degenerate input."""
    import numpy as np

    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)

    fwd = target - eye
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])

    side = np.cross(fwd, up)
    n = np.linalg.norm(side)
    if n < 1e-9:  # up parallel to view direction — pick another up
        up = np.array([0.0, 1.0, 0.0])
        side = np.cross(fwd, up)
        n = np.linalg.norm(side)
    side = side / n
    true_up = np.cross(side, fwd)

    m = np.eye(4)
    m[:3, 0] = side
    m[:3, 1] = true_up
    m[:3, 2] = -fwd
    m[:3, 3] = eye
    return m


def render_pose_mesh(
    robot: Robot,
    q: Optional[dict[str, float]] = None,
    *,
    mesh_dir: Optional[Path] = None,
    fmt: Literal["png"] = "png",
    width: int = 900,
    height: int = 720,
    bg: tuple[float, float, float] = (0.97, 0.97, 0.98),
) -> bytes:
    """Industrial, mesh-accurate render of `robot` at pose `q` -> PNG bytes.

    Requires the `[meshviz]` extra (urchin + pyrender) and a headless GL
    backend. The backend defaults to EGL; override with the env var
    FIELDPILOT_URDF_RENDER_BACKEND (e.g. "osmesa") or by setting
    PYOPENGL_PLATFORM yourself before this module's GL import.

    `mesh_dir` is the root where link meshes were downloaded (see
    `importer.fetch_meshes`); meshes missing from it are skipped. Robots with
    only primitive geometry render without a `mesh_dir`. `fmt` is png-only —
    this is a raster render, unlike the vector-capable tree/pose renderers.
    """
    if fmt != "png":
        raise ValueError(
            f"render_pose_mesh produces raster output; fmt must be 'png', got {fmt!r}"
        )

    # Headless GL backend must be chosen BEFORE pyrender imports PyOpenGL.
    if "PYOPENGL_PLATFORM" not in os.environ:
        os.environ["PYOPENGL_PLATFORM"] = os.environ.get(
            "FIELDPILOT_URDF_RENDER_BACKEND", "egl"
        )

    try:
        import numpy as np
        import pyrender
        import urchin
    except ImportError as e:  # pragma: no cover - exercised via env without extra
        raise ImportError(
            "render_pose_mesh needs the optional mesh-render stack. "
            "Install it with:  pip install 'fieldpilot-urdf[meshviz]'"
        ) from e

    import matplotlib.image as mpimg
    import tempfile

    resolved, _dropped = _resolve_mesh_robot(robot, mesh_dir)

    # urchin loads from a file; absolute mesh paths make CWD irrelevant.
    with tempfile.TemporaryDirectory() as td:
        urdf_path = Path(td) / f"{robot.name or 'robot'}.urdf"
        urdf_path.write_text(to_xml(resolved), encoding="utf-8")
        urdf_u = urchin.URDF.load(str(urdf_path), lazy_load_meshes=False)

    # urchin rejects unknown joint names; keep only actuated ones it knows.
    actuated = set(getattr(urdf_u, "actuated_joint_names", []) or [])
    cfg = {k: v for k, v in (q or {}).items() if k in actuated} or None

    fk = urdf_u.visual_trimesh_fk(cfg=cfg)  # {trimesh.Trimesh: 4x4 world pose}

    scene = pyrender.Scene(
        bg_color=[bg[0], bg[1], bg[2], 1.0],
        ambient_light=[0.35, 0.35, 0.35],
    )

    # Compute the world-space AABB ourselves from the visual trimeshes rather
    # than reading scene.centroid / scene.scale: pyrender's bounds properties
    # call np.infty, which NumPy 2.0 removed, so they raise on a modern stack.
    lo = np.array([np.inf, np.inf, np.inf])
    hi = -lo
    for tm, pose in fk.items():
        T = np.asarray(pose)
        scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False), pose=T)
        b = np.asarray(tm.bounds)  # (2, 3) local-frame min/max
        corners = np.array([[b[i, 0], b[j, 1], b[k, 2]]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        world = (T @ np.c_[corners, np.ones(len(corners))].T).T[:, :3]
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))

    if np.all(np.isfinite(lo)):
        centroid = (lo + hi) / 2.0
        extent = float(np.linalg.norm(hi - lo))  # bounding-box diagonal
    else:  # empty scene (no resolvable visuals)
        centroid = np.zeros(3)
        extent = 1.0
    if not np.isfinite(extent) or extent <= 0:
        extent = 1.0

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=width / height)
    direction = np.array([1.0, -1.0, 0.7])
    direction = direction / np.linalg.norm(direction)
    eye = centroid + direction * (extent * 1.6 + 1e-3)
    cam_pose = _look_at(eye, centroid, up=[0.0, 0.0, 1.0])
    scene.add(cam, pose=cam_pose)

    # Key light rides with the camera; fill light from above for depth.
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0),
              pose=cam_pose)
    top = _look_at(centroid + np.array([0.0, 0.0, extent * 2.0 + 1e-3]),
                   centroid, up=[0.0, 1.0, 0.0])
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0),
              pose=top)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, _depth = renderer.render(scene)
    finally:
        renderer.delete()

    buf = io.BytesIO()
    mpimg.imsave(buf, color, format=fmt)
    return buf.getvalue()
