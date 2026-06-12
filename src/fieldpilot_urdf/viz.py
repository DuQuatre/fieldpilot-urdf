"""Visualisation for URDF Robot models.

Two pure-bytes renderers mirroring app/graph_visualization.py's pattern:

    render_kinematic_tree(robot, q?, highlight?, format)  -> bytes (png|svg)
    render_pose_3d(robot, q?, format)                     -> bytes (png|svg)

Both require external native deps already used by pyDEXPI: graphviz's `dot`
binary for the tree, matplotlib for the 3D pose. No I/O — endpoints write
the bytes to the HTTP response themselves.
"""
from __future__ import annotations

import io
from typing import Iterable, Literal, Optional

import graphviz
import matplotlib

matplotlib.use("Agg")  # headless backend; safe under uvicorn / pytest
import matplotlib.pyplot as plt  # noqa: E402

from .fk import forward_kinematics
from .graph import build_graph
from .models import Robot


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
