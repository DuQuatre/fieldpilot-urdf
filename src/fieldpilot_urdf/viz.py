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

import numpy as np  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from .fk import forward_kinematics
from .graph import build_graph, leaf_links
from .importer import package_uri_parts
from .loader import to_xml
from .models import Mesh, Robot


Format = Literal["png", "svg"]
MotionFormat = Literal["gif", "frames"]

NOMINAL_COLOR = "#2E86DE"   # healthy / commanded motion
FAULT_COLOR = "#E74C3C"     # faulted / observed motion
TRACE_COLOR = "#F39C12"     # end-effector path trail


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


# --- motion animation (3D fault-motion video) ------------------------------
#
# Animate a robot through a sequence of joint configurations (a "trajectory" =
# list of {joint: value} dicts, e.g. plan_path output, TimedTrajectory.as_dicts(),
# or a sim trajectory). render_motion_comparison plays a nominal motion against a
# faulted one so a tech can eyeball "does my robot move like this?". Mesh-free
# stick figures, same as render_pose_3d. GIF assembly uses Pillow (a matplotlib
# dependency, so always present under the [viz] extra); fmt="frames" needs only
# matplotlib and returns the per-frame PNGs.


def _frame_transforms(robot: Robot, frames: list[dict]) -> list[dict]:
    if not frames:
        raise ValueError("no frames to animate (empty trajectory)")
    return [forward_kinematics(robot, q or {}) for q in frames]


def _global_limits(tfs_lists: list[list[dict]]):
    """Camera box enclosing every link across every frame of every trajectory,
    so the view doesn't jump frame to frame."""
    xs, ys, zs = [], [], []
    for tfs_list in tfs_lists:
        for tfs in tfs_list:
            for T in tfs.values():
                xs.append(float(T[0, 3])); ys.append(float(T[1, 3])); zs.append(float(T[2, 3]))
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-3)
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2, span)


def _apply_limits(ax, lim) -> None:
    cx, cy, cz, span = lim
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)
    ax.set_zlim(cz - span / 2, cz + span / 2)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")


def _draw_skeleton(ax, robot: Robot, tfs: dict, *, edge_color: Optional[str] = None,
                   node_color: str = "#3498DB", style: str = "solid",
                   alpha: float = 1.0, label: Optional[str] = None) -> None:
    """Scatter link origins + parent→child segments into a 3D axes. With
    ``edge_color=None`` segments are coloured by joint type (like
    ``render_pose_3d``); set it to draw the whole robot in one colour."""
    xs = [float(T[0, 3]) for T in tfs.values()]
    ys = [float(T[1, 3]) for T in tfs.values()]
    zs = [float(T[2, 3]) for T in tfs.values()]
    ax.scatter(xs, ys, zs, c=node_color, s=40, alpha=alpha, depthshade=True)
    first = True
    for j in robot.joints:
        if j.parent not in tfs or j.child not in tfs:
            continue
        c = edge_color if edge_color is not None else JOINT_STYLE.get(j.type, JOINT_STYLE["fixed"])["color"]
        p, ch = tfs[j.parent], tfs[j.child]
        ax.plot([p[0, 3], ch[0, 3]], [p[1, 3], ch[1, 3]], [p[2, 3], ch[2, 3]],
                color=c, linewidth=2, linestyle=style, alpha=alpha,
                label=(label if first else None))
        first = False


def _fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def _assemble_gif(pngs: list[bytes], fps: float) -> bytes:
    from PIL import Image
    imgs = [Image.open(io.BytesIO(p)).convert("RGB") for p in pngs]
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:],
                 duration=max(1, int(round(1000.0 / max(fps, 1e-6)))), loop=0)
    return buf.getvalue()


def render_motion(
    robot: Robot,
    frames: list[dict],
    *,
    fmt: MotionFormat = "gif",
    fps: float = 10.0,
    elev: float = 22.0,
    azim: float = -60.0,
    track_link: Optional[str] = None,
    title: Optional[str] = None,
):
    """Animate ``robot`` through ``frames`` (a list of ``{joint: value}`` configs).

    ``fmt="gif"`` returns animated-GIF bytes; ``fmt="frames"`` returns the list of
    per-frame PNG bytes. ``track_link`` draws the trailing path of that link's
    origin. The camera box is fixed across all frames so the robot doesn't drift.
    """
    tfs_list = _frame_transforms(robot, frames)
    lim = _global_limits([tfs_list])
    trail: list[tuple[float, float, float]] = []
    pngs: list[bytes] = []
    for i, tfs in enumerate(tfs_list):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        _draw_skeleton(ax, robot, tfs)
        if track_link and track_link in tfs:
            T = tfs[track_link]
            trail.append((float(T[0, 3]), float(T[1, 3]), float(T[2, 3])))
            t = np.array(trail)
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=TRACE_COLOR, linewidth=1.4, alpha=0.85)
        _apply_limits(ax, lim)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title or f"{robot.name} — frame {i + 1}/{len(tfs_list)}", fontsize=10)
        pngs.append(_fig_png(fig))
    return pngs if fmt == "frames" else _assemble_gif(pngs, fps)


def render_motion_comparison(
    robot: Robot,
    nominal: list[dict],
    faulted: list[dict],
    *,
    labels: tuple[str, str] = ("nominal", "faulted"),
    layout: Literal["overlay", "sidebyside"] = "overlay",
    fmt: MotionFormat = "gif",
    fps: float = 10.0,
    elev: float = 22.0,
    azim: float = -60.0,
    track_link: Optional[str] = None,
    title: Optional[str] = None,
):
    """Animate a ``nominal`` motion against a ``faulted`` one so reality can be
    compared to the simulation — the headline 3D fault-motion visual.

    ``layout="overlay"`` draws both robots in one axes (nominal solid blue,
    faulted dashed red) with a dotted line marking the end-effector divergence and
    its magnitude in the title; ``"sidebyside"`` uses two panels. ``track_link``
    is the link whose divergence is measured (defaults to a leaf link). The two
    trajectories are paired frame-by-frame and truncated to the shorter one.
    ``fmt`` is as in :func:`render_motion`.
    """
    n_tfs = _frame_transforms(robot, nominal)
    f_tfs = _frame_transforms(robot, faulted)
    n = min(len(n_tfs), len(f_tfs))
    n_tfs, f_tfs = n_tfs[:n], f_tfs[:n]
    lim = _global_limits([n_tfs, f_tfs])
    if track_link is None:
        leaves = leaf_links(build_graph(robot))
        track_link = sorted(leaves)[0] if leaves else None

    pngs: list[bytes] = []
    for i in range(n):
        nt, ft = n_tfs[i], f_tfs[i]
        diverge = None
        if track_link and track_link in nt and track_link in ft:
            diverge = (nt[track_link][:3, 3], ft[track_link][:3, 3])
        if layout == "sidebyside":
            fig = plt.figure(figsize=(10, 5))
            axn = fig.add_subplot(121, projection="3d")
            axf = fig.add_subplot(122, projection="3d")
            _draw_skeleton(axn, robot, nt, edge_color=NOMINAL_COLOR, node_color=NOMINAL_COLOR)
            _draw_skeleton(axf, robot, ft, edge_color=FAULT_COLOR, node_color=FAULT_COLOR)
            for ax, lab in ((axn, labels[0]), (axf, labels[1])):
                _apply_limits(ax, lim)
                ax.view_init(elev=elev, azim=azim)
                ax.set_title(lab, fontsize=10)
            sup = title or robot.name
            if diverge is not None:
                sup += f" — Δ{track_link} = {float(np.linalg.norm(diverge[0] - diverge[1])) * 1000:.0f} mm"
            fig.suptitle(sup, fontsize=11)
        else:  # overlay
            fig = plt.figure(figsize=(6.5, 5.5))
            ax = fig.add_subplot(111, projection="3d")
            _draw_skeleton(ax, robot, nt, edge_color=NOMINAL_COLOR, node_color=NOMINAL_COLOR,
                           label=labels[0])
            _draw_skeleton(ax, robot, ft, edge_color=FAULT_COLOR, node_color=FAULT_COLOR,
                           style="dashed", alpha=0.95, label=labels[1])
            ttl = title or robot.name
            if diverge is not None:
                pn, pf = diverge
                ax.plot([pn[0], pf[0]], [pn[1], pf[1]], [pn[2], pf[2]],
                        color="#7F8C8D", linestyle=":", linewidth=1.2)
                ttl += f" — Δ{track_link} = {float(np.linalg.norm(pn - pf)) * 1000:.0f} mm"
            _apply_limits(ax, lim)
            ax.view_init(elev=elev, azim=azim)
            ax.legend(loc="upper left", fontsize=8)
            ax.set_title(ttl, fontsize=10)
        pngs.append(_fig_png(fig))
    return pngs if fmt == "frames" else _assemble_gif(pngs, fps)


# --- oscilloscope parameter traces -----------------------------------------
#
# Stacked time-series panels (a multi-channel "scope") for joint parameters —
# position / velocity / effort over time — with an expected (simulated) signal
# overlaid against the observed one, so a tech can see *where* and *how much* the
# real robot diverges from the model. render_scope is the general plotter;
# render_trajectory_scope is the convenience for our TimedTrajectory /
# simulate.Trajectory (anything with joint_ids / times / q / u).


class ScopeSeries(BaseModel):
    """One trace on a panel: a time-series with a label and line style."""

    label: str = Field(..., description="Legend label (e.g. 'expected', 'observed')")
    times: list[float] = Field(..., description="X values (seconds)")
    values: list[float] = Field(..., description="Y values, same length as times")
    style: str = Field("solid", description="matplotlib linestyle")


class ScopePanel(BaseModel):
    """One stacked subplot — typically one joint+parameter — with its traces."""

    ylabel: str = Field(..., description="Y-axis label (e.g. 'j3 position (rad)')")
    series: list[ScopeSeries] = Field(default_factory=list)
    shade_divergence: bool = Field(
        False, description="With exactly 2 same-grid series, shade the gap and annotate max |Δ|")


def render_scope(
    panels: list[ScopePanel],
    *,
    title: Optional[str] = None,
    xlabel: str = "time (s)",
    size: Optional[tuple[float, float]] = None,
    fmt: Format = "png",
) -> bytes:
    """Render stacked time-series panels (a multi-channel oscilloscope) sharing an
    x-axis. Each :class:`ScopePanel` overlays its :class:`ScopeSeries`; when a
    panel has exactly two series on the *same* time grid and ``shade_divergence``
    is set, the gap between them is shaded and the max ``|Δ|`` annotated."""
    if not panels:
        raise ValueError("no panels to plot")
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=size or (7.0, 2.1 * n), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, panel in zip(axes, panels):
        for s in panel.series:
            ax.plot(s.times, s.values, linestyle=s.style, linewidth=1.6, label=s.label)
        if (panel.shade_divergence and len(panel.series) == 2
                and panel.series[0].times == panel.series[1].times):
            a, b = panel.series
            ta = np.asarray(a.times, dtype=float)
            va, vb = np.asarray(a.values, dtype=float), np.asarray(b.values, dtype=float)
            ax.fill_between(ta, va, vb, color=FAULT_COLOR, alpha=0.15)
            dmax = float(np.max(np.abs(va - vb))) if ta.size else 0.0
            ax.text(0.99, 0.05, f"max Δ = {dmax:.3g}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color="#922B21")
        ax.set_ylabel(panel.ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        if panel.series:
            ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel(xlabel, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return buf.getvalue()


_SIGNAL_ATTR = {"position": "q", "velocity": "u"}


def _traj_series(traj, joint: str, attr: str, label: str, style: str) -> ScopeSeries:
    idx = list(traj.joint_ids).index(joint)
    rows = getattr(traj, attr)
    return ScopeSeries(label=label, style=style,
                       times=[float(t) for t in traj.times],
                       values=[float(r[idx]) for r in rows])


def render_trajectory_scope(
    nominal,
    observed=None,
    *,
    joints: Optional[list[str]] = None,
    signals: tuple[str, ...] = ("position", "velocity"),
    labels: tuple[str, str] = ("expected", "observed"),
    title: Optional[str] = None,
    shade: bool = True,
    fmt: Format = "png",
) -> bytes:
    """Scope view of one or two trajectories: a stacked panel per (joint, signal),
    the ``nominal`` (expected/simulated) trace overlaid with the ``observed`` one.

    ``nominal`` / ``observed`` are any objects exposing ``joint_ids`` / ``times`` /
    ``q`` (positions) / ``u`` (velocities) — e.g. a `TimedTrajectory` or a
    `simulate.Trajectory`. ``signals`` chooses the parameters (``"position"`` →
    ``q``, ``"velocity"`` → ``u``); ``joints`` defaults to ``nominal``'s joints.
    With ``observed`` given and ``shade=True`` the divergence is shaded per panel
    (when the two share a time grid). For effort/torque traces, build panels by
    hand and call :func:`render_scope`.
    """
    bad = [s for s in signals if s not in _SIGNAL_ATTR]
    if bad:
        raise ValueError(f"unknown signal(s) {bad}; choose from {list(_SIGNAL_ATTR)}")
    jn = joints if joints is not None else list(nominal.joint_ids)
    units = {"position": "", "velocity": "/s"}
    panels: list[ScopePanel] = []
    for j in jn:
        for sig in signals:
            attr = _SIGNAL_ATTR[sig]
            series = [_traj_series(nominal, j, attr, labels[0], "solid")]
            if observed is not None:
                series.append(_traj_series(observed, j, attr, labels[1], "dashed"))
            panels.append(ScopePanel(
                ylabel=f"{j} {sig}{units[sig]}", series=series,
                shade_divergence=(shade and observed is not None)))
    return render_scope(panels, title=title, fmt=fmt)


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

    * Rewrite every <visual> <mesh> filename to an absolute on-disk path so
      urchin resolves it regardless of where the URDF file lives. Tried in
      order: (1) under `mesh_dir` (matches the `fetch_meshes` layout), then
      (2) the filename exactly as given, if it's already a resolvable path
      (absolute, or relative to the current working directory) — this is
      the common case for meshes built in-process (e.g. `mesh_primitives`)
      rather than fetched from a package registry, where there is no
      `mesh_dir` at all. Visuals whose mesh can't be located either way are
      dropped (the resolvable parts still render).

      BUG FIXED: previously, path (2) didn't exist — passing `mesh_dir=None`
      (or a `mesh_dir` that just doesn't happen to contain the file)
      unconditionally dropped every Mesh visual, even ones whose `filename`
      was already a perfectly valid on-disk path. Confirmed by hand: a
      `Robot` with `Visual(geometry=Mesh(filename=<real absolute .stl>))`
      rendered as a blank frame via `render_pose_mesh(robot)` with no
      `mesh_dir` — same PNG byte count as an empty scene — until `mesh_dir`
      was passed too (at which point `pathlib`'s "absolute path on the right
      of `/` discards the left operand" behaviour happened to make it work
      by accident, regardless of what `mesh_dir` actually was).
    * Drop all <collision> geometry. We render visuals only, and urchin loaded
      with lazy_load_meshes=False would otherwise try (and fail) to load
      collision meshes too.

    Returns (robot_copy, dropped_visual_count).

    package://pkg/sub  ->  {mesh_dir}/pkg/sub      (matches fetch_meshes layout)
    other/relative     ->  {mesh_dir}/{filename}   (best-effort)
    (any of the above) ->  {filename} as given     (if directly resolvable)
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
                direct = Path(geom.filename)
                if direct.is_file():
                    on_disk = direct
            if on_disk is None:
                dropped += 1
                continue
            v.geometry = geom.model_copy(update={"filename": str(on_disk.resolve())})
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
    backend = os.environ["PYOPENGL_PLATFORM"]

    # Distinguish "extra not installed" from "GL backend library won't load".
    # find_spec confirms the packages exist WITHOUT importing pyrender — which
    # eagerly loads the PyOpenGL backend and would otherwise conflate the two
    # (a missing libOSMesa/libEGL surfaces as an ImportError from `import
    # pyrender`, masquerading as a missing [meshviz] extra).
    import importlib.util

    if any(importlib.util.find_spec(m) is None for m in ("urchin", "pyrender")):
        raise ImportError(
            "render_pose_mesh needs the optional mesh-render stack. "
            "Install it with:  pip install 'fieldpilot-urdf[meshviz]'"
        )

    try:
        import numpy as np
        import pyrender
        import urchin
    except (ImportError, OSError) as e:  # pragma: no cover - needs a broken GL box
        # Packages are present (checked above), so the chosen GL backend's
        # native library failed to load — name it instead of blaming the extra.
        raise ImportError(
            f"render_pose_mesh's GL backend {backend!r} could not be loaded: {e}. "
            f"Install the backend's system library (libosmesa6 for 'osmesa', or "
            f"Mesa EGL — libegl1 libgl1-mesa-dri — for 'egl'), or choose another "
            f"via FIELDPILOT_URDF_RENDER_BACKEND."
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
