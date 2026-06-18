"""Oscilloscope parameter visuals: stacked time-series panels overlaying an
expected (simulated) signal against the observed one. Smoke level — confirms the
renderers emit the right image bytes, headless, including the divergence-shading
path and the TimedTrajectory convenience.
"""
from __future__ import annotations

import importlib.util

import pytest

from fieldpilot_urdf.retime import TimedTrajectory

mpl_missing = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="[viz] extra (matplotlib) not installed",
)

PNG_MAGIC = b"\x89PNG"
SVG_PREFIX = b"<?xml"


def _traj(offset=0.0):
    """A 2-joint timed trajectory; `offset` perturbs j2 to fake an 'observed' run."""
    times = [0.0, 0.5, 1.0, 1.5]
    q = [[0.0, 0.0], [0.3, 0.2 + offset], [0.6, 0.4 + offset], [0.9, 0.6 + offset]]
    u = [[0.6, 0.4], [0.6, 0.4], [0.6, 0.4], [0.0, 0.0]]
    return TimedTrajectory(joint_ids=["j1", "j2"], times=times, q=q, u=u)


@mpl_missing
def test_render_scope_png():
    from fieldpilot_urdf.viz import ScopePanel, ScopeSeries, render_scope
    panel = ScopePanel(ylabel="j3 position (rad)", series=[
        ScopeSeries(label="expected", times=[0, 1, 2], values=[0.0, 0.5, 1.0]),
        ScopeSeries(label="observed", times=[0, 1, 2], values=[0.0, 0.3, 0.7], style="dashed"),
    ])
    out = render_scope([panel], title="j3")
    assert out[:4] == PNG_MAGIC and len(out) > 1000


@mpl_missing
def test_render_scope_svg():
    from fieldpilot_urdf.viz import ScopePanel, ScopeSeries, render_scope
    out = render_scope([ScopePanel(ylabel="v", series=[
        ScopeSeries(label="a", times=[0, 1], values=[0, 1])])], fmt="svg")
    assert out[:5] == SVG_PREFIX


@mpl_missing
def test_render_scope_divergence_shading_runs():
    from fieldpilot_urdf.viz import ScopePanel, ScopeSeries, render_scope
    # two same-grid series -> shaded gap + max-Δ annotation
    panel = ScopePanel(ylabel="j1", shade_divergence=True, series=[
        ScopeSeries(label="sim", times=[0, 1, 2, 3], values=[0, 1, 2, 3]),
        ScopeSeries(label="obs", times=[0, 1, 2, 3], values=[0, 1.2, 1.7, 3.4])])
    out = render_scope([panel])
    assert out[:4] == PNG_MAGIC


@mpl_missing
def test_render_scope_empty_raises():
    from fieldpilot_urdf.viz import render_scope
    with pytest.raises(ValueError):
        render_scope([])


@mpl_missing
def test_trajectory_scope_single():
    from fieldpilot_urdf.viz import render_trajectory_scope
    out = render_trajectory_scope(_traj(), signals=("position", "velocity"))
    assert out[:4] == PNG_MAGIC and len(out) > 1000


@mpl_missing
def test_trajectory_scope_overlay_observed():
    from fieldpilot_urdf.viz import render_trajectory_scope
    out = render_trajectory_scope(_traj(), _traj(offset=0.15),   # observed drifts on j2
                                  joints=["j2"], signals=("position",),
                                  labels=("expected", "measured"))
    assert out[:4] == PNG_MAGIC


@mpl_missing
def test_trajectory_scope_unknown_signal_raises():
    from fieldpilot_urdf.viz import render_trajectory_scope
    with pytest.raises(ValueError):
        render_trajectory_scope(_traj(), signals=("torque",))
