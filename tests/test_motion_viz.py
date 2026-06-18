"""3D fault-motion visuals: animate a robot through a trajectory and compare a
nominal motion against a faulted one. Smoke level — confirms the renderers emit
the right bytes (GIF / PNG frames) with the expected frame counts, headless.
"""
from __future__ import annotations

import importlib.util

import pytest

from fieldpilot_urdf.models import Joint, JointLimit, Link, Origin, Robot

mpl_missing = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="[viz] extra (matplotlib) not installed",
)

GIF_MAGIC = b"GIF8"
PNG_MAGIC = b"\x89PNG"


def _arm():
    return Robot(
        name="arm",
        links=[Link(name="base"), Link(name="l1"), Link(name="tool")],
        joints=[
            Joint(name="j1", type="revolute", parent="base", child="l1",
                  origin=Origin(xyz=(0, 0, 0)), axis=(0, 0, 1),
                  limit=JointLimit(lower=-3, upper=3, effort=1, velocity=1)),
            Joint(name="j2", type="revolute", parent="l1", child="tool",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 0, 1),
                  limit=JointLimit(lower=-3, upper=3, effort=1, velocity=1)),
        ],
    )


def _sweep(n=5):
    """Nominal motion: both joints sweep."""
    return [{"j1": 0.3 * k / n, "j2": 1.2 * k / n} for k in range(n + 1)]


def _frozen_j2(n=5):
    """Faulted motion: j2 is stuck at 0 while j1 still moves."""
    return [{"j1": 0.3 * k / n, "j2": 0.0} for k in range(n + 1)]


@mpl_missing
def test_render_motion_gif():
    from fieldpilot_urdf.viz import render_motion
    out = render_motion(_arm(), _sweep(5), fmt="gif")
    assert out[:4] == GIF_MAGIC
    assert len(out) > 1000


@mpl_missing
def test_render_motion_frames_count_and_magic():
    from fieldpilot_urdf.viz import render_motion
    frames = render_motion(_arm(), _sweep(4), fmt="frames")
    assert isinstance(frames, list) and len(frames) == 5     # n+1 configs
    assert all(f[:4] == PNG_MAGIC for f in frames)


@mpl_missing
def test_render_motion_with_track_link():
    from fieldpilot_urdf.viz import render_motion
    out = render_motion(_arm(), _sweep(4), fmt="gif", track_link="tool")
    assert out[:4] == GIF_MAGIC


@mpl_missing
def test_comparison_overlay_gif():
    from fieldpilot_urdf.viz import render_motion_comparison
    out = render_motion_comparison(_arm(), _sweep(5), _frozen_j2(5),
                                   layout="overlay", fmt="gif", track_link="tool")
    assert out[:4] == GIF_MAGIC
    assert len(out) > 1000


@mpl_missing
def test_comparison_sidebyside_frames():
    from fieldpilot_urdf.viz import render_motion_comparison
    frames = render_motion_comparison(_arm(), _sweep(3), _frozen_j2(3),
                                      layout="sidebyside", fmt="frames")
    assert len(frames) == 4 and all(f[:4] == PNG_MAGIC for f in frames)


@mpl_missing
def test_comparison_truncates_to_shorter_trajectory():
    from fieldpilot_urdf.viz import render_motion_comparison
    frames = render_motion_comparison(_arm(), _sweep(6), _frozen_j2(2), fmt="frames")
    assert len(frames) == 3                                  # min(7, 3)


@mpl_missing
def test_comparison_auto_picks_track_link():
    from fieldpilot_urdf.viz import render_motion_comparison
    # no track_link given -> a leaf link is chosen automatically; must still render
    out = render_motion_comparison(_arm(), _sweep(3), _frozen_j2(3), fmt="gif")
    assert out[:4] == GIF_MAGIC


@mpl_missing
def test_empty_trajectory_raises():
    from fieldpilot_urdf.viz import render_motion, render_motion_comparison
    with pytest.raises(ValueError):
        render_motion(_arm(), [])
    with pytest.raises(ValueError):
        render_motion_comparison(_arm(), [], _sweep(2))
