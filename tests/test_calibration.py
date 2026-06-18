"""Multi-pose kinematic calibration. The anchor is a round-trip: bake known
per-joint offsets into a robot, generate observed poses at varied commanded
configs, then recover the offsets by calibration — including LARGE offsets that
the single-shot localizer can't, since Gauss-Newton re-linearizes each step.
"""
from __future__ import annotations

import numpy as np

from fieldpilot_urdf.fk import R_to_rpy, forward_kinematics
from fieldpilot_urdf.kinematic_diagnosis import (
    CalibrationResult, PoseObservation, calibrate_joint_offsets,
)
from fieldpilot_urdf.models import Joint, JointLimit, Link, Origin, Robot


def _lim(lo=-3.0, hi=3.0):
    return JointLimit(lower=lo, upper=hi, effort=1.0, velocity=1.0)


def _spatial_6r():
    axes = [(0, 0, 1), (0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0)]
    offs = [(0, 0, 0.3), (0, 0, 0.2), (0.5, 0, 0), (0.5, 0, 0), (0.1, 0, 0), (0.1, 0, 0)]
    links = [Link(name="base")] + [Link(name=f"l{i}") for i in range(1, 7)]
    joints, parent = [], "base"
    for i in range(6):
        child = f"l{i+1}"
        joints.append(Joint(name=f"j{i+1}", type="revolute", parent=parent,
                            child=child, origin=Origin(xyz=offs[i]), axis=axes[i],
                            limit=_lim()))
        parent = child
    return Robot(name="r6", links=links, joints=joints)


def _planar_3r(l=1.0):
    links = [Link(name="base")] + [Link(name=f"l{i}") for i in range(1, 4)] + [Link(name="tool")]
    joints = [
        Joint(name="j1", type="revolute", parent="base", child="l1",
              origin=Origin(xyz=(0, 0, 0)), axis=(0, 0, 1), limit=_lim()),
        Joint(name="j2", type="revolute", parent="l1", child="l2",
              origin=Origin(xyz=(l, 0, 0)), axis=(0, 0, 1), limit=_lim()),
        Joint(name="j3", type="revolute", parent="l2", child="l3",
              origin=Origin(xyz=(l, 0, 0)), axis=(0, 0, 1), limit=_lim()),
        Joint(name="tip", type="fixed", parent="l3", child="tool",
              origin=Origin(xyz=(l, 0, 0))),
    ]
    return Robot(name="r3", links=links, joints=joints)


def _observations(robot, link, configs, true_offsets, *, with_rpy=True):
    """Build PoseObservations: command `cfg`, but the robot is really at
    cfg + true_offsets, so the measured pose reflects the offsets."""
    obs = []
    for cfg in configs:
        actual = {k: cfg.get(k, 0.0) + true_offsets.get(k, 0.0) for k in cfg}
        T = forward_kinematics(robot, actual)[link]
        rpy = tuple(R_to_rpy(T[:3, :3])) if with_rpy else None
        obs.append(PoseObservation(commanded_q=dict(cfg), observed_xyz=tuple(T[:3, 3]),
                                   observed_rpy=rpy))
    return obs


def _random_configs(joint_names, n, seed):
    rng = np.random.default_rng(seed)
    return [{j: float(rng.uniform(-1.0, 1.0)) for j in joint_names} for _ in range(n)]


# --- the anchor: recover injected offsets -----------------------------------

def test_recovers_multi_joint_offsets():
    r = _spatial_6r()
    names = [f"j{i+1}" for i in range(6)]
    true = {"j2": 0.04, "j4": -0.03, "j5": 0.05}     # three joints miscalibrated
    configs = _random_configs(names, 12, seed=1)
    obs = _observations(r, "l6", configs, true)
    res = calibrate_joint_offsets(r, "l6", obs)
    assert res.converged
    for j in names:
        assert np.isclose(res.offsets[j], true.get(j, 0.0), atol=1e-4), (j, res.offsets[j])
    assert res.position_rms_after < 1e-5
    assert res.orientation_rms_after < 1e-5
    assert res.position_rms_after < res.position_rms_before


def test_recovers_large_offsets_gauss_newton():
    """Offsets far outside the linear regime — single-shot would fail, but the
    iterated solve still nails them."""
    r = _planar_3r()
    names = ["j1", "j2", "j3"]
    true = {"j1": 0.6, "j2": -0.7, "j3": 0.5}
    obs = _observations(r, "tool", _random_configs(names, 10, seed=2), true)
    res = calibrate_joint_offsets(r, "tool", obs)
    assert res.converged
    for j in names:
        assert np.isclose(res.offsets[j], true[j], atol=1e-4)
    assert res.position_rms_after < 1e-5
    assert res.position_rms_before > 0.1            # genuinely large error before


def test_perfect_robot_yields_zero_offsets():
    r = _planar_3r()
    names = ["j1", "j2", "j3"]
    obs = _observations(r, "tool", _random_configs(names, 6, seed=3), {})  # no offsets
    res = calibrate_joint_offsets(r, "tool", obs)
    assert all(abs(v) < 1e-6 for v in res.offsets.values())
    assert res.position_rms_before < 1e-9 and res.position_rms_after < 1e-9


# --- disambiguation: many poses pin down what one can't ---------------------

def test_more_poses_resolve_ambiguity():
    r = _planar_3r()
    names = ["j1", "j2", "j3"]
    true = {"j2": 0.05}
    # one position-only sample under-constrains 3 offsets -> min-norm spreads it
    one = _observations(r, "tool", _random_configs(names, 1, seed=4), true, with_rpy=False)
    res1 = calibrate_joint_offsets(r, "tool", one)
    # the single fit fits its one pose but the offsets need not match the truth
    spread = sum(abs(res1.offsets[j] - true.get(j, 0.0)) for j in names)

    many = _observations(r, "tool", _random_configs(names, 8, seed=4), true, with_rpy=False)
    res8 = calibrate_joint_offsets(r, "tool", many)
    recovered = sum(abs(res8.offsets[j] - true.get(j, 0.0)) for j in names)
    assert recovered < spread                       # more poses -> closer to truth
    assert np.isclose(res8.offsets["j2"], 0.05, atol=1e-3)


def test_position_only_observations_work():
    r = _spatial_6r()
    names = [f"j{i+1}" for i in range(6)]
    true = {"j1": 0.03, "j3": -0.04}
    obs = _observations(r, "l6", _random_configs(names, 20, seed=5), true, with_rpy=False)
    res = calibrate_joint_offsets(r, "l6", obs)
    assert res.orientation_rms_before == 0.0 and res.orientation_rms_after == 0.0
    assert np.isclose(res.offsets["j1"], 0.03, atol=1e-3)
    assert np.isclose(res.offsets["j3"], -0.04, atol=1e-3)
    assert res.position_rms_after < 1e-5


# --- metadata + edges -------------------------------------------------------

def test_result_metadata():
    r = _planar_3r()
    names = ["j1", "j2", "j3"]
    obs = _observations(r, "tool", _random_configs(names, 7, seed=6), {"j1": 0.02})
    res = calibrate_joint_offsets(r, "tool", obs)
    assert isinstance(res, CalibrationResult)
    assert res.n_samples == 7 and res.n_joints == 3
    assert set(res.offsets) == {"j1", "j2", "j3"}
    assert 1 <= res.iterations <= 20


def test_empty_observations_raises():
    r = _planar_3r()
    try:
        calibrate_joint_offsets(r, "tool", [])
        assert False, "empty observations should raise"
    except ValueError:
        pass


def test_fixed_chain_no_joints():
    r = Robot(name="f", links=[Link(name="base"), Link(name="tip")],
              joints=[Joint(name="f", type="fixed", parent="base", child="tip",
                            origin=Origin(xyz=(1, 0, 0)))])
    obs = [PoseObservation(commanded_q={}, observed_xyz=(1.0, 0.0, 0.0))]
    res = calibrate_joint_offsets(r, "tip", obs)
    assert res.n_joints == 0 and res.offsets == {}
    assert res.iterations == 0
