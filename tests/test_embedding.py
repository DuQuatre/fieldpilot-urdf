"""Structural embedding tests (ported from MecAI, re-targeted onto Robot)."""
from __future__ import annotations

import numpy as np

from fieldpilot_urdf.embedding import (
    EMBEDDING_DIM, FEATURE_NAMES, cosine_similarity, embedding_features,
    rank_by_similarity, robot_dof, robot_embedding,
)
from fieldpilot_urdf.models import Inertial, Joint, JointLimit, Link, Origin, Robot


def _lim():
    return JointLimit(lower=-1, upper=1, effort=5, velocity=1)


def _arm(name="arm", masses=(1.0, 2.0, 3.0)):
    mb, m1, m2 = masses
    return Robot(
        name=name,
        links=[
            Link(name="base", inertial=Inertial(mass=mb)),
            Link(name="l1", inertial=Inertial(mass=m1)),
            Link(name="l2", inertial=Inertial(mass=m2)),
        ],
        joints=[
            Joint(name="j1", type="revolute", parent="base", child="l1",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 0, 1), limit=_lim()),
            Joint(name="j2", type="revolute", parent="l1", child="l2",
                  origin=Origin(xyz=(1, 0, 0)), axis=(0, 0, 1), limit=_lim()),
        ],
    )


def _gantry(name="gantry"):
    """A prismatic-prismatic-prismatic cartesian robot — very unlike the arm."""
    return Robot(
        name=name,
        links=[Link(name=f"l{i}", inertial=Inertial(mass=5.0)) for i in range(4)],
        joints=[
            Joint(name="x", type="prismatic", parent="l0", child="l1", axis=(1, 0, 0), limit=_lim()),
            Joint(name="y", type="prismatic", parent="l1", child="l2", axis=(0, 1, 0), limit=_lim()),
            Joint(name="z", type="prismatic", parent="l2", child="l3", axis=(0, 0, 1), limit=_lim()),
        ],
    )


def test_embedding_shape_and_keys():
    vec = robot_embedding(_arm())
    assert vec.shape == (EMBEDDING_DIM,)
    assert set(embedding_features(_arm())) == set(FEATURE_NAMES)


def test_serial_arm_features():
    feats = embedding_features(_arm())
    assert feats["n_links"] == 3
    assert feats["n_joints"] == 2
    assert feats["is_tree"] == 1.0
    assert feats["jt_frac_revolute"] == 1.0          # both joints revolute
    assert feats["jt_frac_prismatic"] == 0.0
    assert feats["dof"] == 2                          # two revolute = 2 DOF
    assert feats["total_mass"] == 6.0


def test_robot_dof_mapping():
    assert robot_dof(_arm()) == 2
    assert robot_dof(_gantry()) == 3


def test_depth_of_chain():
    feats = embedding_features(_arm())
    assert feats["depth"] == 2.0      # base -> l1 -> l2
    assert feats["n_leaf_links"] == 1.0


def test_self_similarity_is_one():
    vec = robot_embedding(_arm())
    assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-12


def test_arm_more_similar_to_itself_than_to_gantry():
    arm, gantry = robot_embedding(_arm()), robot_embedding(_gantry())
    assert cosine_similarity(arm, arm) > cosine_similarity(arm, gantry)


def test_two_arms_more_similar_than_arm_and_gantry():
    a1 = robot_embedding(_arm("a1"))
    a2 = robot_embedding(_arm("a2", masses=(1.1, 2.1, 2.9)))
    gantry = robot_embedding(_gantry())
    assert cosine_similarity(a1, a2) > cosine_similarity(a1, gantry)


def test_cosine_zero_vector():
    assert cosine_similarity(np.zeros(3), np.ones(3)) == 0.0


def test_rank_by_similarity_orders_descending():
    query = robot_embedding(_arm())
    candidates = {
        "arm_copy": robot_embedding(_arm("arm_copy")),
        "gantry": robot_embedding(_gantry()),
    }
    ranked = rank_by_similarity(query, candidates)
    assert [cid for cid, _ in ranked] == ["arm_copy", "gantry"]
    assert ranked[0][1] >= ranked[1][1]
    assert len(rank_by_similarity(query, candidates, top_k=1)) == 1


def test_rank_empty_candidates():
    assert rank_by_similarity(robot_embedding(_arm()), {}) == []
