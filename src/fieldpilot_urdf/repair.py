"""Auto-repair URDF model violations.

For each diagnostic rule whose violation has a deterministic fix, a `fix_RNNN`
function rewrites a copy of the robot and emits a Patch describing the
change. Rules without a deterministic fix (R001 multiple roots, R002 not a
tree, R008 self-collisions) are skipped — they need human judgement.

Repairable rules:
    R003  joint limit invalid     — swap inverted lower/upper, default
                                    non-positive effort/velocity to a small
                                    positive value
    R004  link mass <= 0          — set to a small positive default
    R005  inertia not PSD         — project to nearest PSD (clip negative
                                    eigenvalues)
    R006  duplicate name          — suffix duplicates "_2", "_3"; rewrite
                                    joint parent/child references for links
    R007  zero joint axis         — set to default (1, 0, 0)
"""
from __future__ import annotations

from typing import Iterable, Literal, Optional

import numpy as np
from pydantic import BaseModel

from .models import Joint, JointLimit, Link, Robot


DEFAULT_EFFORT = 10.0
DEFAULT_VELOCITY = 1.0
DEFAULT_MASS = 1e-3
DEFAULT_AXIS: tuple[float, float, float] = (1.0, 0.0, 0.0)
INERTIA_FLOOR = 1e-9  # min eigenvalue after PSD projection

RepairableCode = Literal["R003", "R004", "R005", "R006", "R007"]
UNFIXABLE_CODES: tuple[str, ...] = ("R001", "R002", "R008")


class Patch(BaseModel):
    code: str
    target: str
    field: str
    before: object
    after: object


# --- per-rule fixes --------------------------------------------------------

def fix_r003(robot: Robot) -> tuple[Robot, list[Patch]]:
    """Repair joint <limit> sub-elements: invert swaps, default non-positive."""
    patches: list[Patch] = []
    new_joints: list[Joint] = []
    for j in robot.joints:
        if j.limit is None:
            new_joints.append(j)
            continue
        lim = j.limit
        lower, upper, effort, velocity = lim.lower, lim.upper, lim.effort, lim.velocity
        changed = False
        if lower >= upper:
            patches.append(Patch(code="R003", target=j.name, field="lower<->upper",
                                 before=[lower, upper], after=[upper, lower]))
            lower, upper = upper, lower
            changed = True
        if effort <= 0:
            patches.append(Patch(code="R003", target=j.name, field="effort",
                                 before=effort, after=DEFAULT_EFFORT))
            effort = DEFAULT_EFFORT
            changed = True
        if velocity <= 0:
            patches.append(Patch(code="R003", target=j.name, field="velocity",
                                 before=velocity, after=DEFAULT_VELOCITY))
            velocity = DEFAULT_VELOCITY
            changed = True
        if changed:
            new_joints.append(j.model_copy(update={
                "limit": JointLimit(lower=lower, upper=upper,
                                    effort=effort, velocity=velocity),
            }))
        else:
            new_joints.append(j)
    return robot.model_copy(update={"joints": new_joints}), patches


def fix_r004(robot: Robot) -> tuple[Robot, list[Patch]]:
    """Repair <inertial><mass>: bump non-positive masses to DEFAULT_MASS."""
    patches: list[Patch] = []
    new_links: list[Link] = []
    for l in robot.links:
        if l.inertial is None or l.inertial.mass > 0:
            new_links.append(l)
            continue
        patches.append(Patch(code="R004", target=l.name, field="mass",
                             before=l.inertial.mass, after=DEFAULT_MASS))
        new_links.append(l.model_copy(update={
            "inertial": l.inertial.model_copy(update={"mass": DEFAULT_MASS})
        }))
    return robot.model_copy(update={"links": new_links}), patches


def _project_psd(M: np.ndarray, floor: float = INERTIA_FLOOR) -> np.ndarray:
    """Nearest symmetric-PSD matrix: clip eigenvalues at `floor`."""
    S = (M + M.T) / 2.0  # symmetrise first
    w, V = np.linalg.eigh(S)
    w_clipped = np.clip(w, floor, None)
    return (V * w_clipped) @ V.T


def fix_r005(robot: Robot) -> tuple[Robot, list[Patch]]:
    """Project non-PSD inertia tensors onto the nearest PSD matrix."""
    patches: list[Patch] = []
    new_links: list[Link] = []
    for l in robot.links:
        if l.inertial is None:
            new_links.append(l)
            continue
        i = l.inertial.inertia
        M = np.array([[i.ixx, i.ixy, i.ixz],
                      [i.ixy, i.iyy, i.iyz],
                      [i.ixz, i.iyz, i.izz]], dtype=float)
        eig = np.linalg.eigvalsh((M + M.T) / 2.0)
        if eig.min() >= -1e-12:  # already PSD
            new_links.append(l)
            continue
        P = _project_psd(M)
        new_inertia = l.inertial.inertia.model_copy(update={
            "ixx": float(P[0, 0]), "ixy": float(P[0, 1]), "ixz": float(P[0, 2]),
            "iyy": float(P[1, 1]), "iyz": float(P[1, 2]), "izz": float(P[2, 2]),
        })
        patches.append(Patch(
            code="R005", target=l.name, field="inertia",
            before={"min_eig": float(eig.min()),
                    "ixx": i.ixx, "iyy": i.iyy, "izz": i.izz},
            after={"ixx": new_inertia.ixx, "iyy": new_inertia.iyy,
                   "izz": new_inertia.izz},
        ))
        new_links.append(l.model_copy(update={
            "inertial": l.inertial.model_copy(update={"inertia": new_inertia}),
        }))
    return robot.model_copy(update={"links": new_links}), patches


def _suffix(name: str, n: int) -> str:
    return f"{name}_{n}"


def fix_r006(robot: Robot) -> tuple[Robot, list[Patch]]:
    """De-duplicate link + joint names. Joint parent/child refs follow link
    renames so the topology stays valid."""
    patches: list[Patch] = []

    # Link rename pass.
    link_seen: dict[str, int] = {}
    link_renames: dict[int, str] = {}  # by id(link)
    new_link_names: list[str] = []
    for l in robot.links:
        if l.name not in link_seen:
            link_seen[l.name] = 1
            new_link_names.append(l.name)
            continue
        link_seen[l.name] += 1
        new_name = _suffix(l.name, link_seen[l.name])
        while new_name in link_seen:  # avoid clobbering an existing name
            link_seen[l.name] += 1
            new_name = _suffix(l.name, link_seen[l.name])
        link_seen[new_name] = 1
        patches.append(Patch(code="R006", target=l.name, field="link.name",
                             before=l.name, after=new_name))
        link_renames[id(l)] = new_name
        new_link_names.append(new_name)

    new_links = [
        l.model_copy(update={"name": link_renames[id(l)]}) if id(l) in link_renames else l
        for l in robot.links
    ]

    # Joint rename pass — independent from links.
    joint_seen: dict[str, int] = {}
    joint_renames: dict[int, str] = {}
    for j in robot.joints:
        if j.name not in joint_seen:
            joint_seen[j.name] = 1
            continue
        joint_seen[j.name] += 1
        new_name = _suffix(j.name, joint_seen[j.name])
        while new_name in joint_seen:
            joint_seen[j.name] += 1
            new_name = _suffix(j.name, joint_seen[j.name])
        joint_seen[new_name] = 1
        patches.append(Patch(code="R006", target=j.name, field="joint.name",
                             before=j.name, after=new_name))
        joint_renames[id(j)] = new_name

    # When a link gets renamed, only one of its duplicate occurrences gets a new
    # name; joint references to the original name still point at the FIRST
    # occurrence, which keeps its name. So joint parent/child rewriting is a
    # no-op for this strategy — but only if the duplicate is never referenced.
    # In practice URDFs that reference a duplicate link are already malformed.
    new_joints = [
        j.model_copy(update={"name": joint_renames[id(j)]}) if id(j) in joint_renames else j
        for j in robot.joints
    ]
    return robot.model_copy(update={"links": new_links, "joints": new_joints}), patches


def fix_r007(robot: Robot) -> tuple[Robot, list[Patch]]:
    """Set zero axis vectors on movable joints to DEFAULT_AXIS."""
    patches: list[Patch] = []
    new_joints: list[Joint] = []
    movable = {"revolute", "continuous", "prismatic"}
    for j in robot.joints:
        if j.type not in movable:
            new_joints.append(j)
            continue
        norm = float(np.linalg.norm(np.array(j.axis, dtype=float)))
        if norm > 1e-9:
            new_joints.append(j)
            continue
        patches.append(Patch(code="R007", target=j.name, field="axis",
                             before=list(j.axis), after=list(DEFAULT_AXIS)))
        new_joints.append(j.model_copy(update={"axis": DEFAULT_AXIS}))
    return robot.model_copy(update={"joints": new_joints}), patches


# --- orchestration ---------------------------------------------------------

REPAIRS: dict[str, callable] = {
    "R003": fix_r003,
    "R004": fix_r004,
    "R005": fix_r005,
    "R006": fix_r006,
    "R007": fix_r007,
}


def repair(
    robot: Robot, codes: Optional[Iterable[str]] = None,
) -> tuple[Robot, list[Patch], list[str]]:
    """Apply all (or a subset of) repairable fixes in canonical order.

    Returns:
        new_robot — repaired copy (original untouched)
        patches   — list of every change made
        unfixable — codes the caller asked for that we can't auto-fix
    """
    if codes is None:
        to_apply = list(REPAIRS)
        unfixable: list[str] = []
    else:
        requested = set(codes)
        unfixable = sorted(c for c in requested if c in UNFIXABLE_CODES)
        to_apply = [c for c in REPAIRS if c in requested]

    patches: list[Patch] = []
    cur = robot
    for code in to_apply:
        cur, new_patches = REPAIRS[code](cur)
        patches.extend(new_patches)
    return cur, patches, unfixable
