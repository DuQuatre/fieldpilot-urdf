"""URDF XML <-> Pydantic. Stdlib only.

Pure mapping, no business logic. Round-trip preserves structural fields;
xmlns/comments are dropped.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from .models import (
    Box, Collision, Cylinder, Geometry, Inertia, Inertial, Joint, JointLimit,
    Link, Mesh, Origin, Robot, Sphere, Vec3, Visual,
)


# --- parse helpers ---------------------------------------------------------

def _vec3(s: Optional[str], default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    if not s:
        return default
    parts = s.split()
    if len(parts) != 3:
        raise ValueError(f"expected 3 floats, got: {s!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _origin(el: Optional[ET.Element]) -> Optional[Origin]:
    if el is None:
        return None
    return Origin(xyz=_vec3(el.get("xyz")), rpy=_vec3(el.get("rpy")))


def _geometry(el: ET.Element) -> Geometry:
    child = next(iter(el), None)
    if child is None:
        raise ValueError("<geometry> has no child")
    tag = child.tag
    if tag == "box":
        return Box(size=_vec3(child.get("size")))
    if tag == "cylinder":
        return Cylinder(radius=float(child.get("radius")), length=float(child.get("length")))
    if tag == "sphere":
        return Sphere(radius=float(child.get("radius")))
    if tag == "mesh":
        return Mesh(
            filename=child.get("filename", ""),
            scale=_vec3(child.get("scale"), (1.0, 1.0, 1.0)),
        )
    raise ValueError(f"unknown geometry: <{tag}>")


def _inertial(el: Optional[ET.Element]) -> Optional[Inertial]:
    if el is None:
        return None
    mass_el = el.find("mass")
    inertia_el = el.find("inertia")
    return Inertial(
        origin=_origin(el.find("origin")),
        mass=float(mass_el.get("value", 0.0)) if mass_el is not None else 0.0,
        inertia=Inertia(**{k: float(inertia_el.get(k, 0.0))
                           for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")})
                if inertia_el is not None else Inertia(),
    )


def _visual(el: ET.Element) -> Visual:
    mat = el.find("material")
    return Visual(
        name=el.get("name"),
        origin=_origin(el.find("origin")),
        geometry=_geometry(el.find("geometry")),
        material_name=mat.get("name") if mat is not None else None,
    )


def _collision(el: ET.Element) -> Collision:
    return Collision(
        name=el.get("name"),
        origin=_origin(el.find("origin")),
        geometry=_geometry(el.find("geometry")),
    )


def _link(el: ET.Element) -> Link:
    return Link(
        name=el.get("name", ""),
        inertial=_inertial(el.find("inertial")),
        visuals=[_visual(v) for v in el.findall("visual")],
        collisions=[_collision(c) for c in el.findall("collision")],
    )


def _joint(el: ET.Element) -> Joint:
    parent_el = el.find("parent")
    child_el = el.find("child")
    axis_el = el.find("axis")
    limit_el = el.find("limit")
    return Joint(
        name=el.get("name", ""),
        type=el.get("type", "fixed"),
        parent=parent_el.get("link", "") if parent_el is not None else "",
        child=child_el.get("link", "") if child_el is not None else "",
        origin=_origin(el.find("origin")),
        axis=_vec3(axis_el.get("xyz"), (1.0, 0.0, 0.0)) if axis_el is not None else (1.0, 0.0, 0.0),
        limit=JointLimit(
            lower=float(limit_el.get("lower", 0.0)),
            upper=float(limit_el.get("upper", 0.0)),
            effort=float(limit_el.get("effort")),
            velocity=float(limit_el.get("velocity")),
        ) if limit_el is not None else None,
    )


def from_xml(text: str) -> Robot:
    root = ET.fromstring(text)
    if root.tag != "robot":
        raise ValueError(f"expected <robot>, got <{root.tag}>")
    return Robot(
        name=root.get("name", ""),
        links=[_link(l) for l in root.findall("link")],
        joints=[_joint(j) for j in root.findall("joint")],
    )


def from_file(path: str | Path) -> Robot:
    return from_xml(Path(path).read_text())


# --- serialize -------------------------------------------------------------

def _vec3_str(v: Vec3) -> str:
    return f"{v[0]} {v[1]} {v[2]}"


def _add_origin(parent: ET.Element, o: Optional[Origin]) -> None:
    if o is not None:
        ET.SubElement(parent, "origin", xyz=_vec3_str(o.xyz), rpy=_vec3_str(o.rpy))


def _geometry_el(parent: ET.Element, g: Geometry) -> None:
    geo = ET.SubElement(parent, "geometry")
    if isinstance(g, Box):
        ET.SubElement(geo, "box", size=_vec3_str(g.size))
    elif isinstance(g, Cylinder):
        ET.SubElement(geo, "cylinder", radius=str(g.radius), length=str(g.length))
    elif isinstance(g, Sphere):
        ET.SubElement(geo, "sphere", radius=str(g.radius))
    elif isinstance(g, Mesh):
        ET.SubElement(geo, "mesh", filename=g.filename, scale=_vec3_str(g.scale))


def to_xml(robot: Robot) -> str:
    root = ET.Element("robot", name=robot.name)
    for l in robot.links:
        lel = ET.SubElement(root, "link", name=l.name)
        if l.inertial is not None:
            iel = ET.SubElement(lel, "inertial")
            _add_origin(iel, l.inertial.origin)
            ET.SubElement(iel, "mass", value=str(l.inertial.mass))
            ix = l.inertial.inertia
            ET.SubElement(iel, "inertia",
                          ixx=str(ix.ixx), ixy=str(ix.ixy), ixz=str(ix.ixz),
                          iyy=str(ix.iyy), iyz=str(ix.iyz), izz=str(ix.izz))
        for v in l.visuals:
            vel = ET.SubElement(lel, "visual", **({"name": v.name} if v.name else {}))
            _add_origin(vel, v.origin)
            _geometry_el(vel, v.geometry)
            if v.material_name:
                ET.SubElement(vel, "material", name=v.material_name)
        for c in l.collisions:
            cel = ET.SubElement(lel, "collision", **({"name": c.name} if c.name else {}))
            _add_origin(cel, c.origin)
            _geometry_el(cel, c.geometry)
    for j in robot.joints:
        jel = ET.SubElement(root, "joint", name=j.name, type=j.type)
        _add_origin(jel, j.origin)
        ET.SubElement(jel, "parent", link=j.parent)
        ET.SubElement(jel, "child", link=j.child)
        ET.SubElement(jel, "axis", xyz=_vec3_str(j.axis))
        if j.limit is not None:
            ET.SubElement(jel, "limit",
                          lower=str(j.limit.lower), upper=str(j.limit.upper),
                          effort=str(j.limit.effort), velocity=str(j.limit.velocity))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")
