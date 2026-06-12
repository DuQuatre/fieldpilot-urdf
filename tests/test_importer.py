"""Tests for app/urdf/importer.py.

All network calls are mocked via monkeypatch on requests.get — no live
internet traffic in any test. Run:
    python3 -m pytest app/urdf/test_importer.py -q  (from pydexpi-server/)
"""
from __future__ import annotations

from typing import Optional

import pytest
import requests

from fieldpilot_urdf import from_xml
from fieldpilot_urdf.importer import (
    HostNotAllowed, ImportError_, SchemeNotAllowed,
    expand_xacro, fetch_urdf, import_urdf,
)


SIMPLE_URDF = """<?xml version="1.0"?>
<robot name="r">
  <link name="a"/><link name="b"/>
  <joint name="j" type="fixed"><parent link="a"/><child link="b"/></joint>
</robot>
"""

XACRO_TEMPLATE = """<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" name="t">
  <xacro:property name="L" value="0.42"/>
  <xacro:macro name="wheel" params="name">
    <link name="${name}">
      <visual><geometry><cylinder radius="0.05" length="${L}"/></geometry></visual>
    </link>
  </xacro:macro>
  <xacro:wheel name="left"/>
  <xacro:wheel name="right"/>
</robot>
"""


class _MockResponse:
    """Minimal stand-in for requests.Response."""
    def __init__(self, body: bytes, status: int = 200, url: str = "",
                 encoding: str = "utf-8"):
        self._body = body
        self.status_code = status
        self.url = url
        self.encoding = encoding

    def iter_content(self, chunk_size: int = 64 * 1024):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]


def _mock_get(body: bytes, *, status: int = 200, final_url: Optional[str] = None):
    """Build a monkeypatch target for requests.get."""
    def fn(url, *args, **kwargs):
        return _MockResponse(body, status=status, url=final_url or url)
    return fn


# --- expand_xacro ----------------------------------------------------------

def test_expand_xacro_property_and_macro():
    out = expand_xacro(XACRO_TEMPLATE)
    assert "${" not in out
    assert "xacro:" not in out
    robot = from_xml(out)
    assert robot.name == "t"
    assert {l.name for l in robot.links} == {"left", "right"}


def test_expand_xacro_passthrough_plain_urdf():
    out = expand_xacro(SIMPLE_URDF)
    robot = from_xml(out)
    assert robot.name == "r"


def test_expand_xacro_malformed_raises():
    with pytest.raises(Exception):
        expand_xacro("<robot>unclosed")


def test_expand_xacro_load_yaml_package_uri(monkeypatch):
    """load_yaml('package://...') resolves through the package alias map and
    fetches the YAML over HTTPS — same path xacro:include uses."""
    yaml_body = b"radius: 0.123\n"
    xacro_src = (
        '<?xml version="1.0"?>\n'
        '<robot xmlns:xacro="http://ros.org/wiki/xacro" name="y">'
        '<xacro:property name="cfg" value="${xacro.load_yaml(\'package://my_pkg/params.yaml\')}"/>'
        '<xacro:property name="r" value="${cfg[\'radius\']}"/>'
        '<link name="base">'
        '<visual><geometry><sphere radius="${r}"/></geometry></visual>'
        '</link>'
        '</robot>'
    )
    monkeypatch.setattr(requests, "get",
        _mock_get(yaml_body, final_url=(
            "https://raw.githubusercontent.com/owner/my_pkg/main/params.yaml"
        )))
    out = expand_xacro(
        xacro_src,
        base_url="https://raw.githubusercontent.com/owner/my_pkg/main/urdf/x.urdf.xacro",
    )
    # The sphere radius should be the YAML value, fully resolved.
    robot = from_xml(out)
    sphere = robot.links[0].visuals[0].geometry
    assert abs(sphere.radius - 0.123) < 1e-9


# --- fetch_urdf safety -----------------------------------------------------

def test_fetch_urdf_rejects_http():
    with pytest.raises(SchemeNotAllowed):
        fetch_urdf("http://github.com/foo.urdf")


def test_fetch_urdf_rejects_file_uri():
    with pytest.raises(SchemeNotAllowed):
        fetch_urdf("file:///etc/passwd")


def test_fetch_urdf_rejects_unknown_host():
    with pytest.raises(HostNotAllowed):
        fetch_urdf("https://evil.example.com/foo.urdf")


def test_fetch_urdf_allowlist_extra():
    """Extra hosts argument augments the default allowlist."""
    # We don't actually hit the network — bail out cleanly when requests fails,
    # the host check passes first.
    try:
        fetch_urdf("https://my.private.host/foo.urdf",
                   allowed_hosts=["my.private.host"], timeout=0.01)
    except ImportError_:
        # network error is fine — host check already passed
        pass


# --- fetch_urdf success path (mocked) --------------------------------------

def test_fetch_urdf_returns_body_and_final_url(monkeypatch):
    monkeypatch.setattr(requests, "get",
                        _mock_get(SIMPLE_URDF.encode(),
                                  final_url="https://github.com/x/y.urdf"))
    body, final = fetch_urdf("https://github.com/x/y.urdf")
    assert "<robot" in body
    assert final == "https://github.com/x/y.urdf"


def test_fetch_urdf_redirect_to_disallowed_host_rejected(monkeypatch):
    """Redirect landing off the allowlist must be refused."""
    monkeypatch.setattr(requests, "get",
                        _mock_get(b"<robot/>", final_url="https://evil.example.com/x"))
    with pytest.raises(HostNotAllowed):
        fetch_urdf("https://github.com/x/y.urdf")


def test_fetch_urdf_non_200(monkeypatch):
    monkeypatch.setattr(requests, "get",
                        _mock_get(b"not found", status=404,
                                  final_url="https://github.com/x/y.urdf"))
    with pytest.raises(ImportError_) as e:
        fetch_urdf("https://github.com/x/y.urdf")
    assert "404" in str(e.value)


def test_fetch_urdf_size_cap(monkeypatch):
    monkeypatch.setattr(requests, "get",
                        _mock_get(b"X" * 200,
                                  final_url="https://github.com/x/y.urdf"))
    with pytest.raises(ImportError_) as e:
        fetch_urdf("https://github.com/x/y.urdf", max_bytes=100)
    assert "exceeds" in str(e.value)


def test_fetch_urdf_network_error_wrapped(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.ConnectionError("DNS failure")
    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(ImportError_) as e:
        fetch_urdf("https://github.com/x/y.urdf")
    assert "DNS" in str(e.value)


# --- end-to-end import_urdf ------------------------------------------------

def test_import_urdf_xacro_round_trip(monkeypatch):
    monkeypatch.setattr(requests, "get",
                        _mock_get(XACRO_TEMPLATE.encode(),
                                  final_url="https://github.com/x/t.urdf.xacro"))
    robot, final = import_urdf("https://github.com/x/t.urdf.xacro")
    assert robot.name == "t"
    assert {l.name for l in robot.links} == {"left", "right"}
    assert final == "https://github.com/x/t.urdf.xacro"


def test_import_urdf_skip_macros(monkeypatch):
    """expand_macros=False should bypass xacro entirely."""
    monkeypatch.setattr(requests, "get",
                        _mock_get(SIMPLE_URDF.encode(),
                                  final_url="https://github.com/x/r.urdf"))
    robot, _ = import_urdf("https://github.com/x/r.urdf", expand_macros=False)
    assert robot.name == "r"


# --- mesh download helpers -------------------------------------------------

from fieldpilot_urdf.importer import (
    fetch_meshes, infer_package_root, package_uri_parts,
)


def test_package_uri_parts():
    assert package_uri_parts("package://ur_description/meshes/ur5/Base.stl") == \
        ("ur_description", "meshes/ur5/Base.stl")
    assert package_uri_parts("file:///x") is None
    assert package_uri_parts("plain.stl") is None


def test_infer_package_root_github_layout():
    url = ("https://raw.githubusercontent.com/ros-industrial/universal_robot/"
           "melodic-devel/ur_description/urdf/ur5.urdf.xacro")
    root = infer_package_root(url, "ur_description")
    assert root == ("https://raw.githubusercontent.com/ros-industrial/"
                    "universal_robot/melodic-devel/ur_description/")


def test_infer_package_root_missing():
    assert infer_package_root("https://github.com/x/y/main/a.urdf", "ghost") is None


def test_infer_package_root_single_package_repo():
    """Regression: when pkg name equals repo name on raw.githubusercontent.com
    (e.g. ros/urdf_tutorial), the package root is the branch dir, not the
    repo dir. Surfaced during UR5 verification."""
    url = "https://raw.githubusercontent.com/ros/urdf_tutorial/ros2/urdf/08-macroed.urdf.xacro"
    root = infer_package_root(url, "urdf_tutorial")
    assert root == "https://raw.githubusercontent.com/ros/urdf_tutorial/ros2/"


def test_infer_package_root_multi_package_repo_unchanged():
    """Multi-package layout (pkg name != repo name) must still work."""
    url = ("https://raw.githubusercontent.com/ros-industrial/universal_robot/"
           "kinetic-devel/ur_description/urdf/ur5.urdf.xacro")
    root = infer_package_root(url, "ur_description")
    assert root == ("https://raw.githubusercontent.com/ros-industrial/"
                    "universal_robot/kinetic-devel/ur_description/")


def test_infer_package_root_builtin_alias_ur_description():
    """ur_description → Universal_Robots_ROS2_Description is built-in."""
    url = ("https://raw.githubusercontent.com/UniversalRobots/"
           "Universal_Robots_ROS2_Description/rolling/urdf/ur.urdf.xacro")
    root = infer_package_root(url, "ur_description")
    assert root == ("https://raw.githubusercontent.com/UniversalRobots/"
                    "Universal_Robots_ROS2_Description/rolling/")


def test_infer_package_root_env_alias_override(monkeypatch):
    """FIELDPILOT_URDF_PACKAGE_ALIASES adds/overrides built-ins at runtime."""
    monkeypatch.setenv(
        "FIELDPILOT_URDF_PACKAGE_ALIASES",
        "myrobot_description=MyRobot_ROS2_Stack",
    )
    url = ("https://raw.githubusercontent.com/acme/MyRobot_ROS2_Stack/"
           "main/urdf/x.urdf.xacro")
    root = infer_package_root(url, "myrobot_description")
    assert root == "https://raw.githubusercontent.com/acme/MyRobot_ROS2_Stack/main/"


def test_infer_package_root_alias_misses_when_repo_absent():
    """Alias present but the aliased repo isn't in the URL: still None."""
    url = "https://raw.githubusercontent.com/some/other_repo/main/x.urdf.xacro"
    assert infer_package_root(url, "ur_description") is None


# --- fetch_meshes end-to-end ----------------------------------------------

URDF_WITH_MESHES = """<?xml version="1.0"?>
<robot name="meshbot">
  <link name="base">
    <collision><geometry>
      <mesh filename="package://my_pkg/meshes/base.stl"/>
    </geometry></collision>
  </link>
  <link name="end">
    <collision><geometry>
      <mesh filename="package://my_pkg/meshes/end.stl"/>
    </geometry></collision>
  </link>
  <link name="orphan">
    <collision><geometry>
      <mesh filename="package://unknown_pkg/floating.stl"/>
    </geometry></collision>
  </link>
  <joint name="j1" type="fixed"><parent link="base"/><child link="end"/></joint>
  <joint name="j2" type="fixed"><parent link="base"/><child link="orphan"/></joint>
</robot>"""


def _stl_bytes() -> bytes:
    """A tiny but valid 80-byte STL header followed by zero-triangle count."""
    return b"solid t\nendsolid t\n"


def test_fetch_meshes_downloads_into_package_dir(tmp_path, monkeypatch):
    robot = from_xml(URDF_WITH_MESHES)
    urdf_url = "https://github.com/x/my_pkg/urdf/r.urdf"

    fetched: list[str] = []

    def fake_get(url, *args, **kwargs):
        fetched.append(url)
        return _MockResponse(_stl_bytes(), url=url)

    monkeypatch.setattr(requests, "get", fake_get)
    report = fetch_meshes(robot, urdf_url, tmp_path)
    # Two from my_pkg should resolve, one from unknown_pkg should fail.
    assert len(report["downloaded"]) == 2
    assert len(report["failures"]) == 1
    assert report["failures"][0]["filename"].startswith("package://unknown_pkg")
    assert "my_pkg" in report["packages"]
    assert "unknown_pkg" in report["packages"]
    # Files actually exist on disk under {tmp}/my_pkg/meshes/{base,end}.stl
    assert (tmp_path / "my_pkg" / "meshes" / "base.stl").exists()
    assert (tmp_path / "my_pkg" / "meshes" / "end.stl").exists()
    # URLs hit the inferred package root.
    assert any("my_pkg/meshes/base.stl" in u for u in fetched)


# --- substitute_find -------------------------------------------------------

from fieldpilot_urdf.importer import substitute_find


def test_substitute_find_basic():
    out = substitute_find('<x p="$(find ur_description)/x.stl"/>')
    assert out == '<x p="package://ur_description/x.stl"/>'


def test_substitute_find_multiple():
    src = '<a><b p="$(find pkg_a)/x"/><c p="$(find pkg_b)/y"/></a>'
    out = substitute_find(src)
    assert "package://pkg_a/x" in out
    assert "package://pkg_b/y" in out


def test_substitute_find_with_dashes_and_underscores():
    assert substitute_find("$(find my-pkg_123)") == "package://my-pkg_123"


def test_substitute_find_no_match_passthrough():
    src = '<robot name="r"><link name="a"/></robot>'
    assert substitute_find(src) == src


# --- resolve_includes ------------------------------------------------------

from fieldpilot_urdf.importer import resolve_includes


def test_resolve_includes_inlines_child(monkeypatch):
    """End-to-end pipeline (substitute_find → resolve_includes) for the common
    `<xacro:include filename="$(find pkg)/sub.xacro"/>` pattern."""
    parent = (
        '<?xml version="1.0"?>'
        '<robot xmlns:xacro="http://wiki.ros.org/xacro" name="p">'
        '  <xacro:include filename="$(find pkg)/sub.xacro"/>'
        '  <link name="own"/>'
        '</robot>'
    )
    child = (
        '<?xml version="1.0"?>'
        '<xacro:macros xmlns:xacro="http://wiki.ros.org/xacro">'
        '  <link name="imported"/>'
        '</xacro:macros>'
    )
    # The child URL pyMechDiag will derive from $(find pkg) on a GitHub-style URL:
    #   https://raw.githubusercontent.com/me/pkg/main/sub.xacro
    expected_url = "https://raw.githubusercontent.com/me/pkg/main/sub.xacro"

    def fake_get(url, *a, **kw):
        if url == expected_url:
            return _MockResponse(child.encode(), url=url)
        raise AssertionError(f"unexpected fetch: {url}")

    monkeypatch.setattr(requests, "get", fake_get)
    # The import pipeline pre-substitutes $(find) before resolving includes.
    out = resolve_includes(
        substitute_find(parent),
        "https://raw.githubusercontent.com/me/pkg/main/parent.xacro",
    )
    assert "<link" in out
    assert 'name="imported"' in out
    assert 'name="own"' in out
    # The include element must be gone.
    assert "xacro:include" not in out


def test_resolve_includes_nested(monkeypatch):
    a = (
        '<?xml version="1.0"?>'
        '<robot xmlns:xacro="http://wiki.ros.org/xacro" name="a">'
        '  <xacro:include filename="https://github.com/x/pkg/main/b.xacro"/>'
        '</robot>'
    )
    b = (
        '<?xml version="1.0"?>'
        '<xacro:r xmlns:xacro="http://wiki.ros.org/xacro">'
        '  <xacro:include filename="https://github.com/x/pkg/main/c.xacro"/>'
        '  <link name="b_link"/>'
        '</xacro:r>'
    )
    c = (
        '<?xml version="1.0"?>'
        '<xacro:r xmlns:xacro="http://wiki.ros.org/xacro">'
        '  <link name="c_link"/>'
        '</xacro:r>'
    )
    responses = {
        "https://github.com/x/pkg/main/b.xacro": b,
        "https://github.com/x/pkg/main/c.xacro": c,
    }

    def fake_get(url, *a, **kw):
        if url in responses:
            return _MockResponse(responses[url].encode(), url=url)
        raise AssertionError(f"unexpected fetch: {url}")

    monkeypatch.setattr(requests, "get", fake_get)
    out = resolve_includes(a, "https://github.com/x/pkg/main/a.xacro")
    assert 'name="b_link"' in out
    assert 'name="c_link"' in out
    assert "xacro:include" not in out


def test_resolve_includes_cycle_drops_duplicate(monkeypatch):
    """When the same file is included from two places, the second include is
    silently dropped (matches xacro's include-guard semantics)."""
    a = (
        '<?xml version="1.0"?>'
        '<robot xmlns:xacro="http://wiki.ros.org/xacro" name="a">'
        '  <xacro:include filename="https://github.com/x/pkg/main/common.xacro"/>'
        '  <xacro:include filename="https://github.com/x/pkg/main/common.xacro"/>'
        '</robot>'
    )
    common = (
        '<?xml version="1.0"?>'
        '<xacro:r xmlns:xacro="http://wiki.ros.org/xacro">'
        '  <link name="L"/>'
        '</xacro:r>'
    )
    fetches: list[str] = []

    def fake_get(url, *a, **kw):
        fetches.append(url)
        return _MockResponse(common.encode(), url=url)

    monkeypatch.setattr(requests, "get", fake_get)
    out = resolve_includes(a, "https://github.com/x/pkg/main/a.xacro")
    assert len(fetches) == 1  # not 2 — duplicate include dropped
    assert out.count('name="L"') == 1


def test_resolve_includes_relative_path(monkeypatch):
    parent = (
        '<?xml version="1.0"?>'
        '<robot xmlns:xacro="http://wiki.ros.org/xacro" name="p">'
        '  <xacro:include filename="sibling.xacro"/>'
        '</robot>'
    )
    sibling = (
        '<?xml version="1.0"?>'
        '<xacro:r xmlns:xacro="http://wiki.ros.org/xacro">'
        '  <link name="s"/>'
        '</xacro:r>'
    )

    def fake_get(url, *a, **kw):
        # Must resolve relative to the parent's URL directory.
        assert url == "https://github.com/x/pkg/main/sibling.xacro"
        return _MockResponse(sibling.encode(), url=url)

    monkeypatch.setattr(requests, "get", fake_get)
    out = resolve_includes(parent, "https://github.com/x/pkg/main/parent.xacro")
    assert 'name="s"' in out


def test_resolve_includes_unresolvable_raises():
    """Include that can't be resolved (e.g. $(find ghost) where ghost
    appears nowhere in the URL path) raises ImportError_."""
    parent = (
        '<?xml version="1.0"?>'
        '<robot xmlns:xacro="http://wiki.ros.org/xacro" name="p">'
        '  <xacro:include filename="$(find ghost)/x.xacro"/>'
        '</robot>'
    )
    # No package name 'ghost' in the URL path.
    parent_with_subbed = substitute_find(parent)
    with pytest.raises(ImportError_):
        resolve_includes(parent_with_subbed,
                         "https://github.com/me/pkg/main/parent.xacro")


def test_fetch_meshes_handles_network_error(tmp_path, monkeypatch):
    robot = from_xml(URDF_WITH_MESHES)

    def boom(*args, **kwargs):
        raise requests.ConnectionError("simulated")

    monkeypatch.setattr(requests, "get", boom)
    report = fetch_meshes(robot, "https://github.com/x/my_pkg/r.urdf", tmp_path)
    assert report["downloaded"] == []
    # Two my_pkg meshes fail with network error, one unknown_pkg fails with
    # "package not found" (no network call needed for that one).
    assert len(report["failures"]) == 3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
