"""ROS-package URDF importer.

Fetch a URDF (or URDF.xacro) from the internet and expand it into a parsed
Robot. Two safety knobs:

  - HTTPS only (no http://, no file://, no opaque schemes)
  - Host allowlist (overridable via FIELDPILOT_URDF_ALLOWED_HOSTS env var)

xacro expansion is done in-process via the xacro Python library, so no
subprocess or shell. Mesh files referenced from the URDF are NOT downloaded
— resolve them yourself via fieldpilot_urdf.collisions.MeshResolver.
"""
from __future__ import annotations

import os
import re
import xml.dom.minidom
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urljoin

import requests
import xacro

from .loader import from_xml
from .models import Mesh, Robot


DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = (
    "github.com",
    "raw.githubusercontent.com",
    "gitlab.com",
    "raw.gitlab.com",
    "bitbucket.org",
    "ros-industrial.org",
)
MAX_BYTES = 5 * 1024 * 1024  # 5 MB cap on fetched URDFs (xacro can blow up)
DEFAULT_TIMEOUT = 10.0


class ImportError_(Exception):
    """Generic import failure (network, scheme, host, size, etc.)."""


class HostNotAllowed(ImportError_):
    pass


class SchemeNotAllowed(ImportError_):
    pass


def _allowed_hosts(extra: Optional[list[str]] = None) -> frozenset[str]:
    env = (os.environ.get("FIELDPILOT_URDF_ALLOWED_HOSTS")
           or os.environ.get("MECHDIAG_URDF_ALLOWED_HOSTS", "")).strip()
    env_hosts = tuple(h.strip() for h in env.split(",") if h.strip()) if env else ()
    return frozenset(DEFAULT_ALLOWED_HOSTS) | frozenset(env_hosts) | frozenset(extra or [])


def _fetch_bytes(
    url: str,
    *,
    allowed_hosts: Optional[list[str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = MAX_BYTES,
) -> tuple[bytes, str, Optional[str]]:
    """Internal: HTTPS-only, host-allowlisted, size-capped binary download.
    Returns (body_bytes, final_url, encoding_hint).
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise SchemeNotAllowed(f"scheme must be https, got {parsed.scheme!r}")
    if parsed.hostname is None:
        raise SchemeNotAllowed("URL has no host")
    hosts = _allowed_hosts(allowed_hosts)
    if parsed.hostname not in hosts:
        raise HostNotAllowed(
            f"host {parsed.hostname!r} not in allowlist {sorted(hosts)}"
        )
    try:
        resp = requests.get(url, timeout=timeout, stream=True,
                            allow_redirects=True)
    except requests.RequestException as e:
        raise ImportError_(f"network error: {e}") from e
    if resp.status_code != 200:
        raise ImportError_(f"HTTP {resp.status_code} from {url}")
    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_content(chunk_size=64 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise ImportError_(f"response exceeds {max_bytes} bytes")
        chunks.append(chunk)
    final_url = resp.url
    # A redirect could move us off the allowlist; refuse.
    final_host = urlparse(final_url).hostname
    if final_host not in hosts:
        raise HostNotAllowed(
            f"redirect landed on disallowed host {final_host!r}"
        )
    return b"".join(chunks), final_url, resp.encoding


def fetch_urdf(
    url: str,
    *,
    allowed_hosts: Optional[list[str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = MAX_BYTES,
) -> tuple[str, str]:
    """Download a text URDF/.xacro file. Returns (body, final_url)."""
    raw, final_url, encoding = _fetch_bytes(
        url, allowed_hosts=allowed_hosts, timeout=timeout, max_bytes=max_bytes,
    )
    return raw.decode(encoding or "utf-8"), final_url


_FIND_RE = re.compile(r"\$\(find\s+([A-Za-z0-9_\-]+)\)")
MAX_INCLUDE_DEPTH = 16


def substitute_find(text: str) -> str:
    """Replace every `$(find pkg_name)` with `package://pkg_name` in the
    given URDF/xacro text. Lets xacro skip the ROS-runtime resolver and lets
    fetch_meshes pick the resulting URIs up as plain package:// references.
    """
    return _FIND_RE.sub(lambda m: f"package://{m.group(1)}", text)


def _resolve_include_target(filename: str, base_url: str) -> Optional[str]:
    """Map the value of an <xacro:include filename="..."> attribute to an
    absolute HTTPS URL we can fetch. Returns None when the path can't be
    resolved (caller surfaces as a fetch failure).

    Supported forms:
      package://pkg/sub/path.xacro     → infer_package_root(base_url, pkg) + sub
      https://… (absolute)             → as-is
      relative path                    → joined against base_url's directory
    """
    if filename.startswith("https://"):
        return filename
    if filename.startswith("package://"):
        parts = package_uri_parts(filename)
        if parts is None:
            return None
        pkg, sub = parts
        root = infer_package_root(base_url, pkg)
        if root is None:
            return None
        return root + sub
    # Relative path — resolve against the URDF URL's directory.
    return urljoin(base_url, filename)


def _is_xacro_include(elem) -> bool:
    """True for any <xacro:include> — by far the common case is the literal
    'xacro:include' tagName (minidom keeps the prefix as part of tagName)."""
    if elem.nodeType != elem.ELEMENT_NODE:
        return False
    return elem.tagName.lower() == "xacro:include"


def resolve_includes(
    text: str,
    base_url: str,
    *,
    allowed_hosts: Optional[list[str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = MAX_BYTES,
    max_depth: int = MAX_INCLUDE_DEPTH,
) -> str:
    """Recursively inline every `<xacro:include filename="…"/>` element.

    `filename` is resolved against `base_url`; included content is fetched
    over HTTPS (same safety constraints as fetch_urdf), recursively resolved,
    and spliced in place of the include element. Cycle detection via a
    visited-URL set. Each fetched document is passed through
    `substitute_find` before parsing, so nested `$(find pkg)` references in
    included files get resolved too. Raises ImportError_ on fetch failure
    or depth overflow.
    """
    doc = xml.dom.minidom.parseString(substitute_find(text))
    _splice_includes(doc.documentElement, base_url,
                     allowed_hosts=allowed_hosts, timeout=timeout,
                     max_bytes=max_bytes, depth=0, max_depth=max_depth,
                     visited=set())
    return doc.toxml()


def _splice_includes(elem, base_url, *, allowed_hosts, timeout, max_bytes,
                     depth, max_depth, visited):
    if depth > max_depth:
        raise ImportError_(
            f"xacro include depth exceeded {max_depth} (cycle? {sorted(visited)})"
        )
    # Walk a snapshot of children — we may mutate the tree as we go.
    for child in list(elem.childNodes):
        if _is_xacro_include(child):
            filename = child.getAttribute("filename")
            url = _resolve_include_target(filename, base_url)
            if url is None:
                raise ImportError_(
                    f"unresolvable <xacro:include filename={filename!r}>"
                )
            if url in visited:
                # Already inlined elsewhere — drop the duplicate include rather
                # than refetching. xacro semantics: include guards.
                elem.removeChild(child)
                continue
            visited.add(url)
            raw, final_url, encoding = _fetch_bytes(
                url, allowed_hosts=allowed_hosts,
                timeout=timeout, max_bytes=max_bytes,
            )
            sub_text = substitute_find(raw.decode(encoding or "utf-8"))
            sub_doc = xml.dom.minidom.parseString(sub_text)
            # Recurse into the included document first (so nested includes
            # are resolved before splicing).
            _splice_includes(sub_doc.documentElement, final_url,
                             allowed_hosts=allowed_hosts, timeout=timeout,
                             max_bytes=max_bytes, depth=depth + 1,
                             max_depth=max_depth, visited=visited)
            # Splice the included root's children before `child`, then remove
            # the placeholder include element.
            for sub_child in list(sub_doc.documentElement.childNodes):
                imported = elem.ownerDocument.importNode(sub_child, deep=True)
                elem.insertBefore(imported, child)
            elem.removeChild(child)
        elif child.nodeType == child.ELEMENT_NODE:
            _splice_includes(child, base_url,
                             allowed_hosts=allowed_hosts, timeout=timeout,
                             max_bytes=max_bytes, depth=depth,
                             max_depth=max_depth, visited=visited)


def _make_yaml_resolver(base_url, allowed_hosts, timeout, max_bytes):
    """Build a drop-in for `xacro.load_yaml` that recognises `package://` URIs
    and routes them through `_resolve_include_target` + `_fetch_bytes`. Falls
    back to the original loader for plain file paths.

    `xacro.load_yaml` is called from inside `${load_yaml(...)}` expressions in
    e.g. UR's xacros. Upstream prepends `os.path.dirname('.')` to non-absolute
    paths, which turns `package://...` into `./package://...` and crashes the
    expand. We intercept those at the source.
    """
    import yaml
    from xacro import YamlListWrapper, ConstructUnits, XacroException

    original = xacro.load_yaml

    def patched(filename):
        if not isinstance(filename, str) or not filename.startswith("package://"):
            return original(filename)
        url = _resolve_include_target(filename, base_url)
        if url is None:
            raise XacroException(f"can't resolve {filename!r} via package alias map")
        raw, _final, encoding = _fetch_bytes(
            url, allowed_hosts=allowed_hosts,
            timeout=timeout, max_bytes=max_bytes,
        )
        # Mirror the SafeLoader constructor setup the original load_yaml does
        # so xacro-tagged units like !radians round-trip correctly.
        for unit in ConstructUnits:
            yaml.SafeLoader.add_constructor(unit.value.tag, unit.constructor)
        return YamlListWrapper.wrap(yaml.safe_load(raw.decode(encoding or "utf-8")))

    return patched


def _swap_xacro_load_yaml(new_fn):
    """Install `new_fn` in xacro's expression-eval symbol table, returning the
    originals for restoration. We patch the cached `_global_symbols` table
    rather than the module attribute because xacro builds that table once at
    import time — by the time `process_doc` runs, swapping `xacro.load_yaml`
    is too late.

    Touches xacro internals (`_global_symbols`), so this may drift if xacro
    refactors its symbol bootstrap. Restored unconditionally in expand_xacro.
    """
    gs = xacro._global_symbols
    ns = gs.get("xacro")
    saved = (gs.get("load_yaml"), ns.get("load_yaml") if ns is not None else None)
    gs["load_yaml"] = new_fn
    if ns is not None:
        ns["load_yaml"] = new_fn
    return saved


def _restore_xacro_load_yaml(saved):
    gs = xacro._global_symbols
    ns = gs.get("xacro")
    top, in_ns = saved
    if top is not None:
        gs["load_yaml"] = top
    if ns is not None and in_ns is not None:
        ns["load_yaml"] = in_ns


def expand_xacro(
    text: str,
    *,
    mappings: Optional[dict[str, str]] = None,
    base_url: Optional[str] = None,
    allowed_hosts: Optional[list[str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = MAX_BYTES,
) -> str:
    """Run xacro property/macro/include expansion on the given XML source.

    When `base_url` is provided, `xacro.load_yaml` is patched to resolve
    `package://` URIs the same way `<xacro:include>` does (via the alias map
    and the HTTPS fetcher), so parameterised URDFs that load YAML configs
    over package:// can be expanded without a local ROS install.
    """
    doc = xml.dom.minidom.parseString(text)
    # xacro reads CLI-style mappings (e.g. "use_nominal_extrinsics:=true")
    # from sys.argv; passing them through process_doc keeps the API explicit.
    if base_url:
        patched = _make_yaml_resolver(
            base_url, allowed_hosts, timeout, max_bytes,
        )
        saved = _swap_xacro_load_yaml(patched)
        try:
            xacro.process_doc(doc, mappings=mappings or {})
        finally:
            _restore_xacro_load_yaml(saved)
    else:
        xacro.process_doc(doc, mappings=mappings or {})
    return doc.toxml()


def import_urdf(
    url: str,
    *,
    expand_macros: bool = True,
    allowed_hosts: Optional[list[str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    mappings: Optional[dict[str, str]] = None,
) -> tuple[Robot, str]:
    """One-shot: fetch → resolve includes → substitute $(find) → xacro → parse.

    `expand_macros=False` skips both include resolution AND xacro expansion —
    use it when the URL already points at a plain URDF.
    """
    body, final_url = fetch_urdf(url, allowed_hosts=allowed_hosts, timeout=timeout)
    if expand_macros:
        # Order matters: substitute $(find) into package:// *first*, so the
        # include resolver can map the resulting URI through infer_package_root.
        body = substitute_find(body)
        body = resolve_includes(body, final_url, allowed_hosts=allowed_hosts,
                                timeout=timeout)
        body = expand_xacro(body, mappings=mappings, base_url=final_url,
                            allowed_hosts=allowed_hosts, timeout=timeout)
    robot = from_xml(body)
    return robot, final_url


# --- mesh download ---------------------------------------------------------

# Reasonable per-file ceiling for binary meshes — STL/OBJ/DAE for a single
# robot link rarely exceed a few MB. Refuse anything larger to bound storage.
MAX_MESH_BYTES = 25 * 1024 * 1024  # 25 MB


def package_uri_parts(filename: str) -> Optional[tuple[str, str]]:
    """Parse `package://pkg/sub/path.stl` → ('pkg', 'sub/path.stl'). Returns
    None for non-package URIs."""
    if not filename.startswith("package://"):
        return None
    rest = filename[len("package://"):]
    pkg, _, sub = rest.partition("/")
    return pkg, sub


# ROS package names sometimes differ from the GitHub repo they live in
# (e.g. `ur_description` lives in `Universal_Robots_ROS2_Description`). When
# the URDF references `package://ur_description/...` we can't find the package
# name in the URL — fall back to this alias map. Extend per-deployment with:
#   FIELDPILOT_URDF_PACKAGE_ALIASES="pkg1=repo1,pkg2=repo2"
_BUILTIN_PACKAGE_ALIASES: dict[str, str] = {
    "ur_description": "Universal_Robots_ROS2_Description",
    "franka_description": "franka_ros",
}


def _load_package_aliases() -> dict[str, str]:
    """Built-in aliases merged with env-var overrides (re-read each call so a
    deployment can adjust without restarting the server)."""
    out = dict(_BUILTIN_PACKAGE_ALIASES)
    extra = (os.environ.get("FIELDPILOT_URDF_PACKAGE_ALIASES")
             or os.environ.get("MECHDIAG_PACKAGE_ALIASES", ""))
    for pair in extra.split(","):
        if "=" in pair:
            k, _, v = pair.partition("=")
            k, v = k.strip(), v.strip()
            if k and v:
                out[k] = v
    return out


def infer_package_root(urdf_url: str, package_name: str) -> Optional[str]:
    """Given the URL the URDF was fetched from, find the prefix that maps to
    `package://{package_name}/`. Returns None if the package name can't be
    located in the URL's path (even after alias lookup).

    Handles two common GitHub raw layouts:

      Multi-package repo (e.g. ros-industrial/universal_robot):
        /{owner}/{repo}/{branch}/{pkg}/urdf/x.urdf.xacro
                                ↑ package root here
        → /{owner}/{repo}/{branch}/{pkg}/

      Single-package repo (e.g. ros/urdf_tutorial):
        /{owner}/{pkg=repo}/{branch}/urdf/x.urdf.xacro
                            ↑ package root here (branch, not repo)
        → /{owner}/{pkg=repo}/{branch}/

    Cross-package alias (e.g. ur_description ↔ Universal_Robots_ROS2_Description):
    when the package name isn't in the URL, retry with the aliased name from
    `_load_package_aliases()`. Aliased lookups follow the same single-package
    fallback, so the returned root is the repo's branch dir.
    """
    parsed = urlparse(urdf_url)
    parts = parsed.path.split("/")
    indices = [i for i, p in enumerate(parts) if p == package_name]
    if not indices:
        alias = _load_package_aliases().get(package_name)
        if not alias:
            return None
        indices = [i for i, p in enumerate(parts) if p == alias]
        if not indices:
            return None
    # Deepest occurrence wins: covers the multi-package case where the
    # pkg name also happens to appear earlier (e.g. as the repo name).
    idx = indices[-1]
    # Single-package GitHub repo: the only occurrence is the repo name at
    # idx 2; the actual package root sits one component deeper (the branch).
    if (parsed.netloc == "raw.githubusercontent.com"
            and idx == 2 and len(indices) == 1 and len(parts) > 3):
        idx = 3
    prefix = "/".join(parts[: idx + 1]) + "/"
    return f"{parsed.scheme}://{parsed.netloc}{prefix}"


def _unique_mesh_filenames(robot: Robot) -> list[str]:
    seen: dict[str, None] = {}
    for link in robot.links:
        for shape in (*link.collisions, *link.visuals):
            if isinstance(shape.geometry, Mesh):
                seen.setdefault(shape.geometry.filename, None)
    return list(seen)


def fetch_meshes(
    robot: Robot,
    urdf_url: str,
    dest_dir: Path,
    *,
    allowed_hosts: Optional[list[str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = MAX_MESH_BYTES,
) -> dict:
    """Walk every <mesh filename="package://…"/> in the robot, attempt to
    download each one under dest_dir, preserving the {pkg}/{sub/path} layout
    so MeshResolver(mesh_dir=dest_dir) can resolve them later.

    Returns:
        {
          "downloaded": [{filename, bytes, package, path_on_disk}, …],
          "failures":   [{filename, reason}, …],
          "packages":   [pkg, …],  # unique package names that resolved
        }
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict] = []
    failures: list[dict] = []
    packages_seen: set[str] = set()

    # Resolve package → base URL once per package to amortise lookups.
    pkg_roots: dict[str, Optional[str]] = {}

    for filename in _unique_mesh_filenames(robot):
        parts = package_uri_parts(filename)
        if parts is None:
            failures.append({"filename": filename, "reason": "not a package:// URI"})
            continue
        pkg, sub = parts
        packages_seen.add(pkg)
        if pkg not in pkg_roots:
            pkg_roots[pkg] = infer_package_root(urdf_url, pkg)
        base = pkg_roots[pkg]
        if base is None:
            failures.append({
                "filename": filename,
                "reason": f"package {pkg!r} not found in URDF URL path",
            })
            continue
        mesh_url = base + sub

        out_path = dest_dir / pkg / sub
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw, _final, _enc = _fetch_bytes(
                mesh_url, allowed_hosts=allowed_hosts,
                timeout=timeout, max_bytes=max_bytes,
            )
        except (SchemeNotAllowed, HostNotAllowed) as e:
            failures.append({"filename": filename, "reason": f"refused: {e}"})
            continue
        except ImportError_ as e:
            failures.append({"filename": filename, "reason": str(e)})
            continue
        out_path.write_bytes(raw)
        downloaded.append({
            "filename": filename,
            "bytes": out_path.stat().st_size,
            "package": pkg,
            "path_on_disk": str(out_path),
        })

    return {
        "downloaded": downloaded,
        "failures": failures,
        "packages": sorted(packages_seen),
    }
