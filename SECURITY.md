# Security

## Reporting a vulnerability

Please report security issues privately via GitHub's **"Report a vulnerability"**
(Security → Advisories) on this repository, rather than opening a public issue.
We aim to acknowledge reports within a few working days.

## The fetch surface: `import_urdf` / `fetch_urdf`

`fieldpilot_urdf.importer` fetches a URDF (or `.urdf.xacro`) **from a
user-supplied URL**. Anything that fetches a caller-controlled URL is a
**Server-Side Request Forgery (SSRF)** surface: if you expose `import_urdf` to
untrusted input (e.g. behind a web endpoint), a caller could try to make your
server reach internal addresses or exfiltrate data.

The importer ships several defences, on by default:

| Defence | Behaviour | Where |
|---|---|---|
| **HTTPS only** | non-`https://` schemes (`http`, `file`, opaque) are rejected | `SchemeNotAllowed` |
| **Host allowlist** | only hosts on the allowlist are fetched | `HostNotAllowed` |
| **Size cap** | response capped at **5 MB** (`MAX_BYTES`) — streamed, not buffered whole | `ImportError_` |
| **Timeout** | **10 s** per request (`DEFAULT_TIMEOUT`) | `requests` timeout |
| **Redirect re-validation** | a redirect that lands on a non-allowlisted host is refused | `HostNotAllowed` |

### Default allowlist

```
github.com
raw.githubusercontent.com
gitlab.com
raw.gitlab.com
bitbucket.org
ros-industrial.org
```

These are the common public hosts for ROS robot descriptions. **Keep this list
conservative** — every host you add widens the SSRF surface.

### Configuring the allowlist

- **Env var** (applies process-wide):
  ```bash
  export FIELDPILOT_URDF_ALLOWED_HOSTS="my.internal-mirror.example,git.acme.dev"
  ```
  (The legacy `MECHDIAG_URDF_ALLOWED_HOSTS` name is still read as a deprecated
  fallback.)
- **Per call** (does not affect other callers):
  ```python
  from fieldpilot_urdf import import_urdf
  robot, _ = import_urdf(url, allowed_hosts=["git.acme.dev"])
  ```

Added hosts are **merged with** the defaults, not replacing them.

### Recommendations if you expose this to untrusted input

- Run the import in a network-isolated worker (egress firewall / no metadata
  endpoint access) — defence in depth beyond the allowlist.
- Keep `FIELDPILOT_URDF_ALLOWED_HOSTS` to the minimum set of hosts you trust.
- Treat the parsed model as untrusted data; this library does **not** execute
  anything from the URDF, but downstream consumers should validate before use.

## Mesh downloads

`import_urdf` does **not** download mesh assets by default. Mesh resolution is a
separate, explicit step (`fetch_meshes` / `collisions.MeshResolver`) and is
subject to the same HTTPS + allowlist rules.

## xacro

URDFs may be `.xacro` and are expanded **in-process** via the `xacro` Python
library — no subprocess or shell is spawned. The importer patches some `xacro`
internals; pin `xacro` (see CI) and review on upgrades.
