# Releasing

`fieldpilot-urdf` publishes to PyPI via **Trusted Publishing (OIDC)** — there is
**no API token stored** anywhere. Publishing is triggered by pushing a `vX.Y.Z`
tag, which runs `.github/workflows/publish.yml`.

## One-time setup (PyPI side, done by a maintainer)

> **Already done.** `fieldpilot-urdf` is live on
> [PyPI](https://pypi.org/project/fieldpilot-urdf/) (0.1.0 onward) and the
> Trusted Publisher is wired up — this section is kept for reference only.

1. **Create the PyPI Trusted Publisher** (no project needs to exist yet — use a
   *pending* publisher). On <https://pypi.org> → *Your projects* →
   *Publishing* → *Add a pending publisher*:
   - PyPI Project Name: `fieldpilot-urdf`
   - Owner: `DuQuatre`
   - Repository name: `fieldpilot-urdf`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
2. **Create the `pypi` environment in GitHub**: repo *Settings → Environments →
   New environment → `pypi`* (optionally add required reviewers so a human must
   approve each publish).

> Recommended first: do the same against **TestPyPI** to dry-run, by adding a
> `repository-url: https://test.pypi.org/legacy/` step to the publish action and
> a matching TestPyPI pending publisher.

## Cutting a release

1. Make sure `main` is green (CI passes) and the working tree is clean.
2. **Set the version** in two places (they must match the tag):
   - `pyproject.toml` → `[project] version`
   - `src/fieldpilot_urdf/__init__.py` → `__version__`
   (the two must stay in sync — the publish workflow guards on it.)
3. Update **`CHANGELOG.md`**: move the entry from *unreleased* to dated, and add
   the release link.
4. Commit: `git commit -am "chore(release): X.Y.Z"`.
5. **Tag and push**:
   ```bash
   git tag -a vX.Y.Z -m "fieldpilot-urdf X.Y.Z"
   git push origin main --tags
   ```
   (First release only — already done — also needed
   `gh repo edit DuQuatre/fieldpilot-urdf --visibility public`.)
6. The `Publish to PyPI` workflow runs: it **guards that the tag matches the
   pyproject version**, builds the sdist+wheel, runs `twine check`, and publishes
   via OIDC. Approve the `pypi` environment if you set required reviewers.
7. Verify: `pip install fieldpilot-urdf` and create a GitHub Release from the tag
   (paste the CHANGELOG section).

## After release

- Bump the version to the next `.devN` on `main` so subsequent builds aren't
  confused with the released one.
- `pip index versions fieldpilot-urdf` to confirm it's live.
