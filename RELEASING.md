# Releasing

`fieldpilot-urdf` publishes to PyPI via **Trusted Publishing (OIDC)** — there is
**no API token stored** anywhere. Publishing is triggered by pushing a `vX.Y.Z`
tag, which runs `.github/workflows/publish.yml`.

## One-time setup (PyPI side, done by a maintainer)

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
   (already set to `0.1.0` for the first release.)
3. Update **`CHANGELOG.md`**: move the entry from *unreleased* to dated, and add
   the release link.
4. Commit: `git commit -am "release: v0.1.0"`.
5. **Flip the repo public** (first release only):
   `gh repo edit DuQuatre/fieldpilot-urdf --visibility public`.
6. **Tag and push**:
   ```bash
   git tag -a v0.1.0 -m "fieldpilot-urdf 0.1.0"
   git push origin main --tags
   ```
7. The `Publish to PyPI` workflow runs: it **guards that the tag matches the
   pyproject version**, builds the sdist+wheel, runs `twine check`, and publishes
   via OIDC. Approve the `pypi` environment if you set required reviewers.
8. Verify: `pip install fieldpilot-urdf` and create a GitHub Release from the tag
   (paste the CHANGELOG section).

## After release

- Bump the version to the next `.devN` on `main` so subsequent builds aren't
  confused with the released one.
- `pip index versions fieldpilot-urdf` to confirm it's live.
