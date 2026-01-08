# Release steps (PyPI)

These steps assume the package name on PyPI is `stratcp` and that a GitHub Actions
workflow publishes on a GitHub Release.

## One-time setup

1) Create a PyPI account and project token at:
   https://pypi.org/manage/account/token/
2) Add the token to GitHub repo secrets as `PYPI_API_TOKEN`.

## Releasing a new version

1) Bump the version in `pyproject.toml`.
2) Commit the version bump.
3) Create and push a version tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

4) Create a GitHub Release from the tag (or push the tag and then publish the release).
5) The `publish-pypi` workflow will build and upload the package to PyPI.

## Optional local dry run

Build the package locally:

```bash
uvx --from build pyproject-build --installer uv
```

If you want to upload manually (not recommended for routine releases):

```bash
uvx --from twine twine upload dist/*
```
