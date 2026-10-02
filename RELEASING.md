# Releasing ace-jax

The sdist and wheel are built, checked and published by
`.github/workflows/publish.yml`. (The coupling library, `ace-jax-coupling`, has
its own process: `coupling/RELEASING.md`.)

## One-time setup

1. **Trusted publishers.** On TestPyPI and on PyPI, add a (pending) trusted
   publisher for the project `ace-jax` (Account → Publishing):
   - owner `ACEsuit`, repository `ace-jax`;
   - workflow `publish.yml`;
   - environment `testpypi` on TestPyPI and `pypi` on PyPI.

   No API token is stored anywhere.
2. **GitHub environments** `testpypi` and `pypi` exist already (the coupling
   wheels use them). Optionally require a reviewer on `pypi`.

## Each release

1. **Version.** Set `version` in `pyproject.toml` and add its section to
   `CHANGELOG.md`. `ace_jax.__version__` and `aj --version` read the installed
   metadata, so there is nothing else to bump. If the coupling library changed,
   release it first and update the `ace-jax-coupling==` pin.
2. **Dry run on TestPyPI.** Actions → publish → Run workflow (on `main`). It
   builds, runs `twine check`, installs the wheel in a clean environment and runs
   `aj`, then uploads to TestPyPI. Check it in a fresh environment; the
   dependencies come from PyPI:

   ```bash
   uv venv /tmp/t && uv pip install --python /tmp/t/bin/python \
       --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
       --index-strategy unsafe-best-match ace-jax==<version>
   /tmp/t/bin/aj --version
   ```
3. **Release.** Tag the release commit `v<version>` and push the tag. The workflow
   refuses a tag that does not match `pyproject.toml`. PyPI versions cannot be
   re-uploaded, so a broken release needs a new version.
4. **After the release.** Create the GitHub release from the tag with the
   changelog section. The tutorial notebooks install `ace-jax` from PyPI, so a
   release with a new tutorial API must come before the docs that use it.
