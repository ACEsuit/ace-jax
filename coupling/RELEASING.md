# Releasing ace-jax-coupling

The wheels are built, tested and published by `.github/workflows/coupling-wheels.yml`.
It publishes **wheels only**: Linux x86_64 and aarch64 (manylinux_2_28), macOS arm64 and Windows x64.
There is no sdist, because on an unsupported platform it would build a wheel with no
library. ace-jax's dependency marker already limits the install to these platforms.

## One-time setup

1. **Trusted publishers.** On PyPI and on TestPyPI, add a pending trusted publisher
   for the project `ace-jax-coupling` (Account → Publishing):
   - owner `ACEsuit`, repository `ace-jax`;
   - workflow `coupling-wheels.yml`;
   - environment `pypi` on PyPI and `testpypi` on TestPyPI.

   No API token is stored anywhere.
2. **GitHub environments.** In the repository settings, create the environments `pypi`
   and `testpypi`. Optionally require a reviewer on `pypi`.

## Each release

1. **Pick the version.** Bump `version` in `coupling/python/pyproject.toml`,
   `__version__` in `coupling/python/src/ace_jax_coupling/__init__.py` and
   `COUPLING_LIB_VERSION` in ace-jax, together with ace-jax's `ace-jax-coupling==`
   pin, whenever the coupling output can change (a new EquivariantTensors rev, a
   changed solver). The version is the cache's backend id, so a bump turns stale
   cache entries into misses.
2. **Moving Julia** (even a patch release) means changing together the
   Manifests' `julia_version` (re-resolve both), the Linux tarball `V=` and the
   macOS `setup-julia` version in `coupling-wheels.yml`. `build.jl` refuses to build
   when the running Julia differs from the Manifests.
3. **Update the notices.** In `THIRD_PARTY_NOTICES.md`, update the Version column
   for any changed Manifest or Julia pin. `tests/test_coupling_notices.py` fails
   until it matches.
4. **Tag the EquivariantTensors commit** pinned in `coupling/julia/Project.toml`, for
   example `ace-jax-coupling-v<version>` on the fork. `build_info()` records it, and
   the tag keeps it reachable if the branch is rebased.
5. **Dry run on TestPyPI.** Actions → coupling-wheels → Run workflow, with
   publish = `testpypi`. Then install it in a clean environment:
   `pip install -i https://test.pypi.org/simple/ ace-jax-coupling==<version>`, and
   check `ace_jax_coupling.build_info()`. To test the published wheels on all four
   platforms, including Windows, run Actions → coupling-index-check with
   index = `testpypi`. It downloads each platform's wheel from the index and runs
   the clean-environment test on it.
6. **Release.** Push the tag `coupling-v<version>`. The publish job runs only after
   all three wheels have passed their clean-environment tests and the ACEpotentials
   parity, and only if the tag equals the package version. PyPI versions cannot be
   re-uploaded, so a broken release needs a new version.
7. **Check the release.** Run coupling-index-check with index = `pypi` and the new
   version.
