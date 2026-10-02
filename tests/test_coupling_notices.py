"""coupling/python/THIRD_PARTY_NOTICES.md states the exact version of every bundled
component, which is what makes the LGPL corresponding source findable. Pin the
versions to what the build actually uses: the JLL builds and the EquivariantTensors
commit in coupling/julia/Manifest.toml, the Julia version, and the package version.
The runtime-only libraries (GMP, MPFR, PCRE2, libunwind, zstd) ship with Julia, so
the julia_version check covers them."""
import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTICES = (ROOT / "coupling/python/THIRD_PARTY_NOTICES.md").read_text()
MANIFEST = tomllib.loads((ROOT / "coupling/julia/Manifest.toml").read_text())
ROWS = {r.split(" | ")[0].lstrip("| "): r for r in NOTICES.splitlines() if r.startswith("| ")}
# notices row (prefix) -> the Manifest JLL that ships it
JLL = {"OpenBLAS": "OpenBLAS_jll", "libblastrampoline": "libblastrampoline_jll", "openlibm": "OpenLibm_jll",
       "zlib": "Zlib_jll", "SuiteSparse_config": "SuiteSparse_jll", "SuiteSparse KLU": "SuiteSparse_jll",
       "GCC runtime": "CompilerSupportLibraries_jll", "libquadmath": "CompilerSupportLibraries_jll",
       "GNU libiconv": "Libiconv_jll", "OpenSpecFun": "OpenSpecFun_jll"}


def _row(prefix):
    hits = [r for k, r in ROWS.items() if k.startswith(prefix)]
    assert len(hits) == 1, (prefix, list(ROWS))
    return hits[0]


def test_jll_versions_match_manifest():
    for prefix, pkg in JLL.items():
        v = MANIFEST["deps"][pkg][0]["version"]
        assert f"{pkg} {v}" in _row(prefix), (prefix, pkg, v)


def test_julia_et_and_package_versions():
    assert f"Julia {MANIFEST['julia_version']}" in _row("Julia runtime")
    et = MANIFEST["deps"]["EquivariantTensors"][0]
    row = _row("EquivariantTensors.jl")
    assert et["version"] in row and f"`{et['repo-rev'][:7]}`" in row, (et["version"], et["repo-rev"][:7])
    v = re.search(r'^version = "([^"]+)"', (ROOT / "coupling/python/pyproject.toml").read_text(), re.M)[1]
    assert f"ace-jax-coupling {v}" in _row("ace-jax glue")
