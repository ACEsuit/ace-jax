# PACE `.yace` import/export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load pacemaker `.yace` (PACE C-tilde) potentials into ace-jax as a `PACEModel` that evaluates E/F/virial in JAX (ASE calculator, lammps-jax), and write them back out as `.yace`.

**Architecture:** A YAML parser produces a `PACESpec` (the verbatim tree) and the numeric leaves. An offline numpy builder turns PACE's complex `ms_combs` into ace-jax's real product basis via a sparse map T. A sibling Equinox module `PACEModel` evaluates analytic PACE radials, pools A per (node, neighbour species), forms AA products, contracts against `ctilde_real = T · ctilde_complex`, and applies the embedding and inner-cutoff regime. Every formula is ported from ML-PACE C++ and checked against it: unit-level through a small C++ driver, model-level through python-ace.

**Tech Stack:** Python 3.11+, JAX, Equinox, NumPy, PyYAML (new core dependency), pytest via `uv run`. Reference side: C++17 + CMake against `~/gits/lammps-user-pace` (@ `99aa6e6`), and python-ace built from source in a separate venv.

**Spec:** `docs/pace-yace-spec.md` (branch `feat/pace-yace`). Read its sections "Representation" and "Radials, embedding, core terms" before any task. They cite the C++ source by file and line.

## Global Constraints

- Branch `feat/pace-yace` from `main`, worktree `.worktrees/pace-yace`. Do not touch `cli.py`, `fit/*`, `construct/*`, or `ACEModel`'s existing methods (open PRs #2–#5 edit those).
- Nothing in `src/` calls `jax.config.update`. Tests enable x64 at the top of the test module, following `tests/test_efv.py`.
- The ML-PACE C++ is the authority for every formula (`~/gits/lammps-user-pace/ML-PACE/ace-evaluator/`, commit `99aa6e6`). Do not rewrite formulas from papers or memory.
- The exported StableHLO must contain no `custom_call` (lammps-jax requirement). Pure-JAX ops only.
- Out of scope, and must raise a clear error: `ACE.jl*` radbasename; per-bond differing `radbasename`; per-bond differing `inner_cutoff_type`; `site_basis`/`site_descriptors`/`edge_jacobian` on a `PACEModel`.
- Run tests with `uv run pytest ...` from the worktree root. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Element order in the `.yace` differs from atomic-number order** (e.g. `elements: [Ge, Si]`). Energies must follow element *names*, not position. A fixture uses reversed order (Task 2) and parity covers it (Task 6).
2. **Bonds with different `rcut`.** The neighbour list uses the largest cutoff, and neighbours beyond a bond's own `rcut` must contribute nothing: no g, no core term, not considered for the ZBL nearest neighbour. Task 6 has a dedicated test.
3. **Padded and neighbourless nodes** (lammps-jax buffers, isolated atoms) must give finite energy and zero, not NaN, forces. Task 6 tests both.
4. **f32 evaluation** (`dtype=jnp.float32`) must stay finite and within 1e-4 relative of f64. Task 6 tests this.
5. **Unsupported files** must fail at load with a message naming the unsupported feature, not deep inside `jit`. Task 3 tests each case.

---

## File structure

| path | responsibility |
|---|---|
| `pace_ref/unit_ref.cpp`, `pace_ref/CMakeLists.txt` | C++ driver: dumps g_k, c_r, ZBL, `cutoff_func_poly`, `Fexp`, `FexpShiftedScaled` and complex Y_lm from the ML-PACE sources as JSON |
| `pace_ref/json_to_npz.py` | converts the driver's JSON to `fixtures/pace/unit_ref.npz` |
| `pace_ref/make_fixtures.py` | (python-ace venv) builds random-coefficient `.yace` fixtures plus E/F/stress references |
| `pace_ref/README.md` | how to build both reference environments; pinned versions |
| `src/ace_jax/eval/pace_io.py` | YAML load/dump with tuple keys; `PACESpec`; `parse_yace` (validation plus numeric arrays); `write_yace` |
| `src/ace_jax/eval/pace_radial.py` | pure-JAX ports: radial families, `cutoff_func_poly`, `radcore`, `pace_zbl`, `fexp`, `fexp_shifted_scaled` |
| `src/ace_jax/eval/pace_build.py` | numpy: Y_lm convention map U_l; A/AA index tables; sparse T |
| `src/ace_jax/eval/pace_model.py` | `PACEModel` (Equinox) and `load_yace` |
| `src/ace_jax/eval/io.py` | dispatch `.yace` in `load` (2 lines) |
| `src/ace_jax/eval/__init__.py` | export `PACEModel`, `load_yace`, `write_yace` |
| `tests/test_pace_*.py` | one test module per source module, plus parity and export |
| `.github/workflows/pace-parity.yml` | path-gated reference regeneration and parity run |

---

### Task 1: C++ unit-reference driver

**Files:**
- Create: `pace_ref/CMakeLists.txt`, `pace_ref/unit_ref.cpp`, `pace_ref/json_to_npz.py`, `pace_ref/README.md`
- Create (generated, committed): `fixtures/pace/unit_ref.npz`
- Modify: `.gitignore` (add `pace_ref/build/`, `pace_ref/.venv/`)

**Interfaces:**
- Produces `fixtures/pace/unit_ref.npz` with the keys below. Tasks 4 and 5 test against it.
  - `r` (600,): the radius grid
  - `g_{name}_{inner}` (600, 8) for name ∈ {ChebExpCos, ChebPow, ChebLinear, SBessel} and inner ∈ {density, distance, zbl}, using λ=2.5, rcut=5.0, dcut=0.3, cut_in=1.2, dcut_in=0.4
  - `g_ChebExpCos_zbl_dcutin0` (600, 8): the same, but with dcut_in=0 (the `cut_in = (dcut_in == 0)` quirk)
  - `cr_density`, `cr_distance` (600,): `radcore(r, pre=3.0, lambda=0.7, cutoff=5.0, r_in=1.2, delta_in=0.4)`
  - `cr_zbl` (600,): `ZBL(r, Zi=14, Zj=32, cut_out=5.0, cut_in=4.7, prefactor=1.0)`
  - `fcpoly` (600,): `cutoff_func_poly(r, 3.0, 1.0)`
  - `fx` (N,), `fexp_m{0.5,1,2}`, `fexpss_m{0.5,1,2}` (N,)
  - `ydir` (50, 3) unit vectors; `ylm_re`, `ylm_im` (50, 49) for l ≤ 6, all m = −l..l, index l*l+l+m, with negative m filled as (−1)^m·conj(Y_{l,|m|}), the same way the evaluator fills A

- [ ] **Step 1: Write the CMake project**

`pace_ref/CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.16)
project(pace_unit_ref CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PACE_SRC "$ENV{HOME}/gits/lammps-user-pace" CACHE PATH "ICAMS/lammps-user-pace checkout")
add_subdirectory(${PACE_SRC} libpace EXCLUDE_FROM_ALL)
add_executable(unit_ref unit_ref.cpp)
target_include_directories(unit_ref PRIVATE ${PACE_SRC}/ML-PACE ${PACE_SRC}/yaml-cpp/include)
target_link_libraries(unit_ref PRIVATE pace)
```

- [ ] **Step 2: Write the driver**

`pace_ref/unit_ref.cpp`:
```cpp
// Dumps ML-PACE's own radial / core / embedding / Ylm values as JSON, so the
// JAX ports in src/ace_jax/eval/pace_radial.py are checked against the C++,
// not against a re-derivation.  Regenerate: see pace_ref/README.md.
#include "ace-evaluator/ace_radial.h"
#include "ace-evaluator/ace_abstract_basis.h"
#include "ace-evaluator/ace_spherical_cart.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
using namespace std;

static bool first = true;
static void emit(const string &key, const vector<double> &v, int ncol) {
    printf("%s\"%s\": {\"ncol\": %d, \"data\": [", first ? "" : ",\n", key.c_str(), ncol);
    first = false;
    for (size_t i = 0; i < v.size(); i++) printf("%s%.17g", i ? "," : "", v[i]);
    printf("]}");
}

int main() {
    const int K = 8;
    vector<double> rs;
    for (int i = 0; i < 600; i++) rs.push_back(0.005 + 0.01 * i);
    printf("{\n");
    emit("r", rs, 1);

    for (string name : {"ChebExpCos", "ChebPow", "ChebLinear", "SBessel"}) {
        for (string inner : {"density", "distance", "zbl"}) {
            ACERadialFunctions rf(K, 0, 1, 0.001, 1, {{name}});
            rf.inner_cutoff_type = inner;
            vector<double> out;
            for (double r : rs) {
                rf.radbase(2.5, 5.0, 0.3, name, r, 1.2, 0.4);
                for (int k = 0; k < K; k++) out.push_back(rf.gr(k));
            }
            emit("g_" + name + "_" + inner, out, K);
        }
    }
    {   // zbl quirk: cut_in = (dcut_in == 0)
        ACERadialFunctions rf(K, 0, 1, 0.001, 1, {{"ChebExpCos"}});
        rf.inner_cutoff_type = "zbl";
        vector<double> out;
        for (double r : rs) {
            rf.radbase(2.5, 5.0, 0.3, "ChebExpCos", r, 1.2, 0.0);
            for (int k = 0; k < K; k++) out.push_back(rf.gr(k));
        }
        emit("g_ChebExpCos_zbl_dcutin0", out, K);
    }
    for (string inner : {"density", "distance"}) {
        ACERadialFunctions rf(K, 0, 1, 0.001, 1, {{"ChebExpCos"}});
        rf.inner_cutoff_type = inner;
        vector<double> out;
        for (double r : rs) {
            double cr, dcr;
            rf.radcore(r, 3.0, 0.7, 5.0, cr, dcr, 1.2, 0.4);
            out.push_back(cr);
        }
        emit("cr_" + inner, out, 1);
    }
    {
        vector<double> out;
        for (double r : rs) {
            double cr, dcr;
            ACERadialFunctions::ZBL(r, 14, 32, cr, dcr, 5.0, 4.7, 1.0);
            out.push_back(cr);
        }
        emit("cr_zbl", out, 1);
    }
    {
        vector<double> out;
        for (double r : rs) {
            double fc, dfc;
            cutoff_func_poly(r, 3.0, 1.0, fc, dfc);
            out.push_back(fc);
        }
        emit("fcpoly", out, 1);
    }
    {
        vector<double> xs = {0.0, 1e-12, -1e-12, 1e-9, 1e-7, -1e-7, 2e-6, 3e-6, 4e-6, -5e-6,
                             1e-5, 1e-3, -1e-3, 0.1, -0.1, 0.5, 1.0, -2.0, 7.3, -40.0};
        emit("fx", xs, 1);
        for (double m : {0.5, 1.0, 2.0}) {
            vector<double> a, b;
            for (double x : xs) {
                double F, DF;
                Fexp(x, m, F, DF); a.push_back(F);
                FexpShiftedScaled(x, m, F, DF); b.push_back(F);
            }
            char buf[32];
            snprintf(buf, sizeof buf, "%g", m);
            emit(string("fexp_m") + buf, a, 1);
            emit(string("fexpss_m") + buf, b, 1);
        }
    }
    {
        const int L = 6;
        ACECartesianSphericalHarmonics sh(L);
        mt19937 gen(7);
        normal_distribution<double> nd;
        vector<double> dirs, re, im;
        for (int s = 0; s < 50; s++) {
            double x = nd(gen), y = nd(gen), z = nd(gen), n = sqrt(x * x + y * y + z * z);
            x /= n; y /= n; z /= n;
            dirs.insert(dirs.end(), {x, y, z});
            sh.compute_ylm(x, y, z, L);
            for (int l = 0; l <= L; l++)
                for (int m = -l; m <= l; m++) {
                    int am = abs(m);
                    double sgn = (am % 2 == 0) ? 1.0 : -1.0;
                    double yr = sh.ylm(l, am).real, yi = sh.ylm(l, am).img;
                    if (m < 0) { yr = sgn * yr; yi = -sgn * yi; }
                    re.push_back(yr);
                    im.push_back(yi);
                }
        }
        emit("ydir", dirs, 3);
        emit("ylm_re", re, (L + 1) * (L + 1));
        emit("ylm_im", im, (L + 1) * (L + 1));
    }
    printf("\n}\n");
    return 0;
}
```

- [ ] **Step 3: Write the JSON→npz converter**

`pace_ref/json_to_npz.py`:
```python
"""unit_ref JSON (stdin) -> fixtures/pace/unit_ref.npz.  Records the PACE commit."""
import json, pathlib, subprocess, sys
import numpy as np

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "~/gits/lammps-user-pace").expanduser()
d = json.load(sys.stdin)
out = {k: np.asarray(v["data"], float).reshape(-1, v["ncol"]).squeeze() if v["ncol"] == 1
       else np.asarray(v["data"], float).reshape(-1, v["ncol"]) for k, v in d.items()}
out["pace_commit"] = np.asarray(subprocess.check_output(
    ["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip())
dst = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "pace" / "unit_ref.npz"
dst.parent.mkdir(parents=True, exist_ok=True)
np.savez(dst, **out)
print(f"wrote {dst} ({len(out)} arrays)")
```

- [ ] **Step 4: Build, run, generate**

```bash
cmake -S pace_ref -B pace_ref/build -DCMAKE_BUILD_TYPE=Release
cmake --build pace_ref/build --target unit_ref -j
pace_ref/build/unit_ref | uv run python pace_ref/json_to_npz.py
```
Expected: `wrote .../fixtures/pace/unit_ref.npz (NN arrays)`.

If it doesn't compile because the pace target exposes no include path or needs a different yaml namespace, fix `pace_ref/CMakeLists.txt` only: never edit the PACE sources.

- [ ] **Step 5: Sanity-check three known values**

```bash
uv run python -c "
import numpy as np; z=np.load('fixtures/pace/unit_ref.npz')
d=z['ydir'][0]; print('Y00', z['ylm_re'][0,0], '== 1')
print('Y10', z['ylm_re'][0,2], '==', 3**.5*d[2])
print('Y11re', z['ylm_re'][0,3], '==', -(1.5)**.5*d[0])"
```
Expected: each pair is equal to 1e-15 (the spec's convention: Y00=1, Y10=√3·ẑ, Y11=−√(3/2)(x̂+iŷ)).

- [ ] **Step 6: README and commit**

`pace_ref/README.md` records: the PACE checkout path and commit (`99aa6e6`); the Step 4 commands; the python-ace venv from Task 2; and the fact that both generated fixture sets are committed, so the everyday suite needs neither toolchain.

```bash
printf 'pace_ref/build/\npace_ref/.venv/\n' >> .gitignore
git add pace_ref .gitignore fixtures/pace/unit_ref.npz
git commit -m "test(pace): C++ unit-reference driver against ML-PACE sources

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: python-ace model fixtures

**Files:**
- Create: `pace_ref/make_fixtures.py`
- Create (generated, committed): `fixtures/pace/{name}.yace`, `fixtures/pace/{name}_ref.npz`
- Modify: `pace_ref/README.md` (venv recipe, python-ace commit)

**Interfaces:**
- Produces, for each name in the table below: `fixtures/pace/{name}.yace` (shipped `deltaSplineBins` 0.001), and `fixtures/pace/{name}_ref.npz` with these keys:
  - `struct_names` (S,)
  - per structure s: `Z_s`, `pos_s` (n,3), `cell_s` (3,3), `pbc_s` (3,)
  - `E_tight_s`, `F_tight_s` (n,3), `S_tight_s` (6,) (evaluated with `deltaSplineBins` 1e-4)
  - `E_ship_s`, `F_ship_s`, `S_ship_s` (evaluated with the shipped file)
  - `pyace_version`

| name | elements (file order) | radbase | ndensity / npot | inner | purpose |
|---|---|---|---|---|---|
| `si_chebexpcos` | Si | ChebExpCos | 1, FSShiftedScaled m=1 | density | simplest |
| `si_chebpow_fs` | Si | ChebPow | 2, FinnisSinclair [1,1,1,0.5] | density | Fexp, 2 densities |
| `si_cheblinear` | Si | ChebLinear | 1, FSShiftedScaled | density | 4th family |
| `gesi_sbessel` | **Ge, Si** | SBessel | 2, FSShiftedScaled [1,1,1,0.5] | density | reversed element order; per-bond `nradmax`/`rcut` differ |
| `sige_density_core` | Si, Ge | ChebExpCos | 1 | density, core-repulsion [100, 5], rho_core_cut 50 | regime 1 |
| `sige_distance` | Si, Ge | ChebExpCos | 1 | distance, r_in 1.6, delta_in 0.4 | regime 2 |
| `sige_zbl` | Si, Ge | ChebExpCos | 1 | zbl, r_in 1.6, delta_in 0.4 | regime 3 |

Structures (all fixtures): `bulk` (diamond Si 2×2×2 conventional, rattled 0.05 Å, seed 0; random species for 2-element fixtures, seed 1); `dimer_2.3` (pbc False, 20 Å box); `isolated` (1 atom, pbc False); `close` (bulk with atom 0 moved to 1.1 Å from atom 1).

- [ ] **Step 1: Build the python-ace venv**

```bash
uv venv pace_ref/.venv --python 3.11
uv pip install --python pace_ref/.venv/bin/python numpy ase pyyaml "git+https://github.com/ICAMS/python-ace"
pace_ref/.venv/bin/python -c "import pyace, inspect; print(pyace.__version__ if hasattr(pyace,'__version__') else 'ok'); from pyace import ACEBBasisSet, create_multispecies_basis_config, PyACECalculator"
```
Expected: the imports succeed. If the build fails, stop and report the error (CMake/compiler version). Don't work around it in ace-jax.

- [ ] **Step 2: Confirm the pyace API names this script uses**

```bash
pace_ref/.venv/bin/python -c "
from pyace import ACEBBasisSet; b=[m for m in dir(ACEBBasisSet) if 'coeff' in m.lower() or 'ctilde' in m.lower() or 'save' in m.lower()]; print(b)"
```
Expected to include `all_coeffs`/`set_all_coeffs` and `to_ACECTildeBasisSet`. If they're named differently, adapt only the three marked lines in Step 3.

- [ ] **Step 3: Write the generator**

`pace_ref/make_fixtures.py`:
```python
"""Random-coefficient .yace fixtures + python-ace E/F/stress references.

Run in the python-ace venv (pace_ref/.venv).  Coefficients are random: parity
between two evaluators does not need a physical potential, and random values
exercise every term.  Writes fixtures/pace/{name}.yace and {name}_ref.npz.
"""
import copy, pathlib, tempfile
import numpy as np, yaml
from ase import Atoms
from ase.build import bulk
from pyace import ACEBBasisSet, PyACECalculator, create_multispecies_basis_config

OUT = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "pace"


def emb(nd, npot, fs, rc=1e5, drc=250.0):
    return {"npot": npot, "fs_parameters": fs, "ndensity": nd,
            "rho_core_cut": rc, "drho_core_cut": drc}


def bond(radbase, **kw):
    b = {"radbase": radbase, "radparameters": [5.25], "rcut": 5.0, "dcut": 0.01,
         "NameOfCutoffFunction": "cos"}
    b.update(kw)
    return b


FUNCS1 = {"number_of_functions_per_element": 40,
          "ALL": {"nradmax_by_orders": [8, 4, 3, 2], "lmax_by_orders": [0, 3, 2, 1]}}
FUNCS2 = {"number_of_functions_per_element": 60,
          "ALL": {"nradmax_by_orders": [8, 4, 3], "lmax_by_orders": [0, 3, 2]}}

CASES = {
    "si_chebexpcos": (["Si"], emb(1, "FinnisSinclairShiftedScaled", [1, 1]), {"ALL": bond("ChebExpCos")}, FUNCS1),
    "si_chebpow_fs": (["Si"], emb(2, "FinnisSinclair", [1, 1, 1, 0.5]), {"ALL": bond("ChebPow", radparameters=[2.0])}, FUNCS1),
    "si_cheblinear": (["Si"], emb(1, "FinnisSinclairShiftedScaled", [1, 1]), {"ALL": bond("ChebLinear")}, FUNCS1),
    "gesi_sbessel": (["Ge", "Si"], emb(2, "FinnisSinclairShiftedScaled", [1, 1, 1, 0.5]),
                     {"ALL": bond("SBessel"), ("Ge", "Ge"): bond("SBessel", rcut=4.2)},
                     {"number_of_functions_per_element": 60,
                      "ALL": {"nradmax_by_orders": [8, 4, 3], "lmax_by_orders": [0, 3, 2]},
                      ("Ge", "Ge"): {"nradmax_by_orders": [6, 2, 2], "lmax_by_orders": [0, 2, 1]}}),
    "sige_density_core": (["Si", "Ge"], emb(1, "FinnisSinclairShiftedScaled", [1, 1], rc=50.0, drc=20.0),
                          {"ALL": bond("ChebExpCos", **{"core-repulsion": [100.0, 5.0]})}, FUNCS2),
    "sige_distance": (["Si", "Ge"], emb(1, "FinnisSinclairShiftedScaled", [1, 1]),
                      {"ALL": bond("ChebExpCos", inner_cutoff_type="distance", r_in=1.6, delta_in=0.4,
                                   **{"core-repulsion": [100.0, 5.0]})}, FUNCS2),
    "sige_zbl": (["Si", "Ge"], emb(1, "FinnisSinclairShiftedScaled", [1, 1]),
                 {"ALL": bond("ChebExpCos", inner_cutoff_type="zbl", r_in=1.6, delta_in=0.4)}, FUNCS2),
}


def structures(elements):
    rng = np.random.default_rng(0)
    b = bulk("Si", "diamond", a=5.43, cubic=True).repeat(2)
    b.rattle(0.05, seed=0)
    if len(elements) > 1:
        b.numbers = np.random.default_rng(1).choice(
            [Atoms(e).numbers[0] for e in elements], len(b))
    else:
        b.numbers[:] = Atoms(elements[0]).numbers[0]
    z0 = b.numbers[0]
    dimer = Atoms(numbers=[z0, b.numbers[-1]], positions=[[0, 0, 0], [2.3, 0, 0]],
                  cell=np.eye(3) * 20, pbc=False)
    iso = Atoms(numbers=[z0], positions=[[0, 0, 0]], cell=np.eye(3) * 20, pbc=False)
    close = b.copy()
    d = close.get_distance(0, 1, mic=True, vector=True)
    close.positions[0] = close.positions[1] - 1.1 * d / np.linalg.norm(d)
    return {"bulk": b, "dimer_2.3": dimer, "isolated": iso, "close": close}


def evaluate(path, at):
    at = at.copy()
    at.calc = PyACECalculator(str(path))
    E = at.get_potential_energy()
    F = at.get_forces()
    S = at.get_stress() if at.pbc.all() else np.zeros(6)
    return E, F, S


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (els, embd, bonds, funcs) in CASES.items():
        cfg = {"deltaSplineBins": 0.001, "elements": els, "embeddings": {"ALL": embd},
               "bonds": bonds, "functions": funcs}
        bb = ACEBBasisSet(create_multispecies_basis_config(cfg))
        rng = np.random.default_rng(abs(hash(name)) % 2**32)
        c = np.asarray(bb.all_coeffs)                          # API line 1
        bb.set_all_coeffs(rng.normal(scale=0.3, size=c.shape))  # API line 2
        cb = bb.to_ACECTildeBasisSet()                          # API line 3
        yace = OUT / f"{name}.yace"
        cb.save_yaml(str(yace))
        tight = pathlib.Path(tempfile.mkdtemp()) / f"{name}.yace"
        tight.write_text(yace.read_text().replace("deltaSplineBins: 0.001", "deltaSplineBins: 0.0001"))
        assert "deltaSplineBins: 0.0001" in tight.read_text(), "deltaSplineBins key not found"
        ref = {"struct_names": np.asarray(list(structures(els)))}
        import pyace
        ref["pyace_version"] = np.asarray(getattr(pyace, "__version__", "unknown"))
        for s, at in structures(els).items():
            ref[f"Z_{s}"], ref[f"pos_{s}"] = at.numbers, at.positions
            ref[f"cell_{s}"], ref[f"pbc_{s}"] = at.cell.array, at.pbc
            for tag, p in (("tight", tight), ("ship", yace)):
                E, F, S = evaluate(p, at)
                ref[f"E_{tag}_{s}"], ref[f"F_{tag}_{s}"], ref[f"S_{tag}_{s}"] = E, F, S
        np.savez(OUT / f"{name}_ref.npz", **ref)
        print(f"{name}: {yace.stat().st_size // 1024} kB")


if __name__ == "__main__":
    main()
```

The tight copy is edited textually so it differs from the shipped file *only* in `deltaSplineBins`.

- [ ] **Step 4: Generate and inspect**

```bash
pace_ref/.venv/bin/python pace_ref/make_fixtures.py
grep -m3 -n "inner_cutoff_type\|radbasename" fixtures/pace/sige_zbl.yace fixtures/pace/si_cheblinear.yace
head -3 fixtures/pace/gesi_sbessel.yace
```
Expected: 7 fixtures written; `sige_zbl` shows `inner_cutoff_type: zbl`; `si_cheblinear` shows `radbasename: ChebLinear`; `gesi_sbessel` starts with `elements: [Ge, Si]`.

If pyace's config rejects `ChebLinear`, generate that fixture with `ChebPow` and replace `radbasename: ChebPow` with `radbasename: ChebLinear` in the file text *before* evaluating. The C++ selects the family purely by name. Record that in the README.

Check that each fixture has fewer than 1 MB of YAML, so it's reasonable to commit.

- [ ] **Step 5: Commit**

```bash
git add pace_ref/make_fixtures.py pace_ref/README.md fixtures/pace/*.yace fixtures/pace/*_ref.npz
git commit -m "test(pace): python-ace model fixtures (4 radial families, 3 inner regimes)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `.yace` parsing and validation

**Files:**
- Create: `src/ace_jax/eval/pace_io.py`, `tests/test_pace_io.py`
- Modify: `pyproject.toml` (add `"pyyaml>=6"` to `dependencies`), `uv.lock` (via `uv lock`)

**Interfaces:**
- Produces:
  - `load_tree(path) -> dict`: PyYAML, flow-sequence keys become tuples (`bonds[(0, 1)]`)
  - `dump_tree(tree, path) -> None`
  - `@dataclass PACESpec(tree: dict, element_names: list[str], functions: list[tuple[int, int, int, int, int]])`, where each tuple is `(element, func_index, term_start, num_ms_combs, ndensity)` in term order
  - `parse_yace(path) -> tuple[PACESpec, dict]`. The dict holds numpy arrays and scalars: `E0` (NZ,); `crad` (NZ,NZ,nradmax,lmax+1,nradbase); `radparams` (NZ,NZ,5) [λ, rcut, dcut, rcut_in, dcut_in]; `core` (NZ,NZ,2) [prehc, lambdahc]; `fs_params` (NZ, 2P); `rho_core_cut` (NZ,2); `ctilde_complex` (n_terms, P); `Z` (NZ,) atomic numbers; plus static entries `radbasename`, `inner_cutoff_type`, `npoti` (tuple per element), `ndensity` P (max), `nradmax`, `lmax`, `nradbase`, and `funcs`, a list of dicts with `el, rank, mus, ns, ls, ms (num_ms, rank) array`, in term order.
  - Missing bond fields default to the C++ `init` defaults: `rcut_in` 0, `dcut_in` 1e-5, `prehc` 0, `lambdahc` 1, `inner_cutoff_type` "density".

- [ ] **Step 1: Write the failing tests**

`tests/test_pace_io.py`:
```python
import pathlib
import numpy as np
import pytest
from ace_jax.eval.pace_io import load_tree, parse_yace

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"

MINI = """\
elements: [Si]
E0: [-1.5]
deltaSplineBins: 0.001
embeddings:
  0: {ndensity: 1, FS_parameters: [1, 1], npoti: FinnisSinclairShiftedScaled, rho_core_cutoff: 100000, drho_core_cutoff: 250}
bonds:
  [0, 0]: {radbasename: ChebExpCos, radparameters: [5.25], radcoefficients: [[[1, 0.5]]], prehc: 0, lambdahc: 1, rcut: 5, dcut: 0.01, rcut_in: 0, dcut_in: 0, inner_cutoff_type: density, nradmax: 1, lmax: 0, nradbasemax: 2}
functions:
  0:
    - {mu0: 0, rank: 1, ndensity: 1, num_ms_combs: 1, mus: [0], ns: [1], ls: [0], ms_combs: [0], ctildes: [0.7]}
    - {mu0: 0, rank: 2, ndensity: 1, num_ms_combs: 3, mus: [0, 0], ns: [1, 1], ls: [1, 1], ms_combs: [-1, 1, 0, 0, 1, -1], ctildes: [0.1, -0.2, 0.1]}
"""


@pytest.fixture
def mini(tmp_path):
    p = tmp_path / "mini.yace"
    p.write_text(MINI)
    return p


def test_tuple_bond_keys(mini):
    t = load_tree(mini)
    assert (0, 0) in t["bonds"] and t["embeddings"][0]["ndensity"] == 1


def test_parse_mini(mini):
    spec, a = parse_yace(mini)
    assert spec.element_names == ["Si"] and a["Z"].tolist() == [14]
    assert a["crad"].shape == (1, 1, 1, 1, 2)
    np.testing.assert_array_equal(a["crad"][0, 0, 0, 0], [1, 0.5])
    np.testing.assert_array_equal(a["radparams"][0, 0], [5.25, 5, 0.01, 0, 0])
    assert a["ctilde_complex"].shape == (4, 1)          # 1 + 3 terms
    assert [f[2:] for f in spec.functions] == [(0, 1, 1), (1, 3, 1)]
    np.testing.assert_array_equal(a["funcs"][1]["ms"], [[-1, 1], [0, 0], [1, -1]])


@pytest.mark.parametrize("edit, msg", [
    (("ChebExpCos", "ACE.jl.base"), "ACE.jl"),
    (("inner_cutoff_type: density", "inner_cutoff_type: zbl"), None),   # single bond: fine
])
def test_radbase_checks(tmp_path, edit, msg):
    p = tmp_path / "x.yace"
    p.write_text(MINI.replace(*edit))
    if msg is None:
        parse_yace(p)
    else:
        with pytest.raises(NotImplementedError, match=msg):
            parse_yace(p)


def test_mixed_bond_types_rejected(tmp_path, mini):
    t = load_tree(mini)
    from ace_jax.eval.pace_io import dump_tree
    t["elements"] = ["Si", "Ge"]; t["E0"] = [-1.5, -2.0]
    t["embeddings"][1] = dict(t["embeddings"][0])
    b = t["bonds"][(0, 0)]
    for k in [(0, 1), (1, 0), (1, 1)]:
        t["bonds"][k] = dict(b)
    t["functions"][1] = []
    t["bonds"][(1, 1)]["radbasename"] = "SBessel"
    p = tmp_path / "a.yace"; dump_tree(t, p)
    with pytest.raises(NotImplementedError, match="radbasename"):
        parse_yace(p)
    t["bonds"][(1, 1)]["radbasename"] = "ChebExpCos"
    t["bonds"][(1, 1)]["inner_cutoff_type"] = "distance"
    dump_tree(t, p)
    with pytest.raises(ValueError, match="inner_cutoff_type"):
        parse_yace(p)


@pytest.mark.parametrize("name", ["si_chebexpcos", "gesi_sbessel", "sige_zbl"])
def test_parse_fixtures(name):
    p = FIX / f"{name}.yace"
    if not p.exists():
        pytest.skip("fixture not generated")
    spec, a = parse_yace(p)
    NZ = len(spec.element_names)
    assert a["crad"].shape[:2] == (NZ, NZ)
    assert a["ctilde_complex"].shape[0] == spec.functions[-1][2] + spec.functions[-1][3]
    if name == "gesi_sbessel":
        assert a["Z"].tolist() == [32, 14]
```

- [ ] **Step 2: Run and confirm the failure**

Run: `uv run pytest tests/test_pace_io.py -q`
Expected: `ModuleNotFoundError: No module named 'ace_jax.eval.pace_io'`.

- [ ] **Step 3: Add the dependency**

In `pyproject.toml` `dependencies`, add `"pyyaml>=6",          # .yace (PACE) model files`. Then run `uv lock`.

- [ ] **Step 4: Implement**

`src/ace_jax/eval/pace_io.py`:
```python
"""PACE `.yace` (ACECTildeBasisSet YAML) reading and writing.

Keys and defaults follow ML-PACE `ace_c_basis.cpp::load_yaml` and
`ACERadialFunctions::init` (lammps-user-pace @ 99aa6e6).  Bond keys are flow
sequences (`[0, 1]:`), which PyYAML cannot hash, so they load as tuples.
"""
import copy
from dataclasses import dataclass, field

import numpy as np
import yaml
from ase.data import atomic_numbers

try:
    _BaseLoader, _BaseDumper = yaml.CSafeLoader, yaml.CSafeDumper
except AttributeError:                       # PyYAML without libyaml
    _BaseLoader, _BaseDumper = yaml.SafeLoader, yaml.SafeDumper


class _Loader(_BaseLoader):
    pass


def _mapping(loader, node):
    loader.flatten_mapping(node)
    out = {}
    for kn, vn in node.value:
        k = loader.construct_object(kn, deep=True)
        out[tuple(k) if isinstance(k, list) else k] = loader.construct_object(vn, deep=True)
    return out


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


class _Dumper(_BaseDumper):
    pass


_Dumper.add_representer(tuple, lambda d, t: d.represent_sequence(
    "tag:yaml.org,2002:seq", list(t), flow_style=True))


def load_tree(path):
    with open(path) as f:
        return yaml.load(f, Loader=_Loader)


def dump_tree(tree, path):
    with open(path, "w") as f:
        yaml.dump(tree, f, Dumper=_Dumper, default_flow_style=None,
                  sort_keys=False, width=1 << 30)


@dataclass
class PACESpec:
    """The parsed file, verbatim.  Kept beside the model (not on it: a dict is
    not a hashable Equinox static field) so export never regenerates layout."""
    tree: dict
    element_names: list
    functions: list = field(default_factory=list)   # (el, idx, term_start, num_ms, ndens)


_BOND_DEFAULTS = {"rcut_in": 0.0, "dcut_in": 1e-5, "prehc": 0.0, "lambdahc": 1.0,
                  "inner_cutoff_type": "density"}


def parse_yace(path):
    tree = load_tree(path)
    names = list(tree["elements"])
    NZ = len(names)
    try:
        Z = np.array([atomic_numbers[e] for e in names], np.int64)
    except KeyError as e:
        raise ValueError(f"unknown element {e.args[0]!r} in {path}") from None

    bonds = {k: {**_BOND_DEFAULTS, **v} for k, v in tree["bonds"].items()}
    kinds = {b["radbasename"] for b in bonds.values()}
    if any(k.startswith("ACE.jl") for k in kinds):
        raise NotImplementedError(f"{path}: ACE.jl tabulated radials are not supported")
    if len(kinds) != 1:
        raise NotImplementedError(f"{path}: per-bond differing radbasename {sorted(kinds)}")
    inner = {b["inner_cutoff_type"] for b in bonds.values()}
    if len(inner) != 1:
        raise ValueError(f"{path}: bonds disagree on inner_cutoff_type {sorted(inner)}; "
                         "ML-PACE keeps one global value (last bond wins), so the file is ill-defined")
    for k in [(i, j) for i in range(NZ) for j in range(NZ)]:
        if k not in bonds:
            raise ValueError(f"{path}: missing bond {list(k)}")

    nradmax = max(int(b["nradmax"]) for b in bonds.values())
    lmax = max(int(b["lmax"]) for b in bonds.values())
    K = max(int(b["nradbasemax"]) for b in bonds.values())
    crad = np.zeros((NZ, NZ, nradmax, lmax + 1, K))
    radparams = np.zeros((NZ, NZ, 5))
    core = np.zeros((NZ, NZ, 2))
    for (i, j), b in bonds.items():
        c = np.asarray(b["radcoefficients"], float)          # [n][l][k]
        crad[i, j, :c.shape[0], :c.shape[1], :c.shape[2]] = c
        radparams[i, j] = [b["radparameters"][0], b["rcut"], b["dcut"], b["rcut_in"], b["dcut_in"]]
        core[i, j] = [b["prehc"], b["lambdahc"]]

    embs = tree["embeddings"]
    P = max(int(embs[e]["ndensity"]) for e in range(NZ))
    fs = np.zeros((NZ, 2 * P))
    fs[:, 1::2] = 1.0                                       # padded densities: w=0, m=1
    rho_core_cut = np.zeros((NZ, 2))
    npoti = []
    for e in range(NZ):
        em = embs[e]
        p = np.asarray(em["FS_parameters"], float)
        fs[e, :p.size] = p
        rho_core_cut[e] = [em["rho_core_cutoff"], em["drho_core_cutoff"]]
        npoti.append(em["npoti"])
        if em["npoti"] not in ("FinnisSinclair", "FinnisSinclairShiftedScaled"):
            raise NotImplementedError(f"{path}: embedding {em['npoti']!r}")

    funcs, spec_funcs, ctil = [], [], []
    term = 0
    for e in range(NZ):
        for idx, f in enumerate(tree["functions"].get(e, [])):
            rank, nms, nd = int(f["rank"]), int(f["num_ms_combs"]), int(f["ndensity"])
            ms = np.asarray(f["ms_combs"], np.int64).reshape(nms, rank)
            c = np.zeros((nms, P))
            c[:, :nd] = np.asarray(f["ctildes"], float).reshape(nms, nd)
            funcs.append({"el": e, "rank": rank, "mus": list(f["mus"]), "ns": list(f["ns"]),
                          "ls": list(f["ls"]), "ms": ms})
            spec_funcs.append((e, idx, term, nms, nd))
            ctil.append(c)
            term += nms
    arrays = {
        "E0": np.asarray(tree["E0"], float), "Z": Z, "crad": crad, "radparams": radparams,
        "core": core, "fs_params": fs, "rho_core_cut": rho_core_cut,
        "ctilde_complex": np.concatenate(ctil) if ctil else np.zeros((0, P)),
        "radbasename": kinds.pop(), "inner_cutoff_type": inner.pop(),
        "npoti": tuple(npoti), "ndensity": P, "nradmax": nradmax, "lmax": lmax,
        "nradbase": K, "funcs": funcs,
    }
    return PACESpec(tree=tree, element_names=names, functions=spec_funcs), arrays
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_pace_io.py -q`
Expected: all pass (fixture cases skip if Task 2 hasn't run yet).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/ace_jax/eval/pace_io.py tests/test_pace_io.py
git commit -m "feat(pace): .yace parser with validation (PACESpec + numeric arrays)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Radial, core and embedding ports

**Files:**
- Create: `src/ace_jax/eval/pace_radial.py`, `tests/test_pace_radial.py`

**Interfaces:**
- Consumes: `fixtures/pace/unit_ref.npz` (Task 1).
- Produces (all pure JAX, broadcasting over leading edge axes; `K` is a static int):
  - `cutoff_func_poly(r, r_in, delta) -> array`
  - `radbase(r, name: str, inner: str, lam, rc, dcut, cut_in, dcut_in, K) -> (..., K)`
  - `radcore(r, pre, lamhc, rc, r_in, delta_in, inner: str) -> array`
  - `pace_zbl(r, Zi, Zj, rc, dcut, pre) -> array`
  - `fexp(x, m)`, `fexp_shifted_scaled(x, m) -> array`

- [ ] **Step 1: Write the failing tests**

`tests/test_pace_radial.py`:
```python
import pathlib
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import pace_radial as pr

REF = pathlib.Path(__file__).parent.parent / "fixtures" / "pace" / "unit_ref.npz"


@pytest.fixture(scope="module")
def ref():
    if not REF.exists():
        pytest.skip("unit_ref.npz not generated (Task 1)")
    return np.load(REF)


@pytest.mark.parametrize("name", ["ChebExpCos", "ChebPow", "ChebLinear", "SBessel"])
@pytest.mark.parametrize("inner", ["density", "distance", "zbl"])
def test_radbase(ref, name, inner):
    g = pr.radbase(jnp.asarray(ref["r"]), name, inner, 2.5, 5.0, 0.3, 1.2, 0.4, 8)
    np.testing.assert_allclose(g, ref[f"g_{name}_{inner}"], rtol=1e-12, atol=1e-13)


def test_zbl_cut_in_quirk(ref):
    g = pr.radbase(jnp.asarray(ref["r"]), "ChebExpCos", "zbl", 2.5, 5.0, 0.3, 1.2, 0.0, 8)
    np.testing.assert_allclose(g, ref["g_ChebExpCos_zbl_dcutin0"], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("inner", ["density", "distance"])
def test_radcore(ref, inner):
    c = pr.radcore(jnp.asarray(ref["r"]), 3.0, 0.7, 5.0, 1.2, 0.4, inner)
    np.testing.assert_allclose(c, ref[f"cr_{inner}"], rtol=1e-12, atol=1e-13)


def test_zbl(ref):
    c = pr.pace_zbl(jnp.asarray(ref["r"]), 14, 32, 5.0, 0.3, 1.0)
    np.testing.assert_allclose(c, ref["cr_zbl"], rtol=1e-11, atol=1e-12)


def test_cutoff_func_poly(ref):
    np.testing.assert_allclose(pr.cutoff_func_poly(jnp.asarray(ref["r"]), 3.0, 1.0),
                               ref["fcpoly"], rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("m", [0.5, 1.0, 2.0])
def test_embedding(ref, m):
    x = jnp.asarray(ref["fx"])
    tag = f"{m:g}"
    np.testing.assert_allclose(pr.fexp(x, m), ref[f"fexp_m{tag}"], rtol=1e-12, atol=1e-300)
    np.testing.assert_allclose(pr.fexp_shifted_scaled(x, m), ref[f"fexpss_m{tag}"],
                               rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("fn", [pr.fexp, pr.fexp_shifted_scaled])
@pytest.mark.parametrize("m", [0.5, 1.0, 2.0])
def test_embedding_grad_finite_at_zero(fn, m):
    g = jax.grad(lambda x: fn(x, m))(0.0)
    assert np.isfinite(g)


@pytest.mark.parametrize("name", ["ChebExpCos", "ChebPow", "ChebLinear", "SBessel"])
def test_radbase_grad_finite_at_and_beyond_cutoff(name):
    for r in (5.0, 5.5):
        J = jax.jacfwd(lambda rr: pr.radbase(rr, name, "density", 2.0, 5.0, 0.3, 0.0, 1e-5, 6))(r)
        assert np.all(np.isfinite(J)) and np.all(J == 0)
```

- [ ] **Step 2: Run and confirm the failure**

Run: `uv run pytest tests/test_pace_radial.py -q`
Expected: ImportError on `pace_radial`.

- [ ] **Step 3: Implement**

`src/ace_jax/eval/pace_radial.py`:
```python
"""Pure-JAX ports of ML-PACE radial, core-repulsion and embedding functions.

Source of truth: lammps-user-pace @ 99aa6e6, ML-PACE/ace-evaluator/
  ace_radial.cpp          radbase 241, chebExpCos 294, chebPow 347, chebLinear 366,
                          simplified_bessel 423, cutoff_func_poly 220, radcore 615, ZBL 653
  ace_abstract_basis.cpp  Fexp 37, FexpShiftedScaled 67
Values are checked against the C++ in tests/test_pace_radial.py.  Every branch
not taken is evaluated on a safe argument (double-where), so gradients stay
finite at r >= rcut, x = 0 and m = 1.
"""
import math

import jax.numpy as jnp

PI = math.pi


def _cheb(x, n):
    """T_0..T_n stacked on the last axis."""
    T = [jnp.ones_like(x), x]
    for _ in range(n - 1):
        T.append(2.0 * x * T[-1] - T[-2])
    return jnp.stack(T[:n + 1], axis=-1)


def cutoff_func_poly(r, r_in, delta):
    """1 for r <= r_in - delta, 0 for r >= r_in, quintic in between."""
    d = jnp.where(delta == 0, 1.0, delta)
    x = 1.0 - 2.0 * (1.0 + (r - r_in) / d)
    fc = 0.5 + 7.5 / 2.0 * (x / 4.0 - x ** 3 / 6.0 + x ** 5 / 20.0)
    return jnp.where(r <= r_in - delta, 1.0, jnp.where(r >= r_in, 0.0, fc))


def _cheb_exp_cos(r, lam, rc, dcut, K):
    y1, y2 = jnp.exp(-lam * r / rc), jnp.exp(-lam)
    x = 1.0 - 2.0 * ((y1 - y2) / (1.0 - y2))
    T = _cheb(x, K)
    g = jnp.concatenate([T[..., :1], 0.5 - 0.5 * T[..., 1:K]], axis=-1)
    env = 0.5 * (1.0 + jnp.cos(PI * r / rc))
    fc = jnp.where(r > rc - dcut, 0.5 * (1.0 + jnp.cos(PI * (r - rc + dcut) / dcut)), 1.0)
    return g * (env * fc)[..., None]


def _cheb_pow(r, lam, rc, K):
    y = (1.0 - r / rc) ** lam
    T = _cheb(2.0 * (1.0 - y) - 1.0, K)
    return 0.5 - 0.5 * T[..., 1:K + 1]


def _cheb_linear(r, rc, K):
    T = _cheb(1.0 - r / rc, K)
    return 0.5 - 0.5 * T[..., 1:K + 1]


def _sinc(x):
    xs = jnp.where(x == 0, 1.0, x)
    return jnp.where(x == 0, 1.0, jnp.sin(xs) / xs)


def _sbessel(r, rc, K):
    def f(n):
        pre = ((-1) ** n * math.sqrt(2) * PI * (n + 1) * (n + 2)
               / math.sqrt((n + 1) ** 2 + (n + 2) ** 2))
        return pre / rc ** 1.5 * (_sinc(r * (n + 1) * PI / rc) + _sinc(r * (n + 2) * PI / rc))
    g = [f(0)]
    d_prev = 1.0
    for n in range(1, K):
        en = n ** 2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        dn = 1 - en / d_prev
        g.append((f(n) + math.sqrt(en / d_prev) * g[-1]) / math.sqrt(dn))
        d_prev = dn
    return jnp.stack(g, axis=-1)


def radbase(r, name, inner, lam, rc, dcut, cut_in, dcut_in, K):
    """g_k(r), k < K, zero outside (cut_in - dcut_in, rc)."""
    if inner == "zbl":                       # ace_radial.cpp:246, ported as-is
        cut_in = jnp.where(dcut_in == 0, 1.0, 0.0)
    inside = (r > cut_in - dcut_in) & (r < rc)
    rs = jnp.where(inside, r, 0.5 * rc)      # safe argument for the untaken branch
    if name == "ChebExpCos":
        g = _cheb_exp_cos(rs, lam, rc, dcut, K)
    elif name == "ChebPow":
        g = _cheb_pow(rs, lam, rc, K)
    elif name == "ChebLinear":
        g = _cheb_linear(rs, rc, K)
    elif name == "SBessel":
        g = _sbessel(rs, rc, K)
    else:
        raise NotImplementedError(f"radial basis {name!r}")
    if inner in ("distance", "zbl"):
        g = g * (1.0 - cutoff_func_poly(rs, cut_in, dcut_in))[..., None]
    return jnp.where(inside[..., None], g, 0.0)


def radcore(r, pre, lamhc, rc, r_in, delta_in, inner):
    """Hard-core repulsion c_r(r) for the density / distance regimes."""
    pre, lam = jnp.abs(pre), jnp.abs(lamhc)
    lr2 = lam * r * r
    ok = lr2 < 50.0
    rs = jnp.where(ok & (r > 0), r, 1.0)
    cr = pre * jnp.exp(-lam * rs * rs) / rs * 0.5 * (1.0 + jnp.cos(PI * rs / rc))
    cr = jnp.where(ok, cr, 0.0)
    if inner == "distance":
        cr = cr * cutoff_func_poly(r, r_in, delta_in)
    return cr


_ZC = (0.18175, 0.50986, 0.28022, 0.02817)
_ZE = (-3.19980, -0.94229, -0.40290, -0.20162)
_KZBL = 14.399645351950543


def _zbl_E(r, a):
    """E(r) = phi(r/a)/r and its first two r-derivatives (fun_E_ij_and_deriv2)."""
    x = r / a
    ph = sum(c * jnp.exp(e * x) for c, e in zip(_ZC, _ZE))
    dph = sum(c * e * jnp.exp(e * x) for c, e in zip(_ZC, _ZE))
    d2ph = sum(c * e * e * jnp.exp(e * x) for c, e in zip(_ZC, _ZE))
    E = ph / r
    dE = -ph / r ** 2 + dph / (a * r)
    d2E = 2 * ph / r ** 3 - 2 * dph / (a * r ** 2) + d2ph / (a * a * r)
    return E, dE, d2E


def pace_zbl(r, Zi, Zj, rc, dcut, pre):
    """ACERadialFunctions::ZBL(r, Zi, Zj, cut_out=rc, cut_in=rc-dcut, pre)."""
    cut_out, cut_in = rc, rc - dcut
    a = 0.46850 / (Zi ** 0.23 + Zj ** 0.23)
    rs = jnp.where((r > 0) & (r < cut_out), r, 0.5 * cut_out)
    E, _, _ = _zbl_E(rs, a)
    Ec, dEc, d2Ec = _zbl_E(cut_out, a)
    dr = cut_out - cut_in
    A = (-3 * dEc + dr * d2Ec) / dr ** 2
    B = (2 * dEc - dr * d2Ec) / dr ** 3
    C = -Ec + 0.5 * dr * dEc - dr ** 2 / 12.0 * d2Ec
    t = rs - cut_in
    S = jnp.where(rs <= cut_in, C, A / 3 * t ** 3 + B / 4 * t ** 4 + C)
    cr = pre * _KZBL / 2 * Zi * Zj * (E + S) * cutoff_func_poly(rs, cut_out, dr)
    return jnp.where(r < cut_out, cr, 0.0)


def fexp(x, m):
    """PACE `Fexp`: sign(x)((1-g)|x|^m + lam g |x|), g = exp(-(w|x|)^3), linear
    for |x| <= 1e-10.  No prefactor (w_p is applied by the caller)."""
    w = 1.0e6
    lam = (1.0 / w) ** (m - 1.0)
    a = jnp.abs(x)
    big = a > 1e-10
    as_ = jnp.where(big, a, 1.0)
    w3 = (w * as_) ** 3
    g = jnp.where(w3 > 30.0, 0.0, jnp.exp(-jnp.minimum(w3, 30.0)))
    s = jnp.where(x < 0, -1.0, 1.0)
    return jnp.where(big, s * ((1.0 - g) * as_ ** m + lam * g * as_), lam * x)


def fexp_shifted_scaled(x, m):
    """PACE `FexpShiftedScaled`; identity when |m - 1| < 1e-10."""
    lin = jnp.abs(m - 1.0) < 1e-10
    ms = jnp.where(lin, 2.0, m)
    a = jnp.abs(x)
    e = jnp.exp(-a)
    nu = 1.0 / ms
    xoff = nu ** (nu / (1.0 - nu)) * e
    yoff = nu ** (1.0 / (1.0 - nu)) * e
    s = jnp.where(x < 0, -1.0, 1.0)
    return jnp.where(lin, x, s * ((xoff + a) ** ms - yoff))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_pace_radial.py -q`
Expected: all pass.

If a family differs from the C++, re-read the cited C++ lines and fix the port. Don't loosen tolerances: 1e-12 relative is what line-by-line ports achieve. One exception is `test_zbl` near r → 0, where 1/r magnifies roundoff; there, loosen only `rtol` to 1e-10 and say so in the test.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/eval/pace_radial.py tests/test_pace_radial.py
git commit -m "feat(pace): JAX ports of PACE radials, core repulsion, ZBL, Fexp

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Y_lm convention map and the complex→real basis builder

**Files:**
- Create: `src/ace_jax/eval/pace_build.py`, `tests/test_pace_build.py`

**Interfaces:**
- Consumes: `parse_yace(...)[1]` arrays (Task 3), in particular `funcs`, `nradbase`, `nradmax`, `lmax`, and `Z` (only its length NZ). Also `real_spherical_harmonics(xyz, l_max)` from `eval/harmonics.py`, ordered l*l+l+m.
- Produces:
  - `ylm_map(lmax) -> dict[int, np.ndarray]`: complex `U[l]` of shape (2l+1, 2l+1), rows = PACE m (−l..l), columns = real m′ (−l..l), with Y^PACE_l = U[l] @ Y^R_l
  - `build_basis(arrays) -> dict` with:
    - `a_rad` (n_A,) int, the column into the per-edge radial vector `[g_0..g_{K-1}, R_{0,0}, R_{0,1}, …]`, where R_{n,l} sits at column `K + n*(lmax+1) + l`
    - `a_y` (n_A,) int, the column into Y^R (`l*l+l+m′`)
    - `n_a_local` int
    - `aa_specs`, a tuple of (n_v, order) int arrays indexing the flattened (NZ·n_A) pooled A, index `mu*n_a_local + a`
    - `T_rows`, `T_cols` int, `T_vals` float, with ctilde_real.reshape(n_AA·NZ, P)[T_rows] += T_vals[:, None] · ctilde_complex[T_cols]
    - `n_aa` int

- [ ] **Step 1: Write the failing tests**

`tests/test_pace_build.py`:
```python
import itertools, pathlib
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval.harmonics import real_spherical_harmonics
from ace_jax.eval.pace_build import build_basis, ylm_map

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"


def test_ylm_map_matches_pace_cpp():
    ref_p = FIX / "unit_ref.npz"
    if not ref_p.exists():
        pytest.skip("unit_ref.npz not generated")
    ref = np.load(ref_p)
    U = ylm_map(6)
    YR = np.asarray(real_spherical_harmonics(jnp.asarray(ref["ydir"]), 6))
    YP = ref["ylm_re"] + 1j * ref["ylm_im"]
    for l in range(7):
        s = slice(l * l, (l + 1) ** 2)
        np.testing.assert_allclose(YR[:, s] @ U[l].T, YP[:, s], atol=1e-13)


def _toy_arrays(rng, NZ=2, K=3, nradmax=2, lmax=2):
    """Random functions with random (not necessarily invariant) ms combos."""
    funcs = [{"el": 0, "rank": 1, "mus": [1], "ns": [2], "ls": [0], "ms": np.array([[0]])}]
    for el in range(NZ):
        for rank in (2, 3):
            ls = list(rng.integers(0, lmax + 1, rank))
            ms = np.array([[rng.integers(-l, l + 1) for l in ls] for _ in range(4)])
            funcs.append({"el": el, "rank": rank, "mus": list(rng.integers(0, NZ, rank)),
                          "ns": list(rng.integers(1, nradmax + 1, rank)), "ls": ls, "ms": ms})
    n_terms = sum(len(f["ms"]) for f in funcs)
    return {"funcs": funcs, "nradbase": K, "nradmax": nradmax, "lmax": lmax,
            "Z": np.arange(NZ), "ndensity": 2,
            "ctilde_complex": rng.normal(size=(n_terms, 2))}


def test_real_basis_reproduces_complex_products():
    """sum_t Re(c_t prod A^PACE) == AA . ctilde_real, for random real A."""
    rng = np.random.default_rng(3)
    a = _toy_arrays(rng)
    b = build_basis(a)
    NZ, K, nm, lmax = 2, a["nradbase"], a["nradmax"], a["lmax"]
    ncol = K + nm * (lmax + 1)
    # random real "pooled" A per (mu, radial column, real Ylm column)
    Areal = rng.normal(size=(NZ, ncol, (lmax + 1) ** 2))
    U = ylm_map(lmax)

    def A_pace(mu, col, l, m):                     # complex A for PACE index m
        s = slice(l * l, (l + 1) ** 2)
        return U[l][m + l] @ Areal[mu, col, s]

    rho = np.zeros((NZ, 2))                        # [centre element, density]
    t = 0
    for f in a["funcs"]:
        for ms in f["ms"]:
            prod = 1.0 + 0j
            for k in range(f["rank"]):
                col = (f["ns"][k] - 1) if f["rank"] == 1 else K + (f["ns"][k] - 1) * (lmax + 1) + f["ls"][k]
                prod *= A_pace(f["mus"][k], col, f["ls"][k], ms[k])
            rho[f["el"]] += (prod * a["ctilde_complex"][t]).real
            t += 1

    Aflat = np.stack([Areal[mu][b["a_rad"], b["a_y"]] for mu in range(NZ)]).reshape(-1)
    AA = np.concatenate([np.prod(Aflat[s], axis=-1) for s in b["aa_specs"]])
    ct = np.zeros((b["n_aa"] * NZ, 2))
    np.add.at(ct, b["T_rows"], b["T_vals"][:, None] * a["ctilde_complex"][b["T_cols"]])
    ct = ct.reshape(b["n_aa"], NZ, 2)
    np.testing.assert_allclose(np.einsum("a,aep->ep", AA, ct), rho, rtol=1e-12, atol=1e-12)


def test_aa_specs_sorted_and_unique():
    b = build_basis(_toy_arrays(np.random.default_rng(5)))
    keys = [tuple(r) for s in b["aa_specs"] for r in np.asarray(s)]
    assert len(keys) == len(set(keys))
    assert all(list(k) == sorted(k) for k in keys)
```

- [ ] **Step 2: Run and confirm the failure**

Run: `uv run pytest tests/test_pace_build.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/ace_jax/eval/pace_build.py`:
```python
"""Offline (numpy) translation of a PACE C-tilde basis into ace-jax's real
product basis.

PACE forms products of complex A_{mu,n,l,m} (Y_lm with Y00 = 1, Condon-Shortley
phase, negative m by (-1)^m conj) and keeps Re(c . prod).  ace-jax pools real
A with SpheriCart's L2-normalised real Y.  Y^PACE_l = U_l Y^R_l with U_l having
at most 2 non-zeros per row, so each complex product expands into <= 2^rank real
products; their real parts are merged by sorted index tuple into `aa_specs`,
and T maps the complex coefficients onto the merged real ones.  ctilde_complex
stays the model's leaf, so export never inverts this map.
"""
import itertools
import math
from collections import defaultdict

import numpy as np

_S4PI = math.sqrt(4.0 * math.pi)


def ylm_map(lmax):
    """U[l] (2l+1, 2l+1) complex: rows PACE m = -l..l, cols real m' = -l..l.

    Verified against ML-PACE's compute_ylm in tests/test_pace_build.py."""
    U = {}
    r2 = math.sqrt(2.0)
    for l in range(lmax + 1):
        u = np.zeros((2 * l + 1, 2 * l + 1), complex)
        u[l, l] = _S4PI
        for m in range(1, l + 1):
            sg = (-1) ** m
            u[l + m, l + m] = _S4PI * sg / r2
            u[l + m, l - m] = 1j * _S4PI * sg / r2
            u[l - m, l + m] = _S4PI / r2
            u[l - m, l - m] = -1j * _S4PI / r2
        U[l] = u
    return U


def build_basis(arrays):
    funcs, K = arrays["funcs"], int(arrays["nradbase"])
    lmax = int(arrays["lmax"])
    NZ = len(arrays["Z"])
    U = ylm_map(lmax)

    a_index = {}                                   # (rad_col, y_col) -> local a
    def a_of(col, l, mp):
        key = (col, l * l + l + mp)
        if key not in a_index:
            a_index[key] = len(a_index)
        return a_index[key]

    acc = defaultdict(float)                       # (sorted (mu, a) tuple, el, term) -> val
    term = 0
    for f in funcs:
        rank = f["rank"]
        cols = [(f["ns"][0] - 1) if rank == 1
                else K + (f["ns"][k] - 1) * (lmax + 1) + f["ls"][k] for k in range(rank)]
        ls = [0] * rank if rank == 1 else list(f["ls"])
        for ms in f["ms"]:
            opts = []
            for k in range(rank):
                row = U[ls[k]][ms[k] + ls[k]]
                opts.append([(mp - ls[k], row[mp]) for mp in np.nonzero(row)[0]])
            for combo in itertools.product(*opts):
                coef = np.prod([c for _, c in combo]).real
                if abs(coef) < 1e-14:
                    continue
                key = tuple(sorted((f["mus"][k], a_of(cols[k], ls[k], combo[k][0]))
                                   for k in range(rank)))
                acc[(key, f["el"], term)] += coef
            term += 1

    n_a = len(a_index)
    a_rad = np.zeros(n_a, np.int32)
    a_y = np.zeros(n_a, np.int32)
    for (col, y), a in a_index.items():
        a_rad[a], a_y[a] = col, y

    by_order = defaultdict(dict)                   # order -> {flat tuple: idx}
    for (key, _, _), v in acc.items():
        if v != 0.0:
            flat = tuple(sorted(mu * n_a + a for mu, a in key))
            by_order[len(flat)].setdefault(flat, len(by_order[len(flat)]))
    offsets, aa_specs, off = {}, [], 0
    for order in sorted(by_order):
        offsets[order] = off
        tbl = np.zeros((len(by_order[order]), order), np.int32)
        for flat, i in by_order[order].items():
            tbl[i] = flat
        aa_specs.append(tbl)
        off += len(tbl)

    rows, cols, vals = [], [], []
    for (key, el, t), v in acc.items():
        if v == 0.0:
            continue
        flat = tuple(sorted(mu * n_a + a for mu, a in key))
        aa = offsets[len(flat)] + by_order[len(flat)][flat]
        rows.append(aa * NZ + el)
        cols.append(t)
        vals.append(v)
    return {"a_rad": a_rad, "a_y": a_y, "n_a_local": n_a, "aa_specs": tuple(aa_specs),
            "T_rows": np.asarray(rows, np.int32), "T_cols": np.asarray(cols, np.int32),
            "T_vals": np.asarray(vals, float), "n_aa": off}
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_pace_build.py -q`
Expected: all pass. `test_ylm_map_matches_pace_cpp` is the authority on U. If it fails, correct `ylm_map`'s signs and factors until it passes, then update the spec's formula to match.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/eval/pace_build.py tests/test_pace_build.py
git commit -m "feat(pace): complex C-tilde -> real product basis (Ylm map U, sparse T)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `PACEModel`, `load_yace`, and parity

**Files:**
- Create: `src/ace_jax/eval/pace_model.py`, `tests/test_pace_model.py`
- Modify: `src/ace_jax/eval/io.py` (dispatch at the top of `load`), `src/ace_jax/eval/__init__.py` (exports)

**Interfaces:**
- Consumes: `parse_yace` (Task 3); `pace_radial.*` (Task 4); `build_basis` (Task 5); `real_spherical_harmonics`; `pool_sparse`; `ACEModel.energy_forces_virial` (reused unchanged).
- Produces:
  - `class PACEModel(eqx.Module)` with methods `site_energies(rij, zi, zj, segment_ids, n_nodes, node_z, mask=None)`, `site_energies_dense(rij, zi, zj, mask, node_z)`, `energy_forces_virial(rij, zi, zj, senders, receivers, n_nodes, node_z, mask=None)`, `energy_from_positions(positions, node_z, senders, receivers, edge_mask=None, shifts=None)`, and `ctilde_real() -> (n_AA, NZ, P)`. `site_basis`/`site_descriptors`/`edge_jacobian` raise `NotImplementedError`.
  - `load_yace(path, dtype=jnp.float64) -> (PACEModel, meta, PACESpec)`, where meta = `{"elements": [Z...], "rcut": max bond rcut, "format": "yace"}`
  - `ace_jax.eval.load(path)` returns the same triple for a `.yace` path, so `ACECalculator("x.yace")` works unchanged.

- [ ] **Step 1: Write the failing tests**

`tests/test_pace_model.py`:
```python
import pathlib
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase import Atoms

from ace_jax.eval import ACECalculator, load, sparse_graph
from ace_jax.eval.pace_model import PACEModel, load_yace

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"
NAMES = ["si_chebexpcos", "si_chebpow_fs", "si_cheblinear", "gesi_sbessel",
         "sige_density_core", "sige_distance", "sige_zbl"]


def _fixture(name):
    y, r = FIX / f"{name}.yace", FIX / f"{name}_ref.npz"
    if not (y.exists() and r.exists()):
        pytest.skip("fixtures not generated (Task 2)")
    return y, np.load(r)


def _atoms(ref, s):
    return Atoms(numbers=ref[f"Z_{s}"], positions=ref[f"pos_{s}"],
                 cell=ref[f"cell_{s}"], pbc=ref[f"pbc_{s}"])


@pytest.mark.parametrize("name", NAMES)
def test_parity_tight(name):
    y, ref = _fixture(name)
    calc = ACECalculator(str(y))
    for s in ref["struct_names"]:
        at = _atoms(ref, s)
        at.calc = calc
        n = len(at)
        assert abs(at.get_potential_energy() - ref[f"E_tight_{s}"]) / n < 1e-9, s
        np.testing.assert_allclose(at.get_forces(), ref[f"F_tight_{s}"], atol=1e-8, err_msg=s)
        if at.pbc.all():
            np.testing.assert_allclose(at.get_stress(), ref[f"S_tight_{s}"], atol=1e-9, err_msg=s)


@pytest.mark.parametrize("name", NAMES)
def test_parity_shipped_grid(name):
    """Gap to the C++ at the file's own deltaSplineBins: measured, then pinned."""
    y, ref = _fixture(name)
    calc = ACECalculator(str(y))
    worst = 0.0
    for s in ref["struct_names"]:
        at = _atoms(ref, s)
        at.calc = calc
        worst = max(worst, abs(at.get_potential_energy() - ref[f"E_ship_{s}"]) / len(at),
                    float(np.abs(at.get_forces() - ref[f"F_ship_{s}"]).max()))
    print(f"\n  {name}: worst |dE|/atom or |dF| vs shipped spline grid = {worst:.2e}")
    assert worst < SHIP_TOL


SHIP_TOL = 1e-6   # replace with 10x the largest value printed above (Step 5)


def _fd_forces(calc, at, h=1e-5):
    F = np.zeros((len(at), 3))
    for i in range(len(at)):
        for d in range(3):
            for sg in (1, -1):
                a = at.copy(); a.positions[i, d] += sg * h; a.calc = calc
                F[i, d] -= sg * a.get_potential_energy() / (2 * h)
    return F


@pytest.mark.parametrize("name", ["gesi_sbessel", "sige_zbl"])
def test_forces_match_finite_differences(name):
    y, ref = _fixture(name)
    calc = ACECalculator(str(y))
    at = _atoms(ref, "close")[:12]
    at.calc = calc
    np.testing.assert_allclose(at.get_forces(), _fd_forces(calc, at), atol=1e-6)


def test_rotation_invariance():
    y, ref = _fixture("gesi_sbessel")
    at = _atoms(ref, "dimer_2.3") + _atoms(ref, "dimer_2.3").copy()
    at.positions[2:] += [0.3, 2.0, 0.7]
    at.calc = ACECalculator(str(y))
    E = at.get_potential_energy()
    b = at.copy(); b.rotate(37, [1, 2, 3], center="COM"); b.calc = at.calc
    assert abs(b.get_potential_energy() - E) < 1e-10


def test_mixed_rcut_neighbours_beyond_bond_cutoff():
    """gesi_sbessel: Ge-Ge rcut 4.2 < 5.0.  A Ge at 4.6 A from a Ge must not
    interact, although it is inside the neighbour-list cutoff."""
    y, _ = _fixture("gesi_sbessel")
    calc = ACECalculator(str(y))
    def e(d, zb):
        a = Atoms(numbers=[32, zb], positions=[[0, 0, 0], [d, 0, 0]], cell=np.eye(3) * 30, pbc=False)
        a.calc = calc
        return a.get_potential_energy(), a.get_forces()
    iso = Atoms(numbers=[32], positions=[[0, 0, 0]], cell=np.eye(3) * 30, pbc=False)
    iso.calc = calc
    E_iso = iso.get_potential_energy()
    E, F = e(4.6, 32)
    assert abs(E - 2 * E_iso) < 1e-12 and np.abs(F).max() < 1e-12
    E, _ = e(4.6, 14)                               # Ge-Si rcut 5.0: interacts
    assert abs(E - E_iso - _single(calc, 14)) > 1e-8


def _single(calc, z):
    a = Atoms(numbers=[z], positions=[[0, 0, 0]], cell=np.eye(3) * 30, pbc=False)
    a.calc = calc
    return a.get_potential_energy()


@pytest.mark.parametrize("name", ["si_chebpow_fs", "sige_zbl"])
def test_padded_and_neighbourless_nodes_finite(name):
    y, ref = _fixture(name)
    model, meta, _ = load(str(y))
    at = _atoms(ref, "bulk")
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    z2i = {z: i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers] + [0])      # one extra, isolated node
    pos = jnp.concatenate([jnp.asarray(at.positions), jnp.asarray([[50.0, 50, 50]])])
    send = jnp.concatenate([jnp.asarray(g.senders), jnp.asarray([len(at)] * 5)])
    recv = jnp.concatenate([jnp.asarray(g.receivers), jnp.asarray([len(at)] * 5)])
    shifts = jnp.concatenate([jnp.asarray(g.shifts), jnp.zeros((5, 3))])
    mask = jnp.concatenate([jnp.ones(len(g.senders), bool), jnp.zeros(5, bool)])
    f = lambda p: jnp.sum(model.energy_from_positions(p, nz, send, recv, mask, shifts))
    E, G = jax.value_and_grad(f)(pos)
    assert np.isfinite(E) and np.all(np.isfinite(G)) and np.all(G[-1] == 0)


def test_float32_close_to_float64():
    y, ref = _fixture("sige_distance")
    at = _atoms(ref, "bulk")
    at.calc = ACECalculator(str(y), dtype=jnp.float64)
    E64, F64 = at.get_potential_energy(), at.get_forces()
    at.calc = ACECalculator(str(y), dtype=jnp.float32)
    E32, F32 = at.get_potential_energy(), at.get_forces()
    assert np.isfinite(E32) and abs(E32 - E64) / abs(E64) < 1e-4
    assert np.all(np.isfinite(F32)) and np.abs(F32 - F64).max() < 1e-3


def test_dense_matches_sparse():
    y, ref = _fixture("gesi_sbessel")
    model, meta, _ = load(str(y))
    at = _atoms(ref, "bulk")
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    z2i = {z: i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers])
    s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    rij = jnp.asarray(g.rij)
    e_sp = model.site_energies(rij, nz[s], nz[r], s, len(at), nz)
    K = int(np.bincount(g.senders).max())
    n = len(at)
    R = np.full((n, K, 3), [meta["rcut"], 0, 0]); Zj = np.zeros((n, K), int); M = np.zeros((n, K), bool)
    fill = np.zeros(n, int)
    for e, (i, j) in enumerate(zip(g.senders, g.receivers)):
        R[i, fill[i]] = g.rij[e]; Zj[i, fill[i]] = z2i[int(at.numbers[j])]; M[i, fill[i]] = True; fill[i] += 1
    e_dn = model.site_energies_dense(jnp.asarray(R), jnp.broadcast_to(nz[:, None], (n, K)),
                                     jnp.asarray(Zj), jnp.asarray(M), nz)
    np.testing.assert_allclose(e_dn, e_sp, atol=1e-12)


def test_b_basis_methods_raise():
    y, _ = _fixture("si_chebexpcos")
    model, _, _ = load(str(y))
    with pytest.raises(NotImplementedError, match="B-basis"):
        model.site_descriptors(None, None, None, None, 0, None)
```

- [ ] **Step 2: Run and confirm the failure**

Run: `uv run pytest tests/test_pace_model.py -q`
Expected: ImportError on `pace_model`.

- [ ] **Step 3: Implement the model and loader**

`src/ace_jax/eval/pace_model.py`:
```python
"""PACE C-tilde potential (pacemaker `.yace`) as an Equinox module.

Mirrors ML-PACE `ACECTildeEvaluator::compute_atom` (ace_evaluator.cpp:146-536,
lammps-user-pace @ 99aa6e6) with analytic radials instead of its spline tables.
Neighbour species enter PACE's A as an explicit channel; here edges stay narrow
and are pooled by (node, neighbour species) -- segment id node*NZ + zj -- so per-
edge work does not grow with the number of elements.  See docs/pace-yace-spec.md.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .harmonics import real_spherical_harmonics
from .model import ACEModel, pool_sparse
from .pace_build import build_basis
from .pace_io import parse_yace
from .pace_radial import cutoff_func_poly, fexp, fexp_shifted_scaled, pace_zbl, radbase, radcore


class PACEModel(eqx.Module):
    # trainable leaves (what write_yace serialises)
    crad: jax.Array            # (NZ, NZ, nradmax, lmax+1, K)
    radparams: jax.Array       # (NZ, NZ, 5): lambda, rcut, dcut, rcut_in, dcut_in
    core: jax.Array            # (NZ, NZ, 2): prehc, lambdahc
    ctilde_complex: jax.Array  # (n_terms, P)
    fs_params: jax.Array       # (NZ, 2P): w0, m0, w1, m1, ...
    rho_core_cut: jax.Array    # (NZ, 2)
    E0: jax.Array              # (NZ,)
    # integer structure (leaves, never differentiated)
    Zf: jax.Array              # (NZ,) atomic numbers as floats, for ZBL
    a_rad: jax.Array
    a_y: jax.Array
    aa_specs: tuple
    T_rows: jax.Array
    T_cols: jax.Array
    T_vals: jax.Array
    # static
    radbasename: str = eqx.field(static=True)
    inner_cutoff_type: str = eqx.field(static=True)
    npoti: tuple = eqx.field(static=True)
    ndensity: int = eqx.field(static=True)
    nradbase: int = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    n_a_local: int = eqx.field(static=True)
    n_aa: int = eqx.field(static=True)

    @property
    def nz(self):
        return self.E0.shape[0]

    def ctilde_real(self):
        """(n_AA, NZ, P): T applied to the complex coefficients (cheap, per call)."""
        flat = jax.ops.segment_sum(self.T_vals[:, None] * self.ctilde_complex[self.T_cols],
                                   self.T_rows, num_segments=self.n_aa * self.nz)
        return flat.reshape(self.n_aa, self.nz, self.ndensity)

    # ------------------------------------------------------------ per edge
    def _edges(self, rij, zi, zj, mask):
        r2 = jnp.sum(rij * rij, axis=-1)
        bp = self.radparams[zi, zj]
        lam, rc, dcut, cin, dcin = (bp[:, k] for k in range(5))
        valid = r2 < rc * rc
        if mask is not None:
            valid = valid & mask
        r = jnp.sqrt(jnp.where(valid, r2, 1.0)) * jnp.where(valid, 1.0, 0.5 * rc)
        rij_s = jnp.where(valid[:, None], rij, jnp.stack([r, 0 * r, 0 * r], -1))
        g = radbase(r, self.radbasename, self.inner_cutoff_type, lam, rc, dcut,
                    cin, dcin, self.nradbase)                             # (E, K)
        R = jnp.einsum("ek,enlk->enl", g, self.crad[zi, zj]).reshape(g.shape[0], -1)
        cols = jnp.concatenate([g, R], axis=-1)
        Y = real_spherical_harmonics(rij_s, self.lmax)
        eA = jnp.where(valid[:, None], cols[:, self.a_rad] * Y[:, self.a_y], 0.0)
        if self.inner_cutoff_type == "zbl":
            cr = pace_zbl(r, self.Zf[zi], self.Zf[zj], rc, dcut, self.core[zi, zj, 0])
        else:
            cr = radcore(r, self.core[zi, zj, 0], self.core[zi, zj, 1], rc, cin, dcin,
                         self.inner_cutoff_type)
        cr = jnp.where(valid, cr, 0.0)
        d = jnp.where(valid, r - (cin - dcin), jnp.inf)                   # zbl switch coordinate
        return eA, cr, d, dcin

    # ------------------------------------------------------------ per node
    def _embedding(self, rho, node_z):
        w, m = self.fs_params[node_z, 0::2], self.fs_params[node_z, 1::2]
        ss = jnp.asarray([p == "FinnisSinclairShiftedScaled" for p in self.npoti])[node_z]
        F = jnp.where(ss[:, None], fexp_shifted_scaled(rho, m), fexp(rho, m))
        return jnp.sum(w * F, axis=-1)

    def site_energies(self, rij, zi, zj, segment_ids, n_nodes, node_z, mask=None):
        eA, cr, d, dcin = self._edges(rij, zi, zj, mask)
        A = jax.ops.segment_sum(eA, segment_ids * self.nz + zj, num_segments=n_nodes * self.nz)
        A = A.reshape(n_nodes, self.nz * self.n_a_local)
        AA = jnp.concatenate([jnp.prod(A[:, s], axis=-1) for s in self.aa_specs], axis=-1)
        rho_all = (AA @ self.ctilde_real().reshape(self.n_aa, -1)).reshape(
            n_nodes, self.nz, self.ndensity)
        rho = rho_all[jnp.arange(n_nodes), node_z]
        EF = self._embedding(rho, node_z)
        rho_core = jax.ops.segment_sum(cr, segment_ids, num_segments=n_nodes)
        if self.inner_cutoff_type == "zbl":
            dmin = jax.ops.segment_min(d, segment_ids, num_segments=n_nodes)
            has = jnp.isfinite(dmin)
            is_min = jnp.isfinite(d) & (d == dmin[segment_ids])
            dc = jax.ops.segment_max(jnp.where(is_min, dcin, -jnp.inf), segment_ids,
                                     num_segments=n_nodes)
            dc_s = jnp.where(has, dc, 1.0)
            f = jnp.where(has, cutoff_func_poly(dc_s - jnp.where(has, dmin, 0.0), dc_s, dc_s), 1.0)
            E = EF * f + rho_core * (1.0 - f)
        else:
            rc_, drc_ = self.rho_core_cut[node_z, 0], self.rho_core_cut[node_z, 1]
            E = EF * cutoff_func_poly(rho_core, rc_, drc_) + rho_core
        return E + self.E0[node_z]

    def site_energies_dense(self, rij, zi, zj, mask, node_z):
        n, K = mask.shape
        seg = jnp.repeat(jnp.arange(n), K)
        return self.site_energies(rij.reshape(n * K, 3), zi.reshape(-1), zj.reshape(-1),
                                  seg, n, node_z, mask.reshape(-1))

    def energy_forces_virial(self, rij, zi, zj, senders, receivers, n_nodes,
                             node_z, mask=None):
        """Same algebra as ACEModel (one value_and_grad over edge vectors,
        symmetric-strain virial); it only calls self.site_energies."""
        return ACEModel.energy_forces_virial(self, rij, zi, zj, senders, receivers,
                                             n_nodes, node_z, mask)

    def energy_from_positions(self, positions, node_z, senders, receivers,
                              edge_mask=None, shifts=None):
        """lammps-jax entry point.  Padded edges are masked (and moved off r=0)."""
        rij = positions[receivers] - positions[senders]
        if shifts is not None:
            rij = rij + shifts
        if edge_mask is not None:
            far = jnp.asarray([1.0, 0.0, 0.0], positions.dtype) * jnp.max(self.radparams[..., 1])
            rij = jnp.where(edge_mask[:, None], rij, far)
        zi, zj = node_z[senders], node_z[receivers]
        return self.site_energies(rij, zi, zj, senders, positions.shape[0], node_z, edge_mask)

    # ------------------------------------------------------------ not applicable
    def _no_b(self, *a, **k):
        raise NotImplementedError(
            "PACE models have no B-basis: site_basis / site_descriptors / edge_jacobian "
            "are only defined for ACEModel")
    site_basis = site_descriptors = edge_jacobian = compact_basis = _no_b


def load_yace(path, dtype=jnp.float64):
    spec, a = parse_yace(path)
    b = build_basis(a)
    A = lambda x: jnp.asarray(x, dtype)
    I = lambda x: jnp.asarray(x, jnp.int32)
    model = PACEModel(
        crad=A(a["crad"]), radparams=A(a["radparams"]), core=A(a["core"]),
        ctilde_complex=A(a["ctilde_complex"]), fs_params=A(a["fs_params"]),
        rho_core_cut=A(a["rho_core_cut"]), E0=A(a["E0"]), Zf=A(a["Z"]),
        a_rad=I(b["a_rad"]), a_y=I(b["a_y"]), aa_specs=tuple(I(s) for s in b["aa_specs"]),
        T_rows=I(b["T_rows"]), T_cols=I(b["T_cols"]), T_vals=A(b["T_vals"]),
        radbasename=a["radbasename"], inner_cutoff_type=a["inner_cutoff_type"],
        npoti=a["npoti"], ndensity=int(a["ndensity"]), nradbase=int(a["nradbase"]),
        lmax=int(a["lmax"]), n_a_local=int(b["n_a_local"]), n_aa=int(b["n_aa"]))
    meta = {"elements": [int(z) for z in a["Z"]], "rcut": float(a["radparams"][..., 1].max()),
            "format": "yace"}
    return model, meta, spec
```

- [ ] **Step 4: Dispatch in `load` and export**

In `src/ace_jax/eval/io.py`, directly after the `load(...)` docstring:
```python
    if str(path).endswith(".yace"):          # PACE C-tilde potential: separate model class
        from .pace_model import load_yace
        return load_yace(path, dtype=dtype)
```
In `src/ace_jax/eval/__init__.py`, add `from .pace_model import PACEModel, load_yace` and append `"PACEModel", "load_yace"` to `__all__`.

- [ ] **Step 5: Run the tests, then pin the shipped-grid tolerance**

Run: `uv run pytest tests/test_pace_model.py -q -s`
Expected: all pass, and `test_parity_shipped_grid` prints one `worst` per fixture. Set `SHIP_TOL` to 10× the largest printed value, rounded up to one significant figure, and add a comment giving the measured maximum and the date.

If `test_parity_tight` fails on one regime only, debug that regime against `ace_evaluator.cpp:497–536`. Don't loosen the 1e-9 tolerance. If it fails on every fixture, the likely cause is the U map or species pooling: re-run `tests/test_pace_build.py` first.

- [ ] **Step 6: Commit**

```bash
git add src/ace_jax/eval/pace_model.py src/ace_jax/eval/io.py src/ace_jax/eval/__init__.py tests/test_pace_model.py
git commit -m "feat(pace): PACEModel + load_yace; parity with python-ace on 7 fixtures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `write_yace` and round-trip

**Files:**
- Modify: `src/ace_jax/eval/pace_io.py` (add `write_yace`), `src/ace_jax/eval/__init__.py` (export)
- Create: `tests/test_pace_export.py`

**Interfaces:**
- Consumes: `PACEModel` leaves (Task 6), `PACESpec` (Task 3).
- Produces: `write_yace(model, spec, path) -> None`. It overwrites only numeric fields in a deep copy of `spec.tree`; everything else is verbatim.

- [ ] **Step 1: Write the failing tests**

`tests/test_pace_export.py`:
```python
import pathlib
import equinox as eqx
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
from ase import Atoms

from ace_jax.eval import ACECalculator, load
from ace_jax.eval.pace_io import load_tree, write_yace

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"
NAMES = ["si_chebpow_fs", "gesi_sbessel", "sige_zbl"]


def _need(name):
    p = FIX / f"{name}.yace"
    if not p.exists():
        pytest.skip("fixtures not generated")
    return p


@pytest.mark.parametrize("name", NAMES)
def test_roundtrip_tree_equal(name, tmp_path):
    p = _need(name)
    model, _, spec = load(str(p))
    out = tmp_path / "rt.yace"
    write_yace(model, spec, out)
    assert load_tree(out) == load_tree(p)


@pytest.mark.parametrize("name", NAMES)
def test_modified_model_roundtrips(name, tmp_path):
    p = _need(name)
    model, meta, spec = load(str(p))
    m2 = eqx.tree_at(lambda m: m.ctilde_complex, model, 2.0 * model.ctilde_complex)
    m2 = eqx.tree_at(lambda m: m.crad, m2, 0.9 * m2.crad)
    out = tmp_path / "mod.yace"
    write_yace(m2, spec, out)
    ref = np.load(FIX / f"{name}_ref.npz")
    at = Atoms(numbers=ref["Z_bulk"], positions=ref["pos_bulk"], cell=ref["cell_bulk"], pbc=True)
    at.calc = ACECalculator(m2, meta=meta)
    E_mem = at.get_potential_energy()
    at.calc = ACECalculator(str(out))
    assert at.get_potential_energy() == pytest.approx(E_mem, abs=1e-12)


def test_modified_model_matches_pyace(tmp_path):
    pyace = pytest.importorskip("pyace")
    p = _need("sige_zbl")
    model, meta, spec = load(str(p))
    m2 = eqx.tree_at(lambda m: m.ctilde_complex, model, 1.5 * model.ctilde_complex)
    out = tmp_path / "mod.yace"
    write_yace(m2, spec, out)
    ref = np.load(FIX / "sige_zbl_ref.npz")
    at = Atoms(numbers=ref["Z_bulk"], positions=ref["pos_bulk"], cell=ref["cell_bulk"], pbc=True)
    at.calc = pyace.PyACECalculator(str(out))
    E_cpp = at.get_potential_energy()
    at.calc = ACECalculator(m2, meta=meta)
    assert at.get_potential_energy() == pytest.approx(E_cpp, abs=1e-6 * len(at))
```

- [ ] **Step 2: Run and confirm the failure**

Run: `uv run pytest tests/test_pace_export.py -q`
Expected: ImportError on `write_yace`.

- [ ] **Step 3: Implement**

Append to `src/ace_jax/eval/pace_io.py`:
```python
def write_yace(model, spec, path):
    """Serialise a PACEModel: numeric fields from its leaves, everything else
    verbatim from `spec.tree` (function layout is never regenerated)."""
    from .pace_model import PACEModel
    if not isinstance(model, PACEModel):
        raise TypeError("write_yace only writes PACEModel (PACE radials); see docs/pace-yace-spec.md")
    f64 = lambda x: np.asarray(x, np.float64)
    t = copy.deepcopy(spec.tree)
    t["E0"] = f64(model.E0).tolist()
    fs, rcc = f64(model.fs_params), f64(model.rho_core_cut)
    for e in range(len(spec.element_names)):
        em = t["embeddings"][e]
        em["FS_parameters"] = fs[e, :2 * int(em["ndensity"])].tolist()
        em["rho_core_cutoff"], em["drho_core_cutoff"] = rcc[e].tolist()
    crad, rp, core = f64(model.crad), f64(model.radparams), f64(model.core)
    for (i, j), b in t["bonds"].items():
        n, l, k = int(b["nradmax"]), int(b["lmax"]), int(b["nradbasemax"])
        b["radcoefficients"] = crad[i, j, :n, :l + 1, :k].tolist()
        b["radparameters"] = [float(rp[i, j, 0])] + list(b["radparameters"][1:])
        b["rcut"], b["dcut"] = float(rp[i, j, 1]), float(rp[i, j, 2])
        for key, v in (("rcut_in", rp[i, j, 3]), ("dcut_in", rp[i, j, 4]),
                       ("prehc", core[i, j, 0]), ("lambdahc", core[i, j, 1])):
            if key in b or float(v) != _BOND_DEFAULTS[key]:
                b[key] = float(v)
    ct = f64(model.ctilde_complex)
    for el, idx, start, nms, nd in spec.functions:
        t["functions"][el][idx]["ctildes"] = ct[start:start + nms, :nd].reshape(-1).tolist()
    dump_tree(t, path)
```

Export `write_yace` from `eval/__init__.py`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_pace_export.py -q`
Expected: the tree-equality and modified-roundtrip tests pass; the pyace test skips in the ace-jax venv.

If tree equality fails on integer-valued floats (the YAML `1` vs `1.0` in `FS_parameters`), compare with a normalising helper that casts numbers to float on both sides. That's still exact equality of values, and the file stays valid. Put the helper in the test.

Then run the pyace test inside the reference venv:
```bash
uv pip install --python pace_ref/.venv/bin/python -e . pytest
pace_ref/.venv/bin/python -m pytest tests/test_pace_export.py -q -k pyace
```
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/ace_jax/eval/pace_io.py src/ace_jax/eval/__init__.py tests/test_pace_export.py
git commit -m "feat(pace): write_yace; round-trip and modified-model export tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Deployability guard, CI, docs

**Files:**
- Create: `tests/test_pace_hlo.py`, `.github/workflows/pace-parity.yml`
- Modify: `README.md` (short "PACE `.yace` models" section)

**Interfaces:**
- Consumes: `PACEModel.energy_from_positions` (Task 6).

- [ ] **Step 1: Write the failing HLO test**

`tests/test_pace_hlo.py`:
```python
"""lammps-jax resolves FFI custom calls at run time; a PACE model must lower to
stock StableHLO so it needs none."""
import pathlib
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import load

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"


@pytest.mark.parametrize("name", ["si_chebpow_fs", "sige_zbl", "gesi_sbessel"])
def test_exports_without_custom_calls(name):
    p = FIX / f"{name}.yace"
    if not p.exists():
        pytest.skip("fixtures not generated")
    model, _, _ = load(str(p))
    n, E = 8, 40
    f = jax.jit(lambda pos, z, s, r, m: jnp.sum(model.energy_from_positions(pos, z, s, r, m)))
    args = (jnp.zeros((n, 3)), jnp.zeros(n, jnp.int32), jnp.zeros(E, jnp.int32),
            jnp.zeros(E, jnp.int32), jnp.zeros(E, bool))
    text = jax.export.export(f)(*args).mlir_module()
    assert "custom_call" not in text
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_pace_hlo.py -q`
Expected: PASS. (It's a guard, so it should pass at once. If it fails, find the op that lowers to a custom call and replace it with a pure-JAX equivalent.)

- [ ] **Step 3: Add the CI workflow**

`.github/workflows/pace-parity.yml`:
```yaml
name: pace-parity
# Regenerates the PACE references from the ML-PACE C++ and python-ace, then re-runs
# the PACE tests against them.  CI-only; pip users run against committed fixtures.
on:
  push:
    paths:
      - "src/ace_jax/eval/pace_*.py"
      - "pace_ref/**"
      - "tests/test_pace_*.py"
  schedule:
    - cron: "0 6 * * 1"
  workflow_dispatch:
jobs:
  parity:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - name: ML-PACE sources (pinned)
        run: |
          git clone https://github.com/ICAMS/lammps-user-pace.git $HOME/gits/lammps-user-pace
          git -C $HOME/gits/lammps-user-pace checkout 99aa6e6
      - name: Unit reference
        run: |
          cmake -S pace_ref -B pace_ref/build -DCMAKE_BUILD_TYPE=Release
          cmake --build pace_ref/build --target unit_ref -j
          pace_ref/build/unit_ref | uv run python pace_ref/json_to_npz.py
      - name: python-ace fixtures
        run: |
          uv venv pace_ref/.venv --python 3.11
          uv pip install --python pace_ref/.venv/bin/python numpy ase pyyaml pytest "git+https://github.com/ICAMS/python-ace" -e .
          pace_ref/.venv/bin/python pace_ref/make_fixtures.py
      - name: Parity
        env: { JAX_PLATFORM_NAME: cpu, ACEJAX_REQUIRE_FIXTURES: "1" }
        run: |
          uv run pytest -q tests/test_pace_io.py tests/test_pace_radial.py tests/test_pace_build.py tests/test_pace_model.py tests/test_pace_export.py tests/test_pace_hlo.py
          pace_ref/.venv/bin/python -m pytest -q tests/test_pace_export.py -k pyace
```

- [ ] **Step 4: README section**

Add after "Quickstart":
```markdown
### PACE (pacemaker) potentials

`.yace` files load directly and evaluate in JAX (ASE calculator, lammps-jax):

    atoms.calc = ACECalculator("model.yace")

Supported: ChebExpCos / ChebPow / ChebLinear / SBessel radials, FinnisSinclair
and FinnisSinclairShiftedScaled embeddings, `density` / `distance` / `zbl` inner
cutoffs. `write_yace(model, spec, path)` writes a (possibly modified) model back.
Checked against ML-PACE C++ and python-ace; see `docs/pace-yace-spec.md`.
```

- [ ] **Step 5: Full suite, then commit**

Run: `uv run pytest -q -x`
Expected: the whole suite passes (no regressions in the existing tests).

```bash
git add tests/test_pace_hlo.py .github/workflows/pace-parity.yml README.md
git commit -m "ci(pace): stock-HLO guard, pace-parity workflow, README section

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Deployment check on real LAMMPS (manual, with the user)

No code enters the repo except `docs/pace-yace-results.md`. It needs moriarty and the user's 2FA, so the agent prepares and the user runs. Follow the SSH rules in `~/.claude/CLAUDE.md`: plain `ssh moriarty`, one ControlMaster, and no BatchMode.

- [ ] **Step 1:** Rebuild local LAMMPS with ML-PACE (CPU): `cmake -S ~/gits/lammps/cmake -B ~/gits/lammps/build -D PKG_ML-PACE=ON && cmake --build ~/gits/lammps/build -j`. Record which PACE library version it downloaded (the `PACELIB_URL` in the build).
- [ ] **Step 2:** For `sige_zbl` and one real published `.yace`, run a 2-atom-type bulk snapshot through `pair_style pace` and `pair_coeff * * model.yace Si Ge`. Dump E/F and compare to `ACECalculator` (expect agreement at the shipped-grid tolerance). Repeat with a file written by `write_yace` to confirm LAMMPS reads exported files.
- [ ] **Step 3:** On moriarty (already has ML-PACE): the same comparison with `pace/kk` on GPU, then a timing comparison (atoms·steps/s) of `pace/kk` against ace-jax's jitted `energy_forces_virial` on GPU for a ~4k-atom cell. Also record, for the real published model, how the number of AA products after real conversion compares with its number of PACE functions (the spec's AA-union assumption).
- [ ] **Step 4:** Write `docs/pace-yace-results.md` with the numbers and versions. Commit it.

---

## Self-review notes (resolved inline)

- **Spec coverage:** parser and validation (T3); four radial families, three regimes, `Fexp`/`FexpShiftedScaled` (T4); Y_lm map and T (T5); species pooling, embedding, switches, calculator, lammps-jax entry point, and B-basis errors (T6); writer and round-trip (T7); stock HLO, CI and docs (T8); LAMMPS/moriarty validation, benchmarks and the AA-union measurement (T9); reference environment (T1, T2).
- **Deferred by the spec and not planned:** sub-project 2 (export of ACEpotentials/ace-jax models to ML-PACE); pruning zero-ctilde functions; a per-bond-type `crad` contraction optimisation (only if the T9 benchmarks call for it).
- **Type consistency:** `parse_yace` array keys (T3) are exactly the ones `build_basis` (T5) and `load_yace` (T6) read; `PACESpec.functions` tuples (T3) are what `write_yace` (T7) unpacks; `load` returns `(model, meta, spec)` everywhere.
