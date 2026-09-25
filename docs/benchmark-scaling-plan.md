# Benchmark scaling, Phase A: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproducible speed and memory scaling data and plots for ace-jax (`PACEModel`, linear `ACEModel`) vs ML-PACE vs MACE: standalone and in LAMMPS, on SiGe and Cantor at three model sizes, on moriarty CPU, moriarty A4500 and Modal A100.

**Architecture:**
- An ace-jax → lammps-jax exporter (`ace_jax.export.lammps`) makes ace-jax runnable as `pair_style jax/kk`.
- A harness in `bench/scaling/` covers structures, models, one-case runners (standalone and LAMMPS), parity gates, a resumable sweep, and plots.
- Each case runs in its own process and appends one JSON row to `bench/scaling/results/<host>.jsonl`.
- `plot.py` turns the JSONL into `docs/figs/` and `docs/benchmarks.md`.

**Tech stack:**
- Python 3.12, JAX, ASE, matplotlib, pytest.
- lammps-jax (`~/gits/lammps-jax`, editable), pyace (`pace_ref/.venv`) and mace-torch.
- LAMMPS (develop, ≥ 10 Sep 2025) with KOKKOS / ML-PACE / PYTHON, patched with Symmetrix (`pair_style symmetrix/mace`).
- Julia `julia/export_model.jl` for ACE models.
- Modal.

**Spec:** `docs/benchmark-scaling-spec.md`. Read it first; this plan implements its Phase A.

## Global Constraints

- **Branch:** `feat/bench-scaling`, stacked on `feat/pace-forces`, in worktree `.worktrees/bench-scaling`. Don't edit `fit/*`, `construct/*` or `cli.py`: PRs #2–#5 are being merged in parallel.
- **No timing without a passed parity gate** (spec thresholds). A failure is recorded as a row with `"status": "parity_fail"`, never dropped.
- **One case per process.** Every row carries versions and git SHAs.
- **Results are append-only JSONL.** Figures come only from `plot.py`.
- **SSH to moriarty:** one ControlMaster (`-o ControlPath=~/.ssh/cm-%r@%h-%p`), plain `ssh moriarty`, never BatchMode. Check the node is idle (`/proc/loadavg`, `nvidia-smi`) before launching. Long jobs run under `setsid nohup`.
- **Killing remote jobs:** use a bracket pattern, and never in the same command line that also contains the literal name.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Periodic systems in LAMMPS use ghost atoms.** Senders are owned atoms, receivers may be ghosts, and ghost rows carry no edges. The exporter's energy must match `ACECalculator` on periodic cells, not only on clusters. Task 8's parity gate checks this on bulk cells.
2. **Dense regrouping has to cope with edge order.** The packed LAMMPS edge buffer isn't guaranteed to be in sender order, and an atom may have more neighbours than `k_dense`. Regrouping must work on any order and fail loudly (NaN energy) on overflow. Task 1 tests shuffled order and overflow.
3. **LAMMPS type order must follow the model's element order** (lammps-jax maps type t to species t−1; ML-PACE and Symmetrix map by the element names in `pair_coeff`). The data files are written with `specorder` equal to the model's elements, and Task 5 tests this for a reversed-order model (`gesi_sbessel`: Ge, Si).
4. **A run that ends with fewer atoms, or any NaN, is a failure, not a timing** (random-coefficient models). Task 5 parses the final atom count; Task 6 marks the row.
5. **Out-of-memory is data.** The sweep records it and stops increasing N for that line, rather than crashing or retrying. Task 6 tests this with a fake runner.

---

## File structure

| path | responsibility |
|---|---|
| `src/ace_jax/export/__init__.py`, `src/ace_jax/export/lammps.py` | energy function in the lammps-jax contract (sparse / dense), `export_lammps` bundle writer |
| `tests/test_export_lammps.py` | exporter parity vs `ACECalculator`, dense overflow, bundle contents |
| `bench/scaling/structures.py` | deterministic SiGe / Cantor supercells at any N |
| `bench/scaling/models.py` | model manifest; builders for `.yace` (pyace), `.npz` (Julia), MACE-MP-0 |
| `bench/scaling/run_standalone.py` | one standalone case → one JSON row |
| `bench/scaling/run_lammps.py` | one LAMMPS case: data + input + run + parse → one JSON row |
| `bench/scaling/parity.py` | the three parity gates |
| `bench/scaling/sweep.py` | expand the matrix for a host, resume, handle OOM and failures, append JSONL |
| `bench/scaling/plot.py` | JSONL → `docs/figs/scaling_*.png` + tables in `docs/benchmarks.md` |
| `bench/scaling/envs/moriarty.sh`, `bench/scaling/modal_app.py`, `bench/scaling/README.md` | environments and how to reproduce |
| `tests/test_bench_scaling.py` | unit tests for structures, sweep logic, LAMMPS input and log parsing, plotting |

---

### Task 1: ace-jax → lammps-jax exporter

**Files:**
- Create: `src/ace_jax/export/__init__.py`, `src/ace_jax/export/lammps.py`, `tests/test_export_lammps.py`
- Modify: `pyproject.toml` (extra `lammps = ["lammps-jax"]`), `docs/benchmark-scaling-spec.md`

**Interfaces:**
- Produces:
  - `make_energy_fn(model, n_species, layout, k_dense=None) -> Callable[[positions, species, graph], per_atom_energy]`, where `graph` has `.senders`, `.receivers` and `.edge_mask`.
  - `export_lammps(model, meta, path, *, max_atoms, max_edges, k_dense=None, dtype="float64", layout="auto") -> dict`, the bundle, with `bundle["ace_jax"] = {"layout", "elements", "k_dense"}`.

- [ ] **Step 1: Amend the spec (standalone timing definition)**

In `docs/benchmark-scaling-spec.md` → Metrics, replace the `force_s` bullet with:
```
- `call_s`: median ASE-calculator call (energy + forces + stress), including the
  neighbour list, for every code (MACE builds its graph inside the call, so this
  is the like-for-like standalone number); ace-jax also reports `force_s`, the
  jitted model call alone, and `nlist_s`.
```

- [ ] **Step 2: Make lammps-jax importable in the dev venv**

```bash
uv pip install -e ~/gits/lammps-jax
uv run python -c "from lammps_jax.export import export_model, LammpsNeighborList; print('ok')"
```
Add `lammps = ["lammps-jax"]` under `[project.optional-dependencies]` in `pyproject.toml`.

- [ ] **Step 3: Write the failing tests**

`tests/test_export_lammps.py`:
```python
"""ace-jax models as lammps-jax energy functions.

The exported function sees LAMMPS's packed edge buffer: masked padding, edges
in whatever order LAMMPS built them, receivers that may be ghost atoms.  On an
open cluster (no ghosts needed) it must equal ACECalculator exactly, in both
layouts; periodic parity is checked end-to-end in LAMMPS (bench parity gate).
"""
import json
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase import Atoms

from ace_jax.calc.point import ACECalculator
from ace_jax.eval import load, sparse_graph
from ace_jax.export.lammps import make_energy_fn
from conftest import pace_fixture

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"


class Graph:                                    # the lammps-jax graph fields
    def __init__(self, s, r, m):
        self.senders, self.receivers, self.edge_mask = s, r, m


def _cluster(name="gesi_sbessel"):
    ref = np.load(pace_fixture(FIX / f"{name}_ref.npz"))
    at = Atoms(numbers=ref["Z_bulk"], positions=ref["pos_bulk"], cell=ref["cell_bulk"], pbc=False)
    at.center(vacuum=8.0)
    return at


def _lammps_graph(at, rcut, n_pad=37, seed=0):
    g = sparse_graph(at.positions, at.cell.array, at.pbc, rcut)
    order = np.random.default_rng(seed).permutation(len(g.senders))   # LAMMPS order is not ours
    s, r = g.senders[order], g.receivers[order]
    m = np.ones(len(s), bool)
    s = np.concatenate([s, np.zeros(n_pad, int)]); r = np.concatenate([r, np.zeros(n_pad, int)])
    m = np.concatenate([m, np.zeros(n_pad, bool)])
    return Graph(jnp.asarray(s, jnp.int32), jnp.asarray(r, jnp.int32), jnp.asarray(m)), g


@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_energy_fn_matches_calculator(layout):
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    at = _cluster()
    graph, g = _lammps_graph(at, meta["rcut"])
    z2i = {z: i for i, z in enumerate(meta["elements"])}
    species = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    K = int(np.bincount(g.senders, minlength=len(at)).max())
    f = make_energy_fn(model, len(meta["elements"]), layout, k_dense=K + 3)
    pos = jnp.asarray(at.positions)
    E, G = jax.value_and_grad(lambda p: jnp.sum(f(p, species, graph)))(pos)
    at.calc = ACECalculator(y, layout="sparse")
    assert float(E) == pytest.approx(at.get_potential_energy(), abs=1e-10)
    np.testing.assert_allclose(-np.asarray(G), at.get_forces(), atol=1e-9)


def test_dense_overflow_is_loud():
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    at = _cluster()
    graph, _ = _lammps_graph(at, meta["rcut"])
    species = jnp.zeros(len(at), jnp.int32)
    f = make_energy_fn(model, len(meta["elements"]), "dense", k_dense=2)
    assert np.isnan(float(jnp.sum(f(jnp.asarray(at.positions), species, graph))))


def test_bundle_written(tmp_path):
    pytest.importorskip("lammps_jax")
    from ace_jax.export.lammps import export_lammps
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    b = export_lammps(model, meta, tmp_path / "m.json", max_atoms=256, max_edges=256 * 64,
                      k_dense=64, dtype="float64", layout="dense")
    on_disk = json.loads((tmp_path / "m.json").read_text())
    assert on_disk["contract"]["n_species"] == 2
    assert on_disk["ace_jax"] == {"layout": "dense", "elements": [32, 14], "k_dense": 64}
    assert b["ace_jax"]["layout"] == "dense"
```

- [ ] **Step 4: Run and confirm RED**

Run: `uv run pytest tests/test_export_lammps.py -q`
Expected: `ModuleNotFoundError: ace_jax.export`.

- [ ] **Step 5: Implement**

`src/ace_jax/export/__init__.py`:
```python
"""Exporters: ace-jax models to external runtimes."""
```
`src/ace_jax/export/lammps.py`:
```python
"""ace-jax model -> lammps-jax bundle (`pair_style jax/kk`).

lammps-jax calls energy_fn(positions, species, graph) with LAMMPS's packed edge
buffer (senders = owned centre, receivers = neighbour, possibly a ghost;
edge_mask marks real edges) and species = type - 1.  Per-atom energies are
returned; lammps-jax masks ghost rows.  Both EdgeSiteModel layouts work: sparse
uses the buffer as is; dense regroups it into (n, k_dense) slots inside the
exported function, independent of edge order, and returns NaN energies if an
atom has more than k_dense neighbours (never a silent truncation).
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..eval.edge_model import LAYOUTS, estimate_a_bytes


def make_energy_fn(model, n_species, layout, k_dense=None):
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if layout == "dense" and not k_dense:
        raise ValueError("the dense layout needs k_dense (max neighbours per atom)")

    def energy_fn(positions, species, graph):
        n = positions.shape[0]
        m = graph.edge_mask
        s = jnp.where(m, graph.senders, 0)
        r = jnp.where(m, graph.receivers, 0)
        node_z = jnp.clip(species, 0, n_species - 1)
        pad = jnp.asarray([1.0, 0.0, 0.0], positions.dtype) * model.pad_cutoff()
        rij = jnp.where(m[:, None], positions[r] - positions[s], pad)
        if layout == "sparse":
            return model.site_energies(rij, node_z[s], node_z[r], s, n, node_z, m)
        # dense: rank each edge within its centre's group, whatever the order
        key = jnp.where(m, s, n)                              # padding sorts last
        order = jnp.argsort(key, stable=True)
        ks = key[order]
        counts = jax.ops.segment_sum(jnp.ones_like(ks), ks, num_segments=n + 1)
        starts = jnp.cumsum(counts) - counts
        slot_sorted = jnp.arange(ks.shape[0]) - starts[ks]
        slot = jnp.zeros_like(slot_sorted).at[order].set(slot_sorted)
        ok = m & (slot < k_dense)
        row = jnp.where(ok, s, n)                             # out of range: dropped
        col = jnp.where(ok, slot, 0)
        rd = jnp.broadcast_to(pad, (n, k_dense, 3)).at[row, col].set(rij, mode="drop")
        idx = jnp.zeros((n, k_dense), jnp.int32).at[row, col].set(r, mode="drop")
        md = jnp.zeros((n, k_dense), bool).at[row, col].set(True, mode="drop")
        e = model.site_energies_dense(rd, jnp.broadcast_to(node_z[:, None], idx.shape),
                                      node_z[idx], md, node_z)
        overflow = jnp.any(m & (slot >= k_dense))
        return jnp.where(overflow, jnp.nan, e)

    return energy_fn


def export_lammps(model, meta, path, *, max_atoms, max_edges, k_dense=None,
                  dtype="float64", layout="auto"):
    """Write a lammps-jax JSON bundle for `model`; returns the bundle dict.

    layout="auto" picks dense when k_dense is given and estimate_a_bytes for the
    bundle's capacity fits ace_jax.calc.point.dense_budget_bytes(), else sparse.
    """
    from lammps_jax.export import export_model
    from ..calc.point import dense_budget_bytes
    n_species = len(meta["elements"])
    if layout == "auto":
        itemsize = np.dtype(dtype).itemsize
        fits = k_dense and estimate_a_bytes(model, "dense", max_atoms, max_edges, k_dense,
                                            itemsize) <= dense_budget_bytes()
        layout = "dense" if fits else "sparse"
    energy_fn = make_energy_fn(model, n_species, layout, k_dense)
    bundle = export_model(energy_fn=energy_fn, path=path, max_atoms=max_atoms,
                          max_edges=max_edges, cutoff=float(meta["rcut"]), unit_style="metal",
                          precision=dtype, n_species=n_species)
    bundle["ace_jax"] = {"layout": layout, "elements": [int(z) for z in meta["elements"]],
                         "k_dense": int(k_dense) if layout == "dense" else None}
    Path(path).write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_export_lammps.py -q`
Expected: all pass. `test_bundle_written` runs because lammps-jax is installed in Step 2. If `export_model` rejects f64 for this JAX, record the error and ledger a ruling; don't weaken the dtype.

- [ ] **Step 7: Commit**

```bash
git add src/ace_jax/export tests/test_export_lammps.py pyproject.toml uv.lock docs/benchmark-scaling-spec.md
git commit -m "feat(export): ace-jax -> lammps-jax bundle (sparse and dense layouts)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Structures

**Files:**
- Create: `bench/scaling/__init__.py` (empty), `bench/scaling/structures.py`, `tests/test_bench_scaling.py`

**Interfaces:**
- Produces:
  - `SYSTEMS: dict[str, dict]`
  - `n_ladder(system, n_max) -> list[int]`
  - `supercell(system, n_atoms, seed=0) -> ase.Atoms` (orthorhombic, exact composition)

- [ ] **Step 1: Write the failing tests**

`tests/test_bench_scaling.py`:
```python
import collections
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "bench"))
from scaling.structures import SYSTEMS, n_ladder, supercell


@pytest.mark.parametrize("system", ["SiGe", "Cantor"])
def test_supercell_is_deterministic_with_exact_composition(system):
    a, b = supercell(system, 512), supercell(system, 512)
    assert len(a) == 512 and np.array_equal(a.numbers, b.numbers)
    counts = collections.Counter(a.get_chemical_symbols())
    assert set(counts) == set(SYSTEMS[system]["elements"])
    assert max(counts.values()) - min(counts.values()) <= 1
    assert a.pbc.all() and np.allclose(a.cell.angles(), 90)


def test_ladder_powers_of_two():
    assert n_ladder("SiGe", 4096) == [256, 512, 1024, 2048, 4096]
```

- [ ] **Step 2: Run and confirm RED**

Run: `uv run pytest tests/test_bench_scaling.py -q`
Expected: `ModuleNotFoundError: scaling`.

- [ ] **Step 3: Implement**

`bench/scaling/structures.py`:
```python
"""Deterministic benchmark cells: SiGe (random 50/50 diamond) and Cantor
(random equiatomic CrMnFeCoNi fcc), at any power-of-two atom count."""
import itertools

import numpy as np
from ase.build import bulk

SYSTEMS = {
    "SiGe": {"lattice": "diamond", "a": 5.54, "elements": ("Si", "Ge"), "per_cell": 8},
    "Cantor": {"lattice": "fcc", "a": 3.59, "elements": ("Cr", "Mn", "Fe", "Co", "Ni"),
               "per_cell": 4},
}


def n_ladder(system, n_max, n_min=256):
    out, n = [], n_min
    while n <= n_max:
        out.append(n)
        n *= 2
    return out


def _reps(m):
    """Three near-equal integer factors of m (cells along x, y, z)."""
    best = None
    for a in range(1, int(round(m ** (1 / 3))) + 2):
        if m % a:
            continue
        for b in range(a, int(np.sqrt(m // a)) + 2):
            if (m // a) % b:
                continue
            c = m // a // b
            spread = max(a, b, c) - min(a, b, c)
            if best is None or spread < best[0]:
                best = (spread, (a, b, c))
    return best[1]


def supercell(system, n_atoms, seed=0):
    sp = SYSTEMS[system]
    if n_atoms % sp["per_cell"]:
        raise ValueError(f"{system}: n_atoms must be a multiple of {sp['per_cell']}")
    cell = bulk(sp["elements"][0], sp["lattice"], a=sp["a"], cubic=True)
    at = cell.repeat(_reps(n_atoms // sp["per_cell"]))
    k = len(sp["elements"])
    symbols = [sp["elements"][i % k] for i in range(n_atoms)]
    np.random.default_rng(seed).shuffle(symbols)
    at.set_chemical_symbols(symbols)
    return at
```

- [ ] **Step 4: Run and commit**

Run: `uv run pytest tests/test_bench_scaling.py -q`. Expected: pass.
```bash
git add bench/scaling/__init__.py bench/scaling/structures.py tests/test_bench_scaling.py
git commit -m "bench(scaling): deterministic SiGe / Cantor supercells

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Models and manifest

**Files:**
- Create: `bench/scaling/models.py`
- Modify: `tests/test_bench_scaling.py`, `.gitignore` (`bench/scaling/models/`)

**Interfaces:**
- Produces:
  - `MODELS`: an ordered list of dicts `{name, code, system, size, path, elements, n_params}`. Code is `acejax-pace`, `mlpace`, `acejax-ace` or `mace`; size is `small`, `medium` or `large`. The `mlpace` rows share the `.yace` files of `acejax-pace`.
  - `load_manifest() -> list[dict]` reads `bench/scaling/models/manifest.json`.
  - Builders run in their own environments: `python bench/scaling/models.py pace --venv pace_ref/.venv` builds the `.yace` files with pyace; `models.py ace` runs `julia --project=julia julia/export_model.jl` per (system, size); `models.py mace` fetches MACE-MP-0 small / medium / large and extracts a Symmetrix `.json` per (size, system) with `symmetrix_extract_mace <model> --atomic-numbers <Z...>`. The file is specific to that element set.

- [ ] **Step 1: Write the failing test** (manifest schema and size coverage)

Append to `tests/test_bench_scaling.py`:
```python
from scaling.models import SIZES, planned_models


def test_planned_models_cover_the_matrix():
    rows = planned_models()
    for code in ("acejax-pace", "mlpace", "acejax-ace", "mace"):
        for system in ("SiGe", "Cantor"):
            got = sorted(r["size"] for r in rows if r["code"] == code and r["system"] == system)
            assert got == sorted(SIZES), (code, system)
    pace = {(r["system"], r["size"]): r["path"] for r in rows if r["code"] == "acejax-pace"}
    ml = {(r["system"], r["size"]): r["path"] for r in rows if r["code"] == "mlpace"}
    assert pace == ml                                     # the same .yace files
```

- [ ] **Step 2: Implement `planned_models()` and the builders**

`bench/scaling/models.py`:
```python
"""Benchmark models: what is planned, how each is built, and the manifest.

    python bench/scaling/models.py pace   # pyace venv: random-coefficient .yace x3 x2
    python bench/scaling/models.py ace    # Julia: linear ACE .npz x3 x2
    python bench/scaling/models.py mace   # MACE-MP-0 s/m/l + Symmetrix .json per system
Model files live in bench/scaling/models/ (git-ignored); manifest.json records
provenance (builder, parameters, n_params, sha256).
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIR = ROOT / "bench" / "scaling" / "models"
SIZES = ("small", "medium", "large")
ELEMENTS = {"SiGe": ["Si", "Ge"], "Cantor": ["Cr", "Mn", "Fe", "Co", "Ni"]}
PACE_FUNCS = {"small": 100, "medium": 500, "large": 2000}          # per element
# (ACE_ORDER, ACE_TOTALDEGREE) targeting ~100 / 700 / 2800 basis functions per
# element; `models.py ace` records the actual count and warns if off by > 2x
ACE_DEG = {"small": (3, 8), "medium": (3, 12), "large": (4, 12)}
MACE = {"small": "small", "medium": "medium", "large": "large"}    # MACE-MP-0


def planned_models():
    rows = []
    for system, els in ELEMENTS.items():
        for size in SIZES:
            yace = DIR / f"pace_{system}_{size}.yace"
            rows += [dict(name=f"acejax-pace/{system}/{size}", code="acejax-pace", system=system,
                          size=size, path=str(yace), elements=els),
                     dict(name=f"mlpace/{system}/{size}", code="mlpace", system=system,
                          size=size, path=str(yace), elements=els),
                     dict(name=f"acejax-ace/{system}/{size}", code="acejax-ace", system=system,
                          size=size, path=str(DIR / f"ace_{system}_{size}.npz"), elements=els),
                     dict(name=f"mace/{system}/{size}", code="mace", system=system, size=size,
                          path=str(DIR / f"mace_mp0_{size}.model"), elements=els,
                          symmetrix=str(DIR / f"mace_mp0_{size}_{system}.json"))]
    return rows


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def _update_manifest(entries):
    DIR.mkdir(parents=True, exist_ok=True)
    mf = DIR / "manifest.json"
    cur = json.loads(mf.read_text()) if mf.exists() else {}
    cur.update(entries)
    mf.write_text(json.dumps(cur, indent=1, sort_keys=True))


def load_manifest():
    return json.loads((DIR / "manifest.json").read_text())


def build_pace():
    """Run inside pace_ref/.venv (pyace)."""
    import numpy as np
    from pyace import ACEBBasisSet, create_multispecies_basis_config
    out = {}
    for system, els in ELEMENTS.items():
        for size, nf in PACE_FUNCS.items():
            cfg = {"deltaSplineBins": 0.001, "elements": els,
                   "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled",
                                          "fs_parameters": [1, 1, 1, 0.5], "ndensity": 2}},
                   "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [5.25], "rcut": 5.0,
                                     "dcut": 0.01, "inner_cutoff_type": "distance",
                                     "r_in": 1.0, "delta_in": 0.5,
                                     "core-repulsion": [100.0, 5.0]}},
                   "functions": {"number_of_functions_per_element": nf,
                                 "ALL": {"nradmax_by_orders": [15, 6, 4, 3, 2],
                                         "lmax_by_orders": [0, 4, 3, 2, 1]}}}
            bb = ACEBBasisSet(create_multispecies_basis_config(cfg))
            rng = np.random.default_rng(nf + len(els))
            c = np.asarray(bb.all_coeffs)
            bb.all_coeffs = rng.normal(scale=0.05, size=c.shape)
            p = DIR / f"pace_{system}_{size}.yace"
            DIR.mkdir(parents=True, exist_ok=True)
            bb.to_ACECTildeBasisSet().save_yaml(str(p))
            out[str(p)] = {"builder": "pyace", "functions_per_element": nf,
                           "n_params": int(c.size), "sha256": _sha(p)}
    _update_manifest(out)


def build_ace():
    out = {}
    for system, els in ELEMENTS.items():
        for size, (order, deg) in ACE_DEG.items():
            p = DIR / f"ace_{system}_{size}.npz"
            env = {**os.environ, "ACE_ELEMENTS": ",".join(els), "ACE_ORDER": str(order),
                   "ACE_TOTALDEGREE": str(deg), "ACE_RCUT": "5.0"}
            subprocess.run(["julia", "--project=julia", "julia/export_model.jl", str(p), "ace1"],
                           cwd=ROOT, env=env, check=True)
            import numpy as np
            n_b = int(json.loads(bytes(np.load(p)["meta_json"]).decode())["n_B"])
            out[str(p)] = {"builder": "julia/export_model.jl", "order": order,
                           "totaldegree": deg, "n_params": n_b, "sha256": _sha(p)}
    _update_manifest(out)


def build_mace():
    """Run inside the MACE venv (mace-torch + symmetrix): download MACE-MP-0 and
    extract one Symmetrix .json per (size, system) -- each is element-specific."""
    import shutil
    from ase.data import atomic_numbers
    from mace.calculators.foundations_models import download_mace_mp_checkpoint
    out = {}
    DIR.mkdir(parents=True, exist_ok=True)
    for size, tag in MACE.items():
        src = download_mace_mp_checkpoint(tag)
        p = DIR / f"mace_mp0_{size}.model"
        p.write_bytes(pathlib.Path(src).read_bytes())
        out[str(p)] = {"builder": "mace-mp-0", "tag": tag, "sha256": _sha(p)}
        for system, els in ELEMENTS.items():
            zs = sorted(atomic_numbers[e] for e in els)
            subprocess.run([shutil.which("symmetrix_extract_mace"), str(p), "--atomic-numbers",
                            *map(str, zs)], cwd=DIR, check=True)
            made = DIR / f"mace_mp0_{size}-{'-'.join(map(str, zs))}.json"
            dst = DIR / f"mace_mp0_{size}_{system}.json"
            made.replace(dst)
            out[str(dst)] = {"builder": "symmetrix_extract_mace", "from": str(p),
                             "atomic_numbers": zs, "sha256": _sha(dst)}
    _update_manifest(out)


if __name__ == "__main__":
    {"pace": build_pace, "ace": build_ace, "mace": build_mace}[sys.argv[1]]()
```

Before relying on it, check against the installed packages:
- `download_mace_mp_checkpoint`;
- `symmetrix_extract_mace -h`, including the output file name. Its README gives `my-mace.model --atomic-numbers 1 8` → `my-mace-1-8.json`.

If either differs, adapt only `build_mace` and record a ruling.

- [ ] **Step 3: Run the test; build the models; commit**

Run: `uv run pytest tests/test_bench_scaling.py -q` → pass. Then build locally:
```bash
pace_ref/.venv/bin/python bench/scaling/models.py pace
python bench/scaling/models.py ace            # needs Julia + the julia/ env
```
For `ace`: if the recorded `n_params` (n_B per element) is off the ~100 / 700 / 2800 targets by more than 2×, change `ACE_DEG` for that size, rebuild, and record the final values in the commit message. MACE is built on the target machines (Task 8).
```bash
printf 'bench/scaling/models/\n' >> .gitignore
git add bench/scaling/models.py tests/test_bench_scaling.py .gitignore
git commit -m "bench(scaling): model plan, builders and manifest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Standalone runner

**Files:**
- Create: `bench/scaling/run_standalone.py`
- Modify: `tests/test_bench_scaling.py`

**Interfaces:**
- Consumes: `supercell` (Task 2); a manifest row (Task 3).
- Produces: `run_case(model_row, n_atoms, dtype, device, reps=10) -> dict`, one row with the keys `code, mode="standalone", model, size, system, n_atoms, n_edges, device, dtype, call_s, force_s, nlist_s, compile_s, peak_bytes, layout, edge_a_kind, status, versions`. The CLI prints the row as one JSON line.

- [ ] **Step 1: Write the failing test** (ace-jax path on a PACE fixture, CPU, tiny cell)

Append:
```python
def test_standalone_row_for_acejax(tmp_path):
    from scaling.run_standalone import run_case
    row = {"code": "acejax-pace", "system": "SiGe", "size": "small",
           "path": str(pathlib.Path(__file__).parent.parent / "fixtures" / "pace" / "gesi_sbessel.yace"),
           "elements": ["Si", "Ge"], "name": "acejax-pace/SiGe/small"}
    out = run_case(row, 256, "float64", "cpu", reps=2)
    assert out["status"] == "ok" and out["n_atoms"] == 256 and out["layout"] in ("dense", "sparse")
    for k in ("call_s", "force_s", "nlist_s", "compile_s"):
        assert out[k] > 0
```

- [ ] **Step 2: Implement**

`bench/scaling/run_standalone.py`:
```python
"""One standalone benchmark case -> one JSON row (stdout).

    python bench/scaling/run_standalone.py <model-name> <n_atoms> <dtype> <device>
call_s: median ASE calculator call (energy+forces+stress) incl. neighbour list,
for every code.  ace-jax also: force_s (jitted model call alone), nlist_s.
"""
import json
import platform
import statistics
import sys
import time

from scaling.structures import supercell


def _versions():
    import importlib.metadata as md
    out = {"python": platform.python_version()}
    for pkg in ("ace-jax", "jax", "jaxlib", "mace-torch", "torch", "ase"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return out


def _median_time(f, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _peak(device):
    if device == "gpu":
        try:
            import jax
            return jax.devices()[0].memory_stats().get("peak_bytes_in_use")
        except Exception:
            import torch
            return torch.cuda.max_memory_allocated()
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def run_case(row, n_atoms, dtype, device, reps=10):
    from ase.calculators.calculator import all_changes
    at = supercell(row["system"], n_atoms)
    out = {"code": row["code"], "mode": "standalone", "model": row["name"], "size": row["size"],
           "system": row["system"], "n_atoms": n_atoms, "device": device, "dtype": dtype,
           "versions": _versions(), "status": "ok"}
    try:
        if row["code"].startswith("acejax"):
            import jax
            jax.config.update("jax_enable_x64", dtype == "float64")
            import jax.numpy as jnp
            from ace_jax.calc.point import ACECalculator
            from ace_jax.eval import sparse_graph
            calc = ACECalculator(row["path"], dtype=getattr(jnp, dtype))
            call = lambda: calc.calculate(at, ["energy", "forces", "stress"], all_changes)
            t0 = time.perf_counter(); call(); out["compile_s"] = time.perf_counter() - t0
            out["call_s"] = _median_time(call, reps)
            out["nlist_s"] = _median_time(
                lambda: sparse_graph(at.positions, at.cell.array, at.pbc, calc.cutoff), reps)
            out["force_s"] = max(out["call_s"] - out["nlist_s"], 1e-9)
            out["layout"], out["edge_a_kind"] = calc.last_layout, calc.last_edge_a_kind
            out["n_edges"] = int(len(sparse_graph(at.positions, at.cell.array, at.pbc,
                                                  calc.cutoff).senders))
        elif row["code"] == "mace":
            import torch
            from mace.calculators import mace_mp
            calc = mace_mp(model=row["path"], default_dtype=dtype,
                           device="cuda" if device == "gpu" else "cpu",
                           enable_cueq=(device == "gpu"))
            call = lambda: calc.calculate(at, ["energy", "forces", "stress"], all_changes)
            t0 = time.perf_counter(); call(); out["compile_s"] = time.perf_counter() - t0
            out["call_s"] = _median_time(call, reps)
            if device == "gpu":
                torch.cuda.synchronize()
        else:
            raise ValueError(f"no standalone runner for {row['code']}")
    except Exception as ex:                                        # OOM etc. are data
        msg = repr(ex)
        out["status"] = "oom" if ("RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()) else "error"
        out["error"] = msg[:300]
    out["peak_bytes"] = _peak(device)
    return out


if __name__ == "__main__":
    from scaling.models import load_manifest, planned_models
    name, n, dtype, device = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    row = next(r for r in planned_models() if r["name"] == name)
    print(json.dumps(run_case(row, n, dtype, device)))
```

For ace-jax, `force_s = call_s − nlist_s` approximates the model's own time. The note in the header says so, and `plot.py` labels it that way.

- [ ] **Step 3: Run the test and commit**

Run: `uv run pytest tests/test_bench_scaling.py -q -k standalone` → pass.
```bash
git add bench/scaling/run_standalone.py tests/test_bench_scaling.py
git commit -m "bench(scaling): standalone one-case runner

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: LAMMPS runner

**Files:**
- Create: `bench/scaling/run_lammps.py`
- Modify: `tests/test_bench_scaling.py`

**Interfaces:**
- Produces:
  - `lammps_input(style, model_path, elements, data_path, device, run_steps, dump=None) -> str`
  - `parse_log(text) -> {"step_s", "loop_s", "n_atoms_end", "procs", "nan"}`
  - `run_case(row, n_atoms, dtype, device, lmp, ranks, workdir) -> dict`, with the keys of Task 4 plus `mode="lammps", step_s, atom_steps_per_s, ranks`.
- Styles:
  - `acejax`: `pair_style jax/kk ${pjrt}` / `pair_coeff * * <bundle.json>`, with Kokkos `newton off neigh full`.
  - `mlpace`: `pair_style pace` on CPU; `pair_style pace product` with Kokkos `newton on neigh half` on GPU.
  - `mace`: `pair_style symmetrix/mace` / `pair_coeff * * <mace_mp0_<size>_<system>.json> <elements>`, with `-k on g 1 -sf kk` on GPU. The Kokkos neighbour settings are taken from Symmetrix's own examples and confirmed by the T8 parity gate.

- [ ] **Step 1: Write the failing tests** (input text and log parsing; no LAMMPS needed)

Append:
```python
from scaling.run_lammps import lammps_input, parse_log

LOG = """Step PotEng
       0   -100.0
Loop time of 1.5 on 4 procs for 50 steps with 512 atoms
Step PotEng
      50   -99.9
Loop time of 6.0 on 4 procs for 200 steps with 512 atoms
"""


def test_parse_log_uses_the_timed_segment():
    p = parse_log(LOG)
    assert p["step_s"] == pytest.approx(0.03) and p["n_atoms_end"] == 512 and p["procs"] == 4
    assert p["nan"] is False


def test_parse_log_flags_nan():
    assert parse_log(LOG.replace("-99.9", "nan"))["nan"] is True


@pytest.mark.parametrize("style,device,expect", [
    ("mlpace", "cpu", "pair_style pace\n"), ("mlpace", "gpu", "pair_style pace product"),
    ("acejax", "gpu", "pair_style jax/kk ${pjrt}"), ("mace", "gpu", "pair_style symmetrix/mace")])
def test_lammps_input_pair_lines(style, device, expect):
    txt = lammps_input(style, "/m/model.yace", ["Ge", "Si"], "/d/x.data", device, 200)
    assert expect in txt
    if style in ("mlpace", "mace"):
        assert "pair_coeff * * " in txt and txt.strip().split("pair_coeff * * ")[1].split("\n")[0].endswith("Ge Si")
    assert "run 50" in txt and "run 200" in txt
```

- [ ] **Step 2: Implement**

`bench/scaling/run_lammps.py`:
```python
"""One LAMMPS benchmark case -> one JSON row.

Writes the data file with species in the model's element order (lammps-jax
maps type t -> species t-1; ML-PACE / Symmetrix map by pair_coeff names), runs a
50-step warm-up and a timed segment, and parses the timed "Loop time".  Runs
that lose atoms or produce NaN are failures, never timings.
"""
import json
import os
import pathlib
import re
import subprocess
import sys

from ase.io import write

from scaling.structures import supercell

KOKKOS = {"acejax": "newton off neigh full", "mlpace": "newton on neigh half",
          "mace": "newton on neigh half"}


def _pair(style, model_path, elements, device):
    els = " ".join(elements)
    if style == "acejax":
        return f"pair_style jax/kk ${{pjrt}}\npair_coeff * * {model_path}\n"
    if style == "mlpace":
        return (f"pair_style pace product\n" if device == "gpu" else "pair_style pace\n") + \
               f"pair_coeff * * {model_path} {els}\n"
    if style == "mace":                     # Symmetrix: element-specific .json
        return f"pair_style symmetrix/mace\npair_coeff * * {model_path} {els}\n"
    raise ValueError(style)


def lammps_input(style, model_path, elements, data_path, device, run_steps, dump=None):
    txt = ("units metal\natom_style atomic\nboundary p p p\natom_modify map yes\n"
           f"read_data {data_path}\n" + _pair(style, model_path, elements, device) +
           "neighbor 1.0 bin\nneigh_modify every 1 delay 0 check yes\n"
           "timestep 0.0001\nfix 1 all nve\nthermo_style custom step pe atoms\nthermo 50\n")
    if dump:
        txt += f"dump d all custom 1 {dump} id fx fy fz\ndump_modify d sort id\nrun 0\nundump d\n"
    return txt + f"run 50\nrun {run_steps}\n"


_LOOP = re.compile(r"Loop time of ([\d.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms")


def parse_log(text):
    loops = _LOOP.findall(text)
    loop_s, procs, steps, natoms = loops[-1]
    return {"loop_s": float(loop_s), "step_s": float(loop_s) / int(steps), "procs": int(procs),
            "n_atoms_end": int(natoms), "nan": bool(re.search(r"\bnan\b", text, re.I))}


def run_case(row, n_atoms, dtype, device, lmp, ranks, workdir, pjrt=None):
    work = pathlib.Path(workdir); work.mkdir(parents=True, exist_ok=True)
    at = supercell(row["system"], n_atoms)
    data = work / "x.data"
    write(data, at, format="lammps-data", specorder=row["elements"], masses=True)
    style = {"acejax-pace": "acejax", "acejax-ace": "acejax", "mlpace": "mlpace", "mace": "mace"}[row["code"]]
    model = {"acejax": row.get("bundle"), "mace": row.get("symmetrix")}.get(style) or row["path"]
    (work / "in.bench").write_text(lammps_input(style, model, row["elements"], data, device, 200))
    cmd = [lmp, "-in", "in.bench", "-log", "log.lammps", "-nocite"]
    if device == "gpu":
        cmd += ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", *KOKKOS[style].split()]
    elif ranks > 1 and style != "acejax":
        cmd = ["mpirun", "-np", str(ranks)] + cmd
    if pjrt:
        cmd += ["-var", "pjrt", pjrt]
    out = {"code": row["code"], "mode": "lammps", "model": row["name"], "size": row["size"],
           "system": row["system"], "n_atoms": n_atoms, "device": device, "dtype": dtype,
           "ranks": ranks, "status": "ok"}
    p = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=3600)
    log = (work / "log.lammps").read_text() if (work / "log.lammps").exists() else p.stdout
    if p.returncode != 0 or "Loop time" not in log:
        tail = (p.stderr or p.stdout)[-300:]
        out["status"] = "oom" if "out of memory" in tail.lower() or "RESOURCE_EXHAUSTED" in tail else "error"
        out["error"] = tail
        return out
    parsed = parse_log(log)
    out.update(parsed)
    if parsed["nan"] or parsed["n_atoms_end"] != n_atoms:
        out["status"] = "unstable"
    out["atom_steps_per_s"] = n_atoms / parsed["step_s"]
    return out


if __name__ == "__main__":
    from scaling.models import planned_models
    name, n, dtype, device, lmp, ranks, workdir = sys.argv[1:8]
    row = next(r for r in planned_models() if r["name"] == name)
    row["bundle"] = os.environ.get("ACEJAX_BUNDLE", "")
    print(json.dumps(run_case(row, int(n), dtype, device, lmp, int(ranks), workdir,
                              os.environ.get("PJRT_PLUGIN"))))
```

For `acejax`, the sweep (Task 6) exports the bundle per (model, capacity) before the run, using `export_lammps` with `max_atoms` ≈ 1.6 × n_atoms (owned plus ghost atoms, checked in the parity gate) and `k_dense` = the largest neighbour count + 8.

- [ ] **Step 3: Run the tests and commit**

Run: `uv run pytest tests/test_bench_scaling.py -q -k "lammps or parse"` → pass.
```bash
git add bench/scaling/run_lammps.py tests/test_bench_scaling.py
git commit -m "bench(scaling): LAMMPS one-case runner (jax/kk, pace[/kk], symmetrix/mace)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Parity gates and the sweep

**Files:**
- Create: `bench/scaling/parity.py`, `bench/scaling/sweep.py`
- Modify: `tests/test_bench_scaling.py`

**Interfaces:**
- Produces:
  - `HOSTS` (the matrix per host)
  - `cases(host) -> list[Case]`
  - `run_sweep(host, runner, results_path)`: resumable. It skips cases whose key (model, mode, n, dtype, device) is already in the results, and after an `oom`, `error` or `unstable` row it skips larger N on that line.
  - `parity.gate(host, env) -> list[dict]`: rows with `status` `parity_ok` or `parity_fail`, plus the measured differences.

- [ ] **Step 1: Write the failing tests** (sweep logic with a fake runner)

Append:
```python
from scaling.sweep import HOSTS, cases, run_sweep


def test_cases_cover_host_matrix():
    cs = cases("moriarty-gpu")
    assert {c.dtype for c in cs} == {"float32", "float64"}
    assert all(not (c.code == "mlpace" and c.dtype == "float32") for c in cs)   # ML-PACE is f64 only
    assert {c.mode for c in cs} == {"standalone", "lammps"}


def test_sweep_resumes_and_stops_after_oom(tmp_path):
    calls = []

    def fake(case):
        calls.append(case)
        return {"status": "oom" if case.n_atoms >= 1024 else "ok"}
    res = tmp_path / "r.jsonl"
    sel = lambda c: c.model == "acejax-pace/SiGe/small" and c.mode == "standalone" and c.dtype == "float64"
    run_sweep("moriarty-gpu", fake, res, select=sel)
    ns = [c.n_atoms for c in calls]
    assert ns == [256, 512, 1024]                      # stops after the first OOM
    run_sweep("moriarty-gpu", fake, res, select=sel)   # resume: nothing new to run
    assert [c.n_atoms for c in calls] == ns
    rows = [json.loads(l) for l in res.read_text().splitlines()]
    assert [r["status"] for r in rows] == ["ok", "ok", "oom"]
```

- [ ] **Step 2: Implement `sweep.py`**

`bench/scaling/sweep.py`:
```python
"""Expand the benchmark matrix for a host and run it resumably.

    python bench/scaling/sweep.py moriarty-gpu [--only acejax-pace] [--dry-run]
"""
import dataclasses
import json
import pathlib
import subprocess
import sys

from scaling.models import SIZES, planned_models
from scaling.structures import n_ladder

HOSTS = {
    "moriarty-cpu": {"device": "cpu", "n_max": 32768, "ranks": 32},
    "moriarty-gpu": {"device": "gpu", "n_max": 1 << 20, "ranks": 1},
    "modal-a100": {"device": "gpu", "n_max": 1 << 21, "ranks": 1},
    "local-cpu": {"device": "cpu", "n_max": 8192, "ranks": 8},
}
MODES = {"acejax-pace": ("standalone", "lammps"), "acejax-ace": ("standalone", "lammps"),
         "mlpace": ("lammps",), "mace": ("standalone", "lammps")}
DTYPES = {"acejax-pace": ("float64", "float32"), "acejax-ace": ("float64", "float32"),
          "mlpace": ("float64",), "mace": ("float64", "float32")}


@dataclasses.dataclass(frozen=True)
class Case:
    code: str
    model: str
    mode: str
    n_atoms: int
    dtype: str
    device: str
    ranks: int

    def key(self):
        return (self.model, self.mode, self.n_atoms, self.dtype, self.device)


def cases(host):
    h = HOSTS[host]
    out = []
    for m in planned_models():
        for mode in MODES[m["code"]]:
            for dtype in DTYPES[m["code"]]:
                for n in n_ladder(m["system"], h["n_max"]):
                    out.append(Case(m["code"], m["name"], mode, n, dtype, h["device"], h["ranks"]))
    return out


def _line(c):
    return (c.model, c.mode, c.dtype, c.device)


def run_sweep(host, runner, results_path, select=lambda c: True):
    results_path = pathlib.Path(results_path)
    done, dead = set(), set()
    if results_path.exists():
        for l in results_path.read_text().splitlines():
            r = json.loads(l)
            done.add(tuple(r["_key"]))
            if r["status"] in ("oom", "error", "unstable", "parity_fail"):
                dead.add(tuple(r["_line"]))
    todo = sorted((c for c in cases(host) if select(c)), key=lambda c: (_line(c), c.n_atoms))
    for c in todo:
        if c.key() in done or _line(c) in dead:
            continue
        row = runner(c)
        row["_key"], row["_line"], row["host"] = list(c.key()), list(_line(c)), host
        with results_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if row["status"] != "ok":
            dead.add(_line(c))


def subprocess_runner(host, env):
    """Real runner: one case per fresh process (so peak memory is per case)."""
    here = pathlib.Path(__file__).parent

    def run(c):
        if c.mode == "standalone":
            cmd = [sys.executable, str(here / "run_standalone.py"), c.model, str(c.n_atoms),
                   c.dtype, c.device]
        else:
            cmd = [sys.executable, str(here / "run_lammps.py"), c.model, str(c.n_atoms), c.dtype,
                   c.device, env["lmp"], str(c.ranks), f"/tmp/bench_{host}_{c.n_atoms}"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                           env={**env.get("os_env", {}), "PYTHONPATH": env["pythonpath"]})
        lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
        return json.loads(lines[-1]) if lines else {"status": "error", "error": p.stderr[-300:]}
    return run
```

The CLI (`__main__`) reads host settings (the `lmp` path, PJRT plugin, per-code virtualenv interpreters) from `bench/scaling/envs/<host>.json`. That file is written in Task 8 or 9. Before running the sweep, the CLI calls `parity.gate(host, env)` and aborts any code whose gate failed, writing the gate rows first.

- [ ] **Step 3: Implement `parity.py`** (three gates, on 256-atom cells)

`bench/scaling/parity.py`:
```python
"""Parity gates (spec thresholds), run per host before any timing.

- acejax-pace vs mlpace (same .yace): |dE|/atom <= 1e-6, |dF| <= 1e-5
- acejax standalone vs acejax in LAMMPS (f64): |dE|/atom <= 1e-10, |dF| <= 1e-9
- mace standalone vs mace in LAMMPS: |dE|/atom <= 1e-6
Each LAMMPS side uses `run 0` + a sorted force dump.
"""
```
Implement `gate(host, env)` with `run_lammps.lammps_input(..., dump=...)`, reading the dump with `ase.io.read(format="lammps-dump-text")`. Return one row per (gate, system, size = small) with `status` and the measured maxima. Test it by running it on the host in Task 8.

- [ ] **Step 4: Run the tests and commit**

Run: `uv run pytest tests/test_bench_scaling.py -q` → pass.
```bash
git add bench/scaling/sweep.py bench/scaling/parity.py tests/test_bench_scaling.py
git commit -m "bench(scaling): resumable sweep (OOM stops a line) and parity gates

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Plots

**Files:**
- Create: `bench/scaling/plot.py`
- Modify: `tests/test_bench_scaling.py`

Load the `dataviz` skill before writing `plot.py`. It sets the palette, marks and accessibility rules; follow it for all figures.

**Interfaces:**
- Produces: `make_figures(results_glob, outdir) -> list[path]`, the five figure kinds from the spec (throughput vs N; throughput vs model size; memory; f32 vs f64; a compile table as Markdown), plus `write_doc(figures, tables, "docs/benchmarks.md")`.

- [ ] **Step 1: Write the failing test** (synthetic JSONL → files exist)

Append:
```python
def test_plots_from_synthetic_results(tmp_path):
    from scaling.plot import make_figures
    rows = []
    for code in ("acejax-pace", "mlpace", "mace"):
        for n in (256, 512, 1024):
            rows.append({"code": code, "mode": "lammps", "model": f"{code}/SiGe/small", "size": "small",
                         "system": "SiGe", "n_atoms": n, "device": "gpu", "dtype": "float64",
                         "host": "moriarty-gpu", "status": "ok", "step_s": n * 1e-6,
                         "atom_steps_per_s": 1e6, "peak_bytes": n * 1e5})
    (tmp_path / "r.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    figs = make_figures(str(tmp_path / "*.jsonl"), tmp_path / "figs")
    assert figs and all(pathlib.Path(f).exists() for f in figs)
```

- [ ] **Step 2: Implement `plot.py`** following the dataviz skill

The first figure kind: one panel per (system × host), log-log atom-steps/s vs `n_atoms`. One colour per code family and one marker per model size; solid lines for standalone and dashed for LAMMPS; failed or OOM points marked at the last size that ran. The other figures follow the spec's list.

`write_doc` writes `docs/benchmarks.md`: the figures, the compile-time table, the hardware and version table from the rows' `versions`, and the caveats (random-coefficient models; how `force_s` is defined).

- [ ] **Step 3: Run the test and commit**

Run: `uv run pytest tests/test_bench_scaling.py -q -k plots` → pass.
```bash
git add bench/scaling/plot.py tests/test_bench_scaling.py
git commit -m "bench(scaling): figures and docs/benchmarks.md from results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: moriarty environment and parity (manual, scripted)

**Files:**
- Create: `bench/scaling/envs/moriarty.sh`, `bench/scaling/envs/moriarty-cpu.json`, `bench/scaling/envs/moriarty-gpu.json`, `bench/scaling/README.md`

- [ ] **Step 1: Write `envs/moriarty.sh`.** It is idempotent and installs into `/home/eng/essswb/bench-scaling/` (shared home):
  1. Clone LAMMPS `develop` (≥ 10 Sep 2025) and `wcwitt/symmetrix` (`--recursive`), then patch the tree with `symmetrix/pair_symmetrix/install.sh <lammps>`. Configure with:
     - `PKG_KOKKOS` (CUDA + `Kokkos_ARCH_AMPERE86` + OpenMP, `CMAKE_CXX_COMPILER=lib/kokkos/bin/nvcc_wrapper`)
     - `PKG_ML-PACE`, `PKG_PYTHON`
     - `SYMMETRIX_KOKKOS=ON`, `SYMMETRIX_SPHERICART_CUDA=ON`
     - `CMAKE_CXX_STANDARD=20`, `BUILD_SHARED_LIBS=ON`, MPI on

     First check that the prerequisites are available: `module avail` for CMake ≥ 3.27, GCC ≥ 11 and `CUDA/12.2.0`. If they aren't, build Symmetrix on Modal only (spec risk) and record that in the README. Build, then `install-python`.
  2. Build the lammps-jax plugin against that tree for GPU PJRT (README recipe), and for CPU PJRT if lammps-jax supports it.
  3. Create a venv (Python 3.12) with jax[cuda12], the `ace-jax` checkout (editable), lammps-jax (editable), mace-torch, cuequivariance-torch, symmetrix (`pip install ./symmetrix/symmetrix`, for `symmetrix_extract_mace`) and matplotlib.
- [ ] **Step 2: Run it** under `setsid nohup`, waiting with one remote `while pgrep` loop. Verify:
  - `lmp -h | grep -E "pace|symmetrix|jax"` lists `pace`, `pace/kk`, `symmetrix/mace` (and its `/kk` variant) and `jax/kk`;
  - `python -c "import lammps, jax, mace; print(jax.devices())"` shows the GPU.

  If lammps-jax won't build, stop that part and record it in `README.md`; the ace-jax LAMMPS-mode rows are then deferred, per the spec.
- [ ] **Step 3: Build the MACE models there:** `python bench/scaling/models.py mace`. Copy the `.yace` and `.npz` models built in Task 3 (rsync).
- [ ] **Step 4: Write `envs/moriarty-{cpu,gpu}.json`** (`lmp`, `pjrt`, `pythonpath`, venv interpreters) and run the parity gates:
  - `python bench/scaling/sweep.py moriarty-gpu --parity-only`
  - `python bench/scaling/sweep.py moriarty-cpu --parity-only`

  Expected: every gate `parity_ok`. A failure is debugged with superpowers:systematic-debugging, never skipped.
- [ ] **Step 5: Commit** the env scripts, JSON files, README and parity rows (`bench/scaling/results/moriarty-*.jsonl`).

---

### Task 9: Modal A100 environment and parity

**Files:**
- Create: `bench/scaling/modal_app.py`, `bench/scaling/envs/modal-a100.json`

- [ ] **Step 1: Write `modal_app.py`.** The same build as `envs/moriarty.sh`, as image layers: CUDA 12.4 devel base, LAMMPS develop with `Kokkos_ARCH_AMPERE80` and the stub `libcuda` link flags from `bench/pace_modal/run.py`, the lammps-jax plugin, and the Python venv. The ace-jax source and models are mounted from the checkout.
  - `sweep(host="modal-a100", only=None)` runs the parity gates, then `run_sweep`, and returns the JSONL rows.
  - The local entrypoint appends them to `bench/scaling/results/modal-a100.jsonl`.
  - GPU `A100-80GB`, timeout 24 h.
- [ ] **Step 2: Run the parity gates only:** `modal run bench/scaling/modal_app.py --parity-only`. Expected: all `parity_ok`.
- [ ] **Step 3: Commit.**

---

### Task 10: Sweeps

- [ ] **Step 1: moriarty GPU:** after checking the node is idle, run `python bench/scaling/sweep.py moriarty-gpu` under `setsid nohup`. Watch with a Monitor on the results file for `"status": "(error|unstable|parity_fail)"` and for progress.
- [ ] **Step 2: moriarty CPU:** run `sweep.py moriarty-cpu` once the GPU sweep has finished, so the two don't compete for CPU cores during timings.
- [ ] **Step 3: Modal A100:** `modal run bench/scaling/modal_app.py` (one image, one sweep).
- [ ] **Step 4: Spot-check each file:** no `error` rows without an explanation, and OOM only at the large end. Commit `bench/scaling/results/*.jsonl` with a summary of the counts by status in the message.

---

### Task 11: Figures and docs

- [ ] **Step 1:** `python bench/scaling/plot.py 'bench/scaling/results/*.jsonl' docs/figs` → figures plus `docs/benchmarks.md`.
- [ ] **Step 2:** Read every figure and check it against the rows (spot-check three points per figure by hand). Fix plot bugs in `plot.py`, never in the data.
- [ ] **Step 3:** Link `docs/benchmarks.md` from the README ("Performance"), and note that Phase B (the production ACE model) will add a line.
- [ ] **Step 4:** Commit, push `feat/bench-scaling`, and open the PR, stacked on #7 or on `main` if #7 has merged by then.

---

## Self-review notes

- **Spec coverage:**
  - exporter (T1); structures (T2); models (T3);
  - standalone and LAMMPS runners (T4, T5); parity gates and sweep (T6); plots (T7);
  - environments on moriarty and Modal (T8, T9); sweeps (T10); docs (T11).
  - Phase B is deliberately absent.
- **Spec amendment:** T1 Step 1 redefines standalone timing as the end-to-end calculator call (`call_s`), because MACE builds its graph inside the call. ace-jax's `force_s` is kept as a secondary number.
- **External names to verify in their tasks:**
  - `mace_mp(model=path)`, `download_mace_mp_checkpoint` and `symmetrix_extract_mace` (T3/T4);
  - the lammps-jax CPU PJRT (T8);
  - Symmetrix's Kokkos neighbour settings (T5, confirmed by the T8 parity gate).

  Each step says what to do if a name differs.
