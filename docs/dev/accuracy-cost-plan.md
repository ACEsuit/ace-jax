# Accuracy-matched speed benchmark (Si) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit linear ACE (analytic and learned radials), ACE+GP, PACE, MACE and GAP to one Si train split, score
them all on one test split, time them on lestrade, and plot test force RMSE
against throughput with a "smallest model at target" table.

**Architecture:** A self-contained `bench/accuracy_cost/` package:
- Shared plumbing: `common.py`, `split.py`, `weights.py` and `evaluate.py`.
- One fit driver per family, each writing `runs/<family>/<size>/` on /storage
  with a resumable `status.json` protocol.
- `timing.py`, which reuses `bench/scaling`'s `run_case` functions. That harness
  gains a Si system, a rattle option, a GP standalone branch and a `pair quip`
  LAMMPS style.
- `plot.py`, which turns `results/*.jsonl` into the figures.

**Tech Stack:**
- Python 3.12: ace-jax, numpy, matplotlib, pyyaml and libAtoms `extxyz`.
- Separate uv venvs on /storage: pacemaker (TensorFlow), mace-torch plus
  symmetrix, and quippy-ase (which ships `gap_fit`).
- LAMMPS with KOKKOS (Ada sm_89), ML-PACE, Symmetrix, ML-QUIP and the
  lammps-jax plugin.

**Spec:** `docs/dev/accuracy-cost-spec.md`. Executors read both documents.

## Global Constraints

- **Cutoff:** 5.0 Å for every family (`RCUT = 5.0`).
- **Data source:** `libAtoms/silicon-testing-framework` at commit
  `ec91b17fbbcd6c4a06a257acbfcba77734c0080c`, file
  `models/GAP/gp_iter6_sparse9k.xml.xyz`, sha256
  `74ac5f387aa0149ec9ea8f3e275d1087d7287db68c15ba8bc4c28a450de67e2a`:
  2,475 frames in 27 `config_type`s.
- **Split:** 90/10 stratified by `config_type`, seed 0. Types with fewer than 10
  frames go entirely to train.
- **Virials are kept wherever a code can fit them** (user decision 2026-10-04).
  `split.py` keeps `virial` on the 1,783 frames that have one. ace-jax and GAP
  fit E+F+V with GAP-18's virial sigmas; MACE fits virials with its own global
  weight; pacemaker fits E+F only, unless the installed version supports stress
  fitting (checked in Task 6). The write-up states which families saw virials.
- **Weights:** come only from GAP-18's own `default_sigma` / `config_type_sigma`
  (parsed from `gp_iter6_sparse9k.xml`), converted per code by `weights.py`.
- **E0:** the `isolated_atom` energy, -158.54496821 eV, with the frame always
  in train.
- **Learned radials** (user request 2026-10-04): family `ace_lr` is linear ACE
  fitted with `aj fit --learn-radial` (the CLI's default radial-learning
  settings) on the same `(order, max_degree)` ladder as `ace`. It is timed as
  deployed: `ACECalculator` / `export_lammps` with the default
  `spline_tol="auto"`, which splines a `radial_learned` model at 1e-10.
- **Hyperparameters:** each code's defaults, apart from the size knob and the
  shared settings above.
- **Large files:** all data and runs live under
  `/storage/eng/essswb/projects/ace-jax/` (`$ACEJAX_AC_ROOT`) and are never
  committed. Only `results/*.jsonl`, `figs/*.png`, code and docs are committed.
- **Timing:** on lestrade only. GPU uses N = 32768 with 4096 as the check; CPU
  uses N = 4096 with 512 as the check. A row is flagged `not_plateau` when
  per-atom throughput differs by more than 20% between the two sizes. Each case
  runs 3 repeats and reports the median.
- **Precision:** float64 for ace-jax, ML-PACE and GAP. MACE is timed in
  float32 and float64; its LAMMPS rows are float64 only.
- **Code style:** follow CLAUDE.md (dense, short names, trailing `#` comments
  that explain why, Python 3.11 syntax, no PEP 701 f-strings). Tests are plain
  pytest functions with `jax.config.update("jax_enable_x64", True)` at the top
  of any module that imports jax.
- **Commits:** conventional with a scope, e.g. `feat(bench):`, `test(bench):`
  or `docs(bench):`, ending with the Co-Authored-By line.

## Review Focus

1. **Upper-case label keys.** 60 `sh` frames carry `DFT_energy` / `DFT_force` /
   `DFT_virial`. A case-sensitive key match would silently drop them from every
   fit. *Pinned by `test_split_reads_uppercase_label_keys` (Task 1).*
2. **Isolated atom in the test set.** If the single `isolated_atom` frame went
   to test, E0 would be unfixable and every family's energy RMSE would carry a
   ~1 eV outlier. *Pinned by `test_rare_types_stay_in_train` (Task 1).*
3. **Weight conventions differ by code.** ace-jax multiplies residuals (w = 1/σ),
   pacemaker multiplies squared residuals (w = 1/σ²), and MACE scales the
   whole config loss. A mix-up would quietly reweight one family.
   *Pinned by `test_weights_consistent_across_codes` (Task 2).*
4. **A stale split.** A run fitted on an older split and then scored on a
   newer one would look accurate for the wrong reason. *Pinned by
   `test_evaluate_refuses_mismatched_split` (Task 3).*
5. **Targets no family reaches.** The table must print "not reached" and never
   pick the nearest model. *Pinned by `test_smallest_at_target_not_reached`
   (Task 10).*

---

## File structure

| file | responsibility |
|---|---|
| `bench/accuracy_cost/__init__.py` | empty; makes `accuracy_cost` importable from `bench/` as tests do with `scaling` |
| `bench/accuracy_cost/common.py` | paths, `RCUT`, the sweep table, the run-directory protocol (`fit_once`), and jsonl upsert |
| `bench/accuracy_cost/sweep.yaml` | family → list of size entries |
| `bench/accuracy_cost/data.md` | data provenance: URL, commit, sha256, published GAP-18 errors and citation |
| `bench/accuracy_cost/split.py` | download and verify, normalise labels, stratified split, split hash |
| `bench/accuracy_cost/weights.py` | GAP-18 sigmas → ace-jax, pacemaker, MACE and gap_fit weights |
| `bench/accuracy_cost/evaluate.py` | one scorer for every family → `results/metrics.jsonl` |
| `bench/accuracy_cost/fit_ace.py` | linear ACE (analytic and learned radials) and ACE+GP through `aj fit` |
| `bench/accuracy_cost/fit_pace.py` | pacemaker ladder fit → `.yace` per checkpoint |
| `bench/accuracy_cost/fit_mace.py` | `mace_run_train` and the Symmetrix export |
| `bench/accuracy_cost/fit_gap.py` | the `gap_fit` refit from GAP-18's command line |
| `bench/accuracy_cost/timing.py` | idle check, parity gate, 3 repeats at N and N/8 → `results/lestrade-{cpu,gpu}.jsonl` |
| `bench/accuracy_cost/plot.py` | Pareto plot, smallest-at-target table, per-type heatmap |
| `bench/accuracy_cost/envs/{pacemaker,mace,quip}/pyproject.toml` | the three tool venvs |
| `bench/accuracy_cost/envs/lestrade.sh` | the LAMMPS and venv build for lestrade |
| `bench/accuracy_cost/README.md` | how to run it end to end |
| `bench/scaling/structures.py` | **modify:** add `Si` and the `rattle=` kwarg |
| `bench/scaling/run_standalone.py` | **modify:** an `acejax-gp` branch and `rattle` from the row |
| `bench/scaling/run_lammps.py` | **modify:** a `gap` style (`pair quip`) and `rattle` from the row |
| `bench/scaling/parity.py` | **modify:** a `gap` tolerance and `_quip_ef` |
| `tests/test_bench_accuracy_cost.py` | the new fast tests |
| `tests/test_bench_scaling.py` | **modify:** tests for the four hooks |
| `docs/dev/accuracy-cost.md`, `docs/dev/benchmarks.md` | the write-up and the pointer |

---

### Task 1: Data: download, normalise and split

**Files:**
- Create: `bench/accuracy_cost/__init__.py` (empty)
- Create: `bench/accuracy_cost/common.py`
- Create: `bench/accuracy_cost/split.py`
- Create: `bench/accuracy_cost/data.md`
- Test: `tests/test_bench_accuracy_cost.py`
- Modify: `docs/dev/accuracy-cost-spec.md` (Data section: the E+F-only refinement and the findings below)

**Interfaces:**
- Produces:
  - `common.ROOT`, `common.DATA`, `common.RUNS`, `common.RESULTS`, `common.HERE`, `common.RCUT = 5.0`, `common.E0_SI = -158.54496821`
  - `common.upsert_jsonl(path, row, key: tuple[str, ...]) -> None`
  - `split.LABEL_KEYS = {"energy": "dft_energy", "forces": "dft_force", "virial": "dft_virial"}`
  - `split.normalise(frame: Frame) -> ase.Atoms | None`: `atoms.info["energy"]` and `atoms.info["config_type"]` are set, `atoms.info["virial"]` (3×3, eV) is set when the frame has one, `atoms.arrays["forces"]` is set, and an unlabelled frame (no energy or forces) returns None
  - `split.stratified(types: list[str], frac=0.1, seed=0, min_per_type=10) -> (train_idx: list[int], test_idx: list[int])`
  - `split.split_hash(train_path, test_path) -> str` (the first 12 hex characters of sha256 over both files)
  - `split.ensure_split() -> str`: idempotent, writes `DATA/{train,test}.xyz` and `DATA/split.json`, and returns the hash
  - `split.read_atoms(path) -> list[ase.Atoms]`: reads through `read_extxyz`, so labels stay in `info` / `arrays`

Findings recorded in `data.md` and the spec:
- The local copy at
  `/storage/eng/essswb/ownCloud/Si_DATASET_RERUN_FHIaims/gp_iter6_sparse9k.xml.xyz`
  has the same sha256 as the pinned GitHub file. Despite its directory name,
  it holds the original CASTEP labels.
- GAP-18's fit uses a glue core potential (`core_param_file=glue.xml`).
- The 60 `sh` frames use upper-case label keys.
- 1,783 frames carry virials, which are kept (fitted where the code supports it).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bench_accuracy_cost.py
import json
import pathlib
import sys

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "bench"))
from conftest import FIXTURE_DIR

TINY = FIXTURE_DIR / "si_tiny_train.xyz"          # 53 GAP-18 frames, lower-case dft_* keys


def _frames():
    from ace_jax.fit.xyz import read_extxyz
    return read_extxyz(TINY)


def test_normalise_renames_labels_and_keeps_virial():
    from accuracy_cost.split import normalise
    fr = _frames()
    at = normalise(fr[1])
    assert set(at.info) >= {"energy", "config_type"} and "forces" in at.arrays
    assert not any(k.startswith("gap_") or k.startswith("dft_") for k in (*at.info, *at.arrays))
    with_v = [f for f in fr if any(k.lower() == "dft_virial" for k in f.info)]
    if with_v:                                      # the fixture slice may carry none
        a = normalise(with_v[0])
        assert a.info["virial"].shape == (3, 3)


def test_split_writer_round_trips_virial(tmp_path):
    import ase
    from accuracy_cost.split import _write, read_atoms
    at = ase.Atoms("Si2", positions=[[0, 0, 0], [1.3, 1.3, 1.3]], cell=np.eye(3) * 5.4, pbc=True)
    at.info.update(energy=-300.0, config_type="dia", virial=np.arange(9.0).reshape(3, 3))
    at.arrays["forces"] = np.ones((2, 3))
    b = at.copy(); b.info = {"energy": -301.0, "config_type": "liq"}; b.arrays["forces"] = np.zeros((2, 3))
    _write(tmp_path / "x.xyz", [at, b])
    r = read_atoms(tmp_path / "x.xyz")
    assert np.allclose(np.reshape(r[0].info["virial"], (3, 3)), at.info["virial"]) and "virial" not in r[1].info


def test_split_reads_uppercase_label_keys():
    from accuracy_cost.split import normalise
    from ace_jax.fit.xyz import Frame
    f = _frames()[1]
    info = {("DFT_energy" if k == "dft_energy" else k): v for k, v in f.info.items()}
    arrays = {("DFT_force" if k == "dft_force" else k): v for k, v in f.arrays.items()}
    at = normalise(Frame(f.numbers, f.positions, f.cell, f.pbc, info, arrays))
    assert at is not None and at.info["energy"] == f.info["dft_energy"]
    assert np.array_equal(at.arrays["forces"], f.arrays["dft_force"])


def test_split_is_deterministic_and_stratified():
    from accuracy_cost.split import stratified
    types = ["a"] * 40 + ["b"] * 20 + ["c"] * 5
    tr1, te1 = stratified(types, seed=0)
    tr2, te2 = stratified(types, seed=0)
    assert (tr1, te1) == (tr2, te2) and sorted(tr1 + te1) == list(range(len(types)))
    assert sum(types[i] == "a" for i in te1) == 4 and sum(types[i] == "b" for i in te1) == 2


def test_rare_types_stay_in_train():
    from accuracy_cost.split import stratified
    types = ["isolated_atom"] + ["dia"] * 30 + ["crack"] * 7
    tr, te = stratified(types, seed=0)
    assert 0 in tr and all(types[i] != "crack" for i in te)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -v -n 0`
Expected: FAIL with `ModuleNotFoundError: No module named 'accuracy_cost'`

- [ ] **Step 3: Write `common.py`**

```python
"""Paths, the sweep table and the run-directory protocol shared by every
accuracy-cost driver.  Data and runs live on /storage ($ACEJAX_AC_ROOT): only
results/*.jsonl and figs/ are committed."""
import json
import os
import pathlib
import resource
import socket
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("ACEJAX_AC_ROOT", "/storage/eng/essswb/projects/ace-jax"))
DATA = ROOT / "data" / "si_accuracy_cost"
RUNS = ROOT / "runs" / "accuracy_cost"
RESULTS = HERE / "results"
RCUT = 5.0                                    # every family (spec: GAP-18's SOAP cutoff)
E0_SI = -158.54496821                         # the isolated_atom frame's dft_energy


def load_sweep(path=HERE / "sweep.yaml"):
    import yaml
    return yaml.safe_load(pathlib.Path(path).read_text())


def entry(family, size, sweep=None):
    for e in (sweep or load_sweep())[family]:
        if e["size"] == size:
            return e
    raise KeyError(f"{family}/{size} not in sweep.yaml")


def run_dir(family, size):
    return RUNS / family / size


def read_status(d):
    p = pathlib.Path(d) / "status.json"
    return json.loads(p.read_text()) if p.exists() else None


def write_status(d, **kw):
    d = pathlib.Path(d); d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(json.dumps({"host": socket.gethostname(), **kw}, indent=1))


def tail(path, n=40):
    p = pathlib.Path(path)
    return "\n".join(p.read_text(errors="replace").splitlines()[-n:]) if p.exists() else ""


def run_logged(cmd, cwd, log="fit.log", env=None):
    """Run `cmd` in `cwd` with output to `log`: (returncode, wall_s, peak_rss_gb).
    Peak RSS is RUSAGE_CHILDREN's max, so one fit per driver process."""
    cwd = pathlib.Path(cwd); cwd.mkdir(parents=True, exist_ok=True)
    (cwd / "cmd.txt").write_text(" ".join(map(str, cmd)) + "\n")
    t0 = time.perf_counter()
    with open(cwd / log, "w") as f:
        rc = subprocess.run(list(map(str, cmd)), cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    return rc, time.perf_counter() - t0, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6


def fit_once(family, size, model_name, cmd, env=None, split_hash=None):
    """The resumable protocol: an existing model is never refitted; a failure
    writes status 'failed' with the log tail and is not retried."""
    d = run_dir(family, size)
    if (d / model_name).exists() and (read_status(d) or {}).get("status") == "ok":
        return read_status(d)
    rc, wall, peak = run_logged(cmd, d, env=env)
    ok = rc == 0 and (d / model_name).exists()
    write_status(d, family=family, size=size, status="ok" if ok else "failed", model=str(d / model_name),
                 fit_wall_s=wall, fit_peak_rss_gb=peak, split_hash=split_hash,
                 **({} if ok else {"returncode": rc, "error": tail(d / "fit.log")}))
    return read_status(d)


def upsert_jsonl(path, row, key):
    """Replace the row with the same `key` fields, else append; one row per key."""
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    k = tuple(row.get(f) for f in key)
    rows = [r for r in rows if tuple(r.get(f) for f in key) != k] + [row]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
```

- [ ] **Step 4: Write `split.py`**

```python
"""GAP-18 Si data -> DATA/{train,test}.xyz: labels renamed to energy/forces/virial
(the source mixes dft_* and DFT_* keys), stored GAP predictions dropped, virials
kept where present (fitted by the codes that can), 90/10 stratified by config_type with rare types
kept whole in train (one test frame of a type is noise, not a measurement)."""
import hashlib
import json
import sys
import urllib.request

import numpy as np

from .common import DATA

URL = ("https://raw.githubusercontent.com/libAtoms/silicon-testing-framework/"
       "ec91b17fbbcd6c4a06a257acbfcba77734c0080c/models/GAP/")
SRC, XML, GLUE = "gp_iter6_sparse9k.xml.xyz", "gp_iter6_sparse9k.xml", "glue.xml"
SHA = {SRC: "74ac5f387aa0149ec9ea8f3e275d1087d7287db68c15ba8bc4c28a450de67e2a"}
LABEL_KEYS = {"energy": "dft_energy", "forces": "dft_force", "virial": "dft_virial"}


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    for name in (SRC, XML, GLUE):
        p = DATA / name
        if not p.exists():
            urllib.request.urlretrieve(URL + name, p)
        if name in SHA and hashlib.sha256(p.read_bytes()).hexdigest() != SHA[name]:
            raise RuntimeError(f"{p}: sha256 does not match the pinned commit")


def _get(d, key):
    for k, v in d.items():
        if k.lower() == key:
            return v
    return None


def normalise(f):
    """An ase.Atoms with info energy/config_type and arrays forces, or None when unlabelled."""
    import ase
    E, F = _get(f.info, LABEL_KEYS["energy"]), _get(f.arrays, LABEL_KEYS["forces"])
    if E is None or F is None:
        return None
    at = ase.Atoms(numbers=f.numbers, positions=f.positions, cell=f.cell, pbc=f.pbc)
    at.info.update(energy=float(E), config_type=str(f.info.get("config_type", "default")))
    V = _get(f.info, LABEL_KEYS["virial"])
    if V is not None:
        at.info["virial"] = np.asarray(V, float).reshape(3, 3)
    at.arrays["forces"] = np.asarray(F, float).reshape(-1, 3)
    return at


def stratified(types, frac=0.1, seed=0, min_per_type=10):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for t in sorted(set(types)):
        idx = [i for i, x in enumerate(types) if x == t]
        if len(idx) < min_per_type:
            tr += idx; continue
        idx = list(rng.permutation(idx))
        k = max(1, round(frac * len(idx)))
        te += idx[:k]; tr += idx[k:]
    return sorted(tr), sorted(te)


def _write(path, atoms):
    # plain extxyz, written by hand so no SinglePointCalculator round-trip touches the keys
    with open(path, "w") as f:
        for at in atoms:
            c = at.cell.array.reshape(-1)
            v = at.info.get("virial")
            vs = ' virial="' + " ".join(f"{x:.10f}" for x in np.reshape(v, -1)) + '"' if v is not None else ""
            f.write(f"{len(at)}\n")
            f.write('Lattice="' + " ".join(f"{x:.10f}" for x in c) + '" Properties=species:S:1:pos:R:3:forces:R:3 '
                    f'energy={at.info["energy"]:.10f}{vs} config_type={at.info["config_type"]} pbc="T T T"\n')
            for s, p, F in zip(at.get_chemical_symbols(), at.positions, at.arrays["forces"]):
                f.write(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f} {F[0]:.10f} {F[1]:.10f} {F[2]:.10f}\n")


def split_hash(train, test):
    h = hashlib.sha256()
    for p in (train, test):
        h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


def read_atoms(path):
    """ase.Atoms with labels in info/arrays, via libAtoms extxyz (never ase.io.read)."""
    from ace_jax.fit.xyz import read_extxyz
    import ase
    out = []
    for f in read_extxyz(path):
        at = ase.Atoms(numbers=f.numbers, positions=f.positions, cell=f.cell, pbc=f.pbc)
        at.info.update(f.info); at.arrays.update({k: np.asarray(v) for k, v in f.arrays.items()})
        out.append(at)
    return out


def ensure_split():
    tr_p, te_p, meta_p = DATA / "train.xyz", DATA / "test.xyz", DATA / "split.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text())["hash"]
    fetch()
    from ace_jax.fit.xyz import read_extxyz
    frames = read_extxyz(DATA / SRC)
    atoms = [a for a in map(normalise, frames) if a is not None]
    if len(atoms) != len(frames):
        raise RuntimeError(f"{len(frames) - len(atoms)} frames have no energy/force labels")
    assert all(a.pbc.all() for a in atoms)        # _write states pbc="T T T"
    tr, te = stratified([a.info["config_type"] for a in atoms])
    _write(tr_p, [atoms[i] for i in tr]); _write(te_p, [atoms[i] for i in te])
    h = split_hash(tr_p, te_p)
    meta_p.write_text(json.dumps({"hash": h, "n_train": len(tr), "n_test": len(te), "seed": 0,
                                  "frac": 0.1, "min_per_type": 10}, indent=1))
    return h


if __name__ == "__main__":
    print(ensure_split(), file=sys.stdout)
```

The handwritten writer always says `pbc="T T T"`. Every GAP-18 frame is
periodic (the isolated atom sits in a 20 Å box), and `ensure_split` asserts it.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -v -n 0`
Expected: 5 passed

- [ ] **Step 6: Build the real split and check it**

Run: `uv run --no-sync python -m accuracy_cost.split` from `bench/` (`cd bench && uv run --no-sync python -m accuracy_cost.split`)
Expected: a 12-character hash, and `DATA/split.json` with `n_train + n_test == 2475`.

Then run: `uv run --no-sync python -c "from accuracy_cost.split import read_atoms; from accuracy_cost.common import DATA; a=read_atoms(DATA/'test.xyz'); import collections; print(len(a), collections.Counter(x.info['config_type'] for x in a))"` from `bench/`
Expected: about 240 test frames. No `isolated_atom`, `crack_111_1-10`, `crack_110_1-10` or `surface_111_3x3_das` (all have fewer than 10 frames). `sh` has about 22 frames, so the 60 upper-case frames were read.

- [ ] **Step 7: Write `data.md`**

It contains:
- the URL, the commit, the sha256 and the frame counts;
- the local-copy identity finding;
- the glue core;
- the upper-case `sh` keys;
- the virial decision (kept where the code can fit them);
- the published GAP-18 errors with their citation: Bartók, Kermode, Bernstein
  and Csányi, *Phys. Rev. X* **8**, 041048 (2018). Copy the energy and force
  RMSEs from the paper's validation table into a two-row markdown table
  (`quantity | published GAP-18`) and note "training-set errors: GAP-18 saw
  these configurations".

Update the spec's Data section in the same commit:
- replace "The local FHI-aims rerun is not used" with the identity finding;
- add "Virials (1,783 frames) are fitted by every code that supports them,
  with GAP-18's virial sigmas where the code takes per-type weights";
- add "the GAP refit keeps GAP-18's glue core".

- [ ] **Step 8: Commit**

```bash
git add bench/accuracy_cost/__init__.py bench/accuracy_cost/common.py bench/accuracy_cost/split.py \
        bench/accuracy_cost/data.md tests/test_bench_accuracy_cost.py docs/dev/accuracy-cost-spec.md
git commit -m "feat(bench): accuracy-cost Si data split

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Weights: GAP-18 sigmas into every code's convention

**Files:**
- Create: `bench/accuracy_cost/weights.py`
- Test: `tests/test_bench_accuracy_cost.py` (append)

**Interfaces:**
- Consumes: `common.DATA`, `split.XML`
- Produces:
  - `weights.command_line(xml_path) -> str`: the CDATA in `<command_line>`
  - `weights.gap_sigmas(cl: str) -> tuple[tuple[float, float, float], dict[str, tuple[float, float, float]]]`: `(sE, sF, sV)` default and per config_type (the hessian column is dropped)
  - `weights.acejax(default, per) -> dict`: the ACEfit dict `{"default": {"E": 1/sE, "F": 1/sF, "V": 1/sV}, ct: {...}}` (residual multipliers)
  - `weights.pacemaker(default, per, atoms: list[ase.Atoms]) -> list[tuple[float, np.ndarray]]`: per structure `(w_energy = 1/sE**2, w_forces = full(n_atoms, 1/sF**2))` (squared-residual multipliers)
  - `weights.mace(default, per) -> dict[str, float]`: `config_type_weights`, `{"Default": 1.0, ct: (sF_default/sF_ct)**2}`. MACE scales a config's whole loss, so the force sigma sets the ratio, and its own `energy_weight` / `forces_weight` defaults set the E/F balance (a documented deviation).
  - `weights.write_all(out_dir) -> None`: writes `weights_acejax.json` and `weights_mace.json`

GAP-18's line is `default_sigma={0.001 0.1 0.05 0.0}` and
`config_type_sigma={liq:0.003:0.15:0.2:0.0:amorph:0.01:0.2:0.4:0.0:sp:0.01:0.2:0.4:0.0}`.
Groups are `type:E:F:V:H`.

- [ ] **Step 1: Write the failing tests**

```python
CL = ("at_file=x.xyz gap={soap cutoff=5.0 n_sparse=9000} default_sigma={0.001 0.1 0.05 0.0} "
      "config_type_sigma={liq:0.003:0.15:0.2:0.0:amorph:0.01:0.2:0.4:0.0:sp:0.01:0.2:0.4:0.0} "
      "energy_parameter_name=dft_energy")


def test_gap_sigmas_parse_groups_of_five():
    from accuracy_cost.weights import gap_sigmas
    d, per = gap_sigmas(CL)
    assert d == (0.001, 0.1, 0.05)
    assert per == {"liq": (0.003, 0.15, 0.2), "amorph": (0.01, 0.2, 0.4), "sp": (0.01, 0.2, 0.4)}


def test_weights_consistent_across_codes():
    """Every code must weight a config type relative to the default identically:
    ace-jax (1/sigma on residuals)^2 == pacemaker (1/sigma^2 on squared residuals),
    and MACE's per-config factor == the force ratio."""
    import ase
    from accuracy_cost.weights import acejax, gap_sigmas, mace, pacemaker
    d, per = gap_sigmas(CL)
    aj, mc = acejax(d, per), mace(d, per)
    ats = [ase.Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], info={"config_type": t}) for t in ("dia", "liq")]
    pw = pacemaker(d, per, ats)
    for t, (wE, wF) in zip(("default", "liq"), pw):
        assert np.isclose(aj[t]["E"] ** 2, wE) and np.allclose(aj[t]["F"] ** 2, wF)
    assert np.isclose(mc["liq"], (aj["liq"]["F"] / aj["default"]["F"]) ** 2) and mc["Default"] == 1.0
    assert np.isclose(aj["default"]["V"], 20.0) and np.isclose(aj["liq"]["V"], 5.0)


def test_command_line_from_xml(tmp_path):
    from accuracy_cost.weights import command_line
    p = tmp_path / "g.xml"
    p.write_text(f"<GAP><command_line><![CDATA[ {CL}]]></command_line></GAP>")
    assert command_line(p).strip() == CL
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -k "sigmas or weights or command_line" -v -n 0`
Expected: FAIL with `ModuleNotFoundError: No module named 'accuracy_cost.weights'`

- [ ] **Step 3: Implement `weights.py`**

```python
"""GAP-18's per-config-type sigmas, the one source of fit weights, in each
code's convention.  ace-jax (ACEfit dict) multiplies residuals: w = 1/sigma.
pacemaker multiplies squared residuals: w = 1/sigma^2.  MACE scales a config's
whole loss by config_type_weights, so the force sigma sets the ratio (its own
energy/forces/virials weights set the balance).  Virial sigmas reach ace-jax
and gap_fit; MACE uses one global virials weight; pacemaker fits no virials."""
import json
import pathlib
import re

import numpy as np


def command_line(xml_path):
    t = pathlib.Path(xml_path).read_text()
    m = re.search(r"<command_line><!\[CDATA\[(.*?)\]\]></command_line>", t, re.S)
    if not m:
        raise ValueError(f"{xml_path}: no <command_line>")
    return m.group(1)


def gap_sigmas(cl):
    d = tuple(map(float, re.search(r"default_sigma=\{([^}]*)\}", cl).group(1).split()[:3]))
    per, m = {}, re.search(r"config_type_sigma=\{([^}]*)\}", cl)
    if m:
        tok = m.group(1).split(":")
        if len(tok) % 5:
            raise ValueError("config_type_sigma: expected type:E:F:V:H groups")
        for i in range(0, len(tok), 5):
            per[tok[i]] = (float(tok[i + 1]), float(tok[i + 2]), float(tok[i + 3]))
    return d, per


def acejax(d, per):
    w = lambda s: {"E": 1 / s[0], "F": 1 / s[1], "V": 1 / s[2]}
    return {"default": w(d), **{t: w(s) for t, s in per.items()}}


def pacemaker(d, per, atoms):
    out = []
    for at in atoms:
        sE, sF, _ = per.get(at.info.get("config_type"), d)
        out.append((1 / sE ** 2, np.full(len(at), 1 / sF ** 2)))
    return out


def mace(d, per):
    return {"Default": 1.0, **{t: (d[1] / s[1]) ** 2 for t, s in per.items()}}


def write_all(out_dir):
    from .common import DATA
    from .split import XML
    d, per = gap_sigmas(command_line(DATA / XML))
    out = pathlib.Path(out_dir)
    (out / "weights_acejax.json").write_text(json.dumps(acejax(d, per), indent=1))
    (out / "weights_mace.json").write_text(json.dumps(mace(d, per), indent=1))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -v -n 0`
Expected: 8 passed

- [ ] **Step 5: Write the real tables**

Run: `cd bench && uv run --no-sync python -c "from accuracy_cost.weights import write_all; from accuracy_cost.common import DATA; write_all(DATA)" && cat $ACEJAX_AC_ROOT/data/si_accuracy_cost/weights_acejax.json` (with `ACEJAX_AC_ROOT` unset, read `/storage/eng/essswb/projects/ace-jax/data/si_accuracy_cost/weights_acejax.json`)
Expected: `default` E 1000, F 10, V 20; `liq` E ≈ 333.3, F ≈ 6.667, V 5; `amorph` and `sp` E 100, F 5, V 2.5.

- [ ] **Step 6: Commit**

```bash
git add bench/accuracy_cost/weights.py tests/test_bench_accuracy_cost.py
git commit -m "feat(bench): GAP-18 sigmas as per-code fit weights

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: The shared scorer

**Files:**
- Create: `bench/accuracy_cost/evaluate.py`
- Test: `tests/test_bench_accuracy_cost.py` (append)

**Interfaces:**
- Consumes: `split.read_atoms`, `common.read_status`, `common.upsert_jsonl`, `common.RESULTS`, `common.DATA`
- Produces:
  - `evaluate.errors(atoms: list[ase.Atoms], calc) -> list[dict]`: one dict per frame, `{"config_type", "n", "dE": (E_pred - E_ref)/n, "dF": (n, 3) array, "dV": (3, 3) array / n or None}`. The predicted virial is `-stress * volume` from `calc.get_stress(voigt=False)`, computed only for frames with a reference virial.
  - `evaluate.summarise(rows) -> dict`: `{"rmse_E": eV/atom, "rmse_F": eV/Å over all components, "rmse_V": eV/atom over the components of frames with virials (None if none), "per_type": {ct: {"rmse_E", "rmse_F", "rmse_V", "n_frames"}}}`
  - `evaluate.load_calc(family, model_path, device="cpu")`: an ASE calculator
  - `evaluate.score(family, size) -> dict`: upserts into `results/metrics.jsonl`, keyed `("family", "size")`, with fields `family, size, model, split_hash, n_params, rmse_E, rmse_F, per_type, fit_wall_s, fit_host`
  - `evaluate.SplitMismatch(Exception)`
- `n_params` per family: the readout length for ace-jax (`WB.size + Wpair.size`), plus the GP's inducing count for ACE+GP; the number of functions times the number of densities for PACE (from the `.yace`); `sum(p.numel())` for MACE; and the sparse-point count for GAP (from `sweep.yaml`).

- [ ] **Step 1: Write the failing tests**

```python
class _Const:
    """A calculator with known errors: E off by 0.01 eV/atom, every F component off by 0.1."""
    def __init__(self): self.results = {}
    def get_potential_energy(self, at): return at.info["energy"] + 0.01 * len(at)
    def get_forces(self, at): return at.arrays["forces"] + 0.1
    def get_stress(self, at, voigt=False): return np.zeros((3, 3))


def test_summarise_known_errors():
    from accuracy_cost.evaluate import errors, summarise
    from accuracy_cost.split import normalise
    from ace_jax.fit.xyz import read_extxyz
    ats = [a for a in map(normalise, read_extxyz(TINY)) if a is not None]
    s = summarise(errors(ats, _Const()))
    assert np.isclose(s["rmse_E"], 0.01) and np.isclose(s["rmse_F"], 0.1)
    with_v = [a for a in ats if "virial" in a.info]
    if with_v:                                     # zero predicted stress: rmse_V = rms(V_ref / n)
        ref = np.sqrt(np.mean(np.concatenate([(a.info["virial"] / len(a)).ravel() for a in with_v]) ** 2))
        assert np.isclose(s["rmse_V"], ref)
    assert set(s["per_type"]) == {a.info["config_type"] for a in ats}


def test_load_calc_scores_a_real_ace_model():
    from accuracy_cost.evaluate import errors, load_calc, summarise
    from accuracy_cost.split import normalise
    from ace_jax.fit.xyz import read_extxyz
    ats = [a for a in map(normalise, read_extxyz(TINY)) if a is not None][:5]
    s = summarise(errors(ats, load_calc("ace", FIXTURE_DIR / "si_fitted.npz")))
    assert np.isfinite(s["rmse_E"]) and 0 < s["rmse_F"] < 5


def test_evaluate_refuses_mismatched_split(tmp_path, monkeypatch):
    import accuracy_cost.common as C
    import accuracy_cost.evaluate as ev
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs"); monkeypatch.setattr(C, "DATA", tmp_path / "data")
    monkeypatch.setattr(ev, "DATA", tmp_path / "data")
    (tmp_path / "data").mkdir(); (tmp_path / "data" / "split.json").write_text('{"hash": "new"}')
    C.write_status(C.run_dir("ace", "x"), status="ok", split_hash="old", model="m.npz")
    with pytest.raises(ev.SplitMismatch):
        ev.score("ace", "x")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -k "summarise or load_calc or mismatched" -v -n 0`
Expected: FAIL with `ModuleNotFoundError: No module named 'accuracy_cost.evaluate'`

- [ ] **Step 3: Implement `evaluate.py`**

```python
"""One scorer for every family: ASE calculators on DATA/test.xyz, so no
fitter's own metrics enter the comparison.  Refuses a run fitted on another
split (its hash is in status.json)."""
import json
import sys

import numpy as np

from . import common
from .common import DATA, RESULTS


class SplitMismatch(Exception):
    pass


def errors(atoms, calc):
    out = []
    for at in atoms:
        a = at.copy(); a.calc = None
        E, F = calc.get_potential_energy(a), calc.get_forces(a)
        dV = None
        if "virial" in at.info:                     # virial = -stress * volume (ASE sign convention)
            V = -np.asarray(calc.get_stress(a, voigt=False)) * a.get_volume()
            dV = (V - np.reshape(at.info["virial"], (3, 3))) / len(at)
        out.append({"config_type": at.info["config_type"], "n": len(at),
                    "dE": (E - at.info["energy"]) / len(at), "dF": np.asarray(F) - at.arrays["forces"], "dV": dV})
    return out


def _rmse(rows):
    dE = np.array([r["dE"] for r in rows]); dF = np.concatenate([r["dF"].ravel() for r in rows])
    dV = [r["dV"].ravel() for r in rows if r["dV"] is not None]
    V = float(np.sqrt(np.mean(np.concatenate(dV) ** 2))) if dV else None
    return float(np.sqrt(np.mean(dE ** 2))), float(np.sqrt(np.mean(dF ** 2))), V


def summarise(rows):
    E, F, V = _rmse(rows)
    per = {}
    for t in sorted({r["config_type"] for r in rows}):
        sub = [r for r in rows if r["config_type"] == t]
        e, f, v = _rmse(sub)
        per[t] = {"rmse_E": e, "rmse_F": f, "rmse_V": v, "n_frames": len(sub)}
    return {"rmse_E": E, "rmse_F": F, "rmse_V": V, "per_type": per}


def load_calc(family, path, device="cpu"):
    path = str(path)
    if family in ("ace", "ace_lr"):              # ace_lr: lean + spline_tol="auto", as deployed
        from ace_jax.calc.point import ACECalculator
        return ACECalculator(path)
    if family == "ace_gp":
        from ace_jax.calc.gp import GPCalculator
        return GPCalculator.from_file(path)
    if family == "pace":
        from ace_jax.calc.point import ACECalculator         # ace-jax PACEModel; pyace cross-check in Task 6
        return ACECalculator(path)
    if family == "mace":
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=path, device="cuda" if device == "gpu" else "cpu",
                              default_dtype="float64")
    if family == "gap":
        import quippy
        return quippy.potential.Potential(param_filename=path)
    raise ValueError(family)


def score(family, size, device="cpu"):
    d = common.run_dir(family, size)
    st = common.read_status(d)
    if not st or st.get("status") != "ok":
        raise RuntimeError(f"{family}/{size}: no successful fit in {d}")
    want = json.loads((DATA / "split.json").read_text())["hash"]
    if st.get("split_hash") != want:
        raise SplitMismatch(f"{family}/{size} was fitted on split {st.get('split_hash')}, test is {want}")
    from .split import read_atoms
    s = summarise(errors(read_atoms(DATA / "test.xyz"), load_calc(family, st["model"], device)))
    row = {"family": family, "size": size, "model": st["model"], "split_hash": want,
           "n_params": st.get("n_params"), "fit_wall_s": st.get("fit_wall_s"), "fit_host": st.get("host"), **s}
    common.upsert_jsonl(RESULTS / "metrics.jsonl", row, ("family", "size"))
    return row


if __name__ == "__main__":
    fam, sizes = sys.argv[1], sys.argv[2:]
    for s in sizes:
        r = score(fam, s)
        print(f"{fam}/{s}: rmse_E {r['rmse_E'] * 1e3:.2f} meV/atom  rmse_F {r['rmse_F']:.4f} eV/A")
```

MACE and quippy are imported inside `load_calc`, as `test_core_deps` requires.
`evaluate.py` runs under each family's own venv: the project venv for ace,
ace_gp and pace; `envs/mace` for MACE (with ace-jax installed into it, see
Task 7); and `envs/quip` for GAP (likewise, Task 8). `n_params` is written
into `status.json` by each fit driver.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -v -n 0`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add bench/accuracy_cost/evaluate.py tests/test_bench_accuracy_cost.py
git commit -m "feat(bench): shared accuracy-cost scorer

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Harness hooks in `bench/scaling`: Si, rattle, GP standalone and `pair quip`

**Files:**
- Modify: `bench/scaling/structures.py` (`SYSTEMS`, `supercell`)
- Modify: `bench/scaling/run_standalone.py:70-140` (`run_case`: rattle, plus an `acejax-gp` branch)
- Modify: `bench/scaling/run_lammps.py:20-73` (`KOKKOS`, `_pair`, `lammps_input`) and `:205-215` (`run_case` rattle)
- Modify: `bench/scaling/parity.py:31` (`TOL["gap"]`), plus a new `_quip_ef`
- Test: `tests/test_bench_scaling.py` (append)

**Interfaces:**
- Produces:
  - `SYSTEMS["Si"] = {"lattice": "diamond", "a": 5.461, "elements": ("Si",), "per_cell": 8}`. 5.461 Å is GAP-18's DFT (PW91) diamond lattice constant.
  - `supercell(system, n_atoms, seed=0, rattle=0.0)`: with `rattle > 0`, Gaussian displacements of standard deviation `rattle` (Å) from `np.random.default_rng(seed + 1)`.
  - Both `run_case`s call `supercell(row["system"], n_atoms, rattle=row.get("rattle", 0.0))`.
  - Row code `"acejax-gp"` in `run_standalone.run_case`: `GPCalculator.from_file(row["path"])`, timing `call_s` like the MACE branch.
  - LAMMPS style `"gap"`: `pair_style quip` and `pair_coeff * * <xml> "Potential xml_label=<label>" <Z...>`. The row carries `quip_label`, passed through `lammps_input(..., pair_extra=row.get("quip_label"))`.
  - `run_lammps.run_case` maps `row["code"] == "gap"` to style `"gap"` with `model = row["path"]`.
  - `parity.TOL["gap"] = (1e-8, 1e-7)`: LAMMPS and quippy run the same Fortran.
  - `parity._quip_ef(path, at) -> (E, F)` through `quippy.potential.Potential`.

- [ ] **Step 1: Write the failing tests**

```python
def test_si_supercell_and_rattle_are_deterministic():
    a = supercell("Si", 512)
    b, c = supercell("Si", 512, rattle=0.05), supercell("Si", 512, rattle=0.05)
    assert set(a.get_chemical_symbols()) == {"Si"} and np.array_equal(b.positions, c.positions)
    d = np.linalg.norm(b.positions - a.positions, axis=1)
    assert 0.03 < d.mean() < 0.12 and np.array_equal(a.numbers, b.numbers)


def test_unrattled_supercell_is_unchanged():
    assert np.array_equal(supercell("SiGe", 256).positions, supercell("SiGe", 256, rattle=0.0).positions)


def test_lammps_input_quip_pair_line():
    txt = lammps_input("gap", "/m/gap.xml", ["Si"], "/d/x.data", "cpu", 200, pair_extra="GAP_2017_6_17_60_4_3_56_165")
    assert "pair_style quip\n" in txt
    assert 'pair_coeff * * /m/gap.xml "Potential xml_label=GAP_2017_6_17_60_4_3_56_165" 14\n' in txt


def test_standalone_row_for_acejax_gp(tmp_path):
    from scaling.run_standalone import run_case
    gp = pathlib.Path(__file__).parent.parent / "fixtures" / "pipeline_golden"
    path = next(gp.rglob("gp_model.npz"), None)
    if path is None:
        pytest.skip("no committed gp_model.npz fixture")
    row = {"code": "acejax-gp", "system": "Si", "size": "t", "path": str(path), "elements": ["Si"],
           "name": "acejax-gp/Si/t", "rattle": 0.05}
    out = run_case(row, 64, "float64", "cpu", reps=2)
    assert out["status"] == "ok" and out["call_s"] > 0 and out["md_like"] is True
```

- [ ] **Step 2: Check the GP fixture exists**

Run: `find fixtures -name gp_model.npz | head -3`
- If it prints a path whose elements are Si, keep the test as written.
- If nothing prints, or the model is not Si-only: create a tiny Si GP once with
  `uv run --no-sync aj fit --train fixtures/si_tiny_train.xyz --energy-key dft_energy --force-key dft_force --order 2 --max-degree 4 --m-per-species 8 --rungs map --e0 lsq --out <tmp>`.
  Point the test at it through a session-scoped fixture that runs that command
  into `tmp_path_factory`, mark the test `@pytest.mark.slow`, and drop the skip.
  Never commit a new model file.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_bench_scaling.py -k "si_supercell or unrattled or quip or acejax_gp" -v -n 0`
Expected: FAIL with `KeyError: 'Si'` / `TypeError: unexpected keyword argument 'rattle'` / `'pair_extra'`

- [ ] **Step 4: Implement the structures change**

In `bench/scaling/structures.py`, add `Si` to `SYSTEMS`:

```python
    "Si": {"lattice": "diamond", "a": 5.461, "elements": ("Si",), "per_cell": 8},   # GAP-18 PW91 a0
```

Then add the `rattle` kwarg to `supercell`:

```python
def supercell(system, n_atoms, seed=0, rattle=0.0):
    ...                                           # existing body up to set_chemical_symbols
    if rattle:                                    # a thermal-like cell: no zero-force symmetric start
        at.positions += np.random.default_rng(seed + 1).normal(0, rattle, at.positions.shape)
    return at
```

Update the module docstring's first line to name Si.

- [ ] **Step 5: Implement the runner changes**

In `run_standalone.run_case`, make the first line
`at = supercell(row["system"], n_atoms, rattle=row.get("rattle", 0.0))`. Then
add a branch before `elif row["code"] == "mace":`:

```python
        elif row["code"] == "acejax-gp":
            import jax
            if device == "cpu":
                jax.config.update("jax_platforms", "cpu")
            jax.config.update("jax_enable_x64", dtype == "float64")
            out["platform"] = jax.default_backend()
            from ace_jax.calc.gp import GPCalculator
            calc = GPCalculator.from_file(row["path"])
            call = lambda: calc.calculate(at, ["energy", "forces"], all_changes)
            t0 = time.perf_counter(); call(); out["compile_s"] = time.perf_counter() - t0
            out["energy"] = calc.results["energy"]
            call()
            out["call_s"] = _median_time(call, reps, setup=displace)
            out["rebuilds"], out["md_like"] = None, True
```

The `acejax` branch tests `row["code"].startswith("acejax")`. Move the new
branch **above** it, or change that test to
`row["code"].startswith("acejax") and row["code"] != "acejax-gp"`, so GP rows
don't reach `ACECalculator`.

In `run_lammps.py`:

```python
KOKKOS = {"acejax": "newton on neigh half", "mlpace": "newton on neigh half",
          "mace": "newton on neigh half"}                      # gap: CPU only, no KOKKOS


def _pair(style, model_path, elements, device, extra=None):
    ...
    if style == "gap":                       # QUIP takes atomic numbers, and the xml label picks the Potential
        from ase.data import atomic_numbers
        zs = " ".join(str(atomic_numbers[e]) for e in elements)
        return f'pair_style quip\npair_coeff * * {model_path} "Potential xml_label={extra}" {zs}\n'
    raise ValueError(style)


def lammps_input(style, model_path, elements, data_path, device, run_steps, dump=None,
                 warmup=50, pair_extra=None):
    txt = (... + _pair(style, model_path, elements, device, pair_extra) + ...)
```

In `run_lammps.run_case`:
- make the supercell line `at = supercell(row["system"], n_atoms, rattle=row.get("rattle", 0.0))`;
- make the style line `style = "acejax" if row["code"].startswith("acejax") else row["code"]`. It already maps `gap` to `"gap"`, so no change is needed there;
- in `_run_lammps`, pass `pair_extra=row.get("quip_label")` to `lammps_input`.

The existing `if style != "acejax":` path then sends GAP through `_run_lammps`
with `model = row["path"]`.

In `parity.py`, add:

Add `"gap": (1e-8, 1e-7)` as a key in the existing `TOL` dict literal at
`parity.py:31` (LAMMPS `pair quip` and quippy run the same Fortran), then add:

```python
def _quip_ef(path, at, label=None):
    import quippy
    p = quippy.potential.Potential(param_filename=str(path), **({"args_str": f"Potential xml_label={label}"} if label else {}))
    a = at.copy(); a.calc = p
    return a.get_potential_energy(), a.get_forces()
```

- [ ] **Step 6: Run the new and existing scaling tests**

Run: `uv run pytest tests/test_bench_scaling.py -v -n 0`
Expected: all passed. The GP test is skipped only if Step 2 found no fixture and Step 2's slow path was not taken.

- [ ] **Step 7: Commit**

```bash
git add bench/scaling/structures.py bench/scaling/run_standalone.py bench/scaling/run_lammps.py \
        bench/scaling/parity.py tests/test_bench_scaling.py
git commit -m "feat(bench): scaling harness takes Si, rattled cells, GP and pair quip rows

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: The ace-jax fits (linear ACE, learned-radial linear ACE and ACE+GP) and `sweep.yaml`

**Files:**
- Create: `bench/accuracy_cost/sweep.yaml`
- Create: `bench/accuracy_cost/fit_ace.py`
- Test: `tests/test_bench_accuracy_cost.py` (append)

**Interfaces:**
- Consumes: `common.fit_once`, `common.entry`, `common.DATA`, `common.RCUT`, `split.ensure_split`, `weights.write_all`
- Produces:
  - `fit_ace.command(family, e: dict, out: Path) -> list[str]`, where family is `"ace"`, `"ace_lr"` or `"ace_gp"`
  - `fit_ace.MODEL = {"ace": "model.npz", "ace_lr": "model.npz", "ace_gp": "gp_model.npz"}`
  - the CLI `python -m accuracy_cost.fit_ace <ace|ace_lr|ace_gp> <size|all>`

- [ ] **Step 1: Fix the linear ACE ladder from real basis sizes**

For each candidate, run
`uv run --no-sync aj basis --elements Si --order O --max-degree D --rcut 5.0 --out /tmp/claude-$(id -u)/b.npz`
and read n_B from its log:

| O | D candidates |
|---|---|
| 3 | 6, 8, 10, 12 |
| 4 | 8, 10, 12, 14 |

Check `aj basis --help` for the exact flag names (`add_basis_args` in
`cli.py`), and adjust the command if `--elements` takes atomic numbers.

Choose 6 entries with n_B spaced roughly evenly in log over about 50 to 1,500.
Record each entry's n_B as `n_B:` in `sweep.yaml`.

- [ ] **Step 2: Write `sweep.yaml`**

Use the sizes chosen in Step 1. The ace_gp entries reuse three of the ace
`(order, max_degree)` pairs (smallest, middle, largest-but-one) with `m: 100`
and `m: 500`, minus the largest pair at `m: 500`, for 5 points.

```yaml
# family -> size entries; `size` names the run directory runs/<family>/<size>
ace:          # linear: aj fit --m-per-species 0; n_B from `aj basis` (Task 5 step 1)
  - {size: o3d6,  order: 3, max_degree: 6,  n_B: 0}
  - {size: o3d8,  order: 3, max_degree: 8,  n_B: 0}
  - {size: o3d10, order: 3, max_degree: 10, n_B: 0}
  - {size: o4d10, order: 4, max_degree: 10, n_B: 0}
  - {size: o4d12, order: 4, max_degree: 12, n_B: 0}
  - {size: o4d14, order: 4, max_degree: 14, n_B: 0}
ace_lr:       # linear + aj fit --learn-radial; the same (order, max_degree) pairs and n_B as ace
  - {size: o3d6,  order: 3, max_degree: 6,  n_B: 0}
  - {size: o3d8,  order: 3, max_degree: 8,  n_B: 0}
  - {size: o3d10, order: 3, max_degree: 10, n_B: 0}
  - {size: o4d10, order: 4, max_degree: 10, n_B: 0}
  - {size: o4d12, order: 4, max_degree: 12, n_B: 0}
  - {size: o4d14, order: 4, max_degree: 14, n_B: 0}
ace_gp:       # aj fit --m-per-species m
  - {size: o3d6-m100,  order: 3, max_degree: 6,  m: 100}
  - {size: o3d6-m500,  order: 3, max_degree: 6,  m: 500}
  - {size: o3d10-m100, order: 3, max_degree: 10, m: 100}
  - {size: o3d10-m500, order: 3, max_degree: 10, m: 500}
  - {size: o4d12-m100, order: 4, max_degree: 12, m: 100}
pace:         # one ladder fit; sizes are checkpoints (functions per element)
  - {size: ladder, max_functions: 800, ladder_step: 50, take: [50, 100, 200, 400, 800]}
mace:         # mace_run_train from scratch, 2 layers, max_ell 3, correlation 3
  - {size: c8L0,  channels: 8,  max_L: 0}
  - {size: c16L0, channels: 16, max_L: 0}
  - {size: c32L0, channels: 32, max_L: 0}
  - {size: c16L1, channels: 16, max_L: 1}
  - {size: c32L1, channels: 32, max_L: 1}
  - {size: c64L1, channels: 64, max_L: 1}
gap:          # GAP-18's gap_fit command line, only n_sparse changed
  - {size: s1000, n_sparse: 1000}
  - {size: s3000, n_sparse: 3000}
  - {size: s9000, n_sparse: 9000}
```

Replace every `n_B: 0` with the measured value, and the `(order, max_degree)`
pairs with Step 1's picks. A zero left in the file is a bug, and the Step 3
test catches it.

- [ ] **Step 3: Write the failing tests**

```python
def test_sweep_is_complete_and_measured():
    from accuracy_cost.common import load_sweep
    s = load_sweep()
    assert set(s) == {"ace", "ace_lr", "ace_gp", "pace", "mace", "gap"}
    assert len(s["ace"]) == 6 and all(e["n_B"] > 0 for e in s["ace"])
    assert [(e["order"], e["max_degree"]) for e in s["ace_lr"]] == [(e["order"], e["max_degree"]) for e in s["ace"]]
    nB = [e["n_B"] for e in s["ace"]]
    assert nB == sorted(nB) and nB[0] <= 80 and nB[-1] >= 1000
    for fam, es in s.items():
        assert len({e["size"] for e in es}) == len(es), fam


def test_fit_ace_commands_share_cutoff_weights_and_e0(tmp_path):
    from accuracy_cost.fit_ace import command
    lin = command("ace", {"order": 3, "max_degree": 8}, tmp_path)
    gp = command("ace_gp", {"order": 3, "max_degree": 8, "m": 100}, tmp_path)
    for c in (lin, gp):
        s = " ".join(c)
        assert "--rcut 5.0" in s and "--e0 lsq" in s and "--rungs map" in s and "weights_acejax.json" in s
        assert "--virial-key virial" in s
    assert lin[lin.index("--m-per-species") + 1] == "0" and gp[gp.index("--m-per-species") + 1] == "100"
    lr = command("ace_lr", {"order": 3, "max_degree": 8}, tmp_path)
    assert "--learn-radial" in lr and lr[lr.index("--m-per-species") + 1] == "0"
    assert "--learn-radial" not in lin and "--learn-radial" not in gp
```

- [ ] **Step 4: Run them to verify they fail**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -k "sweep or fit_ace" -v -n 0`
Expected: FAIL with `ModuleNotFoundError: No module named 'accuracy_cost.fit_ace'`. If Step 2 was skipped, the sweep test also fails.

- [ ] **Step 5: Implement `fit_ace.py`**

```python
"""Linear ACE and ACE+GP through `aj fit`: the CLI's defaults, plus the shared
cutoff, weights (E, F and V sigmas) and E0 (--e0 lsq takes the isolated-atom
energy, spec)."""
import sys

import numpy as np

from . import common
from .common import DATA, RCUT

MODEL = {"ace": "model.npz", "ace_lr": "model.npz", "ace_gp": "gp_model.npz"}


def command(family, e, out):
    return ["aj", "fit", "--train", str(DATA / "train.xyz"), "--order", str(e["order"]),
            "--max-degree", str(e["max_degree"]), "--rcut", str(RCUT), "--virial-key", "virial", "--weights",
            str(DATA / "weights_acejax.json"), "--e0", "lsq", "--rungs", "map",
            "--m-per-species", str(e.get("m", 0) if family == "ace_gp" else 0), "--out", str(out),
            *(["--learn-radial"] if family == "ace_lr" else [])]


def n_params(family, model):
    z = np.load(model)
    if family in ("ace", "ace_lr"):
        return int(z["WB"].size + z["Wpair"].size)
    keys = [k for k in z.files if k.endswith("WB") or k.endswith("Wpair")]
    return int(sum(z[k].size for k in keys))                     # + inducing count, added by the caller


def fit(family, size):
    from .split import ensure_split
    from .weights import write_all
    h = ensure_split(); write_all(DATA)
    e = common.entry(family, size)
    d = common.run_dir(family, size)
    st = common.fit_once(family, size, MODEL[family], command(family, e, d), split_hash=h)
    if st["status"] == "ok" and st.get("n_params") is None:
        st["n_params"] = n_params(family, d / MODEL[family]) + (e.get("m", 0) if family == "ace_gp" else 0)
        common.write_status(d, **{k: v for k, v in st.items() if k != "host"})
    return st


if __name__ == "__main__":
    fam, which = sys.argv[1], sys.argv[2]
    sizes = [e["size"] for e in common.load_sweep()[fam]] if which == "all" else [which]
    for s in sizes:
        print(s, fit(fam, s)["status"])
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -v -n 0`
Expected: 13 passed

- [ ] **Step 7: Smoke-fit the smallest model and score it**

On moriarty, from `bench/`, run
`uv run --no-sync python -m accuracy_cost.fit_ace ace o3d6 && uv run --no-sync python -m accuracy_cost.evaluate ace o3d6`.
Expected: `o3d6 ok`, then an `rmse_F` line between 0.05 and 1 eV/Å.

If the `n_params` key lookup for GP files fails, list the keys with
`python -c "import numpy as np; print(np.load('<gp_model.npz>').files)"`, fix
the key filter (the ACE arrays are prefixed `ace/` per
`fit/pipeline/export.py`), and rerun.

- [ ] **Step 8: Commit**

```bash
git add bench/accuracy_cost/sweep.yaml bench/accuracy_cost/fit_ace.py tests/test_bench_accuracy_cost.py \
        bench/accuracy_cost/results/metrics.jsonl
git commit -m "feat(bench): ace-jax linear and GP fits for the accuracy-cost sweep

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: PACE through a pacemaker ladder fit

**Files:**
- Create: `bench/accuracy_cost/envs/pacemaker/pyproject.toml`
- Create: `bench/accuracy_cost/fit_pace.py`
- Test: `tests/test_bench_accuracy_cost.py` (append)

**Interfaces:**
- Consumes: `common.*`, `split.read_atoms`, `weights.gap_sigmas`, `weights.command_line`, `weights.pacemaker`
- Produces:
  - `fit_pace.input_yaml(e: dict) -> dict`: the pacemaker config
  - `fit_pace.dataframes(atoms, wts) -> (data_df, weights_df)`
  - `fit_pace.checkpoint_for(run_dir, n_funcs) -> Path`
  - CLI `python -m accuracy_cost.fit_pace` (run under the pacemaker venv): one ladder run into `runs/pace/ladder/`, then one run directory per `take` value, `runs/pace/f<N>/` holding `model.yace` and `status.json` with `n_params = N`

- [ ] **Step 1: Create the pacemaker venv and pin the version**

`envs/pacemaker/pyproject.toml`:

```toml
[project]
name = "ac-pacemaker"
version = "0"
requires-python = ">=3.9,<3.12"
dependencies = ["python-ace", "tensorflow[and-cuda]>=2.8,<2.16", "pandas", "pyyaml", "extxyz"]
```

Run:
```bash
cd bench/accuracy_cost/envs/pacemaker && UV_PROJECT_ENVIRONMENT=/storage/eng/essswb/venvs/ac-pacemaker uv sync
```
Use `uv pip install -e /home/eng/essswb/projects/ace-jax --no-deps` into the
same venv, so `evaluate` and `split.read_atoms` import. Run `pacemaker --version`
and record the version in `data.md`.

Then run `pacemaker -t`, which writes the default `input.yaml` template, and
read it. **Confirm these key names in the template before writing Step 3:**
- `cutoff`, `seed`;
- `data.filename`;
- `potential.elements`, `potential.bonds.ALL.rcut`, `potential.embeddings.ALL.{npot,fs_parameters,ndensity}`, `potential.functions.number_of_functions_per_element`;
- `fit.loss.kappa`, `fit.weighting` (the `ExternalWeightingPolicy` and its `filename`), `fit.ladder_step`, `fit.ladder_type`, `fit.maxiter`;
- `backend.evaluator: tensorpot`, `backend.batch_size`.

If any name differs in the installed version, use the installed name in
`input_yaml` and its test.

**Stress fitting:** search the installed pyace/pacemaker sources and the template
for stress or virial support (`grep -rn -i "stress\|virial"` over the package
directory). If the installed pacemaker can fit stresses, add its stress option
and a per-structure stress weight of `1/sV**2` (from `gap_sigmas`) to
`input_yaml` / `dataframes`, and extend the tests to match. Otherwise PACE fits
E+F only. Record which in `data.md`.

- [ ] **Step 2: Write the failing test**

```python
def test_pace_input_shares_cutoff_e0_and_ladder():
    from accuracy_cost.fit_pace import input_yaml
    y = input_yaml({"max_functions": 800, "ladder_step": 50})
    assert y["cutoff"] == 5.0 and y["potential"]["bonds"]["ALL"]["rcut"] == 5.0
    assert y["potential"]["functions"]["number_of_functions_per_element"] == 800
    assert y["fit"]["ladder_step"] == 50 and y["fit"]["weighting"]["type"] == "ExternalWeightingPolicy"
    assert y["data"]["reference_energy"] == {"Si": -158.54496821}


def test_pace_dataframes_carry_weights_by_index():
    pd = pytest.importorskip("pandas")
    from accuracy_cost.fit_pace import dataframes
    from accuracy_cost.split import normalise
    from accuracy_cost.weights import pacemaker
    from ace_jax.fit.xyz import read_extxyz
    ats = [a for a in map(normalise, read_extxyz(TINY)) if a is not None][:4]
    data, w = dataframes(ats, pacemaker((0.001, 0.1, 0.05), {}, ats))
    assert list(data.index) == list(w.index) and len(data) == 4
    assert np.isclose(w["w_energy"].iloc[0], 1e6) and np.allclose(w["w_forces"].iloc[1], 100.0)
```

- [ ] **Step 3: Run it to verify it fails, then implement `fit_pace.py`**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -k pace -v -n 0`
Expected: FAIL with `ModuleNotFoundError`

```python
"""PACE via one pacemaker ladder fit (how a PACE practitioner reaches a small
accurate model): each size is a ladder checkpoint, converted to .yace.  The
pacemaker defaults stand, apart from the shared cutoff, E0 and GAP-18 weights.
Run under the pacemaker venv."""
import pathlib
import shutil
import subprocess

from . import common
from .common import DATA, E0_SI, RCUT


def input_yaml(e):
    return {
        "cutoff": RCUT, "seed": 0,
        "data": {"filename": "train.pckl.gzip", "reference_energy": {"Si": E0_SI}},
        "potential": {"deltaSplineBins": 0.001, "elements": ["Si"],
                      "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled",
                                             "fs_parameters": [1, 1, 1, 0.5], "ndensity": 2}},
                      "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [5.25], "rcut": RCUT,
                                        "dcut": 0.01, "NameOfCutoffFunction": "cos"}},
                      "functions": {"number_of_functions_per_element": e["max_functions"],
                                    "ALL": {"nradmax_by_orders": [15, 6, 4, 3, 2, 2],
                                            "lmax_by_orders": [0, 4, 3, 2, 1, 0]}}},
        "fit": {"loss": {"kappa": "auto", "L1_coeffs": 1e-8, "L2_coeffs": 1e-8},
                "weighting": {"type": "ExternalWeightingPolicy", "filename": "weights.pckl.gzip"},
                "optimizer": "BFGS", "maxiter": 1500,
                "ladder_step": e["ladder_step"], "ladder_type": "power_order"},
        "backend": {"evaluator": "tensorpot", "batch_size": 50, "display_step": 50},
    }


def dataframes(atoms, wts):
    import pandas as pd
    data = pd.DataFrame({"ase_atoms": atoms, "energy": [a.info["energy"] for a in atoms],
                         "forces": [a.arrays["forces"] for a in atoms]})
    w = pd.DataFrame({"w_energy": [x[0] for x in wts], "w_forces": [x[1] for x in wts]}, index=data.index)
    return data, w


def checkpoint_for(d, n):
    p = pathlib.Path(d) / f"interim_potential_ladder_step_{n}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"{p}: ladder never reached {n} functions (see {d}/fit.log)")
    return p


def main():
    import yaml
    from .split import XML, ensure_split, read_atoms
    from .weights import command_line, gap_sigmas, pacemaker
    h = ensure_split()
    e = common.entry("pace", "ladder")
    d = common.run_dir("pace", "ladder"); d.mkdir(parents=True, exist_ok=True)
    ats = read_atoms(DATA / "train.xyz")
    data, w = dataframes(ats, pacemaker(*gap_sigmas(command_line(DATA / XML)), ats))
    data.to_pickle(d / "train.pckl.gzip", compression="gzip"); w.to_pickle(d / "weights.pckl.gzip", compression="gzip")
    (d / "input.yaml").write_text(yaml.safe_dump(input_yaml(e)))
    st = common.fit_once("pace", "ladder", "output_potential.yaml", ["pacemaker", "input.yaml"], split_hash=h)
    if st["status"] != "ok":
        raise SystemExit(st.get("error"))
    for n in e["take"]:
        out = common.run_dir("pace", f"f{n}"); out.mkdir(parents=True, exist_ok=True)
        src = d / "output_potential.yaml" if n == e["max_functions"] else checkpoint_for(d, n)
        subprocess.run(["pace_yaml2yace", str(src), "-o", str(out / "model.yace")], check=True)
        shutil.copy(src, out / "model.yaml")
        common.write_status(out, family="pace", size=f"f{n}", status="ok", model=str(out / "model.yace"),
                            n_params=n, split_hash=h, fit_wall_s=st["fit_wall_s"], ladder=str(d))


if __name__ == "__main__":
    main()
```

Spec point: every PACE size takes the ladder's total wall time as its fit time
(`fit_wall_s`), and the write-up says so.

**Checkpoint names:** after the first real run, check how pacemaker names
interim ladder files (`ls runs/pace/ladder`). If the step number in the name
counts ladder steps rather than functions, change `checkpoint_for` to read each
interim YAML's function count (`len(yaml["functions"])` per element) and pick
the file whose count equals `n`. Under `power_order` with `ladder_step: 50`, the
steps may not land exactly on 100/200/400: then take the first checkpoint with
`count >= n`, and record the actual count as `n_params`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -k pace -v -n 0`
Expected: 2 passed, or 1 passed and 1 skipped if the project venv lacks pandas. In that case, also run the dataframe test under the pacemaker venv: `/storage/eng/essswb/venvs/ac-pacemaker/bin/python -m pytest tests/test_bench_accuracy_cost.py -k pace_dataframes -p no:cacheprovider`

- [ ] **Step 5: Run the ladder on moriarty's GPU and score it**

Check `nvidia-smi` shows no other heavy job. Then, from `bench/`, run:
`setsid nohup /storage/eng/essswb/venvs/ac-pacemaker/bin/python -m accuracy_cost.fit_pace > /storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/pace/driver.log 2>&1 < /dev/null &`

When it finishes, from `bench/`, run
`for n in 50 100 200 400 800; do uv run --no-sync python -m accuracy_cost.evaluate pace f$n; done`
in the project venv: ace-jax's `PACEModel` scores the `.yace` files.

Cross-check one size against pyace: in the pacemaker venv, load
`runs/pace/f100/model.yaml` with `pyace.PyACECalculator`, compute five test
frames, and require energies to agree with ace-jax to 1e-6 eV/atom and forces
to 1e-5 eV/Å (the `mlpace` tolerance in `bench/scaling/parity.py`). Record the
result in `runs/pace/f100/pyace_check.txt`.

- [ ] **Step 6: Commit**

```bash
git add bench/accuracy_cost/envs/pacemaker/pyproject.toml bench/accuracy_cost/fit_pace.py \
        tests/test_bench_accuracy_cost.py bench/accuracy_cost/results/metrics.jsonl bench/accuracy_cost/data.md
git commit -m "feat(bench): PACE ladder fit for the accuracy-cost sweep

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: MACE from scratch and its Symmetrix export

**Files:**
- Create: `bench/accuracy_cost/envs/mace/pyproject.toml`
- Create: `bench/accuracy_cost/fit_mace.py`
- Test: `tests/test_bench_accuracy_cost.py` (append)

**Interfaces:**
- Consumes: `common.*`, `weights` (`weights_mace.json`), `scaling.models.symmetrix_cmd(model, zs, head, out)`
- Produces:
  - `fit_mace.command(e, d) -> list[str]`
  - CLI `python -m accuracy_cost.fit_mace <size|all>`, writing `runs/mace/<size>/model.model` (the float64 deployed model), `model-f32.model`, `model.json` (Symmetrix) and a `status.json` with `n_params`

- [ ] **Step 1: Create the MACE venv**

`envs/mace/pyproject.toml`:

```toml
[project]
name = "ac-mace"
version = "0"
requires-python = ">=3.10,<3.13"
dependencies = ["mace-torch", "torch", "pyyaml", "extxyz", "matscipy"]
```

Run `UV_PROJECT_ENVIRONMENT=/storage/eng/essswb/venvs/ac-mace uv sync`. Then
`uv pip install -e /home/eng/essswb/projects/ace-jax --no-deps` into the same
venv, and install `symmetrix` the way `bench/scaling/envs/moriarty.sh` does
(read its `symmetrix_py` step and copy the command). Run
`mace_run_train --help`. **Confirm these flags exist:**
- `--train_file`, `--test_file`, `--valid_fraction`;
- `--E0s`, `--energy_key`, `--forces_key`, `--config_type_weights`;
- `--model`, `--num_interactions`, `--num_channels`, `--max_L`, `--max_ell`, `--correlation`, `--r_max`;
- `--batch_size`, `--max_num_epochs`, `--swa`, `--ema`;
- `--virials_key`, `--virials_weight`, and a `--loss` choice that includes virials (`--loss=virials` in current mace-torch). Check that configs without a virial get zero virial weight rather than an error: train for 1 epoch on a 20-frame slice mixing frames with and without virials.
- `--default_dtype`, `--device`, `--seed`, `--name`, `--results_dir`, `--model_dir`.

Use the installed names. Record the `mace-torch` version in `data.md`.

- [ ] **Step 2: Write the failing test**

```python
def test_mace_command_shares_cutoff_e0_and_weights(tmp_path):
    from accuracy_cost.fit_mace import command
    c = command({"channels": 16, "max_L": 0}, tmp_path)
    s = " ".join(c)
    assert "--r_max=5.0" in s and "--E0s={14: -158.54496821}" in s and "--config_type_weights=" in s
    assert "--num_channels=16" in s and "--max_L=0" in s and "--num_interactions=2" in s
    assert "--correlation=3" in s and "--max_ell=3" in s
    assert "--virials_key=virial" in s and "--loss=virials" in s
```

- [ ] **Step 3: Run it to verify it fails, then implement `fit_mace.py`**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -k mace -v -n 0`
Expected: FAIL with `ModuleNotFoundError`

```python
"""MACE trained from scratch: mace_run_train's defaults, with the shared cutoff,
E0, virials (--loss=virials, MACE's own global virials weight) and GAP-18 per-type weights (as config_type_weights: MACE weights a config's
whole loss, so the force sigma sets the ratio).  2 layers, max_ell 3,
correlation 3; only channels and max_L are swept.  Run under the MACE venv."""
import json
import subprocess
import sys

from . import common
from .common import DATA, E0_SI, RCUT


def command(e, d):
    w = json.loads((DATA / "weights_mace.json").read_text()) if (DATA / "weights_mace.json").exists() else {"Default": 1.0}
    return ["mace_run_train", "--name=model", f"--train_file={DATA / 'train.xyz'}", "--valid_fraction=0.05",
            f"--test_file={DATA / 'test.xyz'}", "--energy_key=energy", "--forces_key=forces",
            "--virials_key=virial", "--loss=virials",
            f"--E0s={{14: {E0_SI}}}", f"--config_type_weights={json.dumps(w)}", "--model=MACE",
            "--num_interactions=2", f"--num_channels={e['channels']}", f"--max_L={e['max_L']}",
            "--max_ell=3", "--correlation=3", f"--r_max={RCUT}", "--batch_size=5",
            "--max_num_epochs=300", "--swa", "--ema", "--default_dtype=float64", "--device=cuda",
            "--seed=0", f"--results_dir={d}", f"--model_dir={d}", f"--checkpoints_dir={d / 'ckpt'}"]


def fit(size):
    from .split import ensure_split
    from .weights import write_all
    h = ensure_split(); write_all(DATA)
    e, d = common.entry("mace", size), common.run_dir("mace", size)
    st = common.fit_once("mace", size, "model.model", command(e, d), split_hash=h)
    if st["status"] != "ok" or st.get("n_params"):
        return st
    import torch
    m = torch.load(d / "model.model", weights_only=False)
    st["n_params"] = int(sum(p.numel() for p in m.parameters()))
    m.float(); torch.save(m, d / "model-f32.model")                       # the float32 timing variant
    sys.path.insert(0, str(common.HERE.parent))
    from scaling.models import symmetrix_cmd
    subprocess.run(symmetrix_cmd(str(d / "model.model"), [14], None, str(d / "model.json")), check=True)
    common.write_status(d, **{k: v for k, v in st.items() if k != "host"})
    return st


if __name__ == "__main__":
    which = sys.argv[1]
    for s in ([e["size"] for e in common.load_sweep()["mace"]] if which == "all" else [which]):
        print(s, fit(s)["status"])
```

Before running, open `bench/scaling/models.py:219` (`symmetrix_cmd`) and check
its argument types (`zs` as ints or symbols, `head=None` allowed). Adapt the
call to match: Symmetrix needs element-specific `.json`, as the scaling README
says.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -k mace -v -n 0`
Expected: 1 passed

- [ ] **Step 5: Train the ladder on moriarty, one at a time, and score it**

From `bench/`, run:
`setsid nohup /storage/eng/essswb/venvs/ac-mace/bin/python -m accuracy_cost.fit_mace all > /storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/mace/driver.log 2>&1 < /dev/null &`

This trains sequentially. Monitor with `tail` on the per-size `fit.log`. When
it is done, still from `bench/`, run
`for s in c8L0 c16L0 c32L0 c16L1 c32L1 c64L1; do /storage/eng/essswb/venvs/ac-mace/bin/python -m accuracy_cost.evaluate mace $s; done`.
Expected: each prints an `rmse_F`, and larger models have lower errors overall.

- [ ] **Step 6: Commit**

```bash
git add bench/accuracy_cost/envs/mace/pyproject.toml bench/accuracy_cost/fit_mace.py \
        tests/test_bench_accuracy_cost.py bench/accuracy_cost/results/metrics.jsonl bench/accuracy_cost/data.md
git commit -m "feat(bench): MACE-from-scratch fits for the accuracy-cost sweep

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: The GAP refit with `gap_fit`

**Files:**
- Create: `bench/accuracy_cost/envs/quip/pyproject.toml`
- Create: `bench/accuracy_cost/fit_gap.py`
- Test: `tests/test_bench_accuracy_cost.py` (append)

**Interfaces:**
- Consumes: `weights.command_line`, `split.XML`, `split.GLUE`, `common.*`
- Produces:
  - `fit_gap.refit_command(cl: str, n_sparse: int, train: str, gp_file: str) -> str`: GAP-18's line with `at_file`, `n_sparse` and `gp_file` substituted, and the label names set to `energy` / `forces` / `virial`
  - `fit_gap.xml_label(xml_path) -> str`: the top-level `Potential` label for `pair quip`
  - CLI `python -m accuracy_cost.fit_gap <size|all>`, writing `runs/gap/<size>/gap.xml` and a status with `n_params = n_sparse` and `quip_label`

- [ ] **Step 1: Write the failing tests**

```python
GAP18 = ("at_file=si_iter6_database.xyz gap={soap l_max=12 n_max=10 atom_sigma=0.5 zeta=4 cutoff=5.0 "
         "cutoff_transition_width=1.0 central_weight=1.0 n_sparse=9000 delta=3.0 f0=0.0 "
         "covariance_type=dot_product sparse_method=cur_points} default_sigma={0.001 0.1 0.05 0.0} "
         "config_type_sigma={liq:0.003:0.15:0.2:0.0} energy_parameter_name=dft_energy "
         "force_parameter_name=dft_force virial_parameter_name=dft_virial config_type_parameter_name=config_type "
         "sparse_jitter=1.0e-8 e0_offset=2.0 core_param_file=glue.xml core_ip_args={IP Glue} gp_file=gp_iter6_sparse9k.xml")


def test_gap_refit_changes_only_data_size_and_labels():
    from accuracy_cost.fit_gap import refit_command
    c = refit_command(GAP18, 1000, "/d/train.xyz", "gap.xml")
    assert "at_file=/d/train.xyz" in c and "n_sparse=1000" in c and "gp_file=gap.xml" in c
    assert "energy_parameter_name=energy" in c and "force_parameter_name=forces" in c
    assert "virial_parameter_name=virial" in c and "n_sparse=9000" not in c
    for keep in ("l_max=12", "cutoff=5.0", "zeta=4", "default_sigma={0.001 0.1 0.05 0.0}",
                 "core_param_file=glue.xml", "core_ip_args={IP Glue}", "e0_offset=2.0", "sparse_method=cur_points"):
        assert keep in c


def test_xml_label_reads_top_level_potential(tmp_path):
    from accuracy_cost.fit_gap import xml_label
    p = tmp_path / "g.xml"
    p.write_text('<Potential label="GAP_X" init_args="Sum init_args_pot1={IP Glue} init_args_pot2={IP GAP label=GAP_Y}"/>\n<GAP_params label="GAP_Y"/>')
    assert xml_label(p) == "GAP_X"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -k "gap_refit or xml_label" -v -n 0`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `fit_gap.py` and the venv**

`envs/quip/pyproject.toml`:

```toml
[project]
name = "ac-quip"
version = "0"
requires-python = ">=3.9,<3.13"
dependencies = ["quippy-ase", "extxyz", "pyyaml"]
```

```python
"""GAP refitted on the train split with GAP-18's own gap_fit command line (the
published recipe, glue core included); only the data, n_sparse and output
change.  Run under the quip
venv, on mnf148: 9k sparse points needs tens of GB."""
import re
import shutil
import sys

from . import common
from .common import DATA


def refit_command(cl, n_sparse, train, gp_file):
    c = re.sub(r"\bat_file=\S+", f"at_file={train}", cl.strip())
    c = re.sub(r"\bn_sparse=\d+", f"n_sparse={n_sparse}", c)
    c = re.sub(r"\bgp_file=\S+", f"gp_file={gp_file}", c)
    c = re.sub(r"\benergy_parameter_name=\S+", "energy_parameter_name=energy", c)
    c = re.sub(r"\bforce_parameter_name=\S+", "force_parameter_name=forces", c)
    return re.sub(r"\bvirial_parameter_name=\S+", "virial_parameter_name=virial", c)


def xml_label(path):
    m = re.search(r'<Potential\s+label="([^"]+)"', open(path).read())
    if not m:
        raise ValueError(f"{path}: no top-level <Potential label=...>")
    return m.group(1)


def fit(size):
    from .split import GLUE, XML, ensure_split
    from .weights import command_line
    h = ensure_split()
    e, d = common.entry("gap", size), common.run_dir("gap", size)
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / GLUE, d / GLUE)                            # core_param_file is read from cwd
    args = refit_command(command_line(DATA / XML), e["n_sparse"], str(DATA / "train.xyz"), "gap.xml")
    st = common.fit_once("gap", size, "gap.xml", ["gap_fit", *args.split()], split_hash=h)
    if st["status"] == "ok" and not st.get("quip_label"):
        st.update(n_params=e["n_sparse"], quip_label=xml_label(d / "gap.xml"))
        common.write_status(d, **{k: v for k, v in st.items() if k != "host"})
    return st


if __name__ == "__main__":
    which = sys.argv[1]
    for s in ([e["size"] for e in common.load_sweep()["gap"]] if which == "all" else [which]):
        print(s, fit(s)["status"])
```

`args.split()` breaks the braces apart: `gap={soap l_max=12 ...}` contains
spaces. gap_fit parses its whole argv joined with spaces, so the split is
harmless, but check this in Step 5 by confirming that `cmd.txt` reproduces the
line exactly. If gap_fit rejects it, pass `["gap_fit", args]` (one argv element)
instead.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -k "gap_refit or xml_label" -v -n 0`
Expected: 2 passed

- [ ] **Step 5: Fit on mnf148 and score**

First check mnf148 is idle: `ssh mnf148 'cat /proc/loadavg; free -g; who'`. It
must show load under 4, at least 150 GB free and no other heavy user.

Build the venv on mnf148, where /storage is shared:
`ssh mnf148 'cd ~/projects/ace-jax/bench/accuracy_cost/envs/quip && UV_PROJECT_ENVIRONMENT=/storage/eng/essswb/venvs/ac-quip ~/.local/bin/uv sync && /storage/eng/essswb/venvs/ac-quip/bin/python -m pip --version'`
Then install ace-jax into it the same way (`uv pip install -e ~/projects/ace-jax --no-deps --python /storage/eng/essswb/venvs/ac-quip/bin/python`).

Launch with
`ssh mnf148 'cd ~/projects/ace-jax/bench && setsid nohup /storage/eng/essswb/venvs/ac-quip/bin/python -m accuracy_cost.fit_gap all > /storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/gap/driver.log 2>&1 < /dev/null &'`.
Watch the log from moriarty directly (shared /storage).

When it is done, from `bench/`, score each size under the same venv:
`/storage/eng/essswb/venvs/ac-quip/bin/python -m accuracy_cost.evaluate gap s1000 s3000 s9000`.

**Published GAP-18 reference:** fetch the published model's sparse-point file
`gp_iter6_sparse9k.xml.sparseX.GAP_2017_6_17_60_4_3_56_1651.bz2` from the same
pinned commit into `DATA`, `bunzip2` it next to `gp_iter6_sparse9k.xml` and
`glue.xml`, load the published XML with `quippy.potential.Potential(param_filename=...)`,
and score it with `evaluate.summarise(errors(...))` on `test.xyz` and on
`train.xyz`. Record both in `data.md` as "GAP-18 published (saw every frame)"
and write a `metrics.jsonl` row with family `gap18_published`, which Task 11
uses as the reference line.

**Sanity check:** the `s9000` training-set force RMSE should be close to
published GAP-18. Score it once on `train.xyz` by hand (a one-off
`summarise(errors(read_atoms(DATA/'train.xyz'), load_calc('gap', ...)))`) and
compare with `data.md`. Expect within about 20%, since GAP-18 also saw the test
frames. Record the result in `data.md`.

- [ ] **Step 6: Commit**

```bash
git add bench/accuracy_cost/envs/quip/pyproject.toml bench/accuracy_cost/fit_gap.py \
        tests/test_bench_accuracy_cost.py bench/accuracy_cost/results/metrics.jsonl bench/accuracy_cost/data.md
git commit -m "feat(bench): GAP-18 refit on the accuracy-cost split

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: The lestrade environment

**Files:**
- Create: `bench/accuracy_cost/envs/lestrade.sh`
- Modify: `bench/scaling/envs/moriarty.sh` (parametrise the Kokkos arch and the root; behaviour unchanged on moriarty)

**Interfaces:**
- Produces:
  - `$ROOT/lammps/build-kk/lmp` (Symmetrix + ML-PACE + ML-QUIP + KOKKOS ADA89)
  - `$ROOT/lammps-dev/build-kk/lmp` (develop + ML-PACE + PLUGIN + KOKKOS ADA89)
  - the lammps-jax plugin
  - `$ROOT/venv`
  - `envs/lestrade-{cpu,gpu}.json` in the same format as `bench/scaling/envs/moriarty-*.json`, which `bench/scaling/sweep.py` reads for host settings

  `ROOT=/storage/eng/essswb/bench-accuracy-cost-lestrade`.

- [ ] **Step 1: Parametrise `moriarty.sh`**

Replace the two literal `-D Kokkos_ARCH_AMPERE86=ON` with
`-D Kokkos_ARCH_${KOKKOS_ARCH:-AMPERE86}=ON`. Add
`EXTRA_PKGS=${EXTRA_PKGS:-}` near the top, and append `$EXTRA_PKGS` to the
`lammps` tree's cmake line (the one at `:64`). Leave every other line as it is.

Run `bash -n bench/scaling/envs/moriarty.sh` (syntax only) and
`grep -n KOKKOS_ARCH bench/scaling/envs/moriarty.sh`.
Expected: no output from `bash -n`, and two matches.

- [ ] **Step 2: Write `lestrade.sh`**

```bash
#!/bin/bash
# Accuracy-cost timing environment on lestrade (RTX 4000 Ada, sm_89, 32 cores):
# bench/scaling's moriarty build with the Ada Kokkos arch and ML-QUIP (GAP's
# pair quip) added to the Symmetrix/ML-PACE tree.  Build trees on /storage.
#   bash bench/accuracy_cost/envs/lestrade.sh [step ...]
set -euo pipefail
[ "$(hostname -s)" = lestrade ] || { echo "run on lestrade (Kokkos arch is per GPU)"; exit 1; }
export BENCH_ROOT=${BENCH_ROOT:-/storage/eng/essswb/bench-accuracy-cost-lestrade}
export ACEJAX_SRC=${ACEJAX_SRC:-$HOME/projects/ace-jax}
export KOKKOS_ARCH=ADA89
export EXTRA_PKGS="-D PKG_ML-QUIP=ON -D DOWNLOAD_QUIP=ON"
exec bash "$ACEJAX_SRC/bench/scaling/envs/moriarty.sh" "$@"
```

`moriarty.sh` loads `CUDA/12.4.0` and `CUDA/12.9.0` modules. Check that both
exist on lestrade with `ssh lestrade 'module avail CUDA 2>&1 | tail -5'`. If
they are named differently, add `CUDA_MAIN` / `CUDA_DEV` overrides to
`moriarty.sh` (as in Step 1, defaulting to today's values) and set them here.

`DOWNLOAD_QUIP=ON` makes LAMMPS build QUIP itself. It needs gfortran and
BLAS/LAPACK from the `gompi/2023a` toolchain. If the cmake step fails on it,
read `lammps/cmake/Modules/Packages/ML-QUIP.cmake` for the required variables
and set them in `EXTRA_PKGS`.

- [ ] **Step 3: Build on lestrade**

Run `ssh lestrade 'cat /proc/loadavg; free -g; nvidia-smi --query-gpu=name,memory.used --format=csv'`
to confirm it's idle. Then run
`ssh lestrade 'cd ~/projects/ace-jax && setsid nohup bash bench/accuracy_cost/envs/lestrade.sh > /storage/eng/essswb/bench-accuracy-cost-lestrade/build.log 2>&1 < /dev/null &'`.
Watch `build.log` from moriarty.

Expected at the end: both `lmp` binaries exist, and
`ssh lestrade '$BENCH_ROOT/lammps/build-kk/lmp -h | grep -E "quip|pace|symmetrix"'`
lists `quip`, `pace` and `symmetrix/mace`.

- [ ] **Step 4: Create the host JSONs**

Copy `bench/scaling/envs/moriarty-cpu.json` and `moriarty-gpu.json` to
`bench/accuracy_cost/envs/lestrade-cpu.json` and `lestrade-gpu.json`, replacing
every moriarty path with the lestrade `BENCH_ROOT` path. Check them with
`python -c "import json,sys; [json.load(open(p)) for p in sys.argv[1:]]" bench/accuracy_cost/envs/lestrade-*.json`.

- [ ] **Step 5: Commit**

```bash
git add bench/scaling/envs/moriarty.sh bench/accuracy_cost/envs/lestrade.sh bench/accuracy_cost/envs/lestrade-*.json
git commit -m "feat(bench): lestrade build for accuracy-cost timing (Ada sm_89, ML-QUIP)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Timing on lestrade, and the plot

**Files:**
- Create: `bench/accuracy_cost/timing.py`
- Create: `bench/accuracy_cost/plot.py`
- Test: `tests/test_bench_accuracy_cost.py` (append)

**Interfaces:**
- Consumes: `scaling.run_standalone.run_case`, `scaling.run_lammps.run_case`, the `parity` helpers, `common.*` and `results/metrics.jsonl`
- Produces:
  - `timing.N = {"gpu": (32768, 4096), "cpu": (4096, 512)}`
  - `timing.cases(device) -> list[dict]`: one row dict per (family code, size, dtype, mode), built from `status.json`s
  - `timing.idle(load_max=2.0) -> dict`: `{"ok", "load1", "gpu_procs"}`
  - `timing.plateau_flag(main_row, check_row, tol=0.2) -> bool`
  - `timing.main(device)`: writes `results/lestrade-{device}.jsonl`
  - `plot.pareto(metrics, timings, targets, out)`
  - `plot.smallest_at_target(metrics, timings, target, device) -> list[dict]`: per family `{family, size, rmse_F, n_params, atom_steps_per_s}`, or `{family, size: None, note: "not reached"}`
  - `plot.heatmap(metrics, sizes, out)`

The table of codes per device is the spec's:

| family | code row | CPU mode | GPU mode | dtypes |
|---|---|---|---|---|
| ace, ace_lr | `acejax-ace` | standalone | lammps | float64 |
| ace_gp | `acejax-gp` | standalone | standalone | float64 |
| pace | `acejax-pace` | standalone | lammps | float64 |
| pace | `mlpace` | lammps (32 ranks) | lammps | float64 |
| mace | `mace` | lammps | lammps | float64; plus standalone float32 on both |
| gap | `gap` | lammps (32 ranks) | — | float64 |

- [ ] **Step 1: Write the failing tests**

```python
def test_plateau_flag():
    from accuracy_cost.timing import plateau_flag
    main, check = {"n_atoms": 32768, "step_s": 1.0}, {"n_atoms": 4096, "step_s": 0.125}
    assert plateau_flag(main, check) is False
    assert plateau_flag(main, {"n_atoms": 4096, "step_s": 0.25}) is True     # check is 2x slower per atom


def _synthetic():
    m = [{"family": "ace", "size": s, "rmse_F": f, "n_params": n} for s, f, n in
         (("a", 0.20, 50), ("b", 0.12, 200), ("c", 0.08, 800))]
    m += [{"family": "mace", "size": "m1", "rmse_F": 0.15, "n_params": 9000}]
    t = [{"family": r["family"], "size": r["size"], "device": "gpu", "n_atoms": 32768, "status": "ok",
          "atom_steps_per_s": 1e7 / r["n_params"] ** 0.5, "not_plateau": False} for r in m]
    return m, t


def test_smallest_at_target_picks_cheapest_meeting_model():
    from accuracy_cost.plot import smallest_at_target
    m, t = _synthetic()
    rows = {r["family"]: r for r in smallest_at_target(m, t, 0.13, "gpu")}
    assert rows["ace"]["size"] == "b"


def test_smallest_at_target_not_reached():
    from accuracy_cost.plot import smallest_at_target
    m, t = _synthetic()
    rows = {r["family"]: r for r in smallest_at_target(m, t, 0.10, "gpu")}
    assert rows["mace"]["size"] is None and rows["mace"]["note"] == "not reached"


def test_pareto_and_heatmap_render(tmp_path):
    from accuracy_cost.plot import heatmap, pareto
    m, t = _synthetic()
    for r in m:
        r["per_type"] = {"dia": {"rmse_F": r["rmse_F"]}, "liq": {"rmse_F": 2 * r["rmse_F"]}}
    pareto(m, t, [0.1, 0.15], tmp_path / "p.png"); heatmap(m, [("ace", "b"), ("mace", "m1")], tmp_path / "h.png")
    assert (tmp_path / "p.png").stat().st_size > 0 and (tmp_path / "h.png").stat().st_size > 0
```

"Smallest" means the fewest `n_params` among the models of a family that meet
the target. Where models tie, the faster one wins. Throughput is only reported
next to the choice, never used to choose it.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -k "plateau or smallest or pareto" -v -n 0`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `timing.py`**

```python
"""Time every fitted model on lestrade with bench/scaling's runners: each code
in its best mode per device, one size plus an N/8 plateau check, 3 repeats.
Refuses a busy box; gates each model on LAMMPS-vs-own-evaluator parity first."""
import json
import os
import socket
import statistics
import subprocess
import sys

from . import common
from .common import RESULTS

sys.path.insert(0, str(common.HERE.parent))
N = {"gpu": (32768, 4096), "cpu": (4096, 512)}
RATTLE, REPEATS = 0.05, 3
# family -> [(code, {device: mode}, dtypes)]
PLAN = {"ace": [("acejax-ace", {"cpu": "standalone", "gpu": "lammps"}, ("float64",))],
        "ace_lr": [("acejax-ace", {"cpu": "standalone", "gpu": "lammps"}, ("float64",))],
        "ace_gp": [("acejax-gp", {"cpu": "standalone", "gpu": "standalone"}, ("float64",))],
        "pace": [("acejax-pace", {"cpu": "standalone", "gpu": "lammps"}, ("float64",)),
                 ("mlpace", {"cpu": "lammps", "gpu": "lammps"}, ("float64",))],
        "mace": [("mace", {"cpu": "lammps", "gpu": "lammps"}, ("float64",)),
                 ("mace", {"cpu": "standalone", "gpu": "standalone"}, ("float32",))],
        "gap": [("gap", {"cpu": "lammps"}, ("float64",))]}


def idle(load_max=2.0):
    load1 = float(open("/proc/loadavg").read().split()[0])
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        q = ""
    procs = [l for l in q.splitlines() if l.strip()]
    return {"ok": load1 <= load_max and not procs, "load1": load1, "gpu_procs": len(procs)}


def plateau_flag(main, check, tol=0.2):
    per = lambda r: r["n_atoms"] / r["step_s"]
    return abs(per(main) - per(check)) / per(main) > tol


def cases(device):
    out = []
    for fam, specs in PLAN.items():
        for e in common.load_sweep()[fam]:
            sizes = [f"f{n}" for n in e["take"]] if fam == "pace" else [e["size"]]
            for size in sizes:
                st = common.read_status(common.run_dir(fam, size))
                if not st or st.get("status") != "ok":
                    continue
                for code, modes, dtypes in specs:
                    if device not in modes:
                        continue
                    path = st["model"]
                    if code == "mace":
                        path = str(common.run_dir(fam, size) / ("model-f32.model" if "float32" in dtypes else "model.model"))
                    row = {"code": code, "family": fam, "system": "Si", "size": size, "elements": ["Si"],
                           "path": path, "name": f"{code}/Si/{size}", "rattle": RATTLE,
                           "symmetrix": str(common.run_dir(fam, size) / "model.json"),
                           "quip_label": st.get("quip_label"), "n_params": st.get("n_params")}
                    for dt in dtypes:
                        out.append({"row": row, "mode": modes[device], "dtype": dt})
    return out


def _one(c, device, n, env, work):
    from scaling import run_lammps, run_standalone
    if c["mode"] == "standalone":
        r = run_standalone.run_case(c["row"], n, c["dtype"], device)
        if r.get("status") == "ok":
            r["step_s"] = r["call_s"]
        return r
    ranks = env.get("cpu_ranks", 32) if device == "cpu" else 1
    return run_lammps.run_case(c["row"], n, c["dtype"], device, env["lmp_dev"] if c["row"]["code"].startswith("acejax")
                               else env["lmp"], ranks, work, pjrt=env.get("pjrt"))


def main(device):
    gate = idle()
    if not gate["ok"]:
        raise SystemExit(f"lestrade busy: {gate}")
    env = json.loads((common.HERE / "envs" / f"lestrade-{device}.json").read_text())
    for c in cases(device):
        reps = {}
        for n in N[device]:
            rows = [_one(c, device, n, env, common.RUNS / "timing" / device / c["row"]["name"] / str(n) / str(k))
                    for k in range(REPEATS)]
            ok = [r for r in rows if r.get("status") == "ok"]
            reps[n] = {**(ok or rows)[0], "step_s": statistics.median(r["step_s"] for r in ok)} if ok else rows[0]
        main_r, check_r = (reps[n] for n in N[device])
        row = {**main_r, "family": c["row"]["family"], "size": c["row"]["size"], "n_params": c["row"]["n_params"],
               "host": socket.gethostname(), "device": device, "load1": gate["load1"], "gpu_procs": gate["gpu_procs"]}
        if main_r.get("status") == "ok" and check_r.get("status") == "ok":
            row["atom_steps_per_s"] = main_r["n_atoms"] / main_r["step_s"]
            row["not_plateau"] = plateau_flag(main_r, check_r)
        common.upsert_jsonl(RESULTS / f"lestrade-{device}.jsonl", row, ("code", "family", "size", "dtype", "mode"))
        print(row["code"], row["size"], row["dtype"], row.get("status"), row.get("atom_steps_per_s"))


if __name__ == "__main__":
    main(sys.argv[1])
```

Before Step 4, read the real return dicts from `run_lammps.run_case` (the dict
built in `_run_lammps` from `parse_log`) and from `run_standalone.run_case`, to
confirm the per-step time key. If the LAMMPS key is not `step_s` (the scaling
`sweep.py` uses `prev.get("step_s")`, so it should be), adapt `_one`. Then read
`bench/scaling/envs/moriarty-gpu.json` to confirm the env keys (`lmp`,
`lmp_dev`, `pjrt`, `cpu_ranks`) and use the real ones.

**The parity gate:** call `parity.gate(...)` once per model before timing,
following the existing `gate_checks` pattern in `bench/scaling/parity.py:116-134`
and `sweep.gate_in_subprocess`. Read those two functions first, add a
`gate_model(row, device, env) -> dict` wrapper in `timing.py` that runs that
model's LAMMPS energy and forces against its own evaluator (`_acejax_ef`,
`_mace_ef`, `_quip_ef`, and pyace for `mlpace`) on three 64-atom rattled Si
cells, and skip any row whose gate fails, writing `status: "parity_failed"`.

- [ ] **Step 4: Implement `plot.py`**

```python
"""Accuracy-cost figures: Pareto (test force RMSE vs throughput, one line per
family, CPU/GPU panels), smallest model at target, per-config-type heatmap."""
import json
import sys

import numpy as np

COLORS = {"ace": "C0", "ace_lr": "C5", "ace_gp": "C1", "pace": "C2", "mace": "C3", "gap": "C4"}


def _join(metrics, timings, device):
    by = {(m["family"], m["size"]): m for m in metrics}
    out = []
    for t in timings:
        m = by.get((t["family"], t["size"]))
        if m and t.get("device") == device and t.get("atom_steps_per_s"):
            out.append({**m, **{k: t[k] for k in ("atom_steps_per_s", "not_plateau", "code", "dtype") if k in t}})
    return out


def smallest_at_target(metrics, timings, target, device):
    rows, fams = [], sorted({m["family"] for m in metrics})
    j = _join(metrics, timings, device)
    for f in fams:
        ok = [r for r in j if r["family"] == f and r["rmse_F"] <= target]
        if not ok:
            rows.append({"family": f, "size": None, "note": "not reached"}); continue
        best = min(ok, key=lambda r: (r["n_params"], -r["atom_steps_per_s"]))
        rows.append({k: best.get(k) for k in ("family", "size", "rmse_F", "n_params", "atom_steps_per_s", "code")})
    return rows


def pareto(metrics, timings, targets, out, refs=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, dev in zip(axes, ("cpu", "gpu")):
        j = _join(metrics, timings, dev)
        for (f, code) in sorted({(r["family"], r.get("code")) for r in j}):
            pts = sorted((r["rmse_F"], r["atom_steps_per_s"], r.get("not_plateau")) for r in j
                         if r["family"] == f and r.get("code") == code)
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(x, y, "o-", color=COLORS.get(f, "k"), label=f"{f} ({code})",
                    ls="-" if code != "mlpace" else "--")
            for xi, yi, flag in pts:
                if flag:
                    ax.plot(xi, yi, "x", color="k")
        for t in targets:
            ax.axvline(t, color="0.7", lw=1)
        for name, v in refs:
            ax.axvline(v, color="0.3", ls=":", lw=1); ax.text(v, ax.get_ylim()[1], name, rotation=90, va="top", fontsize=7)
        ax.set(xscale="log", yscale="log", xlabel="test force RMSE (eV/Å)", title=f"lestrade {dev}")
    axes[0].set_ylabel("atom-steps / s"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def heatmap(metrics, sizes, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {(m["family"], m["size"]): m for m in metrics}
    sel = [by[k] for k in sizes if k in by]
    types = sorted({t for m in sel for t in m["per_type"]})
    A = np.array([[m["per_type"].get(t, {}).get("rmse_F", np.nan) for t in types] for m in sel])
    fig, ax = plt.subplots(figsize=(1 + 0.35 * len(types), 1 + 0.4 * len(sel)))
    im = ax.imshow(A, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(types)), types, rotation=90, fontsize=7)
    ax.set_yticks(range(len(sel)), [f"{m['family']}/{m['size']}" for m in sel], fontsize=7)
    fig.colorbar(im, label="test force RMSE (eV/Å)"); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def _load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


if __name__ == "__main__":
    from .common import HERE, RESULTS
    targets = [float(x) for x in sys.argv[1:]] or [0.1, 0.15]
    m = _load(RESULTS / "metrics.jsonl")
    t = sum((_load(RESULTS / f"lestrade-{d}.jsonl") for d in ("cpu", "gpu") if (RESULTS / f"lestrade-{d}.jsonl").exists()), [])
    figs = HERE / "figs"; figs.mkdir(exist_ok=True)
    refs = json.loads((HERE / "refs.json").read_text()) if (HERE / "refs.json").exists() else []
    pareto(m, t, targets, figs / "pareto.png", refs)
    keep = set()
    for tg in targets:
        for d in ("cpu", "gpu"):
            print(f"\n## target {tg} eV/Å, {d}\n\n| family | size | n_params | rmse_F | atom-steps/s |\n|---|---|---|---|---|")
            for r in smallest_at_target(m, t, tg, d):
                if r["size"] is None:
                    print(f"| {r['family']} | not reached | | | |"); continue
                keep.add((r["family"], r["size"]))
                print(f"| {r['family']} | {r['size']} | {r['n_params']} | {r['rmse_F']:.3f} | {r['atom_steps_per_s']:.3g} |")
    heatmap(m, sorted(keep), figs / "per_type.png")
```

`refs.json` is a list of `[name, rmse_F]` pairs: the refit GAP s9000 test error
and the published GAP-18 error from `data.md`. It's written by hand in Task 11.
The pairs are lists, so `for name, v in refs` unpacks them.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bench_accuracy_cost.py -v -n 0`
Expected: every test passes (about 22).

- [ ] **Step 6: Run the timings on lestrade**

The timing driver runs in the lestrade `$BENCH_ROOT/venv` (Task 9). That venv
needs ace-jax, mace-torch and quippy-ase:
install them with `uv pip install --python $BENCH_ROOT/venv/bin/python quippy-ase mace-torch`
if `moriarty.sh` did not already.

Then, on lestrade:
- `ssh lestrade 'cd ~/projects/ace-jax/bench && setsid nohup $BENCH_ROOT/venv/bin/python -m accuracy_cost.timing gpu > /storage/eng/essswb/projects/ace-jax/runs/accuracy_cost/timing-gpu.log 2>&1 < /dev/null &'`
- when that has finished, the same with `cpu`.

Run them sequentially: the CPU rows need the 32 cores to themselves.

Expected: every row in `results/lestrade-{gpu,cpu}.jsonl` has `status: ok` or
an explicit `oom` / `parity_failed` / `error`, and `not_plateau` is set on each
ok row.

- [ ] **Step 7: Commit**

```bash
git add bench/accuracy_cost/timing.py bench/accuracy_cost/plot.py tests/test_bench_accuracy_cost.py \
        bench/accuracy_cost/results/lestrade-*.jsonl
git commit -m "feat(bench): accuracy-cost timing on lestrade and figures

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Targets, figures and the write-up

**Files:**
- Create: `bench/accuracy_cost/refs.json`
- Create: `bench/accuracy_cost/figs/{pareto,per_type}.png`
- Create: `bench/accuracy_cost/README.md`
- Create: `docs/dev/accuracy-cost.md`
- Modify: `docs/dev/benchmarks.md` (a pointer under the title, plus caveats in "Caveats")
- Modify: `docs/dev/accuracy-cost-spec.md` (an update banner, like `benchmark-scaling-spec.md`'s)

- [ ] **Step 1: Choose the targets**

Read `results/metrics.jsonl`. Take the refit `gap/s9000` test `rmse_F` as `g`.
The targets are `g` rounded up to 2 significant figures, and `1.5 g` rounded
the same way. If no family other than GAP reaches `g`, use `1.5 g` and `2 g`
instead. Write the choice and the reason into `docs/dev/accuracy-cost.md`.

Write `refs.json` as:
`[["GAP refit (s9000, test)", g], ["GAP-18 published (train)", <value from data.md>]]`.

- [ ] **Step 2: Render**

Run: `cd bench && uv run --no-sync python -m accuracy_cost.plot <t1> <t2> > ../docs/dev/accuracy-cost-table.md.tmp`
Expected: `figs/pareto.png` and `figs/per_type.png` are written, and the tables are printed.

Open both PNGs (with the Read tool) and check:
- every family appears in both panels where the spec says it should (GAP only on CPU);
- the `not_plateau` crosses are explained in the caption;
- the axes are labelled.

- [ ] **Step 3: Write `docs/dev/accuracy-cost.md`**

Sections, in order:
1. **What this answers:** quote the reviewer feedback.
2. **Setup:** the data, the split and the fairness rules, copied from the spec's "Fairness rules" and stated as done: which families fitted virials, the shared cutoff, weights, E0 and scorer, and "defaults plus a size sweep". Report `rmse_V` alongside, noting it is only meaningful for the families that fitted virials.
3. **Results:** the Pareto figure, the smallest-at-target tables (pasted from Step 2's output), the per-type heatmap, and the fit-time table (`fit_wall_s` and `fit_host` from `metrics.jsonl`; PACE sizes share the ladder's time).
4. **Caveats:**
   - lestrade numbers are not comparable with `benchmarks.md`;
   - MACE's 10 Å receptive field;
   - GAP-18 published errors are training-set errors;
   - MACE `config_type_weights` scale the whole config loss;
   - `not_plateau` rows;
   - one seed per fit.
5. **Reproduce:** the commands from `bench/accuracy_cost/README.md`.

Reference figures as `../bench/accuracy_cost/figs/pareto.png`. If `mkdocs`
includes `docs/` pages, check with `grep -n accuracy mkdocs.yml`: this page is
**not** added to the user site (`docs/user/`), so no nav change is needed.

- [ ] **Step 4: Point `benchmarks.md` here**

Under the first paragraph, add:
`For speed at matched accuracy (the smallest model of each family that reaches a target test RMSE on the GAP-18 Si data), see [accuracy-cost.md](accuracy-cost.md).`

In "Caveats", after the linear ACE bullet, add one bullet: "The random-weight
size ladder and the foundation MACE models time code paths at fixed sizes, not
models at equal accuracy; [accuracy-cost.md](accuracy-cost.md) does that."

- [ ] **Step 5: Write `bench/accuracy_cost/README.md`**

It contains a file table (the plan's "File structure" rows for
`bench/accuracy_cost/`), followed by the end-to-end run order:
1. `split`;
2. `fit_ace ace all` / `fit_ace ace_lr all` / `fit_ace ace_gp all` on moriarty;
3. `fit_pace` and `fit_mace all` on moriarty, sequentially;
4. `fit_gap all` on mnf148;
5. `evaluate` per family, under that family's venv;
6. `envs/lestrade.sh` once;
7. `timing gpu`, then `timing cpu` on lestrade;
8. `plot <targets>`.

Give each with its exact command from Tasks 1–10.

- [ ] **Step 6: Add the spec update banner and run the full suite**

At the top of `docs/dev/accuracy-cost-spec.md`, add:
`> **Update <date>.** Phase A implemented: harness in bench/accuracy_cost/ (see its README), results in [accuracy-cost.md](accuracy-cost.md). Virials are fitted by every code that supports them.`

Run: `uv run pytest tests/test_bench_accuracy_cost.py tests/test_bench_scaling.py tests/test_core_deps.py tests/test_no_backend_names.py -v`
Expected: all pass.

Run: `uv run ruff check bench/accuracy_cost bench/scaling tests/test_bench_accuracy_cost.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add bench/accuracy_cost/refs.json bench/accuracy_cost/figs bench/accuracy_cost/README.md \
        docs/dev/accuracy-cost.md docs/dev/benchmarks.md docs/dev/accuracy-cost-spec.md
git commit -m "docs(bench): accuracy-matched Si speed comparison

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
