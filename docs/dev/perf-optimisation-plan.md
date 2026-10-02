# ace-jax speed-ups: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring ace-jax's GPU force evaluation to ML-PACE's speed on the same
models, for the ASE calculator and the LAMMPS bundle, with results unchanged to
round-off.

**Architecture:**
- **Shared machinery in `EdgeSiteModel`,** used by both `PACEModel` and
  `ACEModel`:
  - chunked dense evaluation;
  - the reverse-edge force gather;
  - pool-first A assembly, fed by per-model hooks.
- **Per-model changes:**
  - PACE: the SBessel recurrence and a feature-major product basis;
  - `ACEModel`: pool-first and the feature-major product basis, after a
    profile decides which to keep.
- **Calculator:** `ACECalculator` gains a skin neighbour list and a single
  jitted step (`calc/skin.py`).
- **LAMMPS:** the bundle evaluates only owned rows, with slots sized for the
  cutoff.
- **Code shape:** everything replaces the current forms in place. The current
  forms survive only as frozen reference fixtures.

**Tech Stack:** JAX / Equinox, matscipy-neighbours (`neighbour_matrix`,
optionally on the GPU via DLPack), lammps-jax, pytest, Modal A100 for
profiling and micro-benchmarks, and the `bench/scaling/` suite.

**Spec:** `docs/dev/perf-optimisation-spec.md`. Evidence:
`docs/dev/pace-performance-gap.md`, with prototypes in `bench/perf/variants.py`,
`bench/perf/fast_calc.py` and `bench/perf/lammps_variant.py`.

## Global Constraints

- **Parity with the frozen references:**
  - in float64: dE per atom ≤ 1e-12 × |E| per atom, and dF and dV within
    1e-12 relative. That is, `assert_allclose(rtol=1e-12, atol=1e-12)` on
    E, F and V;
  - in float32: measured against the float64 reference, no worse than the
    old float32 result, i.e. at most max(1e-5, 1.25 × old-float32 error)
    relative (Task 2 ruling); forces use 2.0 instead of 1.25 when the float64
    reference forces are below 1e-3 eV/Å (max|F64| < 1e-3: pure rounding
    noise), energy and stress keep 1.25 (Task 5 ruling);
  - existing tests: ML-PACE, python-ace and Julia parity, and
    `tests/test_export_lammps.py` (1e-10 / 1e-9), stay green.
- **Test commands:**
  - the full suite: `uv run pytest tests/ -q -p no:cacheprovider --deselect tests/test_gp_ladder_real.py::test_ladder_on_real_model`
    (that test needs `blackjax`, which isn't installed locally);
  - bundle tests: `uv run --with-editable /Users/u1470235/gits/lammps-jax pytest …`.
- **No run-time flags** select the old or new model forms. Old forms live only
  in `tests/fixtures/perf_ref/`.
- **Skin list:** on by default in `ACECalculator` (`skin=1.0` Å). `skin=0`
  must reproduce rebuild-every-call behaviour exactly.
- **Chunking:** dense layout only. Default block is 16,384 nodes, and below
  one block the path is identical to today's.
- **Branch:** `perf/ace-jax-speedups`, worktree
  `/Users/u1470235/gits/ace-jax/.worktrees/perf`, stacked on
  `feat/bench-scaling`.
- **Hosts:**
  - Modal A100 via `bench/perf/modal_profile.py` (keep total GPU use under 4
    GPU-hours);
  - moriarty only when idle: check `/proc/loadavg`, `who`, `nvidia-smi` first;
  - SSH rules as in the user's CLAUDE.md. Never pass `BatchMode`, reuse one
    `ControlMaster`, and use bracketed `pkill` patterns in their own command.
- **Commits** end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

The five inputs the spec implies but no happy-path test covers, most likely
first. Each has a test in the owning task:
1. **Skin list with atoms that have no neighbours** (an isolated atom, or a
   cluster in vacuum), so a row has `count == 0`. Expect E = E0 for that
   atom, F = 0, and no NaN. Task 6.
2. **Skin list on a triclinic cell,** and on a small cell where
   `rcut + skin` exceeds half the cell so one pair appears as several periodic
   images. Expect results equal to a fresh rebuild, and `rev` matched per
   image, not per atom pair. Tasks 4 and 6.
3. **Calculator reused across unrelated structures** with the same size and
   species, such as a trajectory frame jump or a different configuration in
   the same `Atoms`. Expect a rebuild, then correct results. Task 6.
4. **A node count that isn't a multiple of the chunk size,** or smaller than
   one block. Expect identical results to unchunked evaluation, with padding
   rows contributing nothing. Task 3.
5. **Trainable coefficients under pool-first:** gradients of the energy with
   respect to `crad` (PACE) or the radial weights (ACE) must match the old
   form. The prototype built `W` from `crad` on the host, which would silently
   freeze it. Tasks 5 and 9.

---

## File structure

| file | responsibility | tasks |
|---|---|---|
| `tests/perf_ref.py` | writes the frozen references (run once, before any change) | 1 |
| `tests/fixtures/perf_ref/*.npz` | frozen E, F, V from the current code | 1 |
| `tests/test_perf_parity.py` | new code vs frozen references, every model, layout and dtype | 1 (and runs in every later task) |
| `src/ace_jax/eval/pace_radial.py` | SBessel recurrence | 2 |
| `src/ace_jax/eval/edge_model.py` | chunked dense E, F, V; reverse-edge gather; generic pool-first A | 3, 4, 5 |
| `src/ace_jax/eval/nlist.py` | `reverse_slots` (host) | 4 |
| `src/ace_jax/eval/pace_model.py` | PACE pool-first hooks; feature-major product basis | 5 |
| `src/ace_jax/calc/skin.py` | skin neighbour state and the jitted step | 6 |
| `src/ace_jax/calc/point.py` | `ACECalculator` uses `skin.py` | 6 |
| `src/ace_jax/export/lammps.py` | owned-row bundle | 7 |
| `bench/scaling/run_lammps.py` | `capacity` for `rcut`, `max_owned` | 7 |
| `bench/perf/profile_ace.py`, `docs/dev/perf-ace-profile.md` | `ACEModel` profile and decision | 8 |
| `src/ace_jax/eval/model.py` | `ACEModel` pool-first hooks and feature-major product basis (as decided) | 9 |
| `bench/perf/microbench.py` | before and after per component on Modal | 10 |
| `bench/scaling/run_standalone.py` | MD-like calls, rebuild count | 10 |
| `docs/dev/benchmarks.md`, `docs/dev/perf-optimisation-results.md` | results | 11 |

---

### Task 1: Frozen parity references

**Files:**
- Create: `tests/perf_ref.py`, `tests/test_perf_parity.py`, `tests/fixtures/perf_ref/` (npz files, committed)

**Interfaces:**
- Consumes: the current `load`, `ACECalculator` (`layout="sparse"` and `"dense"`), and the fixtures in `fixtures/pace/*.yace`, `fixtures/si_ace_model.npz` (ACE analytic) and `fixtures/sige_nofit.npz` (ACE spline, two elements, pair term).
- Produces:
  - `tests/fixtures/perf_ref/<model>_<cell>_<layout>_<dtype>.npz` with keys `E`, `F`, `V` (or `S` for stress), `numbers`, `positions`, `cell`, `pbc`;
  - `tests/test_perf_parity.py::CASES`, a list of `(model_path, cell_name, layout, dtype)`.

- [ ] **Step 1: Write the reference generator.** It must run on the unchanged code, so write it before touching anything else.

```python
"""Write frozen E/F/stress references from the CURRENT code (run once, before
any optimisation): tests/test_perf_parity.py holds every later change to them.

    uv run python tests/perf_ref.py
"""
import pathlib

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase import Atoms
from ase.build import bulk

from ace_jax.calc.point import ACECalculator

ROOT = pathlib.Path(__file__).parent.parent
OUT = pathlib.Path(__file__).parent / "fixtures" / "perf_ref"
PACE = sorted((ROOT / "fixtures" / "pace").glob("*.yace"))
ACE = [ROOT / "fixtures" / "si_ace_model.npz", ROOT / "fixtures" / "sige_nofit.npz"]


def cells(elements):
    """Periodic (triclinic, a small cell where rcut+skin spans several images)
    and an open cluster, with every element of the model present."""
    z = [int(e) for e in elements]
    rng = np.random.default_rng(0)
    per = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    per.set_cell(per.cell.array @ np.array([[1, 0.08, 0], [0, 1, 0.05], [0, 0, 1]]),
                 scale_atoms=True)                            # triclinic
    per.numbers = np.asarray(z)[rng.integers(0, len(z), len(per))]
    per.positions += rng.normal(0, 0.05, per.positions.shape)
    small = bulk("Si", "diamond", a=5.43)                      # 2 atoms: many images per pair
    small.numbers = np.asarray(z)[np.arange(len(small)) % len(z)]
    clus = per.copy()
    clus.pbc = False
    clus.center(vacuum=6.0)
    iso = Atoms(numbers=[z[0], z[-1]], positions=[[0, 0, 0], [20.0, 0, 0]],
                cell=[40, 40, 40], pbc=False)                # atoms with no neighbours
    return {"tric": per, "small": small, "cluster": clus, "isolated": iso}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from ace_jax.eval import load
    for path in PACE + ACE:
        _, meta, _ = load(str(path))
        for cname, at in cells(meta["elements"]).items():
            for layout in ("sparse", "dense"):
                for dt in ("float64", "float32"):
                    calc = ACECalculator(str(path), layout=layout, dtype=getattr(jnp, dt))
                    a = at.copy()
                    a.calc = calc
                    E = a.get_potential_energy()
                    F = a.get_forces()
                    S = a.get_stress() if a.pbc.all() else np.zeros(6)
                    np.savez(OUT / f"{path.stem}_{cname}_{layout}_{dt}.npz", E=E, F=F, S=S,
                             numbers=a.numbers, positions=a.positions, cell=a.cell.array,
                             pbc=a.pbc)
    print("wrote", len(list(OUT.glob("*.npz"))), "references to", OUT)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on the unchanged code.**
      Run: `uv run python tests/perf_ref.py`
      Expected: `wrote 144 references` (9 models × 4 cells × 2 layouts × 2 dtypes). The PACE fixtures are git-ignored if not generated; if any are missing, see `pace_ref/README.md`.

- [ ] **Step 3: Write the parity test.**

```python
"""Every optimisation keeps E, F and stress equal to the frozen references
(tests/fixtures/perf_ref, written by tests/perf_ref.py from the code before the
speed-ups): 1e-12 relative in float64, 1e-5 in float32."""
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase import Atoms

from ace_jax.calc.point import ACECalculator

ROOT = pathlib.Path(__file__).parent.parent
REF = pathlib.Path(__file__).parent / "fixtures" / "perf_ref"


def _model_path(stem):
    p = ROOT / "fixtures" / "pace" / f"{stem}.yace"
    return p if p.exists() else ROOT / "fixtures" / f"{stem}.npz"


CASES = sorted(REF.glob("*.npz"))


@pytest.mark.parametrize("skin", [0.0, 1.0])
@pytest.mark.parametrize("ref", CASES, ids=[c.stem for c in CASES])
def test_matches_frozen_reference(ref, skin):
    stem, cname, layout, dt = ref.stem.rsplit("_", 3)
    r = np.load(ref)
    at = Atoms(numbers=r["numbers"], positions=r["positions"], cell=r["cell"], pbc=r["pbc"])
    kw = {} if "skin" not in ACECalculator.__init__.__code__.co_varnames else {"skin": skin}
    at.calc = ACECalculator(str(_model_path(stem)), layout=layout, dtype=getattr(jnp, dt), **kw)
    tol = 1e-12 if dt == "float64" else 1e-5
    scale = max(1.0, abs(float(r["E"])))
    assert abs(at.get_potential_energy() - float(r["E"])) <= tol * scale
    np.testing.assert_allclose(at.get_forces(), r["F"], rtol=tol, atol=tol * scale / len(at))
    if at.pbc.all():
        np.testing.assert_allclose(at.get_stress(), r["S"], rtol=tol, atol=tol * scale)
```

- [ ] **Step 4: Run it. It must pass on the unchanged code** (a self-consistency check).
      Run: `uv run pytest tests/test_perf_parity.py -q -p no:cacheprovider`
      Expected: all pass. With no `skin` argument yet, both parametrisations run the same code.

- [ ] **Step 5: Commit.**

```bash
git add tests/perf_ref.py tests/test_perf_parity.py tests/fixtures/perf_ref
git commit -m "test: frozen E/F/stress references for the speed-ups (every model, layout, dtype)"
```

---

### Task 2: SBessel recurrence (PACE)

**Files:**
- Modify: `src/ace_jax/eval/pace_radial.py` (`_sbessel`, about lines 56–73)
- Test: `tests/test_pace_radial_rec.py`

**Interfaces:**
- Consumes: none.
- Produces: `_sbessel(r, rc, K)`, with the same signature and values.

- [ ] **Step 1: Write the failing test.** It checks exactness against a direct evaluation, and that the transcendental count is O(1) in K.

```python
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from ace_jax.eval import pace_radial as pr


def _direct(r, rc, K):
    """The pre-change definition: one sinc per k."""
    import math
    def f(n):
        pre = ((-1) ** n * math.sqrt(2) * math.pi * (n + 1) * (n + 2)
               / math.sqrt((n + 1) ** 2 + (n + 2) ** 2))
        return pre / rc ** 1.5 * (pr._sinc(r * (n + 1) * math.pi / rc)
                                  + pr._sinc(r * (n + 2) * math.pi / rc))
    g, d_prev = [f(0)], 1.0
    for n in range(1, K):
        en = n ** 2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        dn = 1 - en / d_prev
        g.append((f(n) + math.sqrt(en / d_prev) * g[-1]) / math.sqrt(dn))
        d_prev = dn
    return jnp.stack(g, axis=-1)


def test_sbessel_recurrence_is_exact_and_cheap():
    r = jnp.linspace(1e-6, 4.9, 997)
    for K in (4, 8, 12):
        np.testing.assert_allclose(pr._sbessel(r, 5.0, K), _direct(r, 5.0, K), rtol=1e-12, atol=1e-13)
    hlo = jax.jit(lambda x: pr._sbessel(x, 5.0, 12)).lower(r).as_text()
    assert hlo.count("sine") <= 1 and hlo.count("cosine") <= 1      # not K+1 of each
```

- [ ] **Step 2: Run it. It should fail** on the sine count.
      Run: `uv run pytest tests/test_pace_radial_rec.py -q`
      Expected: FAIL, `assert 13 <= 1` (or similar).

- [ ] **Step 3: Implement.** Replace `_sbessel` with the recurrence from `bench/perf/variants.py::_sbessel_rec`:

```python
def _sbessel(r, rc, K):
    """PACE's SBessel basis.  sin(k x), k = 1..K+1, from one sin and one cos by
    sin((k+1)x) = 2 cos(x) sin(kx) - sin((k-1)x), and one reciprocal instead
    of a division per k: the per-edge transcendental cost no longer grows with K
    (-20% of a PACE force call, docs/dev/pace-performance-gap.md #3)."""
    x = r * PI / rc
    xs = jnp.where(x == 0, 1.0, x)
    s1, c2 = jnp.sin(xs), 2.0 * jnp.cos(xs)
    inv = 1.0 / xs
    s = [jnp.zeros_like(xs), s1]
    for _ in range(2, K + 2):
        s.append(c2 * s[-1] - s[-2])
    sinc = [None] + [jnp.where(x == 0, 1.0, s[k] * inv * (1.0 / k)) for k in range(1, K + 2)]
    rc15 = rc ** 1.5

    def f(n):
        pre = ((-1) ** n * math.sqrt(2) * PI * (n + 1) * (n + 2)
               / math.sqrt((n + 1) ** 2 + (n + 2) ** 2))
        return pre / rc15 * (sinc[n + 1] + sinc[n + 2])
    g, d_prev = [f(0)], 1.0
    for n in range(1, K):
        en = n ** 2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
        dn = 1 - en / d_prev
        g.append((f(n) + math.sqrt(en / d_prev) * g[-1]) / math.sqrt(dn))
        d_prev = dn
    return jnp.stack(g, axis=-1)
```

Add `import math` at the top of the module if it isn't already there.

- [ ] **Step 4: Run the new test and the parity suite.**
      Run: `uv run pytest tests/test_pace_radial_rec.py tests/test_perf_parity.py tests/test_pace_*.py -q -p no:cacheprovider`
      Expected: all pass. The recurrence's round-off stays inside 1e-12 relative. If the SBessel parity fails at 1e-12, record the observed maximum as a ruling and widen only the SBessel cases, explaining why in the test.

- [ ] **Step 5: Commit.**

```bash
git add src/ace_jax/eval/pace_radial.py tests/test_pace_radial_rec.py
git commit -m "perf(pace): SBessel basis by Chebyshev recurrence (one sin/cos per edge)"
```

---

### Task 3: Chunked dense evaluation (shared)

**Files:**
- Modify: `src/ace_jax/eval/edge_model.py` (`energy_forces_virial_dense`)
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `site_energies_dense(rij, zi, zj, mask, node_z)` from both models.
- Produces: `energy_forces_virial_dense(rij, zi, zj, idx, mask, node_z, chunk=CHUNK_NODES)`, same outputs. `CHUNK_NODES = 16384` at module level.

- [ ] **Step 1: Write the failing test.** Use a tiny chunk so a small system spans several blocks, with a node count that isn't a multiple of the block (Review Focus 4).

```python
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ace_jax.eval import load
from ace_jax.eval.nlist import dense_graph
from conftest import pace_fixture

FIX = pathlib.Path(__file__).parent.parent / "fixtures"


def _dense_inputs(at, model, meta, z2i):
    counts_K = 64
    dg = dense_graph(at.positions, at.cell.array, at.pbc, meta["rcut"], counts_K)
    idx = jnp.asarray(dg.idx, jnp.int32)
    mask = jnp.arange(counts_K)[None, :] < jnp.asarray(dg.count)[:, None]
    node_z = jnp.asarray([z2i[int(z)] for z in at.numbers])
    zi = jnp.broadcast_to(node_z[:, None], idx.shape)
    return jnp.asarray(dg.rij), zi, node_z[idx], idx, mask, node_z


@pytest.mark.parametrize("path", [pace_fixture(FIX / "pace" / "gesi_sbessel.yace"),
                                  FIX / "sige_nofit.npz"])
@pytest.mark.parametrize("chunk", [7, 64, 16384])          # 7: 5+ blocks, n % 7 != 0
def test_chunked_equals_unchunked(path, chunk):
    from ase.build import bulk
    model, meta, _ = load(str(path))
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))        # 64 atoms
    at.numbers[::3] = meta["elements"][-1]
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    args = _dense_inputs(at, model, meta, z2i)
    E0, F0, V0 = model.energy_forces_virial_dense(*args, chunk=10 ** 9)   # one block
    E1, F1, V1 = model.energy_forces_virial_dense(*args, chunk=chunk)
    np.testing.assert_allclose(E1, E0, rtol=1e-13)
    np.testing.assert_allclose(F1, F0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(V1, V0, rtol=1e-12, atol=1e-12)
```

- [ ] **Step 2: Run it. It should fail** on `unexpected keyword 'chunk'`.
      Run: `uv run pytest tests/test_chunking.py -q`

- [ ] **Step 3: Implement.** Pad the rows to a multiple of the block, then `lax.map` over blocks inside the energy, with `jax.checkpoint` on the block function. Padding rows are fully masked, parked at the pad cutoff, with species 0. A single block skips the map, so it's the same code path as today.

```python
CHUNK_NODES = 16384   # dense rows per block: peak memory ~ block, not N (spec, component 2)

    def energy_forces_virial_dense(self, rij, zi, zj, idx, mask, node_z, chunk=CHUNK_NODES):
        """`energy_forces_virial` for the dense layout: rij (n, K, 3), zi / zj /
        idx (neighbour index) / mask (n, K).  Same strain trick for the virial.

        Rows are evaluated in blocks of `chunk` (lax.map; jax.checkpoint so the
        backward pass recomputes a block rather than storing all of them): peak
        memory scales with the block.  A site energy depends only on its own row,
        so blocks are independent; padding rows are fully masked."""
        n, K = mask.shape
        nb = max(1, -(-n // chunk))
        B = -(-n // nb)
        pad_n = nb * B - n

        def blocks(a, fill):
            if pad_n:
                a = jnp.concatenate([a, jnp.full((pad_n,) + a.shape[1:], fill, a.dtype)])
            return a.reshape((nb, B) + a.shape[1:])

        park = jnp.asarray([1.0, 0.0, 0.0], rij.dtype) * self.pad_cutoff()

        def total(r, eps):
            sym = 0.5 * (eps + eps.T)
            rs = r + r @ sym
            if nb == 1:
                return jnp.sum(self.site_energies_dense(rs, zi, zj, mask, node_z))
            rb = jnp.concatenate([rs, jnp.broadcast_to(park, (pad_n, K, 3))]) if pad_n else rs
            rb = rb.reshape(nb, B, K, 3)
            live = blocks(jnp.ones((n,), rij.dtype), 0.0)       # padding rows add nothing (E0)
            xs = (rb, blocks(zi, 0), blocks(zj, 0), blocks(mask, False), blocks(node_z, 0), live)
            block = jax.checkpoint(
                lambda a: jnp.sum(self.site_energies_dense(*a[:5]) * a[5]))
            return jnp.sum(jax.lax.map(block, xs))

        eps0 = jnp.zeros((3, 3), rij.dtype)
        E, (g_r, g_eps) = jax.value_and_grad(total, argnums=(0, 1))(rij, eps0)
        g_r = jnp.where(mask[..., None], g_r, 0.0)
        F = (jnp.zeros((n, 3), rij.dtype).at[jnp.arange(n)].add(g_r.sum(axis=1))
             .at[idx.reshape(-1)].add(-g_r.reshape(-1, 3)))
        return E, F, -g_eps
```

- [ ] **Step 4: Run.**
      Run: `uv run pytest tests/test_chunking.py tests/test_perf_parity.py tests/test_dense_a.py tests/test_calc_layout.py -q -p no:cacheprovider`
      Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/ace_jax/eval/edge_model.py tests/test_chunking.py
git commit -m "perf: chunk the dense layout over node blocks (lax.map + checkpoint); memory ~ block"
```

---

### Task 4: Reverse-edge force gather (shared)

**Files:**
- Modify: `src/ace_jax/eval/nlist.py` (add `reverse_slots`), `src/ace_jax/eval/edge_model.py` (`energy_forces_virial_dense` gains `rev=None`)
- Test: `tests/test_reverse_slots.py`

**Interfaces:**
- Consumes: Task 3's `energy_forces_virial_dense`.
- Produces:
  - `nlist.reverse_slots(idx, rij, count, tol=1e-8) -> np.ndarray (n, K) int32`, which raises `ValueError` if a live edge has no reverse;
  - `energy_forces_virial_dense(..., chunk=..., rev=None)`: with `rev` given, it uses the gather.

- [ ] **Step 1: Write the failing tests.** Cover `reverse_slots` on a triclinic cell and on a 2-atom cell with many images per pair (Review Focus 2), and check that forces with `rev` equal the scatter.

```python
import numpy as np
import jax
import pytest
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase.build import bulk

from ace_jax.eval import load
from ace_jax.eval.nlist import dense_graph, reverse_slots


def _cells():
    tric = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    tric.set_cell(tric.cell.array @ np.array([[1, .08, 0], [0, 1, .05], [0, 0, 1]]), scale_atoms=True)
    return {"tric": tric, "small": bulk("Si", "diamond", a=5.43)}      # small: images


@pytest.mark.parametrize("name", ["tric", "small"])
def test_reverse_slots_point_back(name):
    at = _cells()[name]
    dg = dense_graph(at.positions, at.cell.array, at.pbc, 5.0, 128)
    rev = reverse_slots(dg.idx, dg.rij, dg.count)
    live = np.arange(128)[None, :] < dg.count[:, None]
    i, k = np.nonzero(live)
    j = dg.idx[i, k]
    assert (dg.idx[j, rev[i, k]] == i).all()
    np.testing.assert_allclose(dg.rij[j, rev[i, k]], -dg.rij[i, k], atol=1e-10)


def test_rev_gather_equals_scatter():
    model, meta, _ = load("fixtures/sige_nofit.npz")
    at = _cells()["small"].repeat((3, 3, 3))
    at.numbers[::2] = 32
    dg = dense_graph(at.positions, at.cell.array, at.pbc, meta["rcut"], 96)
    node_z = jnp.asarray((at.numbers == 32).astype(int))
    idx = jnp.asarray(dg.idx, jnp.int32)
    mask = jnp.arange(96)[None, :] < jnp.asarray(dg.count)[:, None]
    args = (jnp.asarray(dg.rij), jnp.broadcast_to(node_z[:, None], idx.shape), node_z[idx],
            idx, mask, node_z)
    E0, F0, V0 = model.energy_forces_virial_dense(*args)
    rev = jnp.asarray(reverse_slots(dg.idx, dg.rij, dg.count))
    E1, F1, V1 = model.energy_forces_virial_dense(*args, rev=rev)
    np.testing.assert_allclose(F1, F0, rtol=1e-12, atol=1e-13)
```

- [ ] **Step 2: Run them. They should fail** on `cannot import reverse_slots`.
      Run: `uv run pytest tests/test_reverse_slots.py -q`

- [ ] **Step 3: Implement.** Copy `reverse_slots` from `bench/perf/variants.py` into `nlist.py` verbatim, with the docstring stating it matches per periodic image. Then in `energy_forces_virial_dense`, after `g_r` is masked:

```python
        if rev is None:
            F = (jnp.zeros((n, 3), rij.dtype).at[jnp.arange(n)].add(g_r.sum(axis=1))
                 .at[idx.reshape(-1)].add(-g_r.reshape(-1, 3)))
        else:
            # F_i = sum_k g[i,k] - sum_k g[j, rev[i,k]], j = idx[i,k]: the edge
            # j -> i sits at slot rev[i,k] of row j (a full list is symmetric),
            # so the force assembly is a gather, not a scatter-add
            back = g_r[idx, rev]
            F = g_r.sum(axis=1) - jnp.where(mask[..., None], back, 0.0).sum(axis=1)
```

- [ ] **Step 4: Run.**
      Run: `uv run pytest tests/test_reverse_slots.py tests/test_chunking.py tests/test_perf_parity.py -q -p no:cacheprovider`
      Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/ace_jax/eval/nlist.py src/ace_jax/eval/edge_model.py tests/test_reverse_slots.py
git commit -m "perf: forces by a reverse-edge gather when the list supplies rev (scatter otherwise)"
```

---

### Task 5: Pool-first A (shared helper, PACE hooks) and the PACE feature-major product basis

**Files:**
- Modify: `src/ace_jax/eval/edge_model.py` (add `pool_first_dense` and `pool_first_sparse`), `src/ace_jax/eval/pace_model.py` (hooks, `site_energies[_dense]`, `_node_energies` feature-major)
- Test: `tests/test_pool_first.py`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces:
  - Model hooks:
    - `edge_basis_factors(rij, zi, zj, mask) -> (b (E, n_b), Y (E, n_Y))`: the per-edge fixed basis, zero on invalid edges;
    - `pool_first_weights() -> W (NZ_i, NZ_j, n_a, n_b)`, built in the traced function from the trainable coefficients;
    - `pool_first_sel_y` (n_Y, n_a), a one-hot, static.
  - `EdgeSiteModel.pool_first_dense(b, Y, zj, node_z) -> A_t (C*n_a, n)`, feature-major.
  - `EdgeSiteModel.pool_first_sparse(b, Y, seg, zj, node_z, n_nodes) -> A_t (C*n_a, n)`.
  - `PACEModel._node_energies_t(A_t, cr, d, dcin, seg, n, node_z)`, which replaces `_node_energies`. The zbl tail is kept.

- [ ] **Step 1: Write the failing tests.** They check that pool-first A equals the current A, that gradients with respect to `crad` match (Review Focus 5), and that zbl is still supported.

```python
import dataclasses
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase.build import bulk

from ace_jax.eval import load
from conftest import pace_fixture

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"
REF = pathlib.Path(__file__).parent / "fixtures" / "perf_ref"


@pytest.mark.parametrize("name", ["gesi_sbessel", "sige_zbl", "sige_distance", "si_chebexpcos"])
def test_energy_gradient_wrt_crad_matches_reference(name):
    """Pool-first builds W from crad inside the trace: dE/dcrad must match the
    old path's (frozen as finite differences of the reference energy)."""
    model, meta, _ = load(str(pace_fixture(FIX / f"{name}.yace")))
    at = bulk("Si", "diamond", a=5.43, cubic=True)
    at.numbers[::2] = meta["elements"][-1]
    from ace_jax.eval.nlist import sparse_graph
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers])
    s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)

    def E(crad):
        m = dataclasses.replace(model, crad=crad)
        return jnp.sum(m.site_energies(jnp.asarray(g.rij), nz[s], nz[r], s, len(at), nz))

    grad = jax.grad(E)(model.crad)
    d = np.zeros(model.crad.shape); d.flat[np.argmax(np.abs(np.asarray(grad)))] = 1e-6
    fd = (E(model.crad + d) - E(model.crad - d)) / 2e-6
    assert abs(float(jnp.vdot(grad, d / 1e-6)) - float(fd)) < 1e-6 * max(1.0, abs(float(fd)))
```

The frozen references from Task 1 are the equality oracle for E, F and V, across dense, sparse, zbl and every radial kind. This test adds the gradient-with-respect-to-parameters guard.

- [ ] **Step 2: Run it.** It passes on the current code, since today's path is differentiable in `crad`. That's intended: it must *keep* passing after the change, so it's a regression guard. Confirm it runs.
      Run: `uv run pytest tests/test_pool_first.py -q`
      Expected: PASS (on the current code).

- [ ] **Step 3: Implement the shared pool-first helper** in `EdgeSiteModel`. The prototype's `pool_first_dense` becomes a generic form, feature-major only.

```python
    # -------------------------------------------------- pool-first A (feature-major)
    def pool_first_dense(self, b, Y, zj, node_z):
        """A_t (C*n_a, n) = sum_k W[z_i, z_j] b_k (x) Y_y, pooled BEFORE W.

        Every model here has a radial that is linear in per-pair coefficients
        over a fixed per-edge basis b (PACE: g_k with crad; ACE: polynomials,
        spline or factorised table with its weights), so the per-edge R_nl is
        never formed: pool b (x) Y per (node, neighbour-species channel), then
        apply W per node.  Feature-major output so the product-basis gathers read
        whole rows.  docs/dev/pace-performance-gap.md #4, #5."""
        n, K, nb = b.shape
        C = self.a_channels
        hi = jax.lax.Precision.HIGHEST
        if C > 1:
            oh = jax.nn.one_hot(zj, C, dtype=b.dtype)
            b = (oh[..., :, None] * b[..., None, :]).reshape(n, K, C * nb)
        Ag = jnp.einsum("nkg,nky->ngy", b, Y, precision=hi)                  # (n, C*nb, nY)
        Agy = jnp.matmul(Ag, self.pool_first_sel_y, precision=hi)            # (n, C*nb, n_a)
        Agy = Agy.reshape(n, C, nb, -1)
        W = self.pool_first_weights()                                        # (NZ, C, n_a, nb)
        return jnp.einsum("nmka,nmak->man", Agy, W[node_z], precision=hi).reshape(-1, n)

    def pool_first_sparse(self, b, Y, seg, zj, node_z, n_nodes):
        """`pool_first_dense` for an edge list: pool b (x) Y per (node, channel)
        by segment_sum, then W per node.  Same output layout."""
        C = self.a_channels
        hi = jax.lax.Precision.HIGHEST
        nb = b.shape[1]
        outer = (b[:, :, None] * Y[:, None, :]).reshape(b.shape[0], -1)       # (E, nb*nY)
        Ag = jax.ops.segment_sum(outer, seg * C + (zj if C > 1 else 0),
                                 num_segments=n_nodes * C)
        Ag = Ag.reshape(n_nodes, C, nb, -1)
        Agy = jnp.matmul(Ag, self.pool_first_sel_y, precision=hi)             # (n, C, nb, n_a)
        W = self.pool_first_weights()
        return jnp.einsum("nmka,nmak->man", Agy, W[node_z], precision=hi).reshape(-1, n_nodes)
```

- [ ] **Step 4: Implement the PACE hooks.** `W` comes from `crad` in the trace, using static index arrays computed once at load.

```python
    # in PACEModel fields (static structure, set by load_yace):
    pf_col: jax.Array = None     # (n_a,) radial column of each local A entry
    pf_sel_y: jax.Array = None   # (n_Y, n_a) one-hot

    @property
    def pool_first_sel_y(self):
        return self.pf_sel_y

    def pool_first_weights(self):
        """W (NZ, NZ, n_a, K_rad) from crad, in the trace (crad stays trainable):
        column c < K_rad is g_c itself (a delta row); else R_nl with
        c = K_rad + n*(lmax+1) + l, i.e. the crad[:, :, n, l, :] row."""
        kr, nl = self.nradbase, self.crad.shape[3]
        c = self.pf_col
        is_g = c < kr
        nn, l = jnp.divmod(jnp.maximum(c - kr, 0), nl)
        rows = self.crad[:, :, nn, l, :]                                   # (NZ, NZ, n_a, kr)
        delta = jax.nn.one_hot(jnp.where(is_g, c, 0), kr, dtype=self.crad.dtype)
        return jnp.where(is_g[None, None, :, None], delta[None, None], rows)

    def edge_basis_factors(self, rij, zi, zj, mask=None):
        """(g_k, Y_lm) per edge: the fixed radial basis only, zero on invalid edges."""
        bp, valid, r, rij_s = self._geometry(rij, zi, zj, mask)
        lam, rc, dcut, cin, dcin = (bp[:, k] for k in range(5))
        g = radbase(r, self.radbasename, self.inner_cutoff_type, lam, rc, dcut,
                    cin, dcin, self.nradbase)
        return jnp.where(valid[:, None], g, 0.0), real_spherical_harmonics(rij_s, self.lmax)
```

In `load_yace`, compute the tables:

```python
    ny = (int(a["lmax"]) + 1) ** 2
    sel = np.zeros((ny, len(b["a_y"])))
    sel[np.asarray(b["a_y"]), np.arange(len(b["a_y"]))] = 1.0
    # ... PACEModel(..., pf_col=I(b["a_rad"]), pf_sel_y=A(sel))
```

Then `site_energies_dense` and `site_energies` pool first and go feature-major:

```python
    def site_energies_dense(self, rij, zi, zj, mask, node_z):
        n, K = mask.shape
        flat = lambda a: a.reshape(n * K, *a.shape[2:])
        g, Y = self.edge_basis_factors(flat(rij), flat(zi), flat(zj), flat(mask))
        At = self.pool_first_dense(g.reshape(n, K, -1), Y.reshape(n, K, -1), zj, node_z)
        seg = jnp.repeat(jnp.arange(n), K)
        return self._node_energies_t(At, *self._edge_core(flat(rij), flat(zi), flat(zj),
                                                          flat(mask)), seg, n, node_z)

    def site_energies(self, rij, zi, zj, segment_ids, n_nodes, node_z, mask=None):
        g, Y = self.edge_basis_factors(rij, zi, zj, mask)
        At = self.pool_first_sparse(g, Y, segment_ids, zj, node_z, n_nodes)
        return self._node_energies_t(At, *self._edge_core(rij, zi, zj, mask), segment_ids,
                                     n_nodes, node_z)
```

`_node_energies_t` is `_node_energies` on `At`, with the products taken over rows (`jnp.prod(At[s.T], axis=0)`) and `rho` computed from `ct.T @ AA`. The tail after `rho` (the embedding, the zbl or distance core switch, `E0`) is unchanged: move it into a helper `_energy_tail(rho, cr, d, dcin, seg, n, node_z)` that both branches share. This keeps zbl support, which the prototype had dropped.

```python
    def _node_energies_t(self, At, cr, d, dcin, segment_ids, n_nodes, node_z):
        """Site energies from feature-major A (NZ*n_a, n): the product gathers
        read whole rows and their adjoints add whole rows."""
        AA = jnp.concatenate([jnp.prod(At[s.T], axis=0) for s in self.aa_specs], axis=0)
        ct = self.ctilde_real().reshape(self.n_aa, -1)
        rho_all = jnp.matmul(ct.T, AA, precision=jax.lax.Precision.HIGHEST)
        rho = rho_all.reshape(self.nz, self.ndensity, n_nodes)[node_z, :, jnp.arange(n_nodes)]
        return self._energy_tail(rho, cr, d, dcin, segment_ids, n_nodes, node_z)
```

Delete `edge_a_factors`'s use on the energy path and remove the old `_node_energies`. `edge_a_factors` and `pool_a_*` stay only if something else uses them: check with `grep -rn "pool_a_dense\|pool_a_sparse\|edge_a_factors" src/`. `ACEModel` still does until Task 9.

- [ ] **Step 5: Run.**
      Run: `uv run pytest tests/test_pool_first.py tests/test_perf_parity.py tests/test_pace_*.py tests/test_dense_a.py tests/test_edge_a.py tests/test_calc_layout.py -q -p no:cacheprovider`
      Expected: all pass. The PACE parity tests and the frozen references hold at 1e-12. If `test_edge_a.py` exercised the PACE `edge_a` forms that no longer exist, update it to cover `ACEModel` only, and note the ruling in the ledger.

- [ ] **Step 6: Commit.**

```bash
git add src/ace_jax/eval/edge_model.py src/ace_jax/eval/pace_model.py tests/test_pool_first.py tests/
git commit -m "perf(pace): pool-first A (W from crad in the trace) and a feature-major product basis"
```

---

### Task 6: Skin neighbour state and the one-call step (calculator)

**Files:**
- Create: `src/ace_jax/calc/skin.py`
- Modify: `src/ace_jax/calc/point.py` (`ACECalculator.__init__(..., skin=1.0)`; dense path via `SkinState`)
- Test: `tests/test_skin.py`

**Interfaces:**
- Consumes: `nlist.dense_graph(..., device=)`, `nlist.reverse_slots`, and `energy_forces_virial_dense(..., rev=)`.
- Produces:
  - `skin.SkinState`, which holds `pos0`, `cell`, `pbc`, `numbers`, `idx_s`, `shift_s`, `live_s`, `rev_s`, `node_z`, `K_skin` and `K`;
  - `skin.build(pos, cell, pbc, numbers, lut, rc, skin, K_skin_hint, device) -> SkinState`;
  - `skin.valid(state, pos, cell, pbc, numbers, skin) -> bool`;
  - `skin.step(model, x, state_arrays, rc, K) -> packed (E, F, V, overflow, kmax)`, jitted;
  - on `ACECalculator`: `self.last_timing` gains `"rebuilds"`, and `nlist_s` counts only rebuild time.

- [ ] **Step 1: Write the failing tests.** Cover equality with rebuild-every-call (`skin=0`), the rebuild triggers, and the Review Focus 1–3 cases.

```python
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import all_changes

from ace_jax.calc.point import ACECalculator

M = "fixtures/sige_nofit.npz"


def _efs(calc, at):
    calc.calculate(at, ["energy", "forces", "stress"], all_changes)
    r = calc.results
    return r["energy"], r["forces"].copy(), r.get("stress", np.zeros(6)).copy()


def _cell(kind):
    if kind == "tric":
        a = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
        a.set_cell(a.cell.array @ np.array([[1, .08, 0], [0, 1, .05], [0, 0, 1]]), scale_atoms=True)
    elif kind == "small":
        a = bulk("Si", "diamond", a=5.43)                       # many images per pair
    a.numbers[::2] = 32
    return a


@pytest.mark.parametrize("kind", ["tric", "small"])
def test_skin_matches_rebuild_along_a_trajectory(kind):
    at = _cell(kind)
    fresh, reuse = ACECalculator(M, layout="dense", skin=0.0), ACECalculator(M, layout="dense", skin=1.0)
    rng = np.random.default_rng(1)
    for step in range(12):
        at.positions += rng.normal(0, 0.03, at.positions.shape)     # ~0.4 A over the run
        E0, F0, S0 = _efs(fresh, at)
        E1, F1, S1 = _efs(reuse, at)
        assert abs(E1 - E0) < 1e-12 * max(1, abs(E0))
        np.testing.assert_allclose(F1, F0, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(S1, S0, rtol=1e-12, atol=1e-12)
    assert reuse.last_timing["rebuilds"] < fresh.last_timing["rebuilds"]


def test_rebuild_triggers():
    at = _cell("tric")
    c = ACECalculator(M, layout="dense", skin=1.0)
    _efs(c, at); n0 = c.last_timing["rebuilds"]
    _efs(c, at); assert c.last_timing["rebuilds"] == n0                  # unchanged
    b = at.copy(); b.positions[0] += [0.6, 0, 0]                        # > skin/2
    _efs(c, b); assert c.last_timing["rebuilds"] == n0 + 1
    d = at.copy(); d.set_cell(d.cell * 1.01, scale_atoms=True)          # cell change
    _efs(c, d); assert c.last_timing["rebuilds"] == n0 + 2
    e = at.copy(); e.numbers[1] = 14 if e.numbers[1] == 32 else 32      # species change
    _efs(c, e); assert c.last_timing["rebuilds"] == n0 + 3


def test_unrelated_structure_same_size_rebuilds_and_is_correct():
    a, b = _cell("tric"), _cell("tric")
    b.positions = np.random.default_rng(2).permutation(b.positions)    # same set, relabelled
    c, ref = ACECalculator(M, layout="dense", skin=1.0), ACECalculator(M, layout="dense", skin=0.0)
    _efs(c, a)
    E, F, _ = _efs(c, b)
    E0, F0, _ = _efs(ref, b)
    assert abs(E - E0) < 1e-12 * abs(E0)
    np.testing.assert_allclose(F, F0, rtol=1e-12, atol=1e-12)


def test_isolated_atoms():
    at = Atoms(numbers=[14, 32], positions=[[0, 0, 0], [20, 0, 0]], cell=[40] * 3, pbc=False)
    c = ACECalculator(M, layout="dense", skin=1.0)
    E, F, _ = _efs(c, at)
    E0, F0, _ = _efs(ACECalculator(M, layout="dense", skin=0.0), at)
    assert np.isfinite(E) and np.all(np.isfinite(F))
    assert abs(E - E0) < 1e-12 and np.allclose(F, 0.0)
```

- [ ] **Step 2: Run them. They should fail** on `unexpected keyword 'skin'`.
      Run: `uv run pytest tests/test_skin.py -q`

- [ ] **Step 3: Implement `calc/skin.py`** from `bench/perf/fast_calc.py`. `_step` becomes `step`, with these changes:
  - pass `rev` through. Compute `rev_s = reverse_slots(idx_s, rij_s, count_s)` on the host at build time, over the **skin** list;
  - in `step`, express the gather in skin-slot space: the compact position of skin slot `s` is `csum[:, s] - 1` where `valid`. Compute the forces from `g` on compact slots with

```python
    # skin slot -> compact slot, for the reverse-edge gather (gathers only)
    comp = jnp.where(valid, csum - 1, 0)                                   # (n, Ks)
    j_s, r_s = idx_s, rev_s                                                # (n, Ks)
    back_valid = valid[j_s, r_s]                  # the reverse edge is inside the cutoff
    back_comp = comp[j_s, r_s]                    # ... and sits at this compact slot of row j
```

    Then call the model with `rev=None`, and assemble the forces from `g_r` returned by a variant of `energy_forces_virial_dense` that also returns `g_r`. To keep one code path, add a keyword `return_edge_grad=True` to `energy_forces_virial_dense` that returns `(E, g_r, V)` instead of `(E, F, V)`. The skin step then builds

```python
    F = g_r.sum(axis=1) - jnp.where(valid & back_valid,
                                    g_r[j_s, back_comp], 0.0).sum(axis=1)
```

    summed over the skin slots `s` of row `i`: the reverse edge `(j → i)` has force contribution `g_r[j, back_comp]`;
  - the displacement check `max |x − pos0| > skin/2` is computed in `step` and returned in the packed output next to the overflow flag;
  - `build` vectorises the species map with a 119-entry lookup table, and raises `ValueError` naming the unknown elements.

Then `ACECalculator`:

```python
    def __init__(self, model, meta=None, cutoff=None, dtype=None, edge_a_kind="auto",
                 layout="auto", skin=1.0, **kw):
        ...
        self.skin = float(skin)
        self._skin_state = None
        self._rebuilds = 0
```

In `calculate`:
- **Reuse:** when `self.skin > 0`, the layout resolves to dense, and `skin.valid(...)` holds, call the jitted `skin.step`.
- **Retry:** if the packed flags say the displacement or overflow check failed, rebuild (`skin.build`, `self._rebuilds += 1`) and call it again.
- **Sparse or `skin == 0`:** the current path, unchanged. Count every call there as a rebuild.
- **Timing:** `last_timing = {"nlist_s": <rebuild time this call, 0 if none>, "model_s": ..., "nlist_backend": ..., "rebuilds": self._rebuilds}`.

- [ ] **Step 4: Run.**
      Run: `uv run pytest tests/test_skin.py tests/test_perf_parity.py tests/test_calc_jit.py tests/test_calc_layout.py tests/test_calc_edge_a.py -q -p no:cacheprovider`
      Expected: all pass. `test_perf_parity` now runs both `skin=0` and `skin=1`.

- [ ] **Step 5: Measure locally,** as in the report §8.2, to confirm the direction on CPU.
      Run: `PYTHONPATH=bench:src uv run python bench/perf/e2e.py pace_SiGe_small SiGe 4096` (after copying the models from the main checkout, as `bench/perf` README step 0 says)
      Expected: the skin calculator is faster than `skin=0`. Record both numbers in the ledger.

- [ ] **Step 6: Commit.**

```bash
git add src/ace_jax/calc/skin.py src/ace_jax/calc/point.py src/ace_jax/eval/edge_model.py tests/test_skin.py
git commit -m "perf(calc): skin neighbour list (default 1 A) and one jitted step per call"
```

---

### Task 7: LAMMPS bundle, owned rows and cutoff-sized slots

**Files:**
- Modify: `src/ace_jax/export/lammps.py` (`make_energy_fn`, `export_lammps`), `bench/scaling/run_lammps.py` (`capacity`)
- Test: `tests/test_export_lammps.py` (add cases), `tests/test_bench_scaling.py` (capacity)

**Interfaces:**
- Consumes: Tasks 3 and 5 (the model path).
- Produces:
  - `make_energy_fn(model, n_species, layout, k_dense=None, type_map=None, n_rows=None)`: with `n_rows`, the dense layout evaluates rows `< n_rows` only, and returns NaN if a sender is `≥ n_rows` or a slot overflows;
  - `export_lammps(..., max_owned=None)`, which records `bundle["ace_jax"]["max_owned"]`;
  - `capacity(at, rcut, skin=1.0)`, which returns `max_owned = ⌈1.1 n⌉`, and `k_dense` and `max_edges` sized for `rcut`.

- [ ] **Step 1: Write the failing tests.**

```python
@pytest.mark.parametrize("layout", ["dense"])
def test_owned_rows_bundle_matches_calculator(layout):
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    at = _cluster()
    graph, g = _lammps_graph(at, meta["rcut"])
    z2i = {z: i for i, z in enumerate(meta["elements"])}
    species = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    K = int(np.bincount(g.senders, minlength=len(at)).max())
    f = make_energy_fn(model, len(meta["elements"]), layout, k_dense=K + 3,
                       n_rows=int(np.ceil(1.1 * len(at))))
    E, G = jax.value_and_grad(lambda p: jnp.sum(f(p, species, graph)))(jnp.asarray(at.positions))
    at.calc = ACECalculator(y, layout="sparse", skin=0.0)
    assert float(E) == pytest.approx(at.get_potential_energy(), abs=1e-10)
    np.testing.assert_allclose(-np.asarray(G), at.get_forces(), atol=1e-9)


def test_owned_rows_overflow_is_nan():
    y = str(pace_fixture(FIX / "gesi_sbessel.yace"))
    model, meta, _ = load(y)
    at = _cluster()
    graph, _ = _lammps_graph(at, meta["rcut"])
    f = make_energy_fn(model, len(meta["elements"]), "dense", k_dense=64, n_rows=len(at) // 2)
    assert np.isnan(float(jnp.sum(f(jnp.asarray(at.positions), jnp.zeros(len(at), jnp.int32), graph))))
```

And in `tests/test_bench_scaling.py`:

```python
def test_capacity_sizes_slots_for_rcut_not_skin():
    from scaling.run_lammps import capacity
    at = supercell("Cantor", 8192)
    c = capacity(at, 5.0)
    assert c["max_owned"] == int(np.ceil(1.1 * len(at)))
    c_skin = capacity(at, 6.0)                      # what rcut + skin used to give
    assert c["k_dense"] < c_skin["k_dense"]
```

- [ ] **Step 2: Run them. They should fail** on the `n_rows` keyword and the missing `max_owned`.
      Run: `uv run --with-editable /Users/u1470235/gits/lammps-jax pytest tests/test_export_lammps.py tests/test_bench_scaling.py -q`

- [ ] **Step 3: Implement.** Port `bench/perf/lammps_variant.py::make_energy_fn_owned` into `make_energy_fn` behind `n_rows`, using the same regroup restricted to `s < n_rows`, with the site energies padded to `n` by zeros and NaN on overflow. Then:
  - `export_lammps` passes `n_rows=max_owned` when given;
  - `capacity` sizes `k_max` from a neighbour list at `rcut`, with `k_dense = k_max + 4`, `max_edges = max_owned × k_dense`, and `max_owned = ⌈1.1 n⌉`;
  - `run_lammps.export_bundle` passes `max_owned`.

- [ ] **Step 4: Run** the tests above plus `tests/test_perf_parity.py`.
      Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add src/ace_jax/export/lammps.py bench/scaling/run_lammps.py tests/test_export_lammps.py tests/test_bench_scaling.py
git commit -m "perf(lammps): bundle evaluates owned rows only; slots sized for rcut (not rcut+skin)"
```

---

### Task 8: `ACEModel` profile and decision

**Files:**
- Create: `bench/perf/profile_ace.py`, `docs/dev/perf-ace-profile.md`
- Modify: `bench/perf/modal_profile.py` (an `ace` entrypoint)

**Interfaces:**
- Consumes: the Modal image (`bench/scaling/modal_app.py::base_image`, `with_sources`), and the benchmark's `bench/scaling/models/ace_{SiGe,Cantor}_{small,medium,large}.npz` (copy them from the main checkout).
- Produces: `docs/dev/perf-ace-profile.md` with a per-stage GPU time table for each model, and a **decision table**: for each candidate ACE change (pool-first for the `analytic` / `spline` / `spline_factorised` radials, and a feature-major `_aa`), apply or skip, with the reason.

- [ ] **Step 1:** Write `profile_ace.py` by adapting `bench/perf/profile_acejax.py` to `ACEModel`. Time `energy_forces_virial_dense` at 8192 atoms for each `ace_*` model, with a `jax.profiler` trace and HLO-fusion attribution (`hlo_trace.py` / `hlo_fusion.py`). Attribute time to: radial evaluation, `pool_a_dense`, `_aa` products (forward and adjoint), readout, and the force scatter.
- [ ] **Step 2:** Run on Modal: `uv run --with modal modal run bench/perf/modal_profile.py::ace`.
      Expected: a JSON table per model in `bench/perf/results/ace_*.json`.
- [ ] **Step 3:** Write `docs/dev/perf-ace-profile.md`: the table plus the decision, using these rules:
  - **Pool-first is applied for a radial kind** when its fixed basis width `n_b`, times channels (1 for ACE), is at most the `R_nl` width it replaces, **and** the profile attributes at least 15% of the call to radial plus A assembly.
    - For `spline`, the basis is 4-sparse over `ncoef` columns, so `n_b = ncoef` is normally larger than `n_rnl`: expect **skip**.
    - For `analytic` and `spline_factorised`, expect **apply**.
  - **Feature-major `_aa`** is applied if the `_aa` forward plus adjoint is at least 15% of the call.
  - Record the decision in the ledger as `Task 8: Ruling: ...`.
- [ ] **Step 4:** Commit.

```bash
git add bench/perf/profile_ace.py bench/perf/modal_profile.py bench/perf/results/ace_*.json docs/dev/perf-ace-profile.md
git commit -m "perf(ace): ACEModel GPU profile and the pool-first / feature-major decision"
```

---

### Task 9: `ACEModel` pool-first and feature-major product basis (as decided in Task 8)

**Files:**
- Modify: `src/ace_jax/eval/model.py`
- Test: `tests/test_pool_first.py` (ACE cases), plus the frozen references

**Interfaces:**
- Consumes: Task 5's `pool_first_dense` and `pool_first_sparse` (in `EdgeSiteModel`), and Task 8's decision table.
- Produces, for each radial kind marked **apply**:
  - `ACEModel.edge_basis_factors`: `(b = P(x)·env` for `analytic` / `P(x)·env` for `spline_factorised`, `Y)`;
  - `pool_first_weights`: `Wnlq[zi,zj]` mapped to A entries for `analytic`, `emb[zj][kidx]` on `nidx` for `factorised`;
  - `pool_first_sel_y`;
  - with a feature-major `_aa` and readout if marked apply.

- [ ] **Step 1: Write the failing test** for each applied kind. The gradient of E with respect to the radial weights (`rnl_Wnlq` for analytic, `rnl_embedding` for factorised) matches a central finite difference. This is the same structure as Task 5's test, with `dataclasses.replace(model, rnl_Wnlq=...)`. The frozen references cover E, F and V.

```python
@pytest.mark.parametrize("path,field", [("fixtures/si_ace_model.npz", "rnl_Wnlq")])
def test_ace_energy_gradient_wrt_radial_weights(path, field):
    from ace_jax.eval.nlist import sparse_graph
    model, meta, _ = load(path)
    at = bulk("Si", "diamond", a=5.43, cubic=True)
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    nz = jnp.zeros(len(at), jnp.int32)
    s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)

    def E(w):
        m = dataclasses.replace(model, **{field: w})
        return jnp.sum(m.site_energies(jnp.asarray(g.rij), nz[s], nz[r], s, len(at), nz))

    w0 = getattr(model, field)
    grad = jax.grad(E)(w0)
    d = np.zeros(w0.shape); d.flat[np.argmax(np.abs(np.asarray(grad)))] = 1e-6
    fd = (E(w0 + d) - E(w0 - d)) / 2e-6
    assert abs(float(jnp.vdot(grad, d / 1e-6)) - float(fd)) < 1e-6 * max(1.0, abs(float(fd)))
```

- [ ] **Step 2: Run it.** It passes on the current code, as a regression guard. Confirm it runs.
- [ ] **Step 3: Implement the hooks** for the applied kinds:
  - **`analytic`:** `b = poly_recursion(x, *ABC) · env` has shape (E, n_q). `W[zi, zj, a, q] = Wnlq[zi, zj, rnl(a), q]`, where `rnl(a) = aspec_r[a]`. `a_channels = 1`, so `W` is `(NZ, 1, n_a, n_q)` after indexing by `zj` in the trace. The shared helper's channel is the neighbour species, so with `C = 1` fold the `zj` dependence into `b`. Compute `b_zj = b ⊗ onehot(zj)` inside `edge_basis_factors`, width `NZ · n_q`, and `W[zi, 0, a, (z, q)] = Wnlq[zi, z, rnl(a), q]`.
  - **`spline_factorised`:** `b = P(x)[:, nidx-columns] · env`, and `W` applies `emb[zj][kidx]`, folded the same way.
  - **Readout:** if feature-major is applied, `_aa` becomes `jnp.prod(At[s.T], axis=0)` on `At` (n_A, n), and `_readout_folded` takes `ctilde.T @ AA`.
  - **Kinds marked skip** keep the current `edge_features` / `pool_a_*` path, unchanged.
- [ ] **Step 4: Run.**
      Run: `uv run pytest tests/test_pool_first.py tests/test_perf_parity.py tests/test_fold.py tests/test_roundtrip.py tests/test_efv.py -q -p no:cacheprovider`, and the julia-parity tests if Julia is available locally.
      Expected: all pass.
- [ ] **Step 5: Micro-benchmark on Modal** (`bench/perf/modal_profile.py::ace`) to confirm each applied change is faster at 8k atoms. If one isn't, revert it and record the ruling.
- [ ] **Step 6: Commit.**

```bash
git add src/ace_jax/eval/model.py tests/test_pool_first.py
git commit -m "perf(ace): pool-first A and feature-major product basis where the profile pays"
```

---

### Task 10: Micro-benchmarks and MD-like standalone timing

**Files:**
- Create: `bench/perf/microbench.py`
- Modify: `bench/perf/modal_profile.py` (a `microbench` entrypoint), `bench/scaling/run_standalone.py`
- Test: `tests/test_bench_scaling.py` (standalone row fields)

**Interfaces:**
- Consumes: Tasks 2–9.
- Produces:
  - `microbench.py`, which times the model call and end to end for Cantor medium float64 at 1k / 8k / 65k atoms, and ACE medium at 8k, **at this branch's HEAD and at the merge base**. For the "before" run, create a local worktree of the merge base, `git worktree add /tmp/perf-base $(git merge-base HEAD feat/bench-scaling)`, and set `BENCH_SRC_ROOT=/tmp/perf-base` so that `modal_profile.py` builds its image with `with_sources` rooted there (add that env-var hook to `with_sources`, defaulting to this checkout). Writes `bench/perf/results/microbench_{before,after}.json`;
  - `run_standalone` rows gain `rebuilds` and `md_like: true`; each timed call displaces the atoms by N(0, 1e-3 Å).

- [ ] **Step 1: Write the failing test.**

```python
def test_standalone_is_md_like(tmp_path):
    from scaling.run_standalone import run_case
    row = {"code": "acejax-pace", "system": "SiGe", "size": "small",
           "path": str(pathlib.Path(__file__).parent.parent / "fixtures" / "pace" / "gesi_sbessel.yace"),
           "elements": ["Si", "Ge"], "name": "acejax-pace/SiGe/small"}
    out = run_case(row, 256, "float64", "cpu", reps=4)
    assert out["md_like"] is True and out["rebuilds"] <= 2               # skin reused
```

- [ ] **Step 2: Run it. It should fail** on the `md_like` key.
- [ ] **Step 3: Implement.** In `run_standalone.run_case`'s ace-jax branch, each timed call first does `at.positions += rng.normal(0, 1e-3, at.positions.shape)`, with the `rng` seeded at 0. After timing, record `out["rebuilds"] = calc.last_timing["rebuilds"]` and `out["md_like"] = True`. Apply the same displacement to the MACE branch, so the comparison is fair.
- [ ] **Step 4: Write and run `microbench.py` on Modal.**
      Expected: a before/after table. Targets: the model call at 8k ≤ 4.5 ms, and end to end ≥ 1.1M atom-steps/s. Record the actual numbers in the ledger either way.
- [ ] **Step 5: Commit.**

```bash
git add bench/perf/microbench.py bench/perf/modal_profile.py bench/scaling/run_standalone.py tests/test_bench_scaling.py bench/perf/results/microbench_*.json
git commit -m "bench: MD-like standalone timing (rebuild count); before/after micro-benchmarks"
```

---

### Task 11: Re-run the ace-jax rows, docs, PR

**Files:**
- Modify: `bench/scaling/results/{moriarty-gpu,moriarty-cpu,modal-a100}.jsonl` (the ace-jax rows only), `docs/dev/benchmarks.md`, `docs/dev/figs/*`
- Create: `docs/dev/perf-optimisation-results.md`

- [ ] **Step 1: Keep a copy of the current ace-jax rows** as the "before" series. Copy them to `bench/scaling/results/before-perf/*.jsonl`, then drop them from the live files, standalone and LAMMPS.
- [ ] **Step 2: Modal.** Run `modal run bench/scaling/modal_app.py --only acejax-pace` and `--only acejax-ace`, in parallel, on the Volume-backed runner.
- [ ] **Step 3: moriarty.** Once idle (`/proc/loadavg < 2`, `nvidia-smi` empty, `who`), run a queue script as in the benchmark branch: GPU `acejax-pace` and `acejax-ace`, then CPU `acejax-pace` and `acejax-ace`.
- [ ] **Step 4: Re-render the figures,** and add a before/after figure (throughput vs N, ace-jax before and after, with ML-PACE for reference) to `docs/dev/benchmarks.md`.
- [ ] **Step 5: Write `docs/dev/perf-optimisation-results.md`:**
  - micro-benchmarks before and after (Task 10);
  - the scaling suite summary;
  - each success criterion, met or missed with the number;
  - a link to `docs/dev/pace-performance-gap.md` and `docs/dev/perf-ace-profile.md`.
- [ ] **Step 6: Final review.** Run the full test suite, then the whole-branch review per the execution skill.
- [ ] **Step 7: Open the PR.** Push `perf/ace-jax-speedups` and open a PR against `feat/bench-scaling`, or `main` if that has merged. Description: summary, results table, parity statement, and the ledger's rulings.

```bash
git add bench/scaling/results docs/dev/benchmarks.md docs/dev/figs docs/dev/perf-optimisation-results.md
git commit -m "bench: ace-jax rows after the speed-ups; results doc"
```

---

## Self-review notes

- **Spec coverage:**
  - components 1 → Task 6, 2 → Task 3, 3 → Task 4, 4 → Task 2, 5 → Task 5, 6 → Tasks 8–9, 7 → Task 7, 8 → Tasks 10–11;
  - parity → Task 1, and in every task;
  - success criteria → Tasks 10–11.
- **Spec amendments,** made in the spec in the same commit as this plan:
  - chunking is dense-only;
  - `rev` is built on the host at rebuild time;
  - pool-first is a shared helper fed by per-model hooks, since every radial is linear in per-pair coefficients over a fixed basis.
- **Type consistency:**
  - `energy_forces_virial_dense(rij, zi, zj, idx, mask, node_z, chunk=, rev=, return_edge_grad=)`, used in Tasks 3, 4 and 6;
  - `pool_first_{dense,sparse}` output `A_t (C·n_a, n)`, used in Tasks 5 and 9;
  - `reverse_slots(idx, rij, count)`, used in Tasks 4 and 6;
  - `make_energy_fn(..., n_rows=)`, used in Task 7.
