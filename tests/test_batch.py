"""Batched evaluation (`ace_jax.batch`) against ACECalculator structure by structure: energies to 1e-12
relative, forces to 1e-11 of max|F|, stress to roundoff, over small and odd cells (a primitive cell below
the cutoff, sheared, a slab, a cluster, one periodic atom, unwrapped positions), mixed sizes in one call,
1, 2 and 5 species, a species-embedding model, a PACE .yace, the E0 reference, float32 and the C++
kernel; and the device neighbour builder against `sparse_graph`'s pairs."""
import functools
import os
import pathlib
import shutil

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from ase import Atoms
from ase.build import bulk

from ace_jax.batch import BatchCalculator, bucket_size, complete_cell, neighbours, pack, unpad
from ace_jax.calc.point import ACECalculator
from ace_jax.eval import load, sparse_graph
from conftest import FIXTURE_DIR, pace_fixture

PACE = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"


def structures(elements, seed=0):
    """Small and odd cells with every species of the model: a primitive cell, a cubic cell, a sheared
    supercell, a slab (pbc T T F, vacuum), a cluster (no cell), one periodic atom, a 2x2x2 supercell and
    a cell with unwrapped positions; rattled, species drawn at random."""
    rng = np.random.default_rng(seed)
    metal = len(elements) > 2
    a0 = 3.6 if metal else 5.43
    sym, lat = ("Ni", "fcc") if metal else ("Si", "diamond")
    prim = bulk(sym, lat, a=a0)
    cub = bulk(sym, lat, a=a0, cubic=True)
    sheared = cub.repeat((2, 1, 1))
    sheared.set_cell(sheared.cell.array + [[0, 0, 0], [1.1, 0, 0], [0.4, 0.7, 0]], scale_atoms=True)
    slab = cub.repeat((2, 2, 1))
    slab.center(vacuum=6.0, axis=2)
    slab.pbc = (True, True, False)
    cluster = cub.copy()
    cluster.pbc, cluster.cell = False, np.zeros((3, 3))
    one = Atoms(sym, positions=[[0.1, 0.2, 0.3]], cell=np.diag([2.7, 2.9, 3.1]) if metal else np.diag([2.3] * 3),
                pbc=True)
    big = cub.repeat((2, 2, 2))
    unwrapped = cub.copy()
    unwrapped.positions += 3 * unwrapped.cell[0] - 2 * unwrapped.cell[2]
    unwrapped.positions[::2] += unwrapped.cell[1]
    out = []
    for at in (prim, cub, sheared, slab, cluster, one, big, unwrapped):
        at = at.copy()
        at.rattle(0.05, seed=int(rng.integers(1 << 30)))
        at.numbers = rng.choice(np.asarray(elements, int), len(at))
        out.append(at)
    return out


def reference(calc, ats):
    E, F, S = [], [], []
    for at in ats:
        at = at.copy()
        at.calc = calc
        E.append(at.get_potential_energy())
        F.append(np.asarray(at.get_forces()))
        S.append(np.asarray(at.get_stress()) if at.pbc.all() else np.full(6, np.nan))
    return np.array(E), F, np.array(S)


@functools.cache
def ref_for(name):
    """ACECalculator on structures(elements) for a model, shared by the tests that compare against it:
    each would otherwise compile its 8 shapes again."""
    m, meta = model_for(name)
    return reference(ACECalculator(m, meta), structures(meta["elements"]))


def check(res, ref, e_rel=1e-12, f_rel=1e-11, s_abs=1e-12):
    E, F, S = ref
    assert np.abs(res.energy - E).max() <= e_rel * np.abs(E).max(), np.abs(res.energy - E)
    fmax = max(np.abs(f).max(initial=0.0) for f in F)
    for k, f in enumerate(F):
        assert res.forces[k].shape == f.shape
        assert np.abs(res.forces[k] - f).max(initial=0.0) <= f_rel * max(fmax, 1e-300), k
    assert np.array_equal(np.isnan(res.stress), np.isnan(S))
    fin = np.isfinite(S)
    assert np.abs(res.stress[fin] - S[fin]).max(initial=0.0) <= s_abs * max(np.abs(S[fin]).max(initial=1.0), 1.0)


def embedding_model():
    from conftest import require_coupling_lib
    require_coupling_lib()
    import dataclasses

    from ace_jax.basis.model import build_embedding_model
    from ace_jax.eval.model import fold_readout
    els = [14, 32]
    m, meta = build_embedding_model(els, 3, 6, embedding=(els, np.array([[1.0, 0.3], [0.6, -0.8]]))).eval_pair()
    rng = np.random.default_rng(0)
    m = fold_readout(dataclasses.replace(m, WB=jnp.asarray(rng.normal(size=m.WB.shape)),
                                         Wpair=jnp.asarray(0.1 * rng.normal(size=m.Wpair.shape))))
    return m, meta


def model_for(name):
    if name == "embedding":
        return embedding_model()
    if name.endswith(".yace"):
        return load(str(pace_fixture(PACE / name)))[:2]
    return load(str(FIXTURE_DIR / name))[:2]


@pytest.mark.parametrize("name", ["sige_nofit.npz", "ace_cantor5_small.npz", "sige_density_core.yace", "embedding"])
def test_batched_matches_ACECalculator_per_structure(name):
    """One call over mixed cells and sizes (several buckets) equals ACECalculator on each, in input
    order: the same evaluation model, only the neighbour lists and the summation order differ."""
    m, meta = model_for(name)
    ats = structures(meta["elements"])
    bc = BatchCalculator(m, meta, builder="jax", n_device=256)    # test_matscipy_batched_builder covers "msn"
    res = bc.calculate(ats)
    assert res.timing["single"] == 0 and len(res.timing["buckets"]) >= 3
    check(res, ref_for(name))


def test_order_padding_and_unpad():
    """Results follow the input order whatever the buckets; unpad drops padding structures and atoms."""
    m, meta = model_for("si_ace_model.npz")
    ats = structures(meta["elements"])[::-1] + structures(meta["elements"], seed=1)
    res = BatchCalculator(m, meta, max_batch_atoms=64, n_device=256).calculate(ats)       # several batches per bucket
    assert res.timing["batches"] > len(res.timing["buckets"])
    from ace_jax.batch import _msn_batched
    if jax.default_backend() == "cpu" and _msn_batched() is None:     # auto: all pairs for tiny cells, else a cell list
        assert set(res.timing["builder"]) == {"jax", "host"}
    check(res, reference(ACECalculator(m, meta), ats))
    b = pack(ats, 5.0, meta["elements"])
    seen = {}
    for bt in b:
        for i, p in unpad(bt, bt.positions).items():
            seen[i] = p
            assert len(p) == len(ats[i])
    assert sorted(seen) == list(range(len(ats)))


def test_neighbour_overflow_reruns_with_more_slots():
    """A K too small for a bucket is detected on the device and the batch re-run at a larger K: the
    results are unchanged (never truncated)."""
    m, meta = model_for("si_ace_model.npz")
    ats = structures(meta["elements"])
    bc = BatchCalculator(m, meta, n_device=256)
    bc._K = {bucket_size(len(a)): 4 for a in ats}
    res = bc.calculate(ats)
    assert res.timing["k_retries"] > 0
    check(res, ref_for("si_ace_model.npz"))


def test_large_structures_go_through_ACECalculator():
    m, meta = model_for("si_ace_model.npz")
    ats = structures(meta["elements"])
    res = BatchCalculator(m, meta, n_device=8).calculate(ats)
    assert res.timing["single"] == sum(len(a) > 8 for a in ats)
    check(res, ref_for("si_ace_model.npz"))


def test_energy_reference_E0():
    m, meta = model_for("sige_nofit.npz")
    ats = structures(meta["elements"])
    res = BatchCalculator(m, meta, energy_reference="E0", n_device=256).calculate(ats)
    calc = ACECalculator(m, meta, energy_reference="E0")
    check(res, reference(calc, ats))
    for k, at in enumerate(ats):
        at = at.copy()
        at.calc = calc
        at.get_potential_energy()
        assert res.e0_offset[k] == calc.results["e0_offset"]


def test_float32():
    """dtype=float32 against ACECalculator(dtype=float32): the same model, float32 roundoff."""
    m, meta = model_for("si_ace_model.npz")
    ats = structures(meta["elements"])
    res = BatchCalculator(m, meta, dtype=jnp.float32, n_device=256).calculate(ats)
    check(res, reference(ACECalculator(m, meta, dtype=jnp.float32), ats), e_rel=1e-6, f_rel=1e-4, s_abs=1e-4)


def test_the_kernel_batched():
    """ffi=True: the backend's kernel (C++, or CUDA on a GPU) takes the batch's B x N rows as one block;
    equal to ACECalculator(ffi=True) to roundoff (both read the same radial table)."""
    from ace_jax.eval import ffi
    if (shutil.which("c++") if jax.default_backend() == "cpu" else ffi.cuda_compiler()) is None:
        (pytest.fail if os.environ.get("ACEJAX_REQUIRE_OPTIONAL") else pytest.skip)("no compiler for the kernel")
    m, meta = model_for("ace_cantor5_small.npz")
    ats = structures(meta["elements"])
    bc = BatchCalculator(m, meta, ffi=True)
    assert bc.ffi is not None
    check(bc.calculate(ats), reference(ACECalculator(m, meta, ffi=True), ats), f_rel=1e-10)


def test_to_atoms():
    m, meta = model_for("si_ace_model.npz")
    ats = structures(meta["elements"])[:3]
    res = BatchCalculator(m, meta).calculate(ats)
    out = res.to_atoms(ats)
    assert [a.get_potential_energy() for a in out] == list(res.energy)
    assert np.array_equal(out[2].get_forces(), res.forces[2])


def test_unknown_element_and_bad_input_refused():
    m, meta = model_for("si_ace_model.npz")
    bc = BatchCalculator(m, meta)
    with pytest.raises(ValueError, match="not in the model"):
        bc.calculate([bulk("Cu", "fcc", a=3.6)])
    at = bulk("Si", "diamond", a=5.43)
    at.positions[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        bc.calculate([at])


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_device_builder_matches_sparse_graph(dtype):
    """The all-pairs-over-images builder gives sparse_graph's pairs and edge vectors on every cell kind
    (a primitive cell below the cutoff, unwrapped positions, a slab, a cluster with no cell)."""
    rc = 5.0
    ats = structures([14, 32])
    K = 160
    for b in pack(ats, rc, [14, 32]):
        rij, idx, mask, count = jax.jit(lambda *a: neighbours(*a, rc, K, 2 * rc))(
            jnp.asarray(b.positions, dtype), jnp.asarray(b.cell, dtype), b.images, b.image_mask, b.live)
        rij, idx, mask, count = map(np.asarray, (rij, idx, mask, count))
        assert count.max() <= K
        for s, i in enumerate(b.index):
            if i < 0:
                assert not mask[s].any()
                continue
            at = ats[i]
            g = sparse_graph(at.positions, at.cell.array, at.pbc, rc)
            ref = sorted((int(a), int(c), *np.round(r, 6)) for a, c, r in
                         zip(np.asarray(g.senders), np.asarray(g.receivers), np.asarray(g.rij)))
            n, k = np.nonzero(mask[s])
            got = sorted((int(a), int(idx[s, a, c]), *np.round(rij[s, a, c].astype(float), 6))
                         for a, c in zip(n, k))
            if dtype == "float64":
                assert got == ref, i
            else:
                assert len(got) == len(ref) and [g_[:2] for g_ in got] == [r[:2] for r in ref], i
            assert (count[s, :len(at)] == np.bincount(n, minlength=len(at))).all()


@pytest.mark.parametrize("name", ["sige_nofit.npz", "ace_cantor5_small.npz"])
def test_host_builder(name):
    """builder='host' (a cell list per structure on the host, the CPU default) gives the same results."""
    m, meta = model_for(name)
    ats = structures(meta["elements"])
    res = BatchCalculator(m, meta, builder="host", n_device=256).calculate(ats)
    check(res, ref_for(name))
    bc = BatchCalculator(m, meta, builder="host", n_device=256)
    bc._K = {bucket_size(len(a)): 4 for a in ats}                    # too few slots: re-run, never truncate
    res = bc.calculate(ats)
    assert res.timing["k_retries"] > 0
    check(res, ref_for(name))


def test_matscipy_batched_builder():
    """builder='msn' (matscipy-neighbours' batched XLA FFI, libAtoms/matscipy-neighbours#11) gives the
    same results as the JAX builder.  Skipped until a matscipy-neighbours with it is installed."""
    from ace_jax.batch import _msn_batched
    if _msn_batched() is None:
        (pytest.fail if os.environ.get("ACEJAX_REQUIRE_OPTIONAL") else pytest.skip)(
            "matscipy-neighbours without the batched XLA FFI build")
    m, meta = model_for("sige_nofit.npz")
    ats = structures(meta["elements"])
    res = BatchCalculator(m, meta, builder="msn", n_device=256).calculate(ats)
    assert set(res.timing["builder"]) == {"msn"}
    check(res, ref_for("sige_nofit.npz"))


def test_pack_tail_batch_takes_the_power_of_two_it_needs():
    ats = [bulk("Si", "diamond", 5.43, cubic=True) for _ in range(11)]
    bs = pack(ats, 5.0, [14], max_batch_atoms=8 * 8)                    # B = 8 for the 8-atom bucket
    assert [b.shape[0] for b in bs] == [8, 4]                           # 3 left: 4, not 8
    assert sorted(int(i) for b in bs for i in b.index if i >= 0) == list(range(11))


def test_bucket_size_and_complete_cell():
    sizes = [bucket_size(n) for n in range(1, 2000)]
    assert all(s >= n for n, s in zip(range(1, 2000), sizes)) and sizes == sorted(sizes)
    assert [bucket_size(n) for n in range(1, 10)] == [1, 2, 4, 4, 8, 8, 8, 8, 16]
    assert bucket_size(64) == 64 and all(s % 8 == 0 for s in sizes[8:])
    assert max(s / n for n, s in zip(range(65, 2000), sizes[64:])) < 1.25 and bucket_size(128) == 128
    c = complete_cell(np.diag([3.0, 4.0, 0.0]), [True, True, False])
    assert np.allclose(c[:2], np.diag([3.0, 4.0, 0.0])[:2]) and abs(abs(c[2, 2]) - 1) < 1e-12
    assert abs(abs(np.linalg.det(complete_cell(np.zeros((3, 3)), [False] * 3))) - 1) < 1e-12
    with pytest.raises(ValueError, match="degenerate"):
        complete_cell(np.array([[1.0, 0, 0], [2.0, 0, 0], [0, 0, 1.0]]), [True, True, False])
