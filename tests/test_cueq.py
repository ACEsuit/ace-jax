import jax
jax.config.update("jax_enable_x64", True)
import pytest  # noqa: E402


def test_require_names_the_extra_when_cueq_is_missing(monkeypatch):
    import builtins
    from ace_jax.eval import cueq
    real = builtins.__import__
    def fake(name, *a, **k):
        if name.startswith("cuequivariance"):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    assert not cueq.available()
    with pytest.raises(cueq.CueqUnavailable, match=r"ace-jax\[cueq\]"):
        cueq.require()


def _si():
    from conftest import FIXTURE_DIR
    from ace_jax import load
    from ace_jax.eval.model import lean
    m, meta, _ = load(str(FIXTURE_DIR / "si_fitted.npz"))
    return lean(m)


def test_refuses_multi_species_categorical_unfolded_and_pace():
    import dataclasses
    from conftest import FIXTURE_DIR, pace_fixture
    from ace_jax import load
    from ace_jax.eval import cueq
    from ace_jax.eval.model import lean
    m2, _, _ = load(str(FIXTURE_DIR / "sige_nofit.npz"))             # two species, categorical
    m2 = lean(m2)
    assert m2.ctilde.shape[1] == 2
    with pytest.raises(ValueError, match="one species"):
        cueq.check_supported(m2)
    with pytest.raises(ValueError, match="folded"):
        cueq.check_supported(dataclasses.replace(_si(), folded=False))
    with pytest.raises(ValueError, match="linear ACEModel"):
        cueq.check_supported(load(str(pace_fixture(FIXTURE_DIR / "pace" / "si_chebexpcos.yace")))[0])
    cueq.check_supported(_si())                                       # the supported case passes


def test_n_A_of():
    import numpy as np
    from ace_jax.eval import cueq
    assert cueq.n_A_of([np.zeros((0, 1), int), np.array([[0, 3], [2, 1]])]) == 4


def test_polynomial_chunks_cover_every_monomial_once():
    pytest.importorskip("cuequivariance")
    import numpy as np
    from ace_jax.eval import cueq
    m = _si()
    specs, coef = m.aa_specs, np.asarray(m.ctilde[:, 0])
    plans = cueq.polynomial(specs, coef, chunk=7)
    cols = np.concatenate([c for _, c in plans])
    assert sorted(cols.tolist()) == list(range(sum(np.shape(s)[0] for s in specs if np.size(s))))
    assert all(len(c) <= 7 for _, c in plans)


def test_auto_never_picks_cueq():
    from ace_jax.eval.aa_dag import resolve_aa_dag
    assert resolve_aa_dag(_si(), "auto") != "cueq"


def test_with_cueq_warns_once_without_an_nvrtc_cache(monkeypatch):
    import numpy as np
    from ace_jax.eval import cueq
    monkeypatch.setattr(cueq, "require", lambda: None)
    monkeypatch.setattr(cueq, "polynomial", lambda specs, coef: [(None, np.arange(1))])
    monkeypatch.setattr(cueq, "_nvrtc_warned", False)
    monkeypatch.delenv("CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR", raising=False)
    with pytest.warns(UserWarning, match="CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR"):
        m = cueq.with_cueq(_si())
    assert m.cueq_plan is not None and m.aa_dag is None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cueq.with_cueq(_si())                                         # once per process


def test_aa_dag_none_clears_the_cueq_plans(monkeypatch):
    import numpy as np
    from ace_jax.eval import cueq
    from ace_jax.eval.aa_dag import with_aa_dag
    monkeypatch.setattr(cueq, "require", lambda: None)
    monkeypatch.setattr(cueq, "polynomial", lambda specs, coef: [(None, np.arange(1))])
    monkeypatch.setenv("CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR", "/nonexistent")
    m = with_aa_dag(_si(), "cueq")
    assert m.cueq_plan is not None
    m = with_aa_dag(m, None)
    assert m.cueq_plan is None and m.blk_cueq_plan is None


def test_dag_mode_drops_the_cueq_plans(monkeypatch):
    """The folded readout reads cueq plans first: a later "greedy" / "prefix" must clear them, or the DAG
    would be silently ignored."""
    import numpy as np
    from ace_jax.eval import cueq
    from ace_jax.eval.aa_dag import with_aa_dag
    monkeypatch.setattr(cueq, "require", lambda: None)
    monkeypatch.setattr(cueq, "polynomial", lambda specs, coef: [(None, np.arange(1))])
    monkeypatch.setenv("CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR", "/nonexistent")
    for mode in ("greedy", "prefix"):
        m = with_aa_dag(with_aa_dag(_si(), "cueq"), mode)
        assert m.cueq_plan is None and m.blk_cueq_plan is None and m.aa_dag is not None


def test_refused_model_swap_leaves_the_calculator_unchanged(monkeypatch):
    """A calc.model swap that raises (here in with_cueq) must leave the calculator exactly as it was: the
    old model, its evaluation form and its caches, so a caller who catches the error still gets the old
    model's energies and calc.model reports it."""
    import dataclasses
    import numpy as np
    from ase.build import bulk
    from conftest import FIXTURE_DIR
    from ace_jax import load
    from ace_jax.calc.point import ACECalculator
    from ace_jax.eval import cueq
    from ace_jax.eval.model import fold_readout
    m, meta, _ = load(str(FIXTURE_DIR / "si_fitted.npz"))
    with monkeypatch.context() as mp:             # a cueq calculator, its plan stubbed to the chain form
        mp.setattr(jax, "default_backend", lambda: "gpu")
        mp.setattr(cueq, "with_cueq", lambda model: model)
        calc = ACECalculator(m, meta, aa_dag="cueq")
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat(2)
    at.rattle(0.05, seed=2)
    at.calc = calc
    e_a = at.get_potential_energy()
    eval_a = calc.eval_model
    m2 = fold_readout(dataclasses.replace(m, WB=2.0 * m.WB, folded=False, ctilde=None))
    def refuse(model):
        raise ValueError("refused")
    for gpu in ((True,) if jax.default_backend() == "gpu" else (False, True)):   # at the backend check, in with_cueq
        with monkeypatch.context() as mp:
            if gpu:
                mp.setattr(jax, "default_backend", lambda: "gpu")
            mp.setattr(cueq, "with_cueq", refuse)
            with pytest.raises(ValueError, match="CUDA GPU" if not gpu else "refused"):
                calc.model = m2
        assert calc.model is m and calc.eval_model is eval_a and calc.aa_dag == "cueq"
        at2 = at.copy()
        at2.calc = calc
        np.testing.assert_allclose(at2.get_potential_energy(), e_a, rtol=1e-12)


def _gpu():
    if jax.default_backend() != "gpu":            # first: without its CUDA libraries the cueq import is an OSError
        pytest.skip("cueq evaluation needs a CUDA device")
    pytest.importorskip("cuequivariance_jax")


def _cell(n_atoms):
    from ase.build import bulk
    if n_atoms == 1:
        return bulk("Si", "fcc", a=3.9)                               # one-atom periodic cell
    return bulk("Si", "diamond", a=5.43, cubic=True).repeat(3)[:n_atoms]   # 216, or a periodic slice of it


@pytest.mark.parametrize("n_atoms", [1, 31, 33, 216])
def test_cueq_matches_the_full_model(n_atoms):
    _gpu()
    import numpy as np
    from conftest import FIXTURE_DIR
    from ace_jax import load
    from ace_jax.calc.point import ACECalculator
    full, meta, _ = load(str(FIXTURE_DIR / "si_fitted.npz"))
    at = _cell(n_atoms)
    at.rattle(0.05, seed=1)
    ref = ACECalculator(full, meta, lean=False, aa_dag=None)
    cq = ACECalculator(full, meta, aa_dag="cueq")
    assert cq.aa_dag == "cueq"
    at.calc = ref; e0, f0 = at.get_potential_energy(), at.get_forces()
    at.calc = cq; e1, f1 = at.get_potential_energy(), at.get_forces()
    assert abs(e1 - e0) <= 1e-12 * max(abs(e0), 1)
    assert np.abs(f1 - f0).max() <= 1e-10 * max(np.abs(f0).max(), 1)


@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_cueq_matches_the_chain_form_per_layout(layout):
    _gpu()
    import numpy as np
    from conftest import FIXTURE_DIR
    from ace_jax import load
    from ace_jax.calc.point import ACECalculator
    full, meta, _ = load(str(FIXTURE_DIR / "si_fitted.npz"))
    at = _cell(33)
    at.rattle(0.05, seed=2)
    out = []
    for mode in (None, "cueq"):
        at.calc = ACECalculator(full, meta, layout=layout, aa_dag=mode)
        out.append((at.get_potential_energy(), at.get_forces(), at.get_stress()))
    (e0, f0, s0), (e1, f1, s1) = out
    assert abs(e1 - e0) <= 1e-12 * max(abs(e0), 1)
    assert np.abs(f1 - f0).max() <= 1e-10 * max(np.abs(f0).max(), 1)
    assert np.abs(s1 - s0).max() <= 1e-10 * max(np.abs(s0).max(), 1)


def test_cueq_uses_the_blocked_dense_plan_too():
    _gpu()
    from ace_jax.eval import cueq
    m = cueq.with_cueq(_si())
    assert m.cueq_plan is not None and (m.blk_aa_specs is None or m.blk_cueq_plan is not None)
    assert m.aa_dag is None and m.blk_aa_dag is None


def test_product_energy_sums_the_chunks():
    _gpu()
    import numpy as np
    import jax.numpy as jnp
    from ace_jax.eval import cueq
    from ace_jax.eval.edge_model import product_basis_dot
    m = _si()
    specs, coef = m.blk_aa_specs, np.asarray(m.ctilde[:, 0])
    A = jnp.asarray(np.random.default_rng(0).normal(size=(45, cueq.n_A_of(specs))))   # 45: not a multiple of U
    ref = product_basis_dot(A, specs, m.ctilde[:, np.zeros(45, int)])
    plans = cueq.polynomial(specs, coef, chunk=50)
    assert len(plans) > 1
    np.testing.assert_allclose(cueq.product_energy(plans, A), ref, rtol=1e-12, atol=1e-12)


def test_calculator_refuses_cueq_without_a_cuda_backend():
    if jax.default_backend() == "gpu":
        pytest.skip("this test needs a non-GPU backend")
    from conftest import FIXTURE_DIR
    from ace_jax import load
    from ace_jax.calc.point import ACECalculator
    full, meta, _ = load(str(FIXTURE_DIR / "si_fitted.npz"))
    with pytest.raises(ValueError, match=r"aa_dag='cueq'.*CUDA GPU"):
        ACECalculator(full, meta, aa_dag="cueq")


def test_readout_swap_replans():
    _gpu()
    import dataclasses
    from ase.build import bulk
    from conftest import FIXTURE_DIR
    from ace_jax import load
    from ace_jax.calc.point import ACECalculator
    m, meta, _ = load(str(FIXTURE_DIR / "si_fitted.npz"))
    calc = ACECalculator(m, meta, aa_dag="cueq")
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat(2)
    at.rattle(0.05, seed=2)
    at.calc = calc
    e_a = at.get_potential_energy()
    from ace_jax.eval.model import fold_readout
    m2 = fold_readout(dataclasses.replace(m, WB=2.0 * m.WB, folded=False, ctilde=None))   # a new readout, same basis
    calc.model = m2
    at.calc = calc
    e_b = at.get_potential_energy()
    at.calc = ACECalculator(m2, meta, lean=False, aa_dag=None)
    assert calc.aa_dag == "cueq" and abs(e_b - at.get_potential_energy()) <= 1e-12 * abs(e_b) and e_b != e_a


def test_cueq_float32_matches_the_float32_reference():
    _gpu()
    import numpy as np
    import jax.numpy as jnp
    from conftest import FIXTURE_DIR
    from ace_jax import load
    from ace_jax.calc.point import ACECalculator
    full, meta, _ = load(str(FIXTURE_DIR / "si_fitted.npz"))
    at = _cell(33)
    at.rattle(0.05, seed=3)
    out = []
    for mode in (None, "cueq"):                                   # bounds as test_skin's float32 test
        at.calc = ACECalculator(full, meta, aa_dag=mode, dtype=jnp.float32)
        out.append((at.get_potential_energy(), at.get_forces()))
    (e0, f0), (e1, f1) = out
    assert abs(e1 - e0) < 1e-5 * abs(e0)
    assert np.abs(f1 - f0).max() <= 1e-4


_EXPORT = """
import sys, jax
jax.config.update("jax_enable_x64", True)
from ace_jax import load
from ace_jax.export.lammps import export_lammps
m, meta, _ = load(sys.argv[1])
b = export_lammps(m, meta, sys.argv[2], max_atoms=256, max_edges=256 * 32, k_dense=32, max_owned=200,
                  layout="dense", aa_dag="cueq")
print(b["ace_jax"]["block_rows"])
"""


def test_cueq_bundle_ships_its_replay_and_nvrtc_cache(tmp_path):
    """A cueq bundle on the GPU: the export (in a fresh process: plan handles are registry indices) writes
    <bundle>.ffi-replay and fills <bundle>.cueq-cache, its own NVRTC cache (Review Focus 3)."""
    _gpu()
    pytest.importorskip("lammps_jax")
    import json
    import os
    import subprocess
    import sys
    from conftest import FIXTURE_DIR
    from ace_jax.eval import cueq
    try:
        cueq.recorder()
    except cueq.CueqUnavailable as e:
        pytest.skip(str(e))
    path = tmp_path / "si.json"
    env = {k: v for k, v in os.environ.items() if k != "CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR"}
    p = subprocess.run([sys.executable, "-c", _EXPORT, str(FIXTURE_DIR / "si_fitted.npz"), str(path)],
                       capture_output=True, text=True, env={**env, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
    assert p.returncode == 0, p.stderr[-3000:]
    rec = json.loads(path.read_text())["ace_jax"]
    assert rec["aa_dag"] == "cueq" and int(p.stdout.split()[-1]) == 200          # capped at the rows
    assert rec["cueq"]["replay"] == "si.ffi-replay" and (tmp_path / "si.ffi-replay").is_file()
    assert (tmp_path / "si.ffi-replay").read_bytes()[:8] == b"FFIRPLY1"
    assert rec["cueq"]["cache"] == "si.cueq-cache" and any((tmp_path / "si.cueq-cache").iterdir())
    env = cueq.run_env(rec["cueq"], path)
    assert env["FFI_REPLAY_FILE"] == str(tmp_path / "si.ffi-replay")
    assert env["CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR"] == str(tmp_path / "si.cueq-cache")


def test_run_env_resolves_the_sidecars_next_to_a_moved_bundle(monkeypatch):
    from ace_jax.eval import cueq
    rec = {"handlers": "uniform_1d_cuda=/x/ffi_replay_shim.so:ffi_replay_handler", "replay": "si.ffi-replay",
           "cache": "si.cueq-cache"}
    monkeypatch.delenv("LAMMPS_JAX_CONTRIB", raising=False)
    env = cueq.run_env(rec, "/moved/here/si.json")
    assert env["LAMMPS_JAX_FFI_HANDLERS"] == rec["handlers"]                  # the export host's shim
    assert env["FFI_REPLAY_FILE"] == "/moved/here/si.ffi-replay"
    assert env["CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR"] == "/moved/here/si.cueq-cache"
    assert env["FFI_REPLAY_LIB"].endswith("libcue_ops_jax.so") and "LD_LIBRARY_PATH" in env
    monkeypatch.setenv("LAMMPS_JAX_CONTRIB", "/here/contrib/ffi-replay")       # this host's shim
    assert cueq.run_env(rec, "/moved/here/si.json")["LAMMPS_JAX_FFI_HANDLERS"] == \
        "uniform_1d_cuda=/here/contrib/ffi-replay/ffi_replay_shim.so:ffi_replay_handler"


# ------------------------------------------------------------------ TrACE (species-embedding) models
TRACE = {"SiGe": ([14, 32], 4, 8, None),            # elements, order, degree, d_max: MH-1 rows, fixtures/
         "Cantor": ([24, 25, 26, 27, 28], 3, 6, 4)}  # embedding_ref_mh1_*.npz (as the spike's trace_* models)


_trace_cache = {}


def _trace(name):
    """A small embedding (TrACE) model with a random readout: (full model, meta)."""
    if name not in _trace_cache:
        import dataclasses
        import numpy as np
        import jax.numpy as jnp
        from conftest import FIXTURE_DIR
        from ace_jax.basis.model import build_embedding_model
        els, order, deg, d_max = TRACE[name]
        tab = "SiGe" if len(els) == 2 else "CrMnFeCoNi"
        t = np.load(FIXTURE_DIR / f"embedding_ref_mh1_{tab}.npz")
        m, meta = build_embedding_model(els, order, deg, embedding=([int(z) for z in t["zlist"]], t["table"]),
                                        d_max=d_max).eval_pair()
        rng = np.random.default_rng(7)
        m = dataclasses.replace(m, WB=jnp.asarray(rng.normal(size=m.WB.shape)),
                                Wpair=jnp.asarray(0.1 * rng.normal(size=m.Wpair.shape)))
        _trace_cache[name] = m, meta
    return _trace_cache[name]


def _alloy(meta, n_rep, seed):
    import numpy as np
    from ase.build import bulk
    els = [int(z) for z in meta["elements"]]
    at = (bulk("Ni", "fcc", a=3.6, cubic=True) if len(els) > 2
          else bulk("Si", "diamond", a=5.43, cubic=True)).repeat(n_rep)
    rng = np.random.default_rng(seed)
    at.numbers = np.asarray(els)[np.arange(len(at)) % len(els)]
    rng.shuffle(at.numbers)
    at.rattle(0.05, seed=seed)
    return at


@pytest.mark.parametrize("name", list(TRACE))
def test_check_supported_accepts_embedding_models(name):
    from ace_jax.eval import cueq
    from ace_jax.eval.model import lean
    m, _ = _trace(name)
    cueq.check_supported(m, folded=False)                             # as given (`check_export`)
    L = lean(m)
    assert L.ctilde.shape[1] == len(TRACE[name][0]) > 1
    cueq.check_supported(L)


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("name", list(TRACE))
def test_trace_plan_reproduces_the_folded_product_basis(name, blocked):
    """The channel-free paths and weights W[path, k, z] give the folded product basis exactly (numpy, no cueq):
    e_i = sum_p sum_k W[p, k, z_i] prod_t A'_i[s_pt, k], A' gathered from A."""
    import numpy as np
    from ace_jax.eval import cueq
    from ace_jax.eval.edge_model import product_basis_dot
    from ace_jax.eval.model import lean
    m, _ = _trace(name)
    L = lean(m)
    assert bool(L.blk) and not L.blk_compact
    specs = L.blk_aa_specs if blocked else L.aa_specs
    gather, paths, W, st = cueq.trace_plan(L, blocked)
    n_A, nz = cueq.n_A_of(specs), L.ctilde.shape[1]
    assert st["n_paths"] == len(paths) < st["n_AA"] == sum(np.shape(s)[0] for s in specs)
    rng = np.random.default_rng(1)
    A = rng.normal(size=(23, n_A))
    z = rng.integers(0, nz, 23)
    Ap = np.concatenate([A, np.zeros((23, 1))], 1)[:, np.where(gather < 0, n_A, gather)]    # (n, n_seg, d)
    e = sum((W[j][:, z].T * np.prod(Ap[:, list(p), :], axis=1)).sum(-1) for j, p in enumerate(paths))
    ref = np.asarray(product_basis_dot(A, specs, L.ctilde[:, z]))
    np.testing.assert_allclose(e, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def _site_and_efv(model, meta, at, layout):
    import jax.numpy as jnp
    import numpy as np
    from ace_jax.eval.model import highest_precision
    from ace_jax.eval.nlist import dense_from_sparse, sparse_graph
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    with highest_precision():
        if layout == "sparse":
            s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
            args = (jnp.asarray(g.rij), nz[s], nz[r])
            site = model.site_energies(*args, s, len(at), nz)
            E, F, V = model.energy_forces_virial(*args, s, r, len(at), nz)
        else:
            d = dense_from_sparse(g, meta["rcut"])
            idx = jnp.asarray(d.idx)
            args = (jnp.asarray(d.rij), jnp.broadcast_to(nz[:, None], idx.shape), nz[idx])
            site = model.site_energies_dense(*args, jnp.asarray(d.mask), nz)
            E, F, V = model.energy_forces_virial_dense(*args, idx, jnp.asarray(d.mask), nz)
    return tuple(np.asarray(x) for x in (site, F, V))


@pytest.mark.parametrize("layout", ["sparse", "dense"])
@pytest.mark.parametrize("name", list(TRACE))
def test_cueq_trace_matches_the_full_model(name, layout):
    """aa_dag="cueq" on an embedding model (GPU, float64): site energies <= 1e-12, forces and virial <= 1e-10
    relative to the full model.  sparse reads the aspec-layout plan, dense the l-blocked one."""
    _gpu()
    import numpy as np
    from ace_jax.eval import cueq
    from ace_jax.eval.model import lean
    m, meta = _trace(name)
    at = _alloy(meta, (3, 2, 1), seed=4)                             # 48 / 24 atoms: padded U-atom groups
    cq = cueq.with_cueq(lean(m, spline_tol=None))
    assert cq.cueq_plan is not None and cq.blk_cueq_plan is not None
    e0, f0, v0 = _site_and_efv(m, meta, at, layout)
    e1, f1, v1 = _site_and_efv(cq, meta, at, layout)
    assert np.abs(e1 - e0).max() <= 1e-12 * max(np.abs(e0).max(), 1)
    assert np.abs(f1 - f0).max() <= 1e-10 * max(np.abs(f0).max(), 1)
    assert np.abs(v1 - v0).max() <= 1e-10 * max(np.abs(v0).max(), 1)


def test_cueq_trace_calculator_matches_the_full_model():
    _gpu()
    import numpy as np
    from ace_jax.calc.point import ACECalculator
    m, meta = _trace("SiGe")
    at = _alloy(meta, 2, seed=5)
    out = []
    for kw in ({"lean": False, "aa_dag": None}, {"aa_dag": "cueq"}):
        at.calc = ACECalculator(m, meta, **kw)
        out.append((at.get_potential_energy(), at.get_forces()))
    assert at.calc.aa_dag == "cueq"
    (e0, f0), (e1, f1) = out
    assert abs(e1 - e0) <= 1e-12 * max(abs(e0), 1)
    assert np.abs(f1 - f0).max() <= 1e-10 * max(np.abs(f0).max(), 1)


@pytest.mark.parametrize("name", list(TRACE))
def test_trace_weights_count_in_the_cueq_block_rows(name, monkeypatch):
    """A TrACE plan's per-atom weight operand (rows x paths x d, forward copy and its laid-out residual) counts
    in the bundle's cueq row blocks: sized on the model as given (before lean), equal to the plan's own counts."""
    from ace_jax.eval import cueq
    from ace_jax.eval.model import lean
    from ace_jax.export import lammps as lx
    m, _ = _trace(name)
    st = cueq.trace_plan(lean(m), True)[3]
    assert cueq.trace_size(m) == (st["n_paths"], st["d"])
    w = cueq.trace_weight_bytes_per_row(m, 8)
    assert w == 2 * st["n_paths"] * st["d"] * 8 and cueq.trace_weight_bytes_per_row(_si(), 8) == 0
    monkeypatch.setattr(lx, "estimate_a_bytes", lambda *a: 0)                  # the weights alone
    assert lx.cueq_block_rows_for(m, 100000, 8, 1.01 * 8192 * w, 8) == 8192
    assert lx.cueq_block_rows_for(m, 100000, 8, 0.99 * 8192 * w, 8) == 4096
    assert lx.cueq_block_rows_for(m, 100000, 8, 0.99 * 8192 * w, 8, prep=0.5 * 8192 * w) == 2048
