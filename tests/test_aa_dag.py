"""The recursive (DAG) product basis (`ace_jax.eval.aa_dag`) is the chain form's AA.

Plan construction: every monomial is reachable as parent x one A entry, auxiliary
nodes appear only where no basis monomial serves as a parent, and the "prefix" plan
reproduces `product_basis_t` bitwise ("greedy" to roundoff).  Model parity: E, forces
and virial of ACE (lean, both layouts) and PACE models with the DAG agree with the
chain form to 1e-12.
"""
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from conftest import pace_fixture
from test_lean import MODELS, _assert_same, _efv, _structure, lean_exact

from ace_jax.eval import load
from ace_jax.eval.aa_dag import PLAN_MODES, build_plan, dag_contract, dag_levels, plan_stats, to_device, with_aa_dag
from ace_jax.eval.edge_model import product_basis_t

PACE = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"


def _random_specs(seed=0, n_A=12, sizes=(6, 30, 60, 50)):
    """Per-order unique index rows, unsorted, with repeated indices allowed."""
    rng = np.random.default_rng(seed)
    out = []
    for k, n in enumerate(sizes, start=1):
        rows = {tuple(rng.integers(0, n_A, k)) for _ in range(4 * n)}
        out.append(np.asarray(sorted(rows)[:n], np.int32).reshape(-1, k))
    return tuple(out), n_A


@pytest.mark.parametrize("mode", PLAN_MODES)
def test_plan_reaches_every_monomial(mode):
    specs, n_A = _random_specs()
    levels, readout, n_AA = build_plan(specs, n_A, mode)
    assert n_AA == sum(len(s) for s in specs)
    # rebuild each node's multiset from the plan; every AA column's node must be its monomial
    canon = tuple if mode == "prefix" else lambda t: tuple(sorted(t))   # prefix: ordered rows
    nodes = [[(i,) for i in range(n_A)]]
    for p, a in levels:
        assert (np.diff(p) >= 0).all()                         # sorted parents: sorted scatter
        nodes.append([canon(nodes[-1][pi] + (ai,)) for pi, ai in zip(p.tolist(), a.tolist())])
        assert len(set(nodes[-1])) == len(nodes[-1])           # no duplicate nodes
    cols = np.concatenate([np.asarray(c) for c, _ in readout])
    assert sorted(cols.tolist()) == list(range(n_AA))
    flat = [canon(r) for s in specs for r in s.tolist()]
    for k, (c, nd) in enumerate(readout):
        for cc, n in zip(c.tolist(), nd.tolist()):
            assert nodes[k][n] == flat[cc]
    # auxiliary nodes: a node is either read out or a parent of the next level
    st = plan_stats(specs, (levels, readout, n_AA))
    for k in range(1, len(levels) + 1):
        read = set(readout[k][1].tolist())
        parents = set(levels[k][0].tolist()) if k < len(levels) else set()
        assert read | parents == set(range(len(levels[k - 1][0])))
    assert st["dag_mul"] < st["chain_mul"]


def test_greedy_has_fewer_aux_nodes_than_prefix():
    specs, n_A = _random_specs(1)
    pre, gre = (plan_stats(specs, build_plan(specs, n_A, m)) for m in ("prefix", "greedy"))
    assert gre["dag_nodes"] <= pre["dag_nodes"] and gre["aux_nodes"] <= pre["aux_nodes"]


@pytest.mark.parametrize("mode", PLAN_MODES)
def test_dag_matches_product_basis(mode):
    specs, n_A = _random_specs(2)
    At = jnp.asarray(np.random.default_rng(3).normal(size=(n_A, 7)))
    specs_j = tuple(jnp.asarray(s) for s in specs)
    AA = np.asarray(product_basis_t(At, specs_j))
    dag = to_device(build_plan(specs, n_A, mode))
    L = dag_levels(At, dag)
    got = np.zeros_like(AA)
    for k, (c, nd) in enumerate(dag[1]):
        got[np.asarray(c)] = np.asarray(L[k])[np.asarray(nd)]
    if mode == "prefix":                                   # same multiply order: bitwise
        assert np.array_equal(got, AA)
    else:
        np.testing.assert_allclose(got, AA, rtol=1e-14, atol=0)
    ct = jnp.asarray(np.random.default_rng(4).normal(size=(AA.shape[0], 3)))
    np.testing.assert_allclose(dag_contract(At, dag, ct), ct.T @ AA, rtol=1e-13, atol=1e-13)
    nz = jnp.asarray([0, 2, 1, 1, 0, 2, 2])
    np.testing.assert_allclose(dag_contract(At, dag, ct, nz), np.einsum("ai,ai->i", AA, ct[:, nz]),
                               rtol=1e-13, atol=1e-13)
    # reverse mode through the DAG = through the chain; chunked (padded columns) = unchunked
    f_ch = lambda A: jnp.sum((ct.T @ product_basis_t(A, specs_j)) ** 2)
    for chunk in (None, 3, "auto"):
        f_dag = lambda A, chunk=chunk: jnp.sum(dag_contract(A, dag, ct, chunk=chunk) ** 2)
        np.testing.assert_allclose(jax.grad(f_dag)(At), jax.grad(f_ch)(At), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dag_contract(At, dag, ct, nz, chunk=chunk),
                                   np.einsum("ai,ai->i", AA, ct[:, nz]), rtol=1e-13, atol=1e-13)


@pytest.fixture(scope="module", params=list(MODELS))
def lean_loaded(request):
    m, meta, _ = load(str(MODELS[request.param]))
    return lean_exact(m), meta, _structure(meta)


@pytest.mark.parametrize("mode", PLAN_MODES)
@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_ace_dag_is_exact(lean_loaded, layout, mode):
    m, meta, at = lean_loaded
    m1 = with_aa_dag(m, mode)
    assert m1.aa_dag is not None and (m1.blk_aa_dag is not None) == (m.blk_aa_specs is not None)
    _assert_same(_efv(m, meta, at, layout), _efv(m1, meta, at, layout))
    assert with_aa_dag(m1, None).aa_dag is None


@pytest.mark.parametrize("mode", PLAN_MODES)
@pytest.mark.parametrize("name", ["gesi_sbessel", "sige_zbl", "si_chebpow_fs"])
@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_pace_dag_is_exact(name, layout, mode):
    m, meta, _ = load(str(pace_fixture(PACE / f"{name}.yace")))
    at = _structure(meta)
    (E0, F0, V0), (E1, F1, V1) = _efv(m, meta, at, layout), _efv(with_aa_dag(m, mode), meta, at, layout)
    # relative: these fixtures carry random-weight forces up to ~2e3 eV/A
    assert abs(E0 - E1) <= 1e-12 * max(1.0, abs(E0))
    assert np.abs(F0 - F1).max() <= 1e-12 * max(1.0, np.abs(F0).max())
    assert np.abs(V0 - V1).max() <= 1e-12 * max(1.0, np.abs(V0).max())


def test_calculator_option():
    from ace_jax.calc.point import ACECalculator
    m, meta, _ = load(str(MODELS["si_fitted"]))
    at = _structure(meta)
    out = []
    for dag in (None, "greedy"):
        calc = ACECalculator(m, meta, aa_dag=dag)
        assert (calc.eval_model.blk_aa_dag is not None) == (dag is not None)
        a = at.copy()
        a.calc = calc
        out.append((a.get_potential_energy(), a.get_forces(), a.get_stress()))
    assert abs(out[0][0] - out[1][0]) <= 1e-12 * abs(out[0][0])
    assert np.abs(out[0][1] - out[1][1]).max() < 1e-12
    assert np.abs(out[0][2] - out[1][2]).max() < 1e-12


def test_auto_resolves_on_chain_multiplies(monkeypatch):
    """aa_dag="auto" is "greedy" from AUTO_MIN_CHAIN_MUL chain-form multiplies per site and the chain form
    below, for ACE and PACE models; the calculator (whose default it is) reports the mode it used."""
    from ace_jax.calc.point import ACECalculator
    from ace_jax.eval import aa_dag
    m, meta, _ = load(str(MODELS["si_fitted"]))
    n = aa_dag.chain_multiplies(m.aa_specs)
    assert n == sum(len(s) * (np.shape(s)[1] - 1) for s in m.aa_specs if np.size(s)) > 0
    p, _, _ = load(str(pace_fixture(PACE / "sige_zbl.yace")))
    for thr, want in ((n + 1, None), (n, "greedy")):
        monkeypatch.setattr(aa_dag, "AUTO_MIN_CHAIN_MUL", thr)
        assert aa_dag.resolve_aa_dag(m, "auto") == want
        assert (with_aa_dag(m, "auto").aa_dag is not None) == (want is not None)
        calc = ACECalculator(m, meta)
        assert calc.aa_dag == want and (calc.eval_model.blk_aa_dag is not None) == (want is not None)
    monkeypatch.setattr(aa_dag, "AUTO_MIN_CHAIN_MUL", 0)
    assert aa_dag.resolve_aa_dag(p, "auto") == "greedy"
    assert ACECalculator(m, meta, aa_dag=None).aa_dag is None
    assert aa_dag.resolve_aa_dag(object(), "auto") is None          # nothing to plan: the model as given
