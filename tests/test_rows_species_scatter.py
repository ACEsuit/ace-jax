"""The training rows' species scatter (rows._by_species): one segment_sum keyed on (row, centre species)
replaced NZ masked passes, each adding into an L-wide array.  On a 5-element basis (Cantor, tutorial 3's
data) the rows must equal the NZ-pass form term for term -- unchunked bit for bit, node-chunked to roundoff."""
import jax

jax.config.update("jax_enable_x64", True)
import pathlib  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parents[1] / "docs" / "user" / "tutorials" / "data" / "cantor"


@pytest.fixture(scope="module")
def cantor(tmp_path_factory):
    from ase.io import read, write

    from ace_jax.basis.model import BasisSpec, build_basis
    from ace_jax.fit.pipeline import FitConfig, load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    tr = tmp_path_factory.mktemp("cantor") / "train.xyz"
    write(tr, read(DATA / "cantor_train.xyz", ":12"))
    b = build_basis(BasisSpec(order=2, max_degree=4, elements=("Cr", "Mn", "Fe", "Co", "Ni"), rcut=5.5))
    cfg = FitConfig(model=b, arm="linear", m_per_species=0, e0="lsq", r0=None, rungs=("map",), noise="shared",
                    energy_key="mace_energy", force_key="mace_force", virial_key="mace_virial").validate()
    d = load_fit_data(cfg, train=str(tr), log=lambda *a: None)
    p = build_problem(cfg, d)
    return d.model, p.gpcfg, d.ds_train


def _rows_reference(model, cfg, batch):
    """linear_rows as it was: one masked pass per centre species, each added into its block by _place."""
    from ace_jax.fit.rows import Rows, _e0_columns, _place, _voigt, flat_edges
    Ncap, K = batch.nbr.shape
    C = batch.y_E.shape[0]
    rij, send, recv, mask = flat_edges(batch.rij, batch.nbr, batch.nbr_mask)
    zi = batch.node_z[send]
    X, J = model.edge_jacobian_dense(batch.rij, jnp.broadcast_to(batch.node_z[:, None], (Ncap, K)),
                                     batch.node_z[batch.nbr], batch.nbr_mask)
    seg = lambda a, ids, n: jax.ops.segment_sum(a, ids, num_segments=n)      # noqa: E731
    L = cfg.len_basis
    E, F, V = jnp.zeros((Ncap, L)), jnp.zeros((Ncap, 3, L)), jnp.zeros((C + 1, 6, L))
    edge_cfg = batch.node_cfg[send]
    for z in range(cfg.NZ):
        E = _place(E, jnp.where((batch.node_z == z)[:, None], X, 0.0), z, cfg)
        Jz = jnp.where((zi == z)[:, None, None], J, 0.0)
        F = _place(F, -jnp.swapaxes(seg(Jz, recv, Ncap) - seg(Jz, send, Ncap), 1, 2), z, cfg)
        V = _place(V, jnp.swapaxes(seg(_voigt(Jz, rij), edge_cfg, C + 1), 1, 2), z, cfg)
    E = _e0_columns(seg(E, batch.node_cfg, C + 1)[:C], batch.node_z, batch.node_mask, batch.node_cfg, cfg)
    return Rows(E, F, V[:C])


def test_species_scatter_matches_masked_passes(cantor):
    from ace_jax.fit.rows import linear_rows
    model, g, ds = cantor
    assert g.NZ == 5
    for i in range(min(2, ds.n_batches)):
        bt = jax.tree.map(lambda a: a[i], ds)
        new, ref = jax.jit(lambda b: linear_rows(model, g, b)[0])(bt), jax.jit(lambda b: _rows_reference(model, g, b))(bt)
        for q in "EFV":
            a, r = np.asarray(getattr(new, q)), np.asarray(getattr(ref, q))
            assert a.dtype == r.dtype and a.shape == r.shape, q
            assert np.abs(r).max() > 0, q
            np.testing.assert_array_equal(a, r, err_msg=q)


def test_species_scatter_node_chunked(cantor, monkeypatch):
    from ace_jax.fit import rows
    model, g, ds = cantor
    bt = jax.tree.map(lambda a: a[0], ds)
    Ncap, K = bt.nbr.shape
    monkeypatch.setattr(rows, "ROWS_EDGE_BUDGET", max(1, Ncap // 4) * rows._node_elems(model, g, K))
    assert rows.rows_node_chunk(model, g, Ncap, K) < Ncap
    new = jax.jit(lambda b: rows.linear_rows_bounded(model, g, b))(bt)
    ref = jax.jit(lambda b: _rows_reference(model, g, b))(bt)
    for q in "EFV":
        a, r = np.asarray(getattr(new, q)), np.asarray(getattr(ref, q))
        np.testing.assert_allclose(a, r, rtol=1e-12, atol=1e-12 * np.abs(r).max(), err_msg=q)
