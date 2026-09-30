"""SBessel recurrence: exactness against the pre-change direct evaluation, and
that the transcendental (sin/cos) count is O(1) in K, not O(K)."""
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
    for K in (0, 1, 4, 8, 12):
        np.testing.assert_allclose(pr._sbessel(r, 5.0, K), _direct(r, 5.0, K), rtol=1e-12, atol=1e-13)
    hlo = jax.jit(lambda x: pr._sbessel(x, 5.0, 12)).lower(r).as_text()
    # StableHLO op names, not bare "sine"/"cosine": "cosine" contains "sine" as a
    # substring, so the bare form double-counts stablehlo.cosine as a sine too.
    assert hlo.count("stablehlo.sine") <= 1 and hlo.count("stablehlo.cosine") <= 1  # not K+1 of each


# ------------------------------------------------------------------ matrix form
# g = (S @ M) / rc^1.5 with S[:, k] = sinc((k+1) x): one sin per entry and a
# constant matrix instead of the stacked recurrence, which XLA fuses into one
# kernel that re-evaluates the chain per column (~K^2 per edge).  Faster only
# at large nradbase (docs/ace-vs-pace-gap.md 4.3), so load_yace picks it per
# model: nradbase >= SBESSEL_MATMUL_MIN_K.
import dataclasses  # noqa: E402
import pathlib  # noqa: E402

import pytest  # noqa: E402

PACE_FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "pace"


def test_sbessel_matrix_form_is_exact():
    r = jnp.linspace(0.0, 4.9, 997)                     # includes r = 0 (sinc limit)
    for K in (0, 1, 4, 8, 12, 13, 15, 20):
        np.testing.assert_allclose(pr._sbessel_mm(r, 5.0, K), pr._sbessel(r, 5.0, K),
                                   rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("inner", ["density", "distance", "zbl"])
def test_radbase_matrix_form_matches(inner):
    r = jnp.linspace(0.05, 5.2, 501)
    a = pr.radbase(r, "SBessel", inner, 2.5, 5.0, 0.3, 1.2, 0.4, 13)
    b = pr.radbase(r, "SBessel", inner, 2.5, 5.0, 0.3, 1.2, 0.4, 13, sbessel_form="matmul")
    np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-13)
    ga = jax.grad(lambda x: jnp.sum(pr.radbase(x, "SBessel", inner, 2.5, 5.0, 0.3, 1.2, 0.4, 13)))(r)
    gb = jax.grad(lambda x: jnp.sum(pr.radbase(x, "SBessel", inner, 2.5, 5.0, 0.3, 1.2, 0.4, 13,
                                               sbessel_form="matmul")))(r)
    assert np.isfinite(np.asarray(gb)).all()
    np.testing.assert_allclose(gb, ga, rtol=1e-10, atol=1e-11)


# sige_sbessel13 is the benchmark's SiGe medium .yace (nradbase 13); gesi_sbessel
# has a small radial basis
@pytest.mark.parametrize("name,form", [("sige_sbessel13", "matmul"), ("gesi_sbessel", "rotation")])
def test_load_picks_the_sbessel_form(name, form):
    from ace_jax.eval import load
    from ace_jax.eval.pace_model import SBESSEL_MATMUL_MIN_K
    m, _, _ = load(str(PACE_FIX / f"{name}.yace"))
    assert m.sbessel_form == form
    assert (m.nradbase >= SBESSEL_MATMUL_MIN_K) == (form == "matmul")


@pytest.mark.parametrize("layout", ["sparse", "dense"])
def test_matrix_form_model_is_exact(layout):
    from ase.build import bulk

    from ace_jax.eval import load
    from ace_jax.eval.model import highest_precision
    from ace_jax.eval.nlist import dense_from_sparse, sparse_graph
    m, meta, _ = load(str(PACE_FIX / "sige_sbessel13.yace"))
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 1, 1))
    at.numbers[::3] = 32
    at.rattle(0.08, seed=1)
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    nz = jnp.asarray([z2i[int(z)] for z in at.numbers], jnp.int32)
    g = sparse_graph(at.positions, at.cell.array, at.pbc, meta["rcut"])
    out = []
    for mm in (m, dataclasses.replace(m, sbessel_form="rotation")):
        with highest_precision():
            if layout == "sparse":
                s, r = jnp.asarray(g.senders), jnp.asarray(g.receivers)
                res = mm.energy_forces_virial(jnp.asarray(g.rij), nz[s], nz[r], s, r, len(at), nz)
            else:
                d = dense_from_sparse(g, meta["rcut"])
                idx = jnp.asarray(d.idx)
                res = mm.energy_forces_virial_dense(jnp.asarray(d.rij),
                                                    jnp.broadcast_to(nz[:, None], idx.shape),
                                                    nz[idx], idx, jnp.asarray(d.mask), nz)
        out.append([np.asarray(x) for x in res])
    (E1, F1, V1), (E0, F0, V0) = out
    assert abs(E1 - E0) <= 1e-12 * max(1.0, abs(E0))
    assert np.abs(F1 - F0).max() < 1e-12 and np.abs(V1 - V0).max() < 1e-12
