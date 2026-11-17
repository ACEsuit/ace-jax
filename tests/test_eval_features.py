"""The one-particle basis of a linear ACE without its model (`eval.radial.TensorRadial`,
`eval.features.pooled_a`) is the model's own, bit for bit, for every radial kind in the fixtures; and
the `SiteFeatures` protocol is what a provider has to look like."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ace_jax.eval import load  # noqa: E402
from ace_jax.eval.features import SiteFeatures, pooled_a  # noqa: E402
from ace_jax.eval.product import pool_sparse, product_basis  # noqa: E402
from ace_jax.eval.radial import TensorRadial  # noqa: E402
from conftest import FIXTURE_DIR  # noqa: E402

MODELS = ["si_ace_model.npz", "sige_nofit.npz", "ace_cantor5_small.npz", "si_fitted.npz"]


def _edges(model, n_nodes=6, n_edges=40, seed=0):
    r = np.random.default_rng(seed)
    nz = model.E0.shape[0]
    rc = float(model.pad_cutoff())
    d = r.normal(size=(n_edges, 3))
    rij = d / np.linalg.norm(d, axis=1, keepdims=True) * r.uniform(0.45 * rc, 1.05 * rc, (n_edges, 1))
    seg = np.sort(r.integers(0, n_nodes, n_edges))
    node_z = r.integers(0, nz, n_nodes)
    return jnp.asarray(rij), jnp.asarray(node_z[seg]), jnp.asarray(r.integers(0, nz, n_edges)), jnp.asarray(seg), n_nodes


@pytest.mark.parametrize("name", MODELS)
def test_model_free_one_particle_basis_is_the_models(name):
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(f"no fixture {name}")
    check_one_particle(load(path, fold=False)[0])


def check_one_particle(model):
    """The model-free radial, pooled A and product basis of `model` equal its own (also called by the
    tests that build the element-embedding and spherical-Bessel models)."""
    rij, zi, zj, seg, n = _edges(model)
    rad = TensorRadial.of(model)
    assert rad.radial_kind == model.radial_kind and rad.n_rnl == model.edge_a_widths()[0]
    np.testing.assert_array_equal(rad(jnp.linalg.norm(rij, axis=-1), zi, zj), model.radial(rij, zi, zj)[0])
    A = pooled_a(rad, rij, zi, zj, seg, n, model.aspec_r, model.aspec_y, model.lmax, model.ysolid)
    ref = pool_sparse(model.edge_features(rij, zi, zj)[0], seg, n)
    np.testing.assert_array_equal(A, ref)
    np.testing.assert_array_equal(product_basis(A, model.aa_specs), model._aa(ref))
    # under jit, with the radial a pytree argument (what a provider that is an eqx.Module does)
    f = jax.jit(lambda rad, rij: pooled_a(rad, rij, zi, zj, seg, n, model.aspec_r, model.aspec_y, model.lmax, model.ysolid))
    np.testing.assert_allclose(f(rad, rij), ref, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(float(rad.rcut()), float(model.pad_cutoff()))


def test_site_features_protocol():
    class Provider:
        lps, rcut = ((0, 1), (1, -1)), 5.0

        def n(self, L, p):
            return 3 if (L, p) in self.lps else 0

        def species_index(self, numbers):
            return np.zeros(len(numbers), np.int32)

        def features(self, rij, zi, zj, segment_ids, n_nodes, mask=None):
            return {lp: jnp.zeros((n_nodes, 3, 2 * lp[0] + 1)) for lp in self.lps}

    assert isinstance(Provider(), SiteFeatures)
    assert not isinstance(object(), SiteFeatures)
