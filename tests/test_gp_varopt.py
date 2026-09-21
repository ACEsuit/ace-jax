import jax, numpy as np, jax.numpy as jnp
from conftest import FIXTURE_DIR
jax.config.update("jax_enable_x64", True)
from ace_jax.eval import load, highest_precision
from ace_jax.fit.data import build_dataset, load_configs
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features, build_pmap
from ace_jax.fit.objective import Problem, make_lml, make_lml_embed
from ace_jax.fit.kernels import KernelSpec
from ace_jax.fit.hypers import default_prior, to_array
from ace_jax.fit.embedding import species_onehot
import equinox as eqx

def _tiny_problem(mps=4, ncfg=3):
    model, meta, z = load(FIXTURE_DIR / "sige_nofit.npz")
    els = [int(e) for e in meta["elements"]]
    cfgs = load_configs(FIXTURE_DIR / "si_tiny_train.xyz", "dft_energy", "dft_force", "dft_virial")[:ncfg]
    counts = np.array([[np.sum(c.numbers == e) for e in els] for c in cfgs], float)
    E0, *_ = np.linalg.lstsq(counts, np.array([c.energy for c in cfgs]), rcond=None)
    model = eqx.tree_at(lambda m: m.E0, model, jnp.asarray(E0))
    ds = build_dataset(cfgs, meta, E0, ncfg)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(els), C=ncfg)
    X, S = site_features(model, cfg, ds); scale = descriptor_scale(X, ds.node_mask)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, mps, scale,
                          Pmap=build_pmap(cfg, scale, None), nz=len(els))
    # sige_nofit.npz (unfitted) carries no exported smoothness weights, unlike
    # si_fitted.npz's z["gamma"]; gamma only rescales the prior's linear-Gram
    # diagonal (identically in both make_lml and make_lml_embed below), so any
    # valid (positive) per-basis vector keeps the equivalence check meaningful.
    gamma = jnp.ones(cfg.len_basis)
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg,
                   gamma, default_prior(2.35))
    return prob, ds, els

def test_make_lml_embed_matches_make_lml_at_eye():
    with highest_precision():
        prob, ds, els = _tiny_problem()
        a = to_array(prob.prior.mu)
        eye = jnp.asarray(species_onehot(len(els)))       # ind.embed default IS eye here
        L_ref = float(make_lml(prob, ds)(a))
        L_emb = float(make_lml_embed(prob, ds)(eye, a))
        # sige_nofit's untrained (random) WB weights against default_prior's tight
        # log_sigma_E/F/V give an O(1e6) LML at this tiny fixture (not O(1) as the
        # spec's plain "< 1e-6" implicitly assumed); the jitted make_lml and eager
        # make_lml_embed graphs sum in different orders, so a pure absolute
        # tolerance at that scale bites on float64 rounding, not on correctness.
        # abs+rel combined tolerance keeps the check tight (rtol below is ~40x the
        # actual ~5e-11 relative gap) while being scale-appropriate.
        assert abs(L_emb - L_ref) < 1e-6 + 2e-9 * abs(L_ref)   # same LML when E = the eye default
