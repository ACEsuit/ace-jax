"""Shared problems for the radial-learning tests (test_gp_learn_radial, test_fit_radial): a linear
(M = 0) problem on si_tiny with the analytic Si model, and noiseless labels from given radials."""
import jax
import numpy as np

from conftest import FIXTURE_DIR

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from ace_jax.eval import load  # noqa: E402
from ace_jax.fit.data import build_dataset, load_configs  # noqa: E402
from ace_jax.fit.hypers import Hypers, default_prior  # noqa: E402
from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features  # noqa: E402
from ace_jax.fit.kernels import KernelSpec  # noqa: E402
from ace_jax.fit.objective import Problem  # noqa: E402

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
MODEL = FIXTURE_DIR / "si_ace_model.npz"
HAVE_FIXTURES = XYZ.exists() and MODEL.exists()

THETA = Hypers(log_ell=0.0, log_A=0.0, log_alpha=0.0, log_r0=np.log(2.35), log_eps=0.0,
               log_rho=0.0, log_sigma_c=np.log(0.3), log_sigma_E=np.log(0.01),
               log_sigma_F=np.log(0.01), log_sigma_V=np.log(0.01))


def make_problem(ncfg=6, per_batch=3, start=0, force_key="dft_force"):
    """Linear (M = 0) problem on si_tiny with the analytic Si model; gamma = 1."""
    model, meta, z = load(MODEL)
    configs = load_configs(XYZ, "dft_energy", force_key, "dft_virial")[start:start + ncfg]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=per_batch)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=per_batch)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0, descriptor_scale(X, ds.node_mask))
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg,
                   jnp.ones(cfg.len_basis), default_prior(2.35))
    return prob, ds, meta


def relabel(prob, ds, W, c):
    """ds with targets replaced by the noiseless linear ACE with radials W and readout c."""
    from ace_jax.fit.radial_model import with_radial
    from ace_jax.fit.rows import linear_rows
    m = with_radial(prob.model, W)
    yE, yF, yV = [], [], []
    for i in range(ds.n_batches):
        r, _, _ = linear_rows(m, prob.cfg, jax.tree.map(lambda a: a[i], ds))
        yE.append(r.E @ c); yF.append(r.F @ c); yV.append(r.V @ c)
    return ds._replace(y_E=jnp.stack(yE), y_F=jnp.stack(yF), y_V=jnp.stack(yV))


def perturbed_truth(prob, seed=0, eps=0.2):
    rng = np.random.default_rng(seed)
    Wt = prob.model.rnl_Wnlq
    W0 = Wt + eps * jnp.abs(Wt).mean() * jnp.asarray(rng.standard_normal(Wt.shape))
    c = jnp.asarray(0.1 * rng.standard_normal(prob.cfg.len_basis))
    return Wt, W0, c
