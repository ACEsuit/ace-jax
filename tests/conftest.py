"""Both model families are exercised by every test, so neither branch can rot.

  si_fitted.npz     ace1_model : splined rbasis, splined pair, SPHERICAL Ylm,
                                 ACE1_PolyEnvelope1sR
  si_ace_model.npz  ace_model  : ANALYTIC rbasis (live Wnlq + 3-term recursion),
                                 splined pair, SOLID Ylm, PolyEnvelope1sR

Note the pair basis is splined in both: `ace_model` splinifies it
(ace_heuristics.jl:213), so the radial branches are per-basis, not per-model.
"""
import os
import pathlib

# Two host CPU devices so tests/test_gp_sharding.py runs in the full suite.
# Must be set before ANY jax import (the CPU backend fixes its device count at
# initialisation), so this module imports nothing from jax at top level.
_flag = "--xla_force_host_platform_device_count=2"
if _flag not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _flag).strip()

# Reuse compiled XLA executables across tests, xdist workers, and CI runs. JAX's
# persistent cache is keyed by the computation hash, so it hits even when each
# test builds a fresh jit closure for the same model eval -- the dominant cost in
# this f64-CPU suite. Set before any jax import. CI persists the dir via actions/cache.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(pathlib.Path(__file__).parent.parent / ".jax_cache"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import pytest

ROOT = pathlib.Path(__file__).parent.parent

# ACEJAX_FIXTURE_DIR points the whole suite at a different set of exports -- used
# by the divergence job in CI, which regenerates them from the Julia in the
# working tree so the reference values are fresh rather than committed.
FIXTURE_DIR = pathlib.Path(os.environ.get("ACEJAX_FIXTURE_DIR", ROOT / "fixtures"))

# Missing fixtures normally skip, so a developer without Julia can still run the
# suite.  In CI that would be a silent pass: a regeneration that produced nothing
# would look identical to a green run.  ACEJAX_REQUIRE_FIXTURES turns the skip
# into a failure.
REQUIRE = bool(os.environ.get("ACEJAX_REQUIRE_FIXTURES"))

MODELS = {
    "ace1_spline_spherical": "si_fitted.npz",
    "ace_analytic_solid": "si_ace_model.npz",
    # two species, random (unfitted) weights: exercises every per-species index
    # path -- Wnlq[:,:,iz,jz], E0[z], WB[:,z] and the folded ctilde[:,z]
    "ace1_two_species": "sige_nofit.npz",
}


def pytest_generate_tests(metafunc):
    if "npz" in metafunc.fixturenames:
        ids, paths = [], []
        for name, fname in MODELS.items():
            p = FIXTURE_DIR / fname
            ids.append(name)
            if p.exists():
                paths.append(pytest.param(p))
            elif REQUIRE:
                # Abort collection outright.  An xfail or a skip would still be a
                # green run, which is the failure mode this guard exists to stop.
                raise FileNotFoundError(
                    f"ACEJAX_REQUIRE_FIXTURES is set but {p} is missing -- the "
                    f"Julia export did not produce it")
            else:
                paths.append(pytest.param(p, marks=pytest.mark.skip(
                    reason=f"missing fixture {p.name}; see julia/export_model.jl")))
        metafunc.parametrize("npz", paths, ids=ids)


@pytest.fixture(scope="module")
def tiny_linear_problem():
    """M=0 (BLR-limit) Problem + Dataset on the si_tiny fixtures, shared by the
    hyper-routing ladder/paramset tests.  Mirrors the `blr` construction at the
    top of tests/test_gp_objective.py, returning just (prob, ds)."""
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ace_jax.eval import load
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.inducing import (GPConfig, descriptor_scale, select_inducing,
                                       site_features)
    from ace_jax.fit.kernels import KernelSpec
    from ace_jax.fit.objective import Problem

    xyz = FIXTURE_DIR / "si_tiny_train.xyz"
    design = FIXTURE_DIR / "si_tiny_design.npz"
    fitted = FIXTURE_DIR / "si_fitted.npz"
    if not (xyz.exists() and design.exists() and fitted.exists()):
        pytest.skip("missing GP fixtures")
    model, meta, z = load(fitted)
    configs = load_configs(xyz, "dft_energy", "dft_force", "dft_virial")[:6]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=3)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=3)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0,
                          descriptor_scale(X, ds.node_mask))  # M = 0
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg,
                   jnp.asarray(z["gamma"]), default_prior(2.35))
    return prob, ds


@pytest.fixture(scope="module")
def two_type_synthetic():
    """M=0 (BLR-limit) Problem + Dataset with TWO config-types built from the 6
    si_tiny configs.  Each config appears once as type 0 and once as type 1; the
    type-1 ENERGY label carries 3x the Gaussian noise of its type-0 twin (added
    element-wise, so the per-config weighted-residual ratio is exactly 3
    regardless of the structural weights).  Forces/virials -- which vastly
    outnumber the energy rows -- pin the shared linear model near the truth, so
    each type's energy residual tracks its injected noise and the evidence should
    recover an energy sigma ratio ~3.  Returns (prob, ds, n_types=2)."""
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ace_jax.eval import load
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.inducing import (GPConfig, descriptor_scale, select_inducing,
                                       site_features)
    from ace_jax.fit.kernels import KernelSpec
    from ace_jax.fit.objective import Problem

    xyz = FIXTURE_DIR / "si_tiny_train.xyz"
    fitted = FIXTURE_DIR / "si_fitted.npz"
    if not (xyz.exists() and fitted.exists()):
        pytest.skip("missing GP fixtures")
    model, meta, z = load(fitted)
    base = load_configs(xyz, "dft_energy", "dft_force", "dft_virial")[:6]
    rng = np.random.default_rng(0)
    s = 0.1
    noise = s * rng.normal(size=len(base))
    configs = []
    for c, cfg in enumerate(base):                       # type 0: 1x noise
        e = None if cfg.energy is None else float(cfg.energy + noise[c])
        configs.append(cfg._replace(energy=e, type_idx=0))
    for c, cfg in enumerate(base):                       # type 1: 3x noise (element-wise)
        e = None if cfg.energy is None else float(cfg.energy + 3.0 * noise[c])
        configs.append(cfg._replace(energy=e, type_idx=1))
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=6)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=6)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 0,
                          descriptor_scale(X, ds.node_mask))  # M = 0
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg,
                   jnp.asarray(z["gamma"]), default_prior(2.35))
    return prob, ds, 2


def species_index(z):
    """Map the exported per-atom atomic numbers onto model species indices.

    `node_z` is an index into the model's `elements` (i2z) table, NOT an atomic
    number. Tests used to hardcode `zeros(n_nodes)`, which is correct only for a
    single-species model and silently wrong for any other -- so a two-element
    export could not be tested at all.
    """
    import jax.numpy as jnp
    import numpy as np
    i2z = list(np.asarray(z["elements"]).ravel())
    lookup = {int(zz): i for i, zz in enumerate(i2z)}
    return jnp.asarray([lookup[int(a)] for a in np.asarray(z["test_Z"]).ravel()],
                       dtype=jnp.int32)


# Bound peak memory across the suite: JAX retains compiled executables and traced
# artifacts in-process, which under xdist workers accumulates and can OOM a CI
# runner mid-run. Release them after each test; the persistent on-disk cache
# above keeps the next compile cheap, so this costs little.
@pytest.fixture(autouse=True)
def _release_jax_memory():
    yield
    try:
        import jax
        jax.clear_caches()
    except Exception:
        pass
