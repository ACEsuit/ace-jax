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


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    """A bare `pytest` (the whole suite) runs on parallel workers when pytest-xdist
    is installed: 6 by default (ACEJAX_TEST_WORKERS overrides), the measured
    optimum on a 12-core laptop -- each JAX worker is itself multi-threaded, so
    more workers contend (8: 72 s, 12: 75 s vs 6: 68 s).  Targeted runs
    (`pytest tests/test_x.py`), an explicit -n, --pdb, or no xdist stay serial.
    Runs before xdist's own tryfirst hook (conftest hooks register later).

    Never in a worker: workers re-run this hook with numprocesses reset to None, and
    setting it there makes every worker spawn its own workers, recursively."""
    if (os.environ.get("PYTEST_XDIST_WORKER") or hasattr(config, "workerinput")
            or not config.pluginmanager.hasplugin("xdist") or config.option.numprocesses is not None
            or config.getoption("usepdb", False) or config.args != config.getini("testpaths")):
        return
    config.option.numprocesses = int(os.environ.get("ACEJAX_TEST_WORKERS", min(6, os.cpu_count() or 1)))

# ACEJAX_FIXTURE_DIR points the whole suite at a different set of exports -- used
# by the divergence job in CI, which regenerates them from the Julia in the
# working tree so the reference values are fresh rather than committed.
FIXTURE_DIR = pathlib.Path(os.environ.get("ACEJAX_FIXTURE_DIR", ROOT / "fixtures"))

# Missing fixtures normally skip, so a developer without Julia can still run the
# suite.  In CI that would be a silent pass: a regeneration that produced nothing
# would look identical to a green run.  ACEJAX_REQUIRE_FIXTURES turns the skip
# into a failure.
REQUIRE = bool(os.environ.get("ACEJAX_REQUIRE_FIXTURES"))

def pace_fixture(path):
    """Return `path` if the committed PACE fixture exists; otherwise skip -- or,
    under ACEJAX_REQUIRE_FIXTURES (CI), fail, so a regeneration that wrote
    nothing cannot pass as all-skipped.  Reads the env at call time."""
    if pathlib.Path(path).exists():
        return path
    if os.environ.get("ACEJAX_REQUIRE_FIXTURES"):
        pytest.fail(f"missing PACE fixture {path}")
    pytest.skip(f"PACE fixture {path} not generated (see pace_ref/README.md)")


def require_optional(module):
    """`pytest.importorskip(module)`, except under ACEJAX_REQUIRE_OPTIONAL (the CI
    job that installs lammps-jax and matscipy-neighbours) a missing module fails
    the test, so a broken install cannot pass as all-skipped.  Reads the env at
    call time."""
    if os.environ.get("ACEJAX_REQUIRE_OPTIONAL"):
        import importlib
        try:
            return importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"ACEJAX_REQUIRE_OPTIONAL is set but {module} is not importable: {e}")
    return pytest.importorskip(module)


def require_coupling_lib():
    """Skip (or fail under ACEJAX_REQUIRE_OPTIONAL) unless ace_jax_coupling is
    installed WITH its compiled library (the uv path source builds a lib-less
    dev wheel when no bundle is given)."""
    ajc = require_optional("ace_jax_coupling")
    try:
        ajc.build_info()
    except ajc.CouplingLibError as e:
        if os.environ.get("ACEJAX_REQUIRE_OPTIONAL"):
            pytest.fail(f"ace_jax_coupling has no compiled library: {e}")
        pytest.skip(f"ace_jax_coupling has no compiled library: {e}")
    return ajc


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
def tiny_gp_problem():
    """M > 0 twin of tiny_linear_problem: 8 inducing sites (FPS).  Returns (prob, ds, theta), theta the
    default prior mean with noise scales (0.1, 0.3, 0.3): at the prior mean's sigma_E = 1e-3 the unscaled
    Gram G + Lambda has cond ~1e23 here, past what any reference solve can check against."""
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ace_jax.eval import load
    from ace_jax.fit.data import build_dataset, load_configs
    from ace_jax.fit.hypers import default_prior
    from ace_jax.fit.inducing import GPConfig, descriptor_scale, select_inducing, site_features
    from ace_jax.fit.kernels import KernelSpec
    from ace_jax.fit.objective import Problem

    xyz, fitted = FIXTURE_DIR / "si_tiny_train.xyz", FIXTURE_DIR / "si_fitted.npz"
    if not (xyz.exists() and fitted.exists()):
        pytest.skip("missing GP fixtures")
    model, meta, z = load(fitted)
    configs = load_configs(xyz, "dft_energy", "dft_force", "dft_virial")[:6]
    ds = build_dataset(configs, meta, np.asarray(z["E0"]), configs_per_batch=3)
    cfg = GPConfig(r0=2.35, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"],
                   NZ=len(meta["elements"]), C=3)
    X, S = site_features(model, cfg, ds)
    ind = select_inducing(X, S, ds.node_z, ds.node_mask, 8, descriptor_scale(X, ds.node_mask))
    prior = default_prior(2.35)
    prob = Problem(KernelSpec("cosine", True, cfg.D), model, ind, cfg, jnp.asarray(z["gamma"]), prior)
    theta = prior.mu._replace(log_sigma_E=np.log(0.1), log_sigma_F=np.log(0.3), log_sigma_V=np.log(0.3))
    return prob, ds, theta


def ard_pipe_cfg(**kw):
    """A uq='ard' linear-arm FitConfig for the ARD stage tests."""
    from ace_jax.fit.pipeline import FitConfig
    base = dict(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                virial_key="dft_virial", ntrain=30, ntest=8, batch=4, r0=2.35, arm="linear", uq="ard",
                opt="lbfgs", rungs=("map",), map_steps=5, predict_train=False, ard_variance="kappa", ard_n_min=1)
    return FitConfig(**{**base, **kw})


@pytest.fixture(scope="module")
def ard_map():
    """load_fit_data + build_problem + fit_map for ard_pipe_cfg(): identical for every ARD stage
    test (ard_variance / ard_mode only change the stage that follows), so done once."""
    from ace_jax.eval import highest_precision
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.mapfit import fit_map
    from ace_jax.fit.pipeline.objective import make_objective
    from ace_jax.fit.pipeline.problem import build_problem
    cfg = ard_pipe_cfg().validate()
    d = load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"))
    b = build_problem(cfg, d)
    with highest_precision():
        theta = fit_map(cfg, d, b, make_objective(cfg, d, b), log=lambda *a: None).theta
    return d, b, theta


def _orders(prob):
    """Correlation order of each B column of the tiny problem's model (all body orders present)."""
    import json
    import numpy as np
    z = np.load(FIXTURE_DIR / "si_fitted.npz")
    return [len(x) for x in json.loads(bytes(z["meta_json"]).decode())["nnll"]]


@pytest.fixture
def ard_setup(tiny_linear_problem):
    """(prob, ds, ev, h, post): the joint ARD evidence of the tiny problem and its posterior at h0."""
    import numpy as np
    from ace_jax.eval import highest_precision
    from ace_jax.fit.ard import ARDEvidence, ard_posterior, ard_statistics, body_order_columns
    from ace_jax.fit.hypers import default_prior
    prob, ds = tiny_linear_problem
    theta = default_prior(2.35).mu
    meta = {"nnll": [[None] * o for o in _orders(prob)], "n_B": prob.cfg.n_B, "n_pair": prob.cfg.n_pair,
            "NZ": prob.cfg.NZ, "rcut": prob.cfg.rcut, "elements": [14]}
    with highest_precision():
        ev = ARDEvidence(ard_statistics(theta, prob, ds, "joint"), np.asarray(prob.gamma),
                         body_order_columns(meta, prob.cfg))
        h = ev.h0(theta)
        post = ard_posterior(ev, h, 2.0, meta)
    return prob, ds, ev, h, post


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
# runner mid-run. Release them after each MODULE, not each test: the on-disk cache
# saves XLA compiles but not tracing/lowering, and clearing per test re-traced every
# model in every test (test_lean + test_to_spline: 279 s per test, 134 s per module,
# 120 s never; peak 2.4 / 2.8 / 3.1 GB).
@pytest.fixture(autouse=True, scope="module")
def _release_jax_memory():
    yield
    try:
        import jax
        jax.clear_caches()
    except Exception:
        pass


# A module that compiles many distinct programs per test (the ARD stage tests: every stage
# variant builds its own scans) can instead exhaust the process's memory-map limit within one
# module (vm.max_map_count; XLA's CPU JIT then aborts with "Failed to materialize symbols" /
# "releaseMappedMemory failed").  Such a module sets RELEASE_JAX_PER_TEST = True to clear after
# every test, as the whole suite did before per-module clearing.
@pytest.fixture(autouse=True)
def _release_jax_memory_per_test(request):
    yield
    if getattr(request.module, "RELEASE_JAX_PER_TEST", False):
        try:
            import jax
            jax.clear_caches()
        except Exception:
            pass


@pytest.fixture
def one_config_batch():
    """Factory: ASE Atoms -> the one-config Dataset batch (the calculator's construction)."""
    import jax
    import numpy as np
    from ace_jax.fit.data import Config, build_dataset

    def make(atoms, rcut=6.25):
        meta = {"elements": sorted({int(z) for z in atoms.numbers}), "rcut": rcut}
        c = Config(atoms.get_positions(), atoms.get_atomic_numbers(), atoms.get_cell().array, atoms.get_pbc(),
                   None, None, None, 1.0, 1.0, 1.0)
        return jax.tree.map(lambda a: a[0], build_dataset([c], meta, np.zeros(len(meta["elements"])), 1))
    return make


@pytest.fixture(scope="session")
def aniso_fit(tmp_path_factory):
    """A schema-3 ARD fit with --force-shape aniso (small, 5 MAP steps); shared by the CLI and equivariance tests."""
    from ace_jax.cli import main
    out = tmp_path_factory.mktemp("aniso")
    assert main(["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(FIXTURE_DIR / "si_tiny_train.xyz"),
                 "--ntrain", "30", "--ntest", "8", "--energy-key", "dft_energy", "--force-key", "dft_force",
                 "--virial-key", "dft_virial", "--m-per-species", "0", "--uq", "ard", "--force-shape", "aniso",
                 "--opt", "lbfgs", "--map-steps", "5", "--configs-per-batch", "4", "--r0", "2.35",
                 "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="session")
def fitted(tmp_path_factory):
    """Iso (explicit --force-shape iso) schema-3 ARD fit, shared across the ARD test modules."""
    from ace_jax.cli import main
    out = tmp_path_factory.mktemp("ard")
    assert main(["fit", "--model", str(FIXTURE_DIR / "si_fitted.npz"), "--data", str(FIXTURE_DIR / "si_tiny_train.xyz"), "--ntrain", "30",
                 "--ntest", "8", "--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key",
                 "dft_virial", "--m-per-species", "0", "--uq", "ard", "--force-shape", "iso", "--opt", "lbfgs", "--map-steps", "5",
                 "--configs-per-batch", "4", "--r0", "2.35", "--out", str(out)]) == 0
    return out


def small_si_xyz(path, n=12):
    """The first n frames of si_tiny_train.xyz (the isolated atom first), byte for byte:
    the CLI plumbing and reproduction tests need a fit, not 53 configurations."""
    lines, out, i = (FIXTURE_DIR / "si_tiny_train.xyz").read_text().splitlines(keepends=True), [], 0
    for _ in range(n):
        k = int(lines[i]); out += lines[i:i + k + 2]; i += k + 2
    path.write_text("".join(out))
    return path


CLI_FAST = ["--energy-key", "dft_energy", "--force-key", "dft_force", "--virial-key", "dft_virial",
            "--m-per-species", "0", "--rungs", "map", "--map-steps", "5", "--opt", "adam",
            "--configs-per-batch", "4"]


@pytest.fixture(scope="session")
def built_cli_run(tmp_path_factory):
    """One `aj fit --order 3 --max-degree 10` (basis built inline, default r0) on a
    12-config Si subset, shared by the CLI/run-file tests that only read its outputs:
    a SimpleNamespace(out, xyz, cache, stdout)."""
    import contextlib
    import io
    from types import SimpleNamespace
    from test_basis_build import _primed_cache
    from ace_jax.cli import main
    root = tmp_path_factory.mktemp("built_cli_run")
    xyz = small_si_xyz(root / "si12.xyz")
    cache = _primed_cache(root / "cache")
    buf = io.StringIO()
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(buf):
        mp.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
        main(["fit", "--order", "3", "--max-degree", "10", "--coupling-cache-dir", cache,
              "--train", str(xyz), "--out", str(root / "a"), *CLI_FAST])
    return SimpleNamespace(out=root / "a", xyz=xyz, cache=cache, stdout=buf.getvalue())


def shard_files(files, durations, i, n):
    """The files of shard i (1-based) of n: whole files, by longest-processing-time-first
    over their summed recorded durations (a file not yet recorded weighs the median file).
    Splitting per TEST scattered a module across shards, and each shard then repaid its
    module fixtures and first compiles, so the shards came out far from the balance the
    recorded per-test durations predicted (1238 / 773 / 900 s against 1030 each)."""
    import statistics
    w = {}
    for k, v in durations.items():
        f = k.split("::")[0]
        if f in files:
            w[f] = w.get(f, 0.0) + v
    fill = statistics.median(w.values()) if w else 1.0
    load, out = [0.0] * n, [[] for _ in range(n)]
    for f in sorted(files, key=lambda f: (-w.get(f, fill), f)):
        j = min(range(n), key=lambda j: (load[j], j))
        load[j] += w.get(f, fill); out[j].append(f)
    return sorted(out[i - 1])


def pytest_collection_modifyitems(config, items):
    """ACEJAX_SHARD=i/n keeps only shard i's files (shard_files over .test_durations)."""
    shard = os.environ.get("ACEJAX_SHARD")
    if not shard:
        return
    import json
    i, n = (int(x) for x in shard.split("/"))
    path = ROOT / ".test_durations"
    durations = json.loads(path.read_text()) if path.exists() else {}
    keep = set(shard_files(sorted({it.nodeid.split("::")[0] for it in items}), durations, i, n))
    selected = [it for it in items if it.nodeid.split("::")[0] in keep]
    config.hook.pytest_deselected(items=[it for it in items if it.nodeid.split("::")[0] not in keep])
    items[:] = selected
