"""FitConfig.model may be a Basis or a BasisSpec: the pipeline builds the basis
in memory (species from the data) and fits it exactly as it fits a file."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR, small_si_xyz
from test_basis_build import _primed_cache

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
QUIET = lambda *a, **k: None


@pytest.fixture(scope="module")
def xyz12(tmp_path_factory):
    """12 configs: these are bit-equality and plumbing checks, not accuracy ones."""
    return small_si_xyz(tmp_path_factory.mktemp("si12") / "si12.xyz")


def _cfg(model, **kw):
    from ace_jax.fit.pipeline import FitConfig
    base = dict(arm="linear", m_per_species=0, rungs=("map",), map_steps=5, opt="adam",
                batch=4, r0=2.35, e0="lsq", predict_train=False,
                energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
    base.update(kw)
    return FitConfig(model=model, **base)


def _fit(cfg, tmp_path, name, xyz):
    from ace_jax.fit.pipeline import fit, load_fit_data
    from ace_jax.fit.pipeline.export import save_model
    d = load_fit_data(cfg, train=str(xyz), log=QUIET)
    res = fit(cfg, d, log=QUIET)
    return res, np.load(save_model(res, tmp_path / name, log=QUIET))


def test_basis_spec_fit_equals_file_fit(tmp_path, monkeypatch, xyz12):
    """A basis built inside the pipeline fits and exports bit-identically to the
    same basis saved to a file first (the in-memory hand-off changes nothing)."""
    from ace_jax.basis.export import save_npz
    from ace_jax.basis.model import BasisSpec, build_basis
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    f = tmp_path / "si.npz"
    save_npz(f, build_basis(BasisSpec(order=3, max_degree=10, elements=(14,), coupling_cache_dir=cache)))
    _, za = _fit(_cfg(BasisSpec(order=3, max_degree=10, coupling_cache_dir=cache)), tmp_path, "a", xyz12)
    _, zb = _fit(_cfg(str(f)), tmp_path, "b", xyz12)
    assert sorted(za.files) == sorted(zb.files)
    for k in za.files:
        assert np.array_equal(za[k], zb[k]), k


def test_basis_object_is_accepted(tmp_path, monkeypatch, xyz12):
    from ace_jax.basis.model import BasisSpec, build_basis
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    b = build_basis(BasisSpec(order=3, max_degree=10, elements=(14,), coupling_cache_dir=cache))
    res, z = _fit(_cfg(b), tmp_path, "c", xyz12)
    assert "WB" in z.files and res.data.source.startswith("basis")


def test_inferred_elements_union_of_splits():
    from ace_jax.fit.data import load_configs
    from ace_jax.fit.pipeline.data import infer_elements
    cs = load_configs(str(XYZ))
    assert infer_elements(cs[:2], cs[2:4], []) == [14]


def test_explicit_elements_missing_test_species_errors(tmp_path, monkeypatch):
    from ace_jax.basis.model import BasisSpec
    from ace_jax.fit.pipeline import load_fit_data
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    spec = BasisSpec(order=3, max_degree=10, elements=("Ge",), coupling_cache_dir=_primed_cache(tmp_path))
    with pytest.raises(ValueError, match="Si in the data .*not in the basis elements"):
        load_fit_data(_cfg(spec), train=str(XYZ), log=QUIET)


def test_r0_defaults_to_basis_mean(tmp_path, monkeypatch, xyz12):
    from ace_jax.basis.model import BasisSpec, basis_r0
    from ace_jax.fit.pipeline import load_fit_data
    from ace_jax.fit.pipeline.problem import build_problem
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cfg = _cfg(BasisSpec(order=3, max_degree=10, coupling_cache_dir=_primed_cache(tmp_path)), r0=None)
    d = load_fit_data(cfg, train=str(xyz12), log=QUIET)
    assert d.r0 == pytest.approx(basis_r0(d.meta))
    assert build_problem(cfg, d).gpcfg.r0 == pytest.approx(d.r0)      # where fit() takes r0 from


def test_r0_missing_everywhere_errors(tmp_path):
    from ace_jax.fit.pipeline import fit, load_fit_data
    cfg = _cfg(str(FIXTURE_DIR / "si_ace_model.npz"), r0=None)
    d = load_fit_data(cfg, train=str(XYZ), log=QUIET)
    if d.r0 is None:
        with pytest.raises(ValueError, match="r0"):
            fit(cfg, d, log=QUIET)


def test_ard_on_a_built_basis_equals_file(tmp_path, monkeypatch, xyz12):
    """uq='ard' writes its posterior-mean model arrays during fit (run.py's "model"
    stage) from the arrays the pipeline loaded, so a BasisSpec model (no file to
    re-read) works and equals the same basis fitted from a file."""
    from ace_jax.basis.export import save_npz
    from ace_jax.basis.model import BasisSpec, build_basis
    from ace_jax.fit.pipeline import fit, load_fit_data
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    spec = BasisSpec(order=3, max_degree=10, coupling_cache_dir=cache)
    f = tmp_path / "si.npz"
    save_npz(f, build_basis(BasisSpec(order=3, max_degree=10, elements=("Si",), coupling_cache_dir=cache)))
    kw = dict(uq="ard", opt="lbfgs", map_steps=5, predict_stats="recompute")
    out = {}
    for name, model in (("spec", spec), ("file", str(f))):
        staged = {}
        cfg = _cfg(model, **kw)
        d = load_fit_data(cfg, train=str(xyz12), log=QUIET)
        fit(cfg, d, log=QUIET, on_stage=lambda k, v, s=staged: s.__setitem__(k, v))
        out[name] = staged["model"]
    assert out["spec"].keys() == out["file"].keys()
    for k in ("WB", "Wpair", "E0"):        # ARD is not bit-reproducible run to run (GPU: ~4e-8)
        x, y = out["spec"][k], out["file"][k]
        assert np.abs(x - y).max() <= 1e-6 * max(np.abs(y).max(), 1e-300), k
