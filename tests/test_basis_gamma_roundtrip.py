"""A built basis carries its smoothness prior through save_npz, so loading it never
takes the rebuild-from-meta fallback (or prints its notice)."""
import io

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from ace_jax.basis.export import save_npz
from ace_jax.basis.model import BasisSpec, build_basis
from ace_jax.basis.prior import gamma_from_model, prior_diagonal
from ace_jax.eval import load


def _saved(spec):
    b = build_basis(spec)
    buf = io.BytesIO(); save_npz(buf, b); buf.seek(0)
    return b, buf


def test_saved_basis_keeps_its_gamma_and_loads_silently(capsys):
    b, buf = _saved(BasisSpec(order=2, max_degree=6, elements=("Si",)))
    _, meta, z = load(buf)
    assert "gamma" in z.files
    np.testing.assert_array_equal(np.asarray(z["gamma"]), np.asarray(b.gamma))
    np.testing.assert_array_equal(prior_diagonal(z, meta, "basis"), gamma_from_model(meta))   # same prior
    assert capsys.readouterr().out == ""


def test_no_gamma_basis_saves_none():
    _, buf = _saved(BasisSpec(order=2, max_degree=6, elements=("Si",), no_gamma=True))
    assert "gamma" not in load(buf)[2].files


def test_fit_from_a_basis_prints_nothing(capsys):
    from conftest import FIXTURE_DIR
    from ace_jax.fit.pipeline import FitConfig, load_fit_data
    cfg = FitConfig(model=BasisSpec(order=2, max_degree=6, elements=("Si",)), arm="linear", m_per_species=0,
                    energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial", ntrain=8, ntest=4)
    from ace_jax.fit.pipeline.problem import build_problem
    build_problem(cfg, load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"), log=lambda *a: None))
    assert "gamma missing" not in capsys.readouterr().out


def test_prior_diagonal_fallback_goes_through_log(capsys):
    from conftest import FIXTURE_DIR
    _, meta, z = load(FIXTURE_DIR / "sige_nofit.npz")
    got = []
    prior_diagonal(z, meta, "sige_nofit.npz", log=got.append)
    assert capsys.readouterr().out == "" and "sige_nofit.npz" in got[0]
