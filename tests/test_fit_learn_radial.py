"""aj fit --learn-radial: learned tensor radials as a fit-pipeline stage
(docs/specs/2026-10-01-fit-learned-radials-design.md)."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR

XYZ = FIXTURE_DIR / "si_tiny_train.xyz"
MODEL = FIXTURE_DIR / "si_ace_model.npz"
QUIET = lambda *a, **k: None
KEYS = dict(energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")


def _cfg(**kw):
    from ace_jax.fit.pipeline import FitConfig
    base = dict(model=str(MODEL), arm="linear", m_per_species=0, rungs=("map",), map_steps=20,
                opt="adam", batch=4, r0=2.35, e0="model", predict_train=False, **KEYS)
    base.update(kw)
    return FitConfig(**base)


def test_config_radial_defaults_and_validation():
    c = _cfg().validate()
    assert (c.learn_radial, c.radial_n_q, c.radial_steps, c.radial_lam_grid, c.radial_val_frac) == \
        (False, 12, 40, (0.0, 1e-2), 0.2)
    for bad in (dict(radial_val_frac=0.0), dict(radial_val_frac=1.0), dict(radial_n_q=0),
                dict(radial_steps=-1), dict(radial_lam_grid=())):
        with pytest.raises(ValueError, match="radial"):
            _cfg(learn_radial=True, **bad).validate()
