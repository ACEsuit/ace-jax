"""The curation campaign loop, with the fixture ACE model standing in for the labeller."""
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from conftest import FIXTURE_DIR  # noqa: E402

from ace_jax.tutorials import curation as K  # noqa: E402


def _labeller():
    from ace_jax import ACECalculator
    calc = ACECalculator(str(FIXTURE_DIR / "si_fitted.npz"), skin=0)

    def label(xs):
        out = []
        for a in xs:
            b = a.copy(); b.calc = calc
            e, f = b.get_potential_energy(), b.get_forces()
            c = a.copy(); c.calc = None
            c.info["energy"] = e; c.arrays["forces"] = f
            if c.pbc.all():
                c.info["virial"] = -b.get_stress(voigt=False) * b.get_volume()
            out.append(c)
        return out
    return label


def test_campaign_replays_shipped_pools_and_picks_what_the_selector_asks(tmp_path):
    lab = _labeller()
    live = K.run_campaign("novelty", lab, tmp_path / "live", rounds=1, per_round=2, n_steps=4)
    assert len(live["history"]) == 2 and live["history"][1]["labels"] == 2
    assert np.isfinite([h["err"] for h in live["history"]]).all()
    seen = []
    replay = K.run_campaign("novelty", lab, tmp_path / "replay", rounds=1, per_round=2, pools=live["pools"],
                            pick=lambda pool, train, k, rng, m, p: seen.append(len(pool)) or [0, 1])
    assert seen == [len(live["pools"][0])] and replay["history"][1]["picks"] == [0, 1]
    assert replay["pools"][0] is not live["pools"][0] and len(replay["pools"][0]) == len(live["pools"][0])
