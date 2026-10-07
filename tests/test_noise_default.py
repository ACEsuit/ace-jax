"""noise="auto" (the default): shared noise when the user gives weights, per-quantity otherwise.

With per-quantity noise each learned sigma_q cancels its quantity's weight at a converged MAP (#64), so
explicit E:F:V weights would have no effect; with no weights there is no stated balance, and the evidence
sets it (on Cantor shared noise at the default 1:1:1 weights cost +79 % energy RMSE, D4 of
docs/dev/gp-discrepancy-exploration.md).  "auto" falls back to per-quantity wherever shared noise is not
available, so giving weights never turns a working command into an error."""
import pytest

from ace_jax.fit.pipeline import FitConfig

W = {"default": {"E": 30.0, "F": 1.0, "V": 1.0}}


def _cfg(**kw):
    return FitConfig(**{"model": "x.npz", "arm": "linear", **kw})


def test_default_is_auto():
    assert _cfg().noise == "auto"


@pytest.mark.parametrize("kw, mode", [
    (dict(), "per-quantity"),                                            # no weights: the evidence sets the balance
    (dict(weights=W), "shared"),                                         # weights given: they set the balance
    (dict(factors=[{"Structural": {}}]), "shared"),
    (dict(weights=W, noise="per-quantity"), "per-quantity"),             # an explicit choice is kept
    (dict(noise="shared"), "shared"),
    (dict(weights=W, uq="ard"), "per-quantity"),                         # joint ARD refits sigma_q itself
    (dict(weights=W, uq="ard", ard_mode="sequential"), "shared"),
    (dict(weights=W, sigma_type=True), "per-quantity"),
    (dict(weights=W, solver="lstsq", uq="blr"), "per-quantity"),
])
def test_auto_resolves_on_validate(kw, mode):
    assert _cfg(**kw).validate().noise == mode


def test_cli_weights_select_shared_noise():
    from ace_jax.cli import _fit_config, _parse
    a = _parse(["fit", "--model", "x.npz", "--train", "t.xyz", "--m-per-species", "0", "--r0", "2.35", "--out", "o",
                "--weights", '{"default": {"E": 30, "F": 1, "V": 1}}'])
    assert _fit_config(a).validate().noise == "shared"
    a = _parse(["fit", "--model", "x.npz", "--train", "t.xyz", "--m-per-species", "0", "--r0", "2.35", "--out", "o"])
    assert _fit_config(a).validate().noise == "per-quantity"


def test_stages_resolve_auto_without_validate():
    """Stages called directly (no validate()) see auto resolved: the MAP objective ties the noise when
    weights are given, and not otherwise."""
    from ace_jax.fit.pipeline.config import resolved_noise
    assert resolved_noise(_cfg(weights=W)) == "shared" and resolved_noise(_cfg()) == "per-quantity"
    assert resolved_noise(_cfg(weights=W, uq="ard")) == "per-quantity"
