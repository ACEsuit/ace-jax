"""The E/F/V RMSE table `aj fit` and `aj eval` print, per config type (fit/report.py)."""
import numpy as np

from conftest import FIXTURE_DIR


def _toy():
    types = np.array(["bulk", "bulk", "surf"])
    nat = np.array([2, 2, 1])
    E, Em = np.array([-10.0, -9.0, -4.0]), np.array([-10.002, -9.0, np.nan])
    E = np.array([-10.0, -9.0, np.nan])                         # surf has no energy label
    F = np.zeros((5, 3)); Fm = np.zeros((5, 3)); Fm[0, 0] = 0.3; Fm[4, :] = 0.1
    V = np.full((3, 6), np.nan); Vm = np.zeros((3, 6))
    V[0] = 0.0; Vm[0] = 0.004                                    # one virial label
    return types, nat, E, Em, F, Fm, V, Vm


def test_rmse_by_type_units_and_missing_labels():
    from ace_jax.fit.report import rmse_by_type
    s = rmse_by_type(*_toy())
    assert list(s) == ["bulk", "surf", "all"]
    assert (s["bulk"]["n_cfg"], s["bulk"]["n_atoms"], s["all"]["n_cfg"]) == (2, 4, 3)
    np.testing.assert_allclose(s["bulk"]["E"], 1e3 * np.sqrt(np.mean([(0.002 / 2) ** 2, 0.0])))   # meV/atom
    assert np.isnan(s["surf"]["E"])                               # no label: no number
    np.testing.assert_allclose(s["bulk"]["F"], np.sqrt(0.3 ** 2 / 12))                             # eV/A
    np.testing.assert_allclose(s["surf"]["F"], 0.1)
    np.testing.assert_allclose(s["bulk"]["V"], 1e3 * 0.004 / 2)                                    # meV/atom
    assert np.isnan(s["surf"]["V"])


def test_format_rmse_table_is_aligned_and_marks_missing():
    from ace_jax.fit.report import format_rmse_table, rmse_by_type
    t = format_rmse_table(rmse_by_type(*_toy()), "test")
    lines = t.splitlines()
    assert "test" in lines[0] and "E (meV/atom)" in t and "F (eV/Å)" in t and "V (meV/atom)" in t
    rows = [l for l in lines if l.split() and l.split()[0] in ("bulk", "surf", "all")]
    assert [r.split()[0] for r in rows] == ["bulk", "surf", "all"]
    assert len({len(r) for r in rows}) == 1                        # aligned columns
    assert rows[1].split()[3] == "-"                              # surf: no energy label


def test_load_configs_keeps_the_config_type_name():
    from ace_jax.fit.data import load_configs
    cs = load_configs(FIXTURE_DIR / "si_tiny_train.xyz")
    assert cs[0].config_type == "isolated_atom" and {c.config_type for c in cs} >= {"isolated_atom", "dia"}


def test_fit_logs_an_rmse_table_per_split():
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data
    cfg = FitConfig(model=str(FIXTURE_DIR / "si_fitted.npz"), energy_key="dft_energy", force_key="dft_force",
                    virial_key="dft_virial", ntrain=30, ntest=12, batch=4, r0=2.35, arm="linear",
                    m_per_species=0, opt="lbfgs", rungs=("map",), map_steps=3, predict_train=False)
    lines = []
    fit(cfg, load_fit_data(cfg, data=str(FIXTURE_DIR / "si_tiny_train.xyz"), log=lines.append), log=lines.append)
    text = "\n".join(lines)
    assert "RMSE, test (map)" in text
    table = text[text.index("RMSE, test (map)"):]
    assert "config type" in table and "\nall " in table and ("dia" in table or "bt" in table)


def test_one_config_type_prints_only_the_all_row():
    from ace_jax.fit.report import format_rmse_table, rmse_by_type
    types, nat, E, Em, F, Fm, V, Vm = _toy()
    t = format_rmse_table(rmse_by_type([None] * 3, nat, E, Em, F, Fm, V, Vm), "test")
    rows = [l.split()[0] for l in t.splitlines()[3:] if l and not l.startswith("-")]
    assert rows == ["all"]
