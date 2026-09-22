def test_structural_times_configtype():
    from ace_jax.fit.weights import Structural, ConfigType, compose
    f = compose([Structural(), ConfigType({"defect": {"E": 10, "F": 10, "V": 1}},
                                          default={"E": 1, "F": 1, "V": 1})])
    meta = {"n_atoms": 4, "config_type": "defect"}
    assert abs(f(meta, "E") - 10 * 4 ** -0.5) < 1e-9
    assert abs(f(meta, "F") - 10 * 1.0) < 1e-9        # F exp 0 -> structural 1


def test_perconfig_override():
    from ace_jax.fit.weights import PerConfig, compose
    f = compose([PerConfig(key="w")])
    assert f({"w": 2.5}, "E") == 2.5 and f({}, "E") == 1.0
