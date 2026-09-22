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


def test_configtype_case_insensitive_matches_type_idx_resolution():
    # data.py's type_idx assignment matches config_type case-insensitively
    # (`name.lower() == ct.lower()`); ConfigType.weight must agree, else a
    # case-mismatched config_type gets type_idx=1 but falls to `default`
    # weights here -- an internal inconsistency.
    from ace_jax.fit.weights import ConfigType, compose
    f = compose([ConfigType({"Defect": {"E": 10, "F": 10, "V": 1}},
                            default={"E": 1, "F": 1, "V": 1})])
    meta = {"config_type": "defect"}          # case mismatch vs. table key "Defect"
    assert f(meta, "E") == 10                 # must resolve to the "Defect" entry, not default
