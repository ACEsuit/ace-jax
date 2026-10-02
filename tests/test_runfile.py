"""fit.yaml: one file describes a whole `aj fit` run (CLI > file > default)."""
import pathlib

import pytest
import yaml

from ace_jax.cli import _parse, main


def _write(p, d):
    p.write_text(yaml.safe_dump(d))
    return p


def _fit_ns(argv):
    return _parse(["fit", *argv])


def test_yaml_supplies_required_and_cli_overrides(tmp_path, capsys):
    f = _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o", "m_per_species": 0,
                                       "basis": {"order": 3, "max_degree": 10}})
    a = _fit_ns(["--config", str(f), "--m-per-species", "5"])
    assert a.m_per_species == 5 and a.order == 3 and a.max_degree == 10
    assert pathlib.Path(a.train) == tmp_path / "t.xyz"
    assert "override: m_per_species 0 -> 5 (command line)" in capsys.readouterr().out


def test_unknown_key_did_you_mean(tmp_path, capsys):
    f = _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o", "m_per_specie": 0,
                                       "basis": {"order": 3, "max_degree": 10}})
    with pytest.raises(SystemExit):
        _fit_ns(["--config", str(f)])
    err = capsys.readouterr().err
    assert "unknown key 'm_per_specie'" in err and "did you mean 'm_per_species'" in err


def test_unknown_basis_key(tmp_path, capsys):
    f = _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o",
                                       "basis": {"order": 3, "max_degre": 10}})
    with pytest.raises(SystemExit):
        _fit_ns(["--config", str(f)])
    err = capsys.readouterr().err
    assert "basis: unknown key 'max_degre'" in err and "max_degree" in err


def test_basis_key_at_top_level_is_unknown(tmp_path, capsys):
    f = _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o", "max_degree": 10})
    with pytest.raises(SystemExit):
        _fit_ns(["--config", str(f)])
    assert "unknown key 'max_degree'" in capsys.readouterr().err


def test_model_and_basis_conflict(tmp_path, capsys):
    f = _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o", "model": "m.npz",
                                       "basis": {"order": 3, "max_degree": 10}})
    with pytest.raises(SystemExit):
        _fit_ns(["--config", str(f)])
    assert "'model' and 'basis' are alternatives" in capsys.readouterr().err


def test_not_a_mapping(tmp_path, capsys):
    (tmp_path / "fit.yaml").write_text("- just\n- a list\n")
    with pytest.raises(SystemExit):
        _fit_ns(["--config", str(tmp_path / "fit.yaml")])
    assert "mapping" in capsys.readouterr().err


def test_relative_paths_resolve_against_yaml_dir(tmp_path, monkeypatch):
    sub = tmp_path / "runs"
    sub.mkdir()
    f = _write(sub / "fit.yaml", {"train": "../data/t.xyz", "test": "t2.xyz", "out": "o", "r0": 2.3,
                                  "model": "m.npz"})
    monkeypatch.chdir(tmp_path)
    a = _fit_ns(["--config", str(f)])
    assert pathlib.Path(a.train).resolve() == (tmp_path / "data" / "t.xyz").resolve()
    assert pathlib.Path(a.model) == sub / "m.npz" and pathlib.Path(a.test) == sub / "t2.xyz"
    assert a.out == "o"                                   # out stays relative to the cwd


def test_yaml_lists_and_bools_equal_cli_forms(tmp_path):
    f = _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o", "rungs": ["map", "laplace"],
                                       "no_bump": True,
                                       "basis": {"order": 3, "max_degree": 10, "elements": ["Si", "Ge"],
                                                 "no_gamma": True, "embedding": "identity"}})
    a = _fit_ns(["--config", str(f)])
    b = _fit_ns(["--train", str(tmp_path / "t.xyz"), "--out", "o", "--rungs", "map,laplace", "--no-bump",
                 "--order", "3", "--max-degree", "10", "--elements", "Si,Ge", "--no-gamma",
                 "--basis-embedding", "identity"])
    for k in ("rungs", "no_bump", "elements", "no_gamma", "order", "max_degree", "basis_embedding"):
        assert getattr(a, k) == getattr(b, k), k


def test_aj_basis_reads_the_basis_block(tmp_path, monkeypatch):
    from test_basis_build import _primed_cache
    monkeypatch.setenv("ACEJAX_COUPLING_CACHE_ONLY", "1")
    cache = _primed_cache(tmp_path)
    f = _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o", "seed": 0,
                                       "basis": {"elements": ["Si"], "order": 3, "max_degree": 10,
                                                 "coupling_cache_dir": cache}})
    main(["basis", "--config", str(f), "--out", str(tmp_path / "si.npz")])
    assert (tmp_path / "si.npz").exists()


def test_cli_train_overrides_file_data(tmp_path, capsys):
    """--train on the command line replaces a file's data: (and vice versa); the
    mutually exclusive pair must not silently keep the file's side."""
    f = _write(tmp_path / "fit.yaml", {"data": "all.xyz", "out": "o", "basis": {"order": 3, "max_degree": 10}})
    a = _fit_ns(["--config", str(f), "--train", "t.xyz"])
    assert a.train == "t.xyz" and a.data is None
    assert "override: data" in capsys.readouterr().out
    g = _write(tmp_path / "g.yaml", {"train": "t.xyz", "out": "o", "basis": {"order": 3, "max_degree": 10}})
    a = _fit_ns(["--config", str(g), "--data", "all.xyz"])
    assert a.data == "all.xyz" and a.train is None


def test_cli_switches_basis_source(tmp_path, capsys):
    """CLI > file also for the basis source: --model over a file's basis: block,
    and --order/--max-degree over a file's model:."""
    f = _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o", "r0": 2.4,
                                       "basis": {"order": 3, "max_degree": 10}})
    a = _fit_ns(["--config", str(f), "--model", "m.npz"])
    assert a.model == "m.npz" and a.order is None and a.max_degree is None
    assert "override: order 3 -> None (command line gives --model)" in capsys.readouterr().out
    g = _write(tmp_path / "g.yaml", {"train": "t.xyz", "out": "o", "r0": 2.4, "model": "m.npz"})
    a = _fit_ns(["--config", str(g), "--order", "2", "--max-degree", "6"])
    assert a.model is None and (a.order, a.max_degree) == (2, 6)


@pytest.mark.parametrize("bad, key", [({"uq": "popz"}, "uq"), ({"m_per_species": "many"}, "m_per_species"),
                                      ({"basis": {"order": 3.5, "max_degree": 10}}, "order"),
                                      ({"basis": {"order": 3, "max_degree": 10, "reduction": "pcaa"}}, "reduction")])
def test_yaml_values_are_type_and_choice_checked(tmp_path, capsys, bad, key):
    """argparse checks choices and types only on the command line; file values
    become defaults, so the file reader checks them itself."""
    d = {"train": "t.xyz", "out": "o", "basis": {"order": 3, "max_degree": 10}, **bad}
    f = _write(tmp_path / "fit.yaml", d)
    with pytest.raises(SystemExit):
        _fit_ns(["--config", str(f)])
    assert key in capsys.readouterr().err


def test_scalar_elements_in_yaml(tmp_path):
    f = _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o",
                                       "basis": {"order": 3, "max_degree": 10, "elements": 14}})
    from ace_jax.cli import _basis_spec
    a = _fit_ns(["--config", str(f)])
    assert _basis_spec(a, embedding=None).elements == ("14",)


def test_aj_basis_requires_elements(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _parse(["basis", "--order", "2", "--max-degree", "4", "--out", str(tmp_path / "x.npz")])
    assert "--elements" in capsys.readouterr().err
    f = _write(tmp_path / "fit.yaml", {"basis": {"order": 2, "max_degree": 4}})
    with pytest.raises(SystemExit):
        _parse(["basis", "--config", str(f), "--out", str(tmp_path / "x.npz")])


def test_basis_unavailable_is_a_clean_cli_error(monkeypatch, capsys):
    import ace_jax.cli as C
    from ace_jax.basis.coupling import BasisUnavailable
    def boom(a):
        raise BasisUnavailable("building a new basis is not available on this platform")
    monkeypatch.setattr(C, "run", boom)
    with pytest.raises(SystemExit) as e:
        main(["fit", "--order", "2", "--max-degree", "4", "--train", "t.xyz", "--out", "o"])
    assert e.value.code == 2 and "not available on this platform" in capsys.readouterr().err


ARD_KEYS = {"force_shape": "aniso", "ard_coverage": 0.8, "ard_groups": "none", "ard_cluster_size": 4.5,
            "ard_press": "block", "ard_n_min": 5, "no_ard_support": True, "ard_transfer": "sqrt"}


def _ard_file(tmp_path, **over):
    return _write(tmp_path / "fit.yaml", {"train": "t.xyz", "out": "o", "model": "m.npz", "r0": 2.4,
                                          "uq": "ard", **{**ARD_KEYS, **over}})


def test_ard_keys_accepted_and_cli_overrides(tmp_path):
    a = _fit_ns(["--config", str(_ard_file(tmp_path))])
    assert (a.force_shape, a.ard_coverage, a.ard_groups, a.ard_cluster_size, a.ard_press, a.ard_n_min,
            a.no_ard_support, a.ard_transfer) == ("aniso", 0.8, "none", 4.5, "block", 5, True, "sqrt")
    b = _fit_ns(["--config", str(_ard_file(tmp_path)), "--force-shape", "iso", "--ard-coverage", "0.95",
                 "--ard-groups", "distortion", "--ard-cluster-size", "2", "--ard-press", "exact",
                 "--ard-n-min", "9"])
    assert (b.force_shape, b.ard_coverage, b.ard_groups, b.ard_cluster_size, b.ard_press, b.ard_n_min) == \
        ("iso", 0.95, "distortion", 2.0, "exact", 9)


def test_ard_cluster_size_yaml_inf(tmp_path):
    f = tmp_path / "fit.yaml"
    f.write_text("train: t.xyz\nout: o\nmodel: m.npz\nr0: 2.4\nard_cluster_size: .inf\n")
    assert _fit_ns(["--config", str(f)]).ard_cluster_size == float("inf")


@pytest.mark.parametrize("bad, key", [({"force_shape": "foo"}, "force_shape"), ({"ard_n_min": "x"}, "ard_n_min"),
                                      ({"ard_n_min": 2.5}, "ard_n_min"), ({"ard_coverage": "high"}, "ard_coverage"),
                                      ({"ard_press": "fast"}, "ard_press"), ({"ard_groups": "all"}, "ard_groups"),
                                      ({"no_ard_support": "yes"}, "no_ard_support"),
                                      ({"ard_transfer": "half"}, "ard_transfer")])
def test_ard_bad_values_rejected(tmp_path, capsys, bad, key):
    with pytest.raises(SystemExit):
        _fit_ns(["--config", str(_ard_file(tmp_path, **bad))])
    assert f"'{key}'" in capsys.readouterr().err
