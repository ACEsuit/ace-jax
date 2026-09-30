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
