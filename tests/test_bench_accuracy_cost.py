import pathlib
import sys

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "bench"))
from conftest import FIXTURE_DIR

TINY = FIXTURE_DIR / "si_tiny_train.xyz"          # 53 GAP-18 frames, lower-case dft_* keys


def _frames():
    from ace_jax.fit.xyz import read_extxyz
    return read_extxyz(TINY)


def test_normalise_renames_labels_and_keeps_virial():
    from accuracy_cost.split import normalise
    fr = _frames()
    at = normalise(fr[1])
    assert set(at.info) >= {"energy", "config_type"} and "forces" in at.arrays
    assert not any(k.startswith("gap_") or k.startswith("dft_") for k in (*at.info, *at.arrays))
    with_v = [f for f in fr if any(k.lower() == "dft_virial" for k in f.info)]
    if with_v:                                      # the fixture slice may carry none
        a = normalise(with_v[0])
        assert a.info["virial"].shape == (3, 3)


def test_split_writer_round_trips_virial(tmp_path):
    import ase
    from accuracy_cost.split import _write, read_atoms
    at = ase.Atoms("Si2", positions=[[0, 0, 0], [1.3, 1.3, 1.3]], cell=np.eye(3) * 5.4, pbc=True)
    at.info.update(energy=-300.0, config_type="dia", virial=np.arange(9.0).reshape(3, 3))
    at.arrays["forces"] = np.ones((2, 3))
    b = at.copy(); b.info = {"energy": -301.0, "config_type": "liq"}; b.arrays["forces"] = np.zeros((2, 3))
    _write(tmp_path / "x.xyz", [at, b])
    r = read_atoms(tmp_path / "x.xyz")
    assert np.allclose(np.reshape(r[0].info["virial"], (3, 3)), at.info["virial"]) and "virial" not in r[1].info


def test_split_reads_uppercase_label_keys():
    from accuracy_cost.split import normalise
    from ace_jax.fit.xyz import Frame
    f = _frames()[1]
    info = {("DFT_energy" if k == "dft_energy" else k): v for k, v in f.info.items()}
    arrays = {("DFT_force" if k == "dft_force" else k): v for k, v in f.arrays.items()}
    at = normalise(Frame(f.numbers, f.positions, f.cell, f.pbc, info, arrays))
    assert at is not None and at.info["energy"] == f.info["dft_energy"]
    assert np.array_equal(at.arrays["forces"], f.arrays["dft_force"])


def test_split_is_deterministic_and_stratified():
    from accuracy_cost.split import stratified
    types = ["a"] * 40 + ["b"] * 20 + ["c"] * 5
    tr1, te1 = stratified(types, seed=0)
    tr2, te2 = stratified(types, seed=0)
    assert (tr1, te1) == (tr2, te2) and sorted(tr1 + te1) == list(range(len(types)))
    assert sum(types[i] == "a" for i in te1) == 4 and sum(types[i] == "b" for i in te1) == 2


def test_rare_types_stay_in_train():
    from accuracy_cost.split import stratified
    types = ["isolated_atom"] + ["dia"] * 30 + ["crack"] * 7
    tr, te = stratified(types, seed=0)
    assert 0 in tr and all(types[i] != "crack" for i in te)


CL = ("at_file=x.xyz gap={soap cutoff=5.0 n_sparse=9000} default_sigma={0.001 0.1 0.05 0.0} "
      "config_type_sigma={liq:0.003:0.15:0.2:0.0:amorph:0.01:0.2:0.4:0.0:sp:0.01:0.2:0.4:0.0} "
      "energy_parameter_name=dft_energy")


def test_gap_sigmas_parse_groups_of_five():
    from accuracy_cost.weights import gap_sigmas
    d, per = gap_sigmas(CL)
    assert d == (0.001, 0.1, 0.05)
    assert per == {"liq": (0.003, 0.15, 0.2), "amorph": (0.01, 0.2, 0.4), "sp": (0.01, 0.2, 0.4)}


def test_weights_consistent_across_codes():
    """Every code must weight a config type relative to the default identically:
    ace-jax (relative 1/sigma on residuals, default 1)^2 == pacemaker's type/default
    ratio per quantity, and MACE's per-config factor == ace-jax's (F_type/F_default)^2."""
    import ase
    from accuracy_cost.weights import acejax, gap_sigmas, mace, pacemaker
    d, per = gap_sigmas(CL)
    aj, mc = acejax(d, per), mace(d, per)
    from accuracy_cost.weights import ACEPOTENTIALS_DEFAULT as D0
    assert aj["default"] == D0 == {"E": 30.0, "F": 1.0, "V": 1.0}
    ats = [ase.Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], info={"config_type": t}) for t in ("dia", "liq")]
    (wE0, wF0), (wE, wF) = pacemaker(d, per, ats)
    assert np.isclose((aj["liq"]["E"] / D0["E"]) ** 2, wE / wE0) and np.allclose((aj["liq"]["F"] / D0["F"]) ** 2, wF / wF0)
    assert np.isclose(mc["liq"], (aj["liq"]["F"] / D0["F"]) ** 2) and mc["Default"] == 1.0
    assert np.isclose(aj["liq"]["V"], D0["V"] * d[2] / per["liq"][2])


def test_command_line_from_xml(tmp_path):
    from accuracy_cost.weights import command_line
    p = tmp_path / "g.xml"
    p.write_text(f"<GAP><command_line><![CDATA[ {CL}]]></command_line></GAP>")
    assert command_line(p).strip() == CL



class _Const:
    """A calculator with known errors: E off by 0.01 eV/atom, every F component off by 0.1."""
    def __init__(self): self.results = {}
    def get_potential_energy(self, at): return at.info["energy"] + 0.01 * len(at)
    def get_forces(self, at): return at.arrays["forces"] + 0.1
    def get_stress(self, at): return np.zeros((3, 3))


def test_summarise_known_errors():
    from accuracy_cost.evaluate import errors, summarise
    from accuracy_cost.split import normalise
    from ace_jax.fit.xyz import read_extxyz
    ats = [a for a in map(normalise, read_extxyz(TINY)) if a is not None]
    s = summarise(errors(ats, _Const()))
    assert np.isclose(s["rmse_E"], 0.01) and np.isclose(s["rmse_F"], 0.1)
    with_v = [a for a in ats if "virial" in a.info]
    assert with_v                                  # zero predicted stress: rmse_V = rms(V_ref / n)
    ref = np.sqrt(np.mean(np.concatenate([(a.info["virial"] / len(a)).ravel() for a in with_v]) ** 2))
    assert np.isclose(s["rmse_V"], ref)
    assert set(s["per_type"]) == {a.info["config_type"] for a in ats}


def test_voigt_stress_converts_to_virial():
    from accuracy_cost.evaluate import errors
    from accuracy_cost.split import normalise
    from ace_jax.fit.xyz import read_extxyz
    at = next(a for a in map(normalise, read_extxyz(TINY)) if a is not None and "virial" in a.info)

    class _Voigt(_Const):
        def get_stress(self, at): return np.array([1., 2, 3, 4, 5, 6])    # ASE order xx yy zz yz xz xy
    S = np.array([[1, 6, 5], [6, 2, 4], [5, 4, 3]], float)
    dV = errors([at], _Voigt())[0]["dV"]
    assert np.allclose(dV, (-S * at.get_volume() - np.reshape(at.info["virial"], (3, 3))) / len(at), rtol=0, atol=1e-13)


def test_load_calc_scores_a_real_ace_model():
    from accuracy_cost.evaluate import errors, load_calc, summarise
    from accuracy_cost.split import normalise
    from ace_jax.fit.xyz import read_extxyz
    ats = [a for a in map(normalise, read_extxyz(TINY)) if a is not None]
    iv = next(i for i, a in enumerate(ats) if "virial" in a.info)
    ats = ats[:5] if iv < 5 else ats[:4] + [ats[iv]]
    calc = load_calc("ace", FIXTURE_DIR / "si_fitted.npz")
    rows = errors(ats, calc)
    s = summarise(rows)
    assert np.isfinite(s["rmse_E"]) and 0 < s["rmse_F"] < 5
    assert s["rmse_V"] is not None and np.isfinite(s["rmse_V"])
    at = next(a for a in ats if "virial" in a.info)
    row = next(r for r in rows if r["dV"] is not None)
    Vpred = row["dV"] * len(at) + np.reshape(at.info["virial"], (3, 3))
    a = at.copy(); a.calc = None
    from accuracy_cost.evaluate import _full
    Vaj = -_full(calc.get_stress(a)) * a.get_volume()
    assert np.allclose(Vpred, Vaj, rtol=1e-8, atol=1e-8 * np.abs(Vaj).max())
    # independent sign check: virial = -dE/d(strain), central differences
    h = 1e-5
    def E(eps):
        b = at.copy(); b.calc = None; M = np.eye(3) + eps
        b.set_cell(b.cell.array @ M, scale_atoms=True)
        return calc.get_potential_energy(b)
    for (i, j) in [(0, 0), (0, 1)]:
        eps = np.zeros((3, 3)); eps[i, j] = eps[j, i] = h
        d = (E(eps) - E(-eps)) / (2 * h)
        d = d if i == j else d / 2                  # symmetric pair strain counts the component twice
        assert np.isclose(-d, Vpred[i, j], rtol=1e-4, atol=1e-6 * np.abs(Vpred).max())


def test_evaluate_refuses_mismatched_split(tmp_path, monkeypatch):
    import accuracy_cost.common as C
    import accuracy_cost.evaluate as ev
    monkeypatch.setattr(C, "RUNS", tmp_path / "runs"); monkeypatch.setattr(C, "DATA", tmp_path / "data")
    monkeypatch.setattr(ev, "DATA", tmp_path / "data")
    (tmp_path / "data").mkdir(); (tmp_path / "data" / "split.json").write_text('{"hash": "new"}')
    C.write_status(C.run_dir("ace", "x"), status="ok", split_hash="old", model="m.npz")
    with pytest.raises(ev.SplitMismatch):
        ev.score("ace", "x")


def test_sweep_is_complete_and_measured():
    from accuracy_cost.common import load_sweep
    s = load_sweep()
    assert set(s) == {"ace", "ace_lr", "ace_gp", "pace", "mace", "gap", "ace_pshape", "ace_pshape_lr", "ace_wl", "gap_noglue", "gap_E30", "ace_yace", "ace_tab"}
    assert len(s["ace"]) == 7 and all(e["n_B"] > 0 for e in s["ace"])
    assert [(e["order"], e["max_degree"]) for e in s["ace_lr"]] == [(e["order"], e["max_degree"]) for e in s["ace"]]
    nB = [e["n_B"] for e in s["ace"]]
    assert nB == sorted(nB) and nB[0] <= 80 and nB[-1] >= 1000
    for fam, es in s.items():
        assert len({e["size"] for e in es}) == len(es), fam


def test_fit_ace_commands_share_cutoff_weights_and_e0(tmp_path):
    from accuracy_cost.fit_ace import command
    lin = command("ace", {"order": 3, "max_degree": 8}, tmp_path)
    gp = command("ace_gp", {"order": 3, "max_degree": 8, "m": 100}, tmp_path)
    for c in (lin, gp):
        s = " ".join(c)
        assert "--rcut 5.0" in s and "--e0 lsq" in s and "--rungs map" in s and "weights_acejax.json" in s
        assert "--virial-key virial" in s
        assert c[c.index("--test") + 1].endswith("test.xyz")  # else aj fit scores the training set as "test"
    assert lin[lin.index("--m-per-species") + 1] == "0" and gp[gp.index("--m-per-species") + 1] == "100"
    lr = command("ace_lr", {"order": 3, "max_degree": 8}, tmp_path)
    assert "--learn-radial" in lr and lr[lr.index("--m-per-species") + 1] == "0"
    assert "--learn-radial" not in lin and "--learn-radial" not in gp
    for c in (lin, lr):
        assert c[c.index("--noise") + 1] == "shared"
    assert "--configs-per-batch" not in lin
    big = command("ace", {"order": 4, "max_degree": 16, "configs_per_batch": 2}, tmp_path)
    assert big[big.index("--configs-per-batch") + 1] == "2"


def test_pace_input_shares_cutoff_e0_and_ladder():
    from accuracy_cost.fit_pace import input_yaml
    y = input_yaml({"max_functions": 800, "ladder_step": 50})
    assert y["cutoff"] == 5.0 and y["potential"]["bonds"]["ALL"]["rcut"] == 5.0
    assert y["potential"]["functions"]["number_of_functions_per_element"] == 800
    assert y["fit"]["ladder_step"] == 50 and y["fit"]["weighting"]["type"] == "ExternalWeightingPolicy"
    assert y["data"]["filename"] == "train.pckl.gzip" and y["backend"]["evaluator"] == "tensorpot"


def test_pace_dataframe_tests_pass_in_the_pacemaker_venv():
    import subprocess
    py = pathlib.Path("/storage/eng/essswb/venvs/ac-pacemaker/bin/python")
    if not py.exists():
        pytest.skip("no pacemaker venv")
    t = pathlib.Path(__file__).with_name("test_bench_pace_dataframes.py")
    r = subprocess.run([str(py), "-m", "pytest", str(t), "-p", "no:cacheprovider", "-q"],
                       capture_output=True, text=True, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0 and "passed" in r.stdout and "skipped" not in r.stdout, r.stdout[-1500:] + r.stderr[-500:]


def test_pace_checkpoint_counts_functions_not_steps(tmp_path):
    import yaml
    from accuracy_cost.fit_pace import checkpoint_for
    for k, c in enumerate([10, 20, 40]):            # files are named by ladder step
        (tmp_path / f"interim_potential_ladder_step_{k}.yaml").write_text(
            yaml.safe_dump({"species": [{"nbody": [{"x": i} for i in range(c)]}]}))
    assert checkpoint_for(tmp_path, 20)[1] == 20
    p, c = checkpoint_for(tmp_path, 25)
    assert p.name.endswith("_2.yaml") and c == 40   # first checkpoint with >= n
    with pytest.raises(FileNotFoundError):
        checkpoint_for(tmp_path, 41)


def test_read_atoms_fallback_matches_extxyz(tmp_path):
    from accuracy_cost import split
    from ace_jax.fit.xyz import read_extxyz
    ats = [a for a in map(split.normalise, read_extxyz(TINY)) if a is not None][:6]
    split._write(tmp_path / "t.xyz", ats)
    a, b = split.read_atoms(tmp_path / "t.xyz"), split._read_plain(tmp_path / "t.xyz")
    for x, y in zip(a, b):
        assert np.allclose(x.positions, y.positions) and np.allclose(x.cell, y.cell) and np.allclose(x.arrays["forces"], y.arrays["forces"])
        assert x.info["energy"] == y.info["energy"] and x.info["config_type"] == y.info["config_type"]
        assert ("virial" in x.info) == ("virial" in y.info)
        if "virial" in x.info:
            assert np.allclose(x.info["virial"], y.info["virial"])


def test_pace_with_e0_rewrites_the_yace_e0_line(tmp_path):
    from accuracy_cost.fit_pace import with_e0
    p = tmp_path / "m.yace"; p.write_text("elements: [Si]\nE0: [0]\ndeltaSplineBins: 0.001\n")
    with_e0(p)
    assert p.read_text().splitlines()[1] == "E0: [-158.54496821]" and "deltaSplineBins" in p.read_text()


def test_n_params_gp_requires_ace_keys(tmp_path):
    import pytest
    from accuracy_cost.fit_ace import n_params
    p = tmp_path / "gp.npz"
    np.savez(p, **{"ace/WB": np.zeros((2, 3)), "ace/Wpair": np.zeros(4), "other": np.zeros(9)})
    assert n_params("ace_gp", p) == 10
    np.savez(p, other=np.zeros(9))
    with pytest.raises(ValueError, match="WB"):
        n_params("ace_gp", p)


def test_pace_convert_dedupes_by_function_count(tmp_path, monkeypatch, capsys):
    import yaml
    from accuracy_cost import fit_pace
    for k, c in enumerate([50, 60, 150]):            # steps overshoot: take 100 and 150 both land on 150
        (tmp_path / f"interim_potential_ladder_step_{k}.yaml").write_text(
            yaml.safe_dump({"species": [{"nbody": [{"x": i} for i in range(c)]}]}))
    (tmp_path / "output_potential.yaml").write_text(yaml.safe_dump({"species": [{"nbody": [{}] * 200}]}))
    rows = fit_pace.plan([50, 100, 150, 200], tmp_path, 200)
    assert [(n, c) for n, _, c in rows] == [(50, 50), (100, 150), (200, 200)]
    assert "duplicate of take 100" in capsys.readouterr().out


def test_mace_command_shares_cutoff_e0_and_weights(tmp_path):
    from accuracy_cost.fit_mace import command
    c = command({"channels": 16, "max_L": 0}, tmp_path)
    s = " ".join(c)
    assert "--r_max=5.0" in s and "--E0s={14: -158.54496821}" in s and "--config_type_weights=" in s
    assert "--num_channels=16" in s and "--max_L=0" in s and "--num_interactions=2" in s
    assert "--correlation=3" in s and "--max_ell=3" in s
    assert "--virials_key=virial" in s and "--loss=virials" in s


GAP18 = ("at_file=si_iter6_database.xyz gap={soap l_max=12 n_max=10 atom_sigma=0.5 zeta=4 cutoff=5.0 "
         "cutoff_transition_width=1.0 central_weight=1.0 n_sparse=9000 delta=3.0 f0=0.0 "
         "covariance_type=dot_product sparse_method=cur_points} default_sigma={0.001 0.1 0.05 0.0} "
         "config_type_sigma={liq:0.003:0.15:0.2:0.0} energy_parameter_name=dft_energy "
         "force_parameter_name=dft_force virial_parameter_name=dft_virial config_type_parameter_name=config_type "
         "sparse_jitter=1.0e-8 e0_offset=2.0 core_param_file=glue.xml core_ip_args={IP Glue} gp_file=gp_iter6_sparse9k.xml")


def test_gap_refit_changes_only_data_size_and_labels():
    from accuracy_cost.fit_gap import refit_command
    c = refit_command(GAP18, 1000, "/d/train.xyz", "gap.xml")
    assert "at_file=/d/train.xyz" in c and "n_sparse=1000" in c and "gp_file=gap.xml" in c
    assert "energy_parameter_name=energy" in c and "force_parameter_name=forces" in c
    assert "virial_parameter_name=virial" in c and "n_sparse=9000" not in c
    for keep in ("l_max=12", "cutoff=5.0", "zeta=4", "default_sigma={0.001 0.1 0.05 0.0}",
                 "core_param_file=glue.xml", "core_ip_args={IP Glue}", "e0_offset=2.0", "sparse_method=cur_points"):
        assert keep in c


def test_xml_label_reads_top_level_potential(tmp_path):
    from accuracy_cost.fit_gap import xml_label
    p = tmp_path / "g.xml"
    p.write_text('<Potential label="GAP_X" init_args="Sum init_args_pot1={IP Glue} init_args_pot2={IP GAP label=GAP_Y}"/>\n<GAP_params label="GAP_Y"/>')
    assert xml_label(p) == "GAP_X"


def test_gap_fit_exe_prefers_interpreter_venv(tmp_path, monkeypatch):
    from accuracy_cost import fit_gap
    (tmp_path / "gap_fit").write_text("")
    monkeypatch.setattr(fit_gap.sys, "executable", str(tmp_path / "python"))
    assert fit_gap.gap_fit_exe() == str(tmp_path / "gap_fit")
    (tmp_path / "gap_fit").unlink()
    assert fit_gap.gap_fit_exe() == "gap_fit"


# --- Task 10: timing and figures -------------------------------------------------

def test_plateau_flag():
    from accuracy_cost.timing import plateau_flag
    main, check = {"n_atoms": 32768, "step_s": 1.0}, {"n_atoms": 4096, "step_s": 0.125}
    assert plateau_flag(main, check) is False
    assert plateau_flag(main, {"n_atoms": 4096, "step_s": 0.25}) is True     # check is 2x slower per atom


def _fake_runs(tmp_path, monkeypatch):
    import json as _json
    from accuracy_cost import common
    monkeypatch.setattr(common, "RUNS", tmp_path / "runs")
    st = {("ace", "o3d8"): ("model.npz", {}), ("ace_lr", "o3d8"): ("model.npz", {}),
          ("ace_gp", "o3d8-m100"): ("gp_model.npz", {}), ("pace", "f50"): ("model.yace", {}),
          ("mace", "c8L0"): ("model.model", {}), ("gap", "s1000"): ("gap.xml", {"quip_label": "GAP_L"}),
          ("ace", "o3d10"): ("model.npz", {"status": "failed"})}
    for (fam, size), (name, extra) in st.items():
        d = common.run_dir(fam, size); d.mkdir(parents=True)
        (d / "status.json").write_text(_json.dumps({"family": fam, "size": size, "status": "ok",
                                                   "model": str(d / name), "n_params": 7, **extra}))


def test_cases_follow_the_code_table(tmp_path, monkeypatch):
    from accuracy_cost.timing import cases
    _fake_runs(tmp_path, monkeypatch)
    gpu = {(c["row"]["family"], c["row"]["code"], c["mode"], c["dtype"]) for c in cases("gpu")}
    cpu = {(c["row"]["family"], c["row"]["code"], c["mode"], c["dtype"]) for c in cases("cpu")}
    XL = {("mace", "mace-xl", "lammps", "float64"), ("mace", "mace-xl-f32", "lammps", "float32")}
    assert gpu == {("ace", "acejax-ace", "lammps", "float64"), ("ace_lr", "acejax-ace", "lammps", "float64"),
                   ("ace_gp", "acejax-gp", "standalone", "float64"), ("pace", "acejax-pace", "lammps", "float64"),
                   ("pace", "mlpace", "lammps", "float64"), ("mace", "mace", "lammps", "float64"),
                   ("mace", "mace", "standalone", "float32"), *XL}                   # no gap on the GPU
    assert cpu == {("ace", "acejax-ace", "standalone", "float64"), ("ace_lr", "acejax-ace", "standalone", "float64"),
                   ("ace", "acepotentials-trim", "lammps", "float64"),               # CPU only
                   ("ace_gp", "acejax-gp", "standalone", "float64"),
                   ("pace", "acejax-pace", "standalone", "float64"), ("pace", "mlpace", "lammps", "float64"),
                   ("mace", "mace", "lammps", "float64"), ("mace", "mace", "standalone", "float32"),
                   ("gap", "gap", "lammps", "float64"), *XL}
    by = {(c["row"]["family"], c["mode"], c["dtype"]): c["row"] for c in cases("cpu")}
    assert by[("gap", "lammps", "float64")]["quip_label"] == "GAP_L"
    assert by[("mace", "standalone", "float32")]["path"].endswith("model-f32.model")
    assert by[("mace", "lammps", "float64")]["symmetrix"].endswith("c8L0/model.json")
    assert all(c["row"]["size"] != "o3d10" for c in cases("cpu"))                      # failed fits are skipped
    assert all(c["row"]["family"] != "gap18_published" for c in cases("cpu"))
    assert [c["row"]["family"] for c in cases("cpu", only="ace/o3d8")] == ["ace", "ace"]       # ace-jax, trim


def test_mace_xl_rows_name_the_xl_model_its_lammps_and_a_cpu_reference(tmp_path, monkeypatch):
    """Symmetrix-XL: model-xl.json (compact v2) in its own LAMMPS (`lmp_xl`); the gate's PyTorch reference is
    the CPU copy of the fit on a GPU-less node (model.model may hold CUDA tensors), model.model on the GPU."""
    from accuracy_cost import timing
    from accuracy_cost.common import run_dir
    _fake_runs(tmp_path, monkeypatch)
    xl = lambda dev: {c["row"]["code"]: c["row"] for c in timing.cases(dev, only="mace") if c["row"]["code"] in timing.XL_CODES}
    assert xl("cpu")["mace-xl"]["path"].endswith("c8L0/model.model")              # no CPU copy yet
    (run_dir("mace", "c8L0") / "model-cpu.model").write_text("")
    cpu, gpu = xl("cpu"), xl("gpu")
    assert cpu["mace-xl"]["path"].endswith("model-cpu.model") and cpu["mace-xl-f32"]["path"].endswith("model-cpu.model")
    assert gpu["mace-xl"]["path"].endswith("c8L0/model.model") and gpu["mace-xl"]["symmetrix_xl"].endswith("c8L0/model-xl.json")
    env = {"lmp": "/old/lmp", "lmp_xl": "/xl/lmp", "pjrt": "/p.so"}
    assert timing.lammps_for("mace-xl-f32", "cpu", env) == ("/xl/lmp", None) and timing.lammps_for("mace", "gpu", env)[0] == "/old/lmp"
    assert timing.STYLE["mace-xl-f32"] == "mace-xl-f32"                              # its own (looser) gate
    g = timing.gate_model(cpu["mace-xl"], "cpu", env, tmp_path / "g")                # no model-xl.json: unsupported
    assert g["status"] == "unsupported" and "model-xl.json" in g["error"]


def test_main_drops_mace_xl_without_its_lammps(tmp_path, monkeypatch):
    import json as _json
    from accuracy_cost import timing
    _fake_runs(tmp_path, monkeypatch)
    got = []
    monkeypatch.setattr(timing, "run_cases", lambda dev, cs, *a, **k: got.append({c["row"]["code"] for c in cs}))
    monkeypatch.setattr(timing, "subprocess_runner", lambda env, dev: (None, None))
    monkeypatch.setattr(timing, "wait_idle", lambda **k: {})
    for env in ({"lmp": "/l"}, {"lmp": "/l", "lmp_xl": "/x"}):
        (tmp_path / "env.json").write_text(_json.dumps(env))
        timing.main(["gpu", "--only", "mace", "--env", str(tmp_path / "env.json"), "--results", str(tmp_path / "r.jsonl")])
    timing.main(["gpu", "--only", "mace", "--env", str(tmp_path / "env.json"), "--results", str(tmp_path / "r.jsonl"),
                 "--codes", "mace-xl"])
    assert got == [{"mace"}, {"mace", "mace-xl", "mace-xl-f32"}, {"mace-xl"}]


def test_trim_rows_name_their_library_lammps_and_plugin(tmp_path, monkeypatch):
    """acepotentials-trim: <run dir>/trim/libace.so in the PR 309 LAMMPS (`lmp_ace`, `ace_plugin`), gated
    against the ace-jax fit; main drops the rows on a host whose env json has no `lmp_ace`."""
    import json as _json
    from accuracy_cost import timing
    from accuracy_cost.common import run_dir
    _fake_runs(tmp_path, monkeypatch)
    row = next(c["row"] for c in timing.cases("cpu", only="ace/o3d8") if c["row"]["code"] == timing.TRIM)
    assert row["trim_lib"] == str(run_dir("ace", "o3d8") / "trim" / "libace.so")
    env = {"lmp": "/l", "lmp_ace": "/ace/lmp", "ace_plugin": "/p/aceplugin.so"}
    assert timing.lammps_for(timing.TRIM, "cpu", env) == ("/ace/lmp", None)
    got = []
    monkeypatch.setattr(timing, "run_cases", lambda dev, cs, *a, **k: got.append({c["row"]["code"] for c in cs}))
    monkeypatch.setattr(timing, "subprocess_runner", lambda env, dev: (None, None))
    monkeypatch.setattr(timing, "wait_idle", lambda **k: {})
    for e in ({"lmp": "/l"}, env):
        (tmp_path / "env.json").write_text(_json.dumps(e))
        timing.main(["cpu", "--only", "ace/o3d8", "--env", str(tmp_path / "env.json"), "--results", str(tmp_path / "r.jsonl")])
    assert got == [{"acejax-ace"}, {"acejax-ace", timing.TRIM}]


def test_trim_record_reads_the_build_sidecar(tmp_path):
    from scaling.run_lammps import _trim_build, _trim_status
    so = tmp_path / "libace.so"
    assert _trim_status({"trim_lib": str(so)})[0] == "error"                         # never built
    so.with_suffix(".build.json").write_text('{"status": "unsupported", "unsupported": "refused"}')
    assert _trim_status({"trim_lib": str(so)}) == ("unsupported", f"trim export failed for {so}: refused")
    so.write_bytes(b""); so.with_suffix(".build.json").write_text('{"build_id": "ab", "sha256": "cd", "gates": {}}')
    assert _trim_status({"trim_lib": str(so)}) is None and _trim_build({"trim_lib": str(so)}) == {"build_id": "ab", "sha256": "cd"}


def test_cases_refuse_gap_on_the_gpu(monkeypatch):
    from accuracy_cost import timing
    monkeypatch.setitem(timing.PLAN, "gap", [("gap", {"gpu": "lammps"}, ("float64",))])
    with pytest.raises(ValueError, match="CPU only"):
        timing.cases("gpu")


def test_idle_reads_load_and_gpu_processes(monkeypatch):
    from accuracy_cost import timing
    monkeypatch.setattr(timing, "_load1", lambda: 0.5)
    monkeypatch.setattr(timing, "_gpu_procs", lambda: 0)
    assert timing.idle() == {"ok": True, "load1": 0.5, "gpu_procs": 0}
    monkeypatch.setattr(timing, "_gpu_procs", lambda: 1)
    assert timing.idle()["ok"] is False
    monkeypatch.setattr(timing, "_load1", lambda: 9.0); monkeypatch.setattr(timing, "_gpu_procs", lambda: 0)
    assert timing.idle()["ok"] is False


def _case(code="acejax-ace", mode="lammps", fam="ace"):
    return {"row": {"code": code, "family": fam, "size": "o3d8", "name": f"{fam}/o3d8/{code}", "n_params": 62},
            "mode": mode, "dtype": "float64"}


def test_run_times_gated_cases_and_records_the_plateau(tmp_path):
    import json as _json
    from accuracy_cost.timing import run_cases
    calls = []

    def runner(c, n, work, prev):
        calls.append((n, prev and prev["n_atoms"]))
        return {"code": c["row"]["code"], "n_atoms": n, "status": "ok", "step_s": n * 1e-4 * (1 + len(calls) % 3)}
    res = tmp_path / "t.jsonl"
    gate = lambda c: {"status": "parity_ok", "gate": "acejax"}
    run_cases("gpu", [_case()], runner, gate, res, n=(64, 8), repeats=3, work=tmp_path,
              meta={"load1": 0.1, "gpu_procs": 0, "forced_busy": False})
    rows = [_json.loads(l) for l in res.read_text().splitlines()]
    assert [r["mode"] for r in rows] == ["parity", "lammps"]
    r = rows[1]
    assert r["status"] == "ok" and r["family"] == "ace" and r["n_params"] == 62 and r["n_atoms"] == 64
    assert r["step_s"] == pytest.approx(64 * 2e-4) and len(r["step_s_reps"]) == 3       # median of 1x, 2x, 3x
    assert r["atom_steps_per_s"] == pytest.approx(64 / r["step_s"]) and r["not_plateau"] is False
    assert r["forced_busy"] is False and r["parity"] == "parity_ok"
    assert [c[0] for c in calls] == [8, 8, 8, 64, 64, 64] and calls[3][1] == 8          # check first; sizes steps


def test_run_skips_a_failed_gate(tmp_path):
    import json as _json
    from accuracy_cost.timing import run_cases
    res = tmp_path / "t.jsonl"
    runner = lambda *a: pytest.fail("timed an ungated model")
    run_cases("gpu", [_case()], runner, lambda c: {"status": "parity_fail", "max_dF": 1.0}, res, n=(64, 8),
              repeats=1, work=tmp_path, meta={})
    rows = [_json.loads(l) for l in res.read_text().splitlines()]
    assert rows[-1]["status"] == "parity_failed" and "atom_steps_per_s" not in rows[-1]


def test_standalone_rows_take_call_s_and_need_no_lammps_gate(tmp_path):
    import json as _json
    from accuracy_cost.timing import run_cases
    res = tmp_path / "t.jsonl"
    runner = lambda c, n, w, p: {"code": "acejax-ace", "n_atoms": n, "status": "ok", "call_s": n * 1e-3}
    run_cases("cpu", [_case(mode="standalone")], runner, lambda c: pytest.fail("gated a standalone row"), res,
              n=(64, 8), repeats=1, work=tmp_path, meta={})
    r = [_json.loads(l) for l in res.read_text().splitlines()][-1]
    assert r["status"] == "ok" and r["step_s"] == pytest.approx(0.064) and r["parity"] == "not_applicable"


def test_main_refuses_a_busy_box_unless_forced(monkeypatch, tmp_path):
    from accuracy_cost import timing
    monkeypatch.setattr(timing, "idle", lambda: {"ok": False, "load1": 9.0, "gpu_procs": 1})
    monkeypatch.setattr(timing, "MAX_WAIT_S", 0)
    with pytest.raises(SystemExit, match="busy"):
        timing.main(["cpu", "--results", str(tmp_path / "t.jsonl")])
    seen = {}
    monkeypatch.setattr(timing, "cases", lambda device, only=None: [])
    monkeypatch.setattr(timing, "run_cases", lambda *a, **kw: seen.update(kw))
    timing.main(["cpu", "--force-busy", "--n", "64", "8", "--results", str(tmp_path / "t.jsonl")])
    assert seen["meta"]["forced_busy"] is True and seen["n"] == (64, 8)
    assert seen["ready"]() == {"load1": 9.0, "gpu_procs": 1}             # sampled per case, never waits


def test_main_takes_a_bare_device_string(monkeypatch):
    from accuracy_cost import timing
    seen = {}
    monkeypatch.setattr(timing, "idle", lambda: {"ok": True, "load1": 0.1, "gpu_procs": 0})
    monkeypatch.setattr(timing, "cases", lambda device, only=None: seen.setdefault("device", device) and [])
    monkeypatch.setattr(timing, "run_cases", lambda *a, **kw: None)
    timing.main("gpu")
    assert seen["device"] == "gpu"


class _Clock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s); self.t += s


def test_wait_idle_polls_until_the_box_is_free():
    from accuracy_cost.timing import wait_idle
    seq = iter([{"ok": False, "load1": 9.0, "gpu_procs": 0}] * 2 + [{"ok": True, "load1": 0.5, "gpu_procs": 0}])
    clk = _Clock()
    s = wait_idle(lambda: next(seq), clock=clk, sleep=clk.sleep, poll=60, max_wait=1800)
    assert s == {"load1": 0.5, "gpu_procs": 0, "waited_s": 120.0} and clk.slept == [60, 60]


def test_wait_idle_stops_after_max_wait_naming_the_load():
    from accuracy_cost.timing import wait_idle
    clk = _Clock()
    with pytest.raises(SystemExit, match=r"busy.*load1 9\.0.*gpu_procs 2"):
        wait_idle(lambda: {"ok": False, "load1": 9.0, "gpu_procs": 2}, clock=clk, sleep=clk.sleep,
                  poll=60, max_wait=1800)
    assert sum(clk.slept) == 1800


def test_wait_idle_forced_samples_without_waiting():
    from accuracy_cost.timing import wait_idle
    clk = _Clock()
    s = wait_idle(lambda: {"ok": False, "load1": 9.0, "gpu_procs": 2}, clock=clk, sleep=clk.sleep, force=True)
    assert s == {"load1": 9.0, "gpu_procs": 2} and clk.slept == []


def test_gpu_procs_excludes_this_drivers_own_children(monkeypatch):
    from accuracy_cost import timing
    monkeypatch.setattr(timing, "_nvidia_pids", lambda: [111, 222, 333])
    monkeypatch.setattr(timing, "_own_pids", lambda: {222})
    assert timing._gpu_procs() == 2


def test_each_case_and_gate_records_its_own_idle_sample(tmp_path):
    import json as _json
    from accuracy_cost.timing import run_cases
    n = iter(range(100))
    ready = lambda: {"load1": float(next(n)), "gpu_procs": 0}
    runner = lambda c, k, w, p: {"n_atoms": k, "status": "ok", "step_s": k * 1e-4}
    res = tmp_path / "t.jsonl"
    run_cases("gpu", [_case(), _case(fam="ace_lr")], runner, lambda c: {"status": "parity_ok"}, res, n=(64, 8),
              repeats=1, work=tmp_path, meta={"forced_busy": False}, ready=ready)
    rows = [_json.loads(l) for l in res.read_text().splitlines()]
    assert [r["load1"] for r in rows] == [0.0, 1.0, 2.0, 3.0]               # gate, case, gate, case


def test_unsupported_gate_is_recorded_and_not_regated(tmp_path):
    import json as _json
    from accuracy_cost.timing import run_cases
    res, gates = tmp_path / "t.jsonl", []

    def gate(c):
        gates.append(c)
        return {"status": "unsupported", "error": "no Symmetrix model"}
    for _ in range(2):
        run_cases("gpu", [_case(code="mace", fam="mace")], lambda *a: pytest.fail("timed"), gate, res,
                  n=(64, 8), repeats=1, work=tmp_path, meta={})
    rows = [_json.loads(l) for l in res.read_text().splitlines()]
    assert len(gates) == 1 and rows[-1]["status"] == "unsupported" and rows[-1]["mode"] == "lammps"


def test_failed_check_marks_plateau_unknown_and_main_reuses_the_check_layout(tmp_path):
    import json as _json
    from accuracy_cost.timing import run_cases
    prevs = []

    def runner(c, n, w, prev):
        prevs.append(prev)
        return {"n_atoms": n, "status": "ok", "step_s": 1e-3, "layout": "dense", "matrix_overflow": True}
    res = tmp_path / "t.jsonl"
    run_cases("gpu", [_case()], runner, lambda c: {"status": "parity_ok"}, res, n=(64, 8), repeats=1,
              work=tmp_path, meta={})
    r = [_json.loads(l) for l in res.read_text().splitlines()][-1]
    assert prevs[1] == {"step_s": 1e-3, "n_atoms": 8, "layout": "dense", "matrix_overflow": True}
    assert r["check_layout"] == "dense"
    fail = lambda c, n, w, prev: {"n_atoms": n, "status": "oom"} if n == 8 else \
        {"n_atoms": n, "status": "ok", "step_s": 1e-3}
    run_cases("gpu", [_case(fam="ace_lr")], fail, lambda c: {"status": "parity_ok"}, res, n=(64, 8), repeats=1,
              work=tmp_path, meta={})
    r = [_json.loads(l) for l in res.read_text().splitlines()][-1]
    assert r["status"] == "ok" and r["not_plateau"] == "unknown" and r["check_status"] == "oom"


def _synthetic():
    m = [{"family": "ace", "size": s, "rmse_F": f, "n_params": n} for s, f, n in
         (("a", 0.20, 50), ("b", 0.12, 200), ("c", 0.08, 800))]
    m += [{"family": "mace", "size": "m1", "rmse_F": 0.15, "n_params": 9000}]
    t = [{"family": r["family"], "size": r["size"], "device": "gpu", "n_atoms": 32768, "status": "ok",
          "atom_steps_per_s": 1e7 / r["n_params"] ** 0.5, "not_plateau": False} for r in m]
    return m, t


def test_smallest_at_target_picks_cheapest_meeting_model():
    from accuracy_cost.plot import smallest_at_target
    m, t = _synthetic()
    rows = {r["family"]: r for r in smallest_at_target(m, t, 0.13, "gpu")}
    assert rows["ace"]["size"] == "b"


def test_smallest_at_target_not_reached():
    from accuracy_cost.plot import smallest_at_target
    m, t = _synthetic()
    rows = {r["family"]: r for r in smallest_at_target(m, t, 0.10, "gpu")}
    assert rows["mace"]["size"] is None and rows["mace"]["note"] == "not reached"


def test_smallest_at_target_ties_go_to_the_faster_and_reference_rows_are_skipped():
    from accuracy_cost.plot import smallest_at_target
    m, t = _synthetic()
    m += [{"family": "ace", "size": "b2", "rmse_F": 0.11, "n_params": 200},
          {"family": "gap18_published", "size": "published", "rmse_F": 0.08, "n_params": 9000}]
    t += [{"family": "ace", "size": "b2", "device": "gpu", "status": "ok", "atom_steps_per_s": 1e9}]
    rows = {r["family"]: r for r in smallest_at_target(m, t, 0.13, "gpu")}
    assert rows["ace"]["size"] == "b2" and "gap18_published" not in rows
    rows = {r["family"]: r for r in smallest_at_target(m, t, 0.13, "cpu")}      # nothing timed on the CPU
    assert rows["ace"]["size"] == "b" and rows["ace"]["atom_steps_per_s"] is None
    assert rows["ace"]["note"] == "not timed"


def test_reference_lines_merge_metrics_and_refs_json():
    from accuracy_cost.plot import reference_lines
    m, _ = _synthetic()
    m.append({"family": "gap18_published", "size": "published", "rmse_F": 0.0825, "n_params": 9000})
    assert reference_lines(m, []) == [("GAP-18 published", 0.0825)]
    assert reference_lines(m, [["GAP-18 published", 0.08], ["GAP refit s9000", 0.09]]) == \
        [("GAP-18 published", 0.08), ("GAP refit s9000", 0.09)]


def test_pareto_and_heatmap_render(tmp_path):
    from accuracy_cost.plot import heatmap, pareto
    m, t = _synthetic()
    for r in m:
        r["per_type"] = {"dia": {"rmse_F": r["rmse_F"]}, "liq": {"rmse_F": 2 * r["rmse_F"]}}
    t[0]["not_plateau"], t[1]["not_plateau"] = True, "unknown"
    pareto(m, t, [0.1, 0.15], tmp_path / "p.png"); heatmap(m, [("ace", "b"), ("mace", "m1")], tmp_path / "h.png")
    assert (tmp_path / "p.png").stat().st_size > 0 and (tmp_path / "h.png").stat().st_size > 0


def test_pareto_marks_unknown_plateau_open_and_says_so():
    from accuracy_cost.plot import plateau_markers
    rows = [{"not_plateau": False}, {"not_plateau": True}, {"not_plateau": "unknown"}, {}]
    assert plateau_markers(rows) == ["clean", "below", "unknown", "unknown"]


def test_run_root_overrides_and_status_stamp(tmp_path, monkeypatch):
    import importlib
    import json
    import accuracy_cost.common as C
    monkeypatch.setenv("ACEJAX_AC_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("ACEJAX_AC_RUNS", str(tmp_path / "smoke"))
    monkeypatch.setenv("ACEJAX_AC_RESULTS", str(tmp_path / "res"))
    monkeypatch.setenv("ACEJAX_AC_HOST", "modal"); monkeypatch.setenv("ACEJAX_AC_GPU", "NVIDIA A100-SXM4-40GB")
    try:
        importlib.reload(C)
        assert C.DATA == tmp_path / "root" / "data" / "si_accuracy_cost"     # data stays under the root
        assert C.RUNS == tmp_path / "smoke" and C.RESULTS == tmp_path / "res"
        C.write_status(C.run_dir("mace", "x"), status="ok")
        st = json.loads((tmp_path / "smoke" / "mace" / "x" / "status.json").read_text())
        assert st["host"] == "modal" and st["gpu"] == "NVIDIA A100-SXM4-40GB"
        C.write_status(C.run_dir("mace", "y"), status="ok", host="kept")     # an explicit host wins
        assert C.read_status(C.run_dir("mace", "y"))["host"] == "kept"
    finally:
        for k in ("ACEJAX_AC_ROOT", "ACEJAX_AC_RUNS", "ACEJAX_AC_RESULTS", "ACEJAX_AC_HOST", "ACEJAX_AC_GPU"):
            monkeypatch.delenv(k)
        importlib.reload(C)
    assert C.RUNS == C.ROOT / "runs" / "accuracy_cost"


class _Entry:
    def __init__(self, path, size, mtime, kind="FILE"):
        import types
        self.path, self.size, self.mtime, self.type = path, size, mtime, types.SimpleNamespace(name=kind)


class _FakeVolume:
    """modal.Volume's listdir/read_file_into_fileobj over a dict {path: bytes}."""
    def __init__(self, files, mtime=1_700_000_000):
        self.files, self.mtime, self.reads = files, mtime, []

    def listdir(self, path, recursive=False):
        hits = [_Entry(p, len(b), self.mtime) for p, b in self.files.items() if p.lstrip("/").startswith(path + "/")]
        if not hits:
            raise FileNotFoundError(path)
        return hits + [_Entry(path + "/d", 0, 0, "DIRECTORY")]

    def read_file_into_fileobj(self, path, f):
        self.reads.append(path); f.write(self.files.get(path, self.files.get("/" + path)))


def test_collect_downloads_relocates_and_reruns(tmp_path):
    import json
    from accuracy_cost.modal_collect import collect, statuses
    st = {"status": "ok", "host": "modal", "gpu": "A100", "model": "/vol/runs/ac_x/mace/c8L0/model.model",
          "fit_wall_s": 12.0, "note": "/vol/elsewhere"}
    vol = _FakeVolume({"runs/ac_x/mace/c8L0/status.json": json.dumps(st).encode(),
                       "runs/ac_x/mace/c8L0/model.model": b"weights",
                       "/runs/ac_x/pace/f10/model.yace": b"yace"})          # a leading / is tolerated
    dest = collect(vol, "runs/ac_x", dest=tmp_path / "local", log=lambda *a: None)
    got = statuses(dest)
    assert got["mace/c8L0"]["model"] == str(tmp_path / "local" / "mace" / "c8L0" / "model.model")
    assert got["mace/c8L0"]["note"] == "/vol/elsewhere" and got["mace/c8L0"]["gpu"] == "A100"
    assert (dest / "pace" / "f10" / "model.yace").read_bytes() == b"yace"
    vol.reads.clear()
    collect(vol, "runs/ac_x", dest=tmp_path / "local", log=lambda *a: None)        # unchanged files are skipped
    assert vol.reads == ["runs/ac_x/mace/c8L0/status.json"]
    assert statuses(dest)["mace/c8L0"]["model"].startswith(str(tmp_path))
    class Vanishing(_FakeVolume):                         # listed, then deleted before the read
        def read_file_into_fileobj(self, path, f):
            if path.endswith(".pt"):
                raise FileNotFoundError(path)
            super().read_file_into_fileobj(path, f)
    v2 = Vanishing({**vol.files, "runs/ac_x/mace/c8L0/ckpt/e1.pt": b"x"})
    collect(v2, "runs/ac_x", dest=tmp_path / "v2", log=lambda *a: None)
    assert (tmp_path / "v2" / "mace" / "c8L0" / "model.model").exists() and not list((tmp_path / "v2").rglob("*.part"))
    vol.files["runs/ac_x/mace/c8L0/model.model"] = b"stage2!"; vol.mtime += 60       # same size, newer: re-read
    collect(vol, "runs/ac_x", dest=tmp_path / "local", log=lambda *a: None)
    assert (dest / "mace" / "c8L0" / "model.model").read_bytes() == b"stage2!"


def test_modal_mace_sizes_match_the_sweep():
    from accuracy_cost.common import load_sweep
    from accuracy_cost.modal_collect import MACE_SIZES
    # the one-layer sizes (interactions: 1) are trained on Sulis, not Modal
    assert list(MACE_SIZES) == [e["size"] for e in load_sweep()["mace"] if "interactions" not in e]


def test_mace_overrides_subset_and_restart(tmp_path, monkeypatch):
    import ase
    from accuracy_cost import fit_mace
    from accuracy_cost.split import _write, read_atoms
    ats = []
    for i in range(10):
        a = ase.Atoms("Si", positions=[[0, 0, 0]], cell=np.eye(3) * 5, pbc=True)
        a.info.update(energy=-float(i), config_type="dia"); a.arrays["forces"] = np.zeros((1, 3)); ats.append(a)
    (tmp_path / "data").mkdir(); _write(tmp_path / "data" / "train.xyz", ats)
    monkeypatch.setattr(fit_mace, "DATA", tmp_path / "data")
    p = fit_mace.subset(tmp_path / "run", 3)
    assert [a.info["energy"] for a in read_atoms(p)] == [0.0, -3.0, -6.0]
    c = fit_mace.command({"channels": 8, "max_L": 0, "epochs": 2, "train_file": p}, tmp_path / "run")
    assert f"--train_file={p}" in c and "--max_num_epochs=2" in c and "--restart_latest" in c


def test_pace_resume_spans_segments_and_sums_wall(tmp_path):
    import os
    import yaml
    from accuracy_cost import fit_pace
    pot = lambda c: yaml.safe_dump({"species": [{"nbody": [{"x": i} for i in range(c)]}]})
    for k, c in enumerate([50, 100]):                 # the interrupted first segment
        (tmp_path / f"interim_potential_ladder_step_{k}.yaml").write_text(pot(c))
    (tmp_path / "resume1").mkdir()
    for k, c in enumerate([150, 200]):                # its resumption restarts the step numbering
        (tmp_path / "resume1" / f"interim_potential_ladder_step_{k}.yaml").write_text(pot(c))
    assert [c for _, c in fit_pace.checkpoints(tmp_path)] == [50, 100, 150, 200]
    p, c = fit_pace.checkpoint_for(tmp_path, 120)
    assert c == 150 and p == tmp_path / "resume1" / "interim_potential_ladder_step_0.yaml"
    inp = fit_pace.resume_input({"max_functions": 800, "ladder_step": 50}, tmp_path, p)
    assert inp["potential"]["initial_potential"] == str(p)
    assert inp["data"]["filename"] == str(tmp_path / "train.pckl.gzip")
    assert inp["fit"]["weighting"]["filename"] == str(tmp_path / "weights.pckl.gzip")
    assert inp["potential"]["functions"] == fit_pace.input_yaml({"max_functions": 800, "ladder_step": 50})["potential"]["functions"]
    for s, (t0, t1) in ((tmp_path, (1000, 1600)), (tmp_path / "resume1", (2000, 2300))):
        (s / "cmd.txt").write_text("x"); (s / "fit.log").write_text("x")
        os.utime(s / "cmd.txt", (t0, t0)); os.utime(s / "fit.log", (t1, t1))
    assert fit_pace.segment_wall(tmp_path) == 900


def test_symmetrix_waits_for_finish_and_reexports_a_stale_json(tmp_path):
    import json
    import os
    from accuracy_cost.fit_mace import symmetrix_stale
    from accuracy_cost.modal_collect import complete, symmetrix_due
    d = tmp_path / "mace" / "c8L0"; d.mkdir(parents=True)
    (d / "model.model").write_text("stage one")
    (d / "status.json").write_text(json.dumps({"status": "ok"}))          # fit_once done, finish() not yet
    assert not complete("mace", json.loads((d / "status.json").read_text()))
    assert symmetrix_due(tmp_path, ["c8L0"], log=lambda *a: None) == []
    (d / "status.json").write_text(json.dumps({"status": "ok", "n_params": 10}))
    assert symmetrix_due(tmp_path, ["c8L0"], log=lambda *a: None) == ["c8L0"]
    (d / "model.json").write_text("{}"); os.utime(d / "model.model", (1000, 1000)); os.utime(d / "model.json", (2000, 2000))
    assert symmetrix_due(tmp_path, ["c8L0"], log=lambda *a: None) == [] and not symmetrix_stale(d)
    os.utime(d / "model.model", (3000, 3000))                                 # the stage-two swap landed later
    assert symmetrix_due(tmp_path, ["c8L0"], log=lambda *a: None) == ["c8L0"] and symmetrix_stale(d)
    assert complete("pace", {"status": "ok"})


def test_collect_reads_tolerate_half_written_files(tmp_path, capsys):
    import json
    from accuracy_cost import fit_pace
    from accuracy_cost.modal_collect import relocate, statuses
    (tmp_path / "mace" / "a").mkdir(parents=True); (tmp_path / "mace" / "b").mkdir()
    (tmp_path / "mace" / "a" / "status.json").write_text('{"status": "ok", "model": "/vol/r')     # truncated
    (tmp_path / "mace" / "b" / "status.json").write_text(json.dumps({"status": "ok", "model": "/vol/r/m"}))
    warn = []
    assert relocate(tmp_path / "mace" / "a" / "status.json", "/vol/r", "/l", warn.append) is False
    assert relocate(tmp_path / "mace" / "b" / "status.json", "/vol/r", "/l", warn.append) is True
    assert list(statuses(tmp_path, log=warn.append)) == ["mace/b"] and len(warn) == 2
    assert not list(tmp_path.rglob("*.tmp"))
    lad = tmp_path / "ladder"; lad.mkdir()
    (lad / "interim_potential_ladder_step_0.yaml").write_text("species:\n- nbody: [{x: 1}]\n")
    (lad / "interim_potential_ladder_step_1.yaml").write_text("species:\n- nbody: [{x: 1}, {x")      # truncated
    assert [c for _, c in fit_pace.checkpoints(lad)] == [1]
    assert "skipping unreadable checkpoint" in capsys.readouterr().out


def test_write_status_is_atomic(tmp_path):
    import accuracy_cost.common as C
    C.write_status(tmp_path / "r", status="ok")
    assert C.read_status(tmp_path / "r")["status"] == "ok" and [p.name for p in (tmp_path / "r").iterdir()] == ["status.json"]


def test_pace_resume_refits_when_settings_changed(tmp_path, capsys):
    import json
    from accuracy_cost import fit_pace
    e = {"max_functions": 800, "ladder_step": 50, "n_train": None, "maxiter": 1500}
    d = tmp_path / "ladder"; d.mkdir()
    (d / "interim_potential_ladder_step_0.yaml").write_text("species:\n- nbody: [{x: 1}]\n")
    assert fit_pace.check_settings(d, e) and "no ladder_settings.json" in capsys.readouterr().out   # legacy: kept
    (d / "ladder_settings.json").write_text(json.dumps(fit_pace.ladder_settings(e)))
    assert fit_pace.check_settings(d, e) and fit_pace.checkpoints(d)
    assert not fit_pace.check_settings(d, {**e, "max_functions": 20})                              # moved aside
    assert not fit_pace.checkpoints(d) and d.is_dir()
    assert fit_pace.checkpoints(tmp_path / "ladder.stale1")


def test_non_cpu_tensors_walks_buffers_and_attributes():
    torch = pytest.importorskip("torch")
    from accuracy_cost.fit_mace import non_cpu_tensors
    m = torch.nn.Sequential(torch.nn.Linear(2, 2)); m.register_buffer("b", torch.zeros(1)); m.cache = {"k": [torch.ones(1)]}
    assert non_cpu_tensors(m) == []
    m.cache["k"][0] = torch.empty(1, device="meta"); m[0].weight = torch.nn.Parameter(torch.empty(2, 2, device="meta"))   # meta: not cpu, no GPU
    assert sorted(p for p, _ in non_cpu_tensors(m)) == ["model.0.weight", "model.cache['k'][0]"]


def test_load_calc_scores_the_noglue_control_like_gap(monkeypatch):
    import types
    from accuracy_cost.evaluate import load_calc
    q = types.SimpleNamespace(potential=types.SimpleNamespace(Potential=lambda param_filename: ("quip", param_filename)))
    monkeypatch.setitem(sys.modules, "quippy", q)
    assert load_calc("gap_noglue", "x.xml") == load_calc("gap", "x.xml") == ("quip", "x.xml")
