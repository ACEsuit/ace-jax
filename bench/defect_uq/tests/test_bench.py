"""Bench plumbing tests: ARD_ARMS validity, served-array saving, validate_shape on a synthetic run.

bench/ is outside pyproject's testpaths, so run explicitly:  uv run pytest bench/defect_uq/tests -q
"""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parents[1]


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, HERE / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ard_arms = _load("modal/ard_arms.py", "ard_arms")
served = _load("modal/served_arrays.py", "served_arrays")
vs = _load("scoring/validate_shape.py", "validate_shape")
te = _load("modal/train_extra.py", "train_extra")


def test_arm_names():
    assert set(ard_arms.ARD_ARMS) == {"ard_legacy", "ard_A", "ard_AB", "ard_ABblk", "ard_aniso", "ard_ell2",
                                      "ard_ell4", "ard_ell6", "ard_f1", "ard_f3"}


@pytest.mark.parametrize("arm", sorted(ard_arms.ARD_ARMS))
def test_every_arm_is_a_valid_fitconfig(arm):
    cfg = ard_arms.make_config(arm, dict(model="x.npz", arm="linear", uq="ard"))
    cfg.validate()
    for k, v in ard_arms.ARD_ARMS[arm].items():
        assert getattr(cfg, k) == v


def test_unknown_arm_is_none():
    assert ard_arms.make_config("ard", {}) is None and ard_arms.make_config("ard_c15", {}) is None


class _Calc:
    """Stands in for ACECalculator: serves posterior properties from fixed arrays."""
    def get_property(self, name, at):
        n = len(at)
        return {"forces_std": np.full(n, 0.1), "forces_q": np.full(n, 0.3), "forces_group": np.arange(n) % 3,
                "forces_cov": np.tile(np.eye(3), (n, 1, 1)) * 0.01}[name]


def test_served_arrays_concatenate():
    from ase import Atoms
    ats = [Atoms("H3", positions=np.zeros((3, 3))), Atoms("H2", positions=np.zeros((2, 3)))]
    out = served.collect(_Calc(), ats, ("forces_std", "forces_q", "forces_group", "forces_cov"))
    assert out["sd"].shape == (5,) and out["forces_q"].shape == (5,) and out["forces_cov"].shape == (5, 3, 3)
    assert out["forces_group"].dtype == np.int16 and list(out["forces_group"]) == [0, 1, 2, 0, 1]


def _synthetic_run(tmp, n_cells=60, per=10, aniso=False):
    """cells of `per` atoms (3 rattled copies share cfg//3).  Group 0: all hit; group 1: hit iff cell even."""
    ncfg = n_cells * 3
    cfg = np.repeat(np.arange(ncfg), per)
    n = len(cfg)
    cell = cfg // 3
    grp = np.tile(np.arange(per) % 2, ncfg)                       # alternating groups
    hit = np.where(grp == 0, True, cell % 2 == 0)
    q = np.full(n, 1.0)
    err = np.where(hit, 0.5, 2.0)
    fam = np.where(cell < n_cells // 2, "crack", "edge")
    rc = np.tile(np.linspace(0, 40, per), ncfg)
    d = pathlib.Path(tmp)
    json.dump({"shape": {"variant": "press", "mode": "exact", "ell": 18.75, "K": 100, "K_fit": 80, "rank_R": 7,
                         "lev_p50": 0.1, "lev_p99": 0.5, "lev_max": 0.9, "n_lev_near1": 0},
               "groups": {"alpha": 0.1, "n_min": 20, "lam_rms": [1.0, 2.0], "q": [1.0, 1.0],
                          "n_cfg_val": [30, 10]},
               "transfer": {"f": 0.2, "N_fit": 800, "N": 1000}}, open(d / "ard.json", "w"))
    np.savez(d / "big3_err.npz", err=err, sd=np.full(n, 0.4) + 0.01 * np.arange(n) % 7, family=fam, r_core=rc,
             fixed=np.zeros(n, bool), cfg=cfg, forces_q=q, forces_group=grp.astype(np.int16))
    return d, grp, hit, fam, rc, cell


def test_validate_shape_reproduces_known_coverage(tmp_path):
    d, grp, hit, fam, rc, cell = _synthetic_run(tmp_path)
    R = vs.load_run(d)
    cov = vs.coverage_table(R, B=1000, seed=1)
    row = {r["label"]: r for r in cov}
    assert row["group 0"]["atom"] == pytest.approx(1.0) and row["group 0"]["cell"] == pytest.approx(1.0)
    assert row["group 1"]["atom"] == pytest.approx(hit[grp == 1].mean())
    # group 1 is hit iff cell even -> exactly half the cells
    assert row["group 1"]["cell"] == pytest.approx(0.5)
    lo, hi = row["group 1"]["ci"]
    assert lo <= 0.5 <= hi and hi - lo > 0.05            # the bootstrap resamples whole cells
    assert row["group 0"]["ci"] == (1.0, 1.0)
    # crack tip: crack atoms within 10 A, group-pooled
    tip = (fam == "crack") & (rc <= 10)
    assert row["crack tip"]["atom"] == pytest.approx(hit[tip].mean())


def test_validate_shape_degrades_on_old_run(tmp_path):
    d, *_ = _synthetic_run(tmp_path)
    z = dict(np.load(d / "big3_err.npz"))
    for k in ("forces_q", "forces_group"):
        z.pop(k)
    np.savez(d / "big3_err.npz", **z)
    (d / "ard.json").write_text(json.dumps({"kappa": 1.0}))
    R = vs.load_run(d)
    cov = vs.coverage_table(R, B=50, seed=0)
    assert all(r["atom"] is None for r in cov)
    md = vs.report([R])
    assert "n/a" in md and "rho" in md.lower()


def test_report_full(tmp_path):
    d, *_ = _synthetic_run(tmp_path)
    md = vs.report([vs.load_run(d)], B=100)
    for key in ("lev p50 / p99 / max", "lambda_rms", "crack tip", "group 1"):
        assert key in md


def test_aniso_mahalanobis_and_volume(tmp_path):
    d, *_ = _synthetic_run(tmp_path)
    z = dict(np.load(d / "big3_err.npz"))
    n = len(z["err"])
    z["forces_cov"] = np.tile(np.eye(3) * 4.0, (n, 1, 1)).astype(np.float32)     # lam_g^2 * I with lam = 2 (g1)
    z["dF"] = np.tile([0.0, 0.0, 1.0], (n, 1)) * z["err"][:, None]
    np.savez(d / "big3_err.npz", **z)
    np.savez(d / "posterior.npz", force_shape=np.array("aniso"), eps=1e-3)
    R = vs.load_run(d)
    assert R["force_shape"] == "aniso" and R["eps"] == 1e-3
    # per-atom lam: g0 = 1, g1 = 2 -> V = cov / lam^2 differs per group, but the score stays finite and monotone
    cov = vs.coverage_table(R, B=20)
    assert all(0.0 <= r["atom"] <= 1.0 for r in cov)
    vol, vol_eq = vs.volumes(R)
    assert vol > 0 and vol_eq > 0


def test_sweeps(tmp_path):
    runs = []
    for k, (nfit, ell) in enumerate([(100, 2.0), (200, 4.0), (400, 6.0)]):
        d = tmp_path / f"r{k}"
        d.mkdir()
        _synthetic_run(d)
        a = json.load(open(d / "ard.json"))
        a["transfer"]["N_fit"] = nfit
        a["shape"]["ell"] = ell
        a["groups"]["lam_rms"] = [1.0 * nfit ** -0.5, 2.0 * nfit ** -0.5]
        json.dump(a, open(d / "ard.json", "w"))
        runs.append(vs.load_run(d))
    fs = vs.f_sweep(runs)
    assert fs["group 0"] == pytest.approx(-0.5) and fs["group 1"] == pytest.approx(-0.5)
    es = vs.ell_sweep(runs)
    assert [e[0] for e in es] == [2.0, 4.0, 6.0] and es[0][1] is not None


def test_bootstrap_resamples_only_subset_cells(tmp_path):
    """A 2-cell subset in a 12-cell run: all its atoms covered -> CI is exactly (1, 1), never a 0 bound;
    with one of its two cells missed, the CI brackets the estimate."""
    n_cells, per = 12, 6
    cfg = np.repeat(np.arange(n_cells * 3), per)
    cell = cfg // 3
    fam = np.where(cell < 2, "edge", "crack")
    for hit_edge, name in ((np.ones(len(cfg), bool), "all"), (cell != 1, "half")):
        hit = np.where(fam == "edge", hit_edge, True)
        d = tmp_path / name
        d.mkdir()
        np.savez(d / "big3_err.npz", err=np.where(hit, 0.5, 2.0), sd=np.full(len(cfg), 0.3), family=fam,
                 r_core=np.full(len(cfg), 30.0), fixed=np.zeros(len(cfg), bool), cfg=cfg,
                 forces_q=np.ones(len(cfg)), forces_group=np.zeros(len(cfg), np.int16))
        row = {r["label"]: r for r in vs.coverage_table(vs.load_run(d), B=1000, seed=3)}["edge"]
        if name == "all":
            assert row["ci"] == (1.0, 1.0) and row["ci_atom"] == (1.0, 1.0)
        else:
            lo, hi = row["ci"]
            assert row["cell"] == pytest.approx(0.5) and lo <= 0.5 <= hi and 0.0 <= lo


def test_file_selector(tmp_path):
    d, *_ = _synthetic_run(tmp_path)
    z = dict(np.load(d / "big3_err.npz"))
    np.savez(d / "big3x_r2-3_err.npz", **z)
    assert len(vs.load_run(d)["A"]["err"]) == 2 * len(z["err"])
    R = vs.load_run(d, "big3x_r2-3_err.npz")
    assert len(R["A"]["err"]) == len(z["err"])
    assert len(vs.load_run(d, "big3_err.npz,nomatch*")["A"]["err"]) == len(z["err"])


def _frames(path, n=3, virial=True, keys=("mace_energy", "mace_force"), nfixed=2):
    from ase import Atoms
    from ase.io import write
    out = []
    for i in range(n):
        a = Atoms("Ni4", positions=np.random.default_rng(i).random((4, 3)), cell=[8, 8, 8], pbc=True)
        if "mace_energy" in keys:
            a.info["mace_energy"] = -1.0
        if "mace_force" in keys:
            a.arrays["mace_force"] = np.zeros((4, 3))
        if virial and i != 1:
            a.info["mace_virial"] = np.zeros(9)
        a.arrays["fixed"] = np.arange(4) < nfixed
        out.append(a)
    write(path, out, format="extxyz")


def test_train_extra_validates_and_counts(tmp_path):
    _frames(tmp_path / "train.xyz", virial=True, nfixed=0)
    _frames(tmp_path / "x.xyz")
    c = te.prepare(tmp_path / "train.xyz", [tmp_path / "x.xyz"], tmp_path / "out.xyz", log=lambda *a: None)
    assert c == {"n_train": 3, "n_extra": 3, "n_extra_no_virial": 1, "n_extra_fixed_atoms": 6}
    from ase.io import read
    assert len(read(tmp_path / "out.xyz", ":")) == 6
    with pytest.raises(ValueError, match="x.xyz frame 0.*mace_force"):
        _frames(tmp_path / "x.xyz", keys=("mace_energy",))
        te.prepare(tmp_path / "train.xyz", [tmp_path / "x.xyz"], tmp_path / "o2.xyz", log=lambda *a: None)
    _frames(tmp_path / "x.xyz")
    with pytest.raises(ValueError, match="fixed"):
        te.prepare(tmp_path / "train.xyz", [tmp_path / "x.xyz"], tmp_path / "o3.xyz", log=lambda *a: None,
                   fixed="error")


def test_report_shows_scalar_lam_beside_lambda_rms(tmp_path):
    """m6: the 30-Sep scalar-lambda comparison reads report["lam"] (the #18 scalar, still reported) when the
    run has it; lambda_rms stays the per-group configuration-weighted pooled value."""
    d, *_ = _synthetic_run(tmp_path)
    rep = json.loads((d / "ard.json").read_text())
    rep.update(variance="sandwich", lam=1.2345)
    (d / "ard.json").write_text(json.dumps(rep))
    R = vs.load_run(d)
    assert vs.scalar_lam(R) == pytest.approx(1.2345)
    md = vs.report([R], B=50)
    assert "lam (scalar)" in md and "1.2345" in md
    (d / "ard.json").write_text(json.dumps({"kappa": 1.0}))
    assert vs.scalar_lam(vs.load_run(d)) is None
