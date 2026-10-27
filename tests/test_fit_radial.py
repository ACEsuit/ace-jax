"""fit.radial_learn: the held-out gate over learned candidates (fit_radial), its scores and
checkpoints, and saving the selected radials."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from ace_jax.eval import load  # noqa: E402
from radial_learn_problem import (HAVE_FIXTURES, MODEL, THETA, make_problem,  # noqa: E402
                                  perturbed_truth, relabel)

pytestmark = pytest.mark.skipif(not HAVE_FIXTURES, reason="missing fixtures")


@pytest.fixture(scope="module")
def small():
    return make_problem()


def test_gate_ties_go_to_first():
    from ace_jax.fit.radial_learn import gate
    label, scores = gate({"init": 1, "learned": 2}, lambda k, w: 0.5)
    assert label == "init" and scores == {"init": 0.5, "learned": 0.5}
    label, _ = gate({"init": 1, "learned": 2}, lambda k, w: 1.0 if w == 1 else 0.1)
    assert label == "learned"
    seen = []
    gate({"init": 1, "learned": 1}, lambda k, w: seen.append(k) or 0.0)   # equal values, distinct labels
    assert seen == ["init", "learned"]


def test_holdout_score_energy_only_split(small):
    from ace_jax.fit.hypers import to_array
    from ace_jax.fit.radial_learn import holdout_score
    prob, ds_fit, _ = small
    _, ds_val, _ = make_problem(ncfg=6, start=6, force_key=None)            # no force labels
    from ace_jax.fit.objective import posterior_from_qr
    from ace_jax.fit.stats import linear_qr_statistics, linear_statistics
    a = to_array(THETA)
    W = prob.model.rnl_Wnlq
    s = holdout_score(W, a, a, prob, ds_fit, ds_val)
    val = linear_statistics(prob.model, prob.cfg, ds_val)
    assert float(val.n_F) == 0 and float(val.n_E) > 0
    c, _ = posterior_from_qr(THETA, linear_qr_statistics(prob.model, prob.cfg, ds_fit), prob)
    e_term = float(val.yy_E - 2.0 * c @ val.b_E + c @ val.G_E @ c) / (float(val.n_E) * 0.01 ** 2)
    assert np.isfinite(s) and s > 0.0
    np.testing.assert_allclose(s, e_term, rtol=1e-8)      # the Gram reference cancels: ~1e-9 here


def test_holdout_score_weights_quantities_as_the_objective():
    """The gate sums each quantity's rows (SSE_t / sigma_t^2), as the learner's objective does, over the
    validation configs: forces count by their rows, not as one per-row mean that each energy row
    outweighs ~100x (the Si GAP-18 gate kept init radials that lowered the validation force error)."""
    from ace_jax.fit.hypers import from_array, to_array
    from ace_jax.fit.objective import posterior_from_qr
    from ace_jax.fit.radial_learn import holdout_score
    from ace_jax.fit.stats import linear_qr_statistics, linear_statistics
    prob, ds_fit, _ = make_problem(ncfg=6)
    _, ds_val, _ = make_problem(ncfg=6, start=6)
    a = to_array(THETA)
    s = holdout_score(prob.model.rnl_Wnlq, a, a, prob, ds_fit, ds_val)
    val = linear_statistics(prob.model, prob.cfg, ds_val)
    assert float(val.n_F) > float(val.n_E) > 0
    c, _ = posterior_from_qr(THETA, linear_qr_statistics(prob.model, prob.cfg, ds_fit), prob)
    th = from_array(a)
    ref = sum(float(getattr(val, f"yy_{t}") - 2.0 * c @ getattr(val, f"b_{t}") + c @ getattr(val, f"G_{t}") @ c)
              / float(np.exp(2.0 * float(getattr(th, f"log_sigma_{t}")))) for t in "EF") / float(val.n_E)
    np.testing.assert_allclose(s, ref, rtol=1e-6)


def test_fit_radial_gate_prefers_learned_on_recoverable_problem():
    from ace_jax.fit.radial_learn import fit_radial
    prob, ds_fit, _ = make_problem(ncfg=12, per_batch=3, start=0)
    _, ds_val, _ = make_problem(ncfg=12, per_batch=3, start=12)
    Wt, W0, c = perturbed_truth(prob)
    ds_fit, ds_val = relabel(prob, ds_fit, Wt, c), relabel(prob, ds_val, Wt, c)
    # theta0=None: a0 is the (unconverged) MAP at the init, so this exercises
    # the gate's common re-MAP for every candidate, init included
    W, info = fit_radial(prob, ds_fit, ds_val, W0, lam_grid=(0.0,), theta0=None,
                         profile=False, steps=30, map_steps=50)
    print(info["scores"], info["scores_at_a0"])
    assert info["selected"].startswith("learned")
    assert info["scores"][info["selected"]] < info["scores"]["init"]
    assert info["scores_at_a0"][info["selected"]] < info["scores_at_a0"]["init"]


def test_fit_radial_zero_steps_gate_is_fair(small):
    """steps=0: learned == init, so both are scored by the identical procedure
    and must tie exactly (-> init), whatever the MAP's convergence."""
    from ace_jax.fit.radial_learn import fit_radial
    prob, ds_fit, _ = small
    _, ds_val, _ = make_problem(ncfg=6, start=6)
    W, info = fit_radial(prob, ds_fit, ds_val, prob.model.rnl_Wnlq, lam_grid=(0.0,), theta0=None,
                         steps=0, map_steps=50)
    s = info["scores"]
    print(s, info["scores_at_a0"], info["map_diag"])
    np.testing.assert_allclose(s["learned_lam=0"], s["init"], rtol=1e-6)
    np.testing.assert_allclose(info["scores_at_a0"]["learned_lam=0"], info["scores_at_a0"]["init"],
                               rtol=1e-6)
    assert info["selected"] == "init"
    assert set(info["map_diag"]) == {"init", "learned_lam=0"}
    assert all(np.isfinite(d["grad_norm"]) and np.isfinite(d["dloss_last"])
               for d in info["map_diag"].values())
    assert info["readout"].shape == (prob.cfg.len_basis,)


@pytest.fixture(scope="module")
def checkpointed(small, tmp_path_factory):
    """One fit_radial over two roughness weights with a checkpoint callback, shared by
    the tests that only read its result: (W, info, ds_val, checkpoint dir, labels)."""
    from ace_jax.fit.radial_learn import fit_radial, save_result
    prob, ds_fit, _ = small
    _, ds_val, _ = make_problem(ncfg=6, start=6)
    out, calls = tmp_path_factory.mktemp("ckpt"), []

    def ckpt(label, W, info):
        calls.append(label)
        save_result(out / f"lam_{label}", W, info)

    W, info = fit_radial(prob, ds_fit, ds_val, prob.model.rnl_Wnlq, lam_grid=(0.0, 1e-2),
                         theta0=THETA, profile=False, steps=2, map_steps=10, checkpoint=ckpt)
    return W, info, ds_val, out, calls


def test_fit_radial_checkpoints_each_lambda(checkpointed):
    W, info, _, tmp_path, calls = checkpointed
    assert calls == ["0", "0.01"]
    import json
    saved = {}
    for lab in calls:
        saved[lab] = np.load(tmp_path / f"lam_{lab}" / "rnl_Wnlq.npy")
        js = json.loads((tmp_path / f"lam_{lab}" / "radial_info.json").read_text())
        assert js["lam_rough"] == float(lab) and js["steps"] == 2
    sel = info["selected"]
    if sel != "init":
        np.testing.assert_array_equal(saved[sel.split("=")[1]], np.asarray(W))


def test_learned_radials_reject_inducing_points(small):
    from ace_jax.fit.radial_learn import fit_radial, learn_radial
    prob, ds, _ = small
    ind = prob.ind._replace(XM=jnp.zeros((3, prob.ind.XM.shape[1])))
    bad = prob._replace(ind=ind)
    with pytest.raises(ValueError, match="M = 0"):
        learn_radial(bad, ds, prob.model.rnl_Wnlq, theta0=THETA, steps=0)
    with pytest.raises(ValueError, match="M = 0"):
        fit_radial(bad, ds, ds, prob.model.rnl_Wnlq, steps=0)


def test_fit_radial_spec_and_gap_grid_labels(small):
    """Both suffixes in one grid (gap > 0 also exercises the uniform-Gram precompute)."""
    from ace_jax.fit.radial_learn import fit_radial
    prob, ds_fit, _ = small
    _, ds_val, _ = make_problem(ncfg=6, start=6)
    spec_grid, gap_grid = (0.0, 1.0), (0.0, 1.0)
    W, info = fit_radial(prob, ds_fit, ds_val, prob.model.rnl_Wnlq, lam_grid=(0.0,), spec_grid=spec_grid,
                         gap_grid=gap_grid, theta0=THETA, steps=0, map_steps=10)
    expected = {"init"} | {f"learned_lam=0_spec={s:g}_gap={g:g}" for s in spec_grid for g in gap_grid}
    assert set(info["scores"]) == expected


def test_fit_radial_rejects_duplicate_lambdas(small):
    from ace_jax.fit.radial_learn import fit_radial
    prob, ds, _ = small
    with pytest.raises(ValueError, match="duplicate"):
        fit_radial(prob, ds, ds, prob.model.rnl_Wnlq, lam_grid=(0.0, 1e-2, 0.01), steps=0)


def test_save_result_roundtrip(tmp_path, small):
    from ace_jax.fit.radial_learn import save_result
    prob, _, _ = small
    W = 1.5 * prob.model.rnl_Wnlq
    c = np.arange(prob.cfg.len_basis, dtype=float)
    save_result(tmp_path, W, {"selected": "learned", "trace": [1.0, 0.5], "theta": [np.zeros(3)],
                              "readout": c}, src_npz=MODEL, model=prob.model)
    assert np.array_equal(np.load(tmp_path / "rnl_Wnlq.npy"), np.asarray(W))
    assert np.array_equal(np.load(tmp_path / "readout.npy"), c)
    back, meta, _ = load(tmp_path / "model.npz")
    np.testing.assert_array_equal(np.asarray(back.rnl_Wnlq), np.asarray(W))
    nB, nP, NZ = prob.cfg.n_B, prob.cfg.n_pair, prob.cfg.NZ
    np.testing.assert_array_equal(np.asarray(back.WB)[:, 0], c[:nB])
    np.testing.assert_array_equal(np.asarray(back.Wpair)[:, 0], c[NZ * nB:NZ * nB + nP])
    import json
    js = json.loads((tmp_path / "radial_info.json").read_text())
    assert js["selected"] == "learned" and "readout" not in js


def test_save_result_refuses_stale_readout(tmp_path, small):
    from ace_jax.fit.radial_learn import save_result
    prob, _, _ = small
    with pytest.raises(ValueError, match="readout"):
        save_result(tmp_path, prob.model.rnl_Wnlq, {"selected": "init"}, src_npz=MODEL, model=prob.model)
    assert not (tmp_path / "model.npz").exists()
    save_result(tmp_path, prob.model.rnl_Wnlq, {"selected": "init"})      # no model.npz: fine
    assert (tmp_path / "rnl_Wnlq.npy").exists()


def test_readout_to_npz_layout():
    from ace_jax.basis.export import readout_to_npz
    nB, nP, NZ = 3, 2, 2
    c = np.arange((nB + nP) * NZ, dtype=float)
    WB, Wpair = readout_to_npz(c, nB, nP, NZ)
    np.testing.assert_array_equal(WB, [[0, 3], [1, 4], [2, 5]])
    np.testing.assert_array_equal(Wpair, [[6, 8], [7, 9]])
    with pytest.raises(ValueError, match="readout shape"):
        readout_to_npz(c[:-1], nB, nP, NZ)


def test_saved_model_energies_match_fitted_readout(tmp_path, small, checkpointed):
    """model.npz written by save_result evaluates to linear_rows @ readout
    (+ E0): the readout is the one fitted for the selected radials."""
    from ace_jax.fit.radial_learn import save_result
    from ace_jax.fit.radial_model import with_radial
    from ace_jax.fit.rows import linear_rows
    prob, ds_fit, _ = small
    W, info, ds_val, _, _ = checkpointed
    save_result(tmp_path, W, info, src_npz=MODEL, model=prob.model)
    back, _, z = load(tmp_path / "model.npz")
    E0 = np.asarray(z["E0"])
    m = with_radial(prob.model, W)
    worst = 0.0
    for i in range(ds_val.n_batches):
        b = jax.tree.map(lambda a: a[i], ds_val)
        rows, _, _ = linear_rows(m, prob.cfg, b)
        C = b.y_E.shape[0]
        Ncap, K = b.nbr.shape
        zi = jnp.broadcast_to(b.node_z[:, None], (Ncap, K))
        e = back.site_energies_dense(b.rij, zi, b.node_z[b.nbr], b.nbr_mask, b.node_z)
        E_model = np.asarray(jax.ops.segment_sum(e, b.node_cfg, num_segments=C + 1)[:C])
        live = np.asarray(b.node_mask)
        E0sum = np.asarray(jax.ops.segment_sum(jnp.where(b.node_mask, jnp.asarray(E0)[b.node_z], 0.0),
                                               b.node_cfg, num_segments=C + 1)[:C])
        E_lin = np.asarray(rows.E @ jnp.asarray(info["readout"])) + E0sum
        assert live.any()
        worst = max(worst, float(np.max(np.abs(E_model - E_lin) / np.abs(E_lin))))
    print(f"max relative |E_npz - (rows @ c + E0)| / |E| = {worst:.3e}")
    assert worst < 1e-8
