"""Plan step 1 analysis (docs/dev/plans/2026-10-15-fp32-gpus.md) of the rows_dump.py outputs, in
numpy float64 on the CPU: row error, conditioning, the effect of fp32 rows on the solve, readout
cancellation, and the host fp64 QR merge cost.

    uv run --no-sync python bench/fp32/analyse.py o3d8 o3d12 o4d12 o4d16 > results.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla

RUNS = Path("/storage/eng/essswb/projects/ace-jax/runs")
FIT = RUNS / "accuracy_cost/ace_per_quantity_noise"
DUMP = RUNS / "fp32_step1"
EPS32 = float(np.finfo(np.float32).eps)


def load(m, tag):
    z = np.load(DUMP / f"{m}_{tag}.npz")
    return {k: z[k] for k in z.files if k != "info"} | {"info": json.loads(str(z["info"]))}


def row_err(a, b):
    """fp32 block a vs fp64 block b: Frobenius and per-column (norm-relative) error."""
    d = a.astype(np.float64) - b
    cn = np.linalg.norm(b, axis=0)
    live = cn > 0
    col = np.linalg.norm(d, axis=0)[live] / cn[live]
    return dict(frob=float(np.linalg.norm(d) / np.linalg.norm(b)), col_median=float(np.median(col)),
                col_max=float(col.max()), col_max_idx=int(np.flatnonzero(live)[col.argmax()]))


def design(r, th, dtype=np.float64):
    """The fit's weighted rows (solve._weighted_rows): stacked E|F|V times w / sigma_t."""
    s = {k: np.exp(th["log_sigma_" + k]) for k in "EFV"}
    W = np.concatenate([r["E"].astype(dtype) * (r["wE"] / s["E"])[:, None].astype(dtype),
                        r["F"].astype(dtype) * (r["wF"] / s["F"])[:, None].astype(dtype),
                        r["V"].astype(dtype) * (r["wV"] / s["V"])[:, None].astype(dtype)])
    y = np.concatenate([r["yE"] * r["wE"] / s["E"], r["yF"] * r["wF"] / s["F"], r["yV"] * r["wV"] / s["V"]])
    return W, y.astype(dtype)


def kappa(A):
    R = np.linalg.qr(A, mode="r")
    sv = np.linalg.svd(R, compute_uv=False)
    return float(sv[0] / sv[-1]), sv


def qr_solve(A, y):
    """Householder QR least squares in A's dtype (LAPACK geqrf/ormqr, as the fit's merges)."""
    Q, R = np.linalg.qr(A)
    return sla.solve_triangular(R, Q.T @ y)


def metrics(r, c):
    """Training-set RMSEs of readout c on the fp64 rows: meV/atom, meV/A, meV/atom (virial)."""
    eE = (r["E"] @ c - r["yE"]) / r["nat"]
    eF = r["F"] @ c - r["yF"]
    eV = (r["V"] @ c - r["yV"]) / np.repeat(r["nat"], 6)
    live = r["wV"] > 0
    return dict(E=1e3 * float(np.sqrt(np.mean(eE ** 2))), F=1e3 * float(np.sqrt(np.mean(eF ** 2))),
                V=1e3 * float(np.sqrt(np.mean(eV[live] ** 2))))


def diff(r, c, c_ref):
    """RMS difference of the predictions of c and c_ref on the fp64 rows, in the metrics' units."""
    d = c - c_ref
    return dict(E=1e3 * float(np.sqrt(np.mean(((r["E"] @ d) / r["nat"]) ** 2))),
                F=1e3 * float(np.sqrt(np.mean((r["F"] @ d) ** 2))),
                V=1e3 * float(np.sqrt(np.mean(((r["V"] @ d) / np.repeat(r["nat"], 6))[r["wV"] > 0] ** 2))),
                coef_rel=float(np.linalg.norm(d) / np.linalg.norm(c_ref)))


def merge_time(L, rows):
    """Seconds for one host fp64 merge of `rows` new rows into an L x L R (stats._host_merge)."""
    import scipy.linalg.lapack as la
    rng = np.random.default_rng(0)
    R = np.triu(rng.standard_normal((L, L))) + 10 * np.eye(L)
    A = np.asfortranarray(np.vstack([R, rng.standard_normal((rows, L))]))
    lwork = int(la.dgeqrf_lwork(*A.shape)[0])
    ts = []
    for _ in range(3):
        B = A.copy(order="F")
        t = time.perf_counter()
        la.dgeqrf(B, lwork=max(lwork, L + 1), overwrite_a=1)
        ts.append(time.perf_counter() - t)
    return min(ts)


def analyse(m):
    r64, r32, rtf = load(m, "f64"), load(m, "f32"), load(m, "tf32")
    z = np.load(FIT / m / "model.npz")
    th = json.load(open(FIT / m / "theta_map.json"))
    c_fit = np.concatenate([z["WB"].T.ravel(), z["Wpair"].T.ravel()])
    L = r64["E"].shape[1]
    assert c_fit.shape[0] == L, (c_fit.shape, L)
    out = dict(model=m, L=L, n_configs=r64["info"]["n_configs"], n_rows=int(sum(len(r64[k]) for k in "EFV")))
    out["timing"] = {t: dict(device=r["info"]["device"], t_batch=r["info"]["t_batch_median"],
                             batch_shape=r["info"]["batch_shape"]) for t, r in (("f64", r64), ("f32", r32), ("tf32", rtf))}

    # 1. row error
    out["row_err"] = {t: {k: row_err(r[k], r64[k]) for k in "EFVX"} for t, r in (("f32", r32), ("tf32", rtf))}

    # 2. conditioning
    W64, y = design(r64, th)
    W32, _ = design(r32, th)
    P = np.diag(z["gamma"] / np.exp(th["log_sigma_c"]))
    D = np.linalg.norm(W64, axis=0)
    D[D == 0] = 1.0
    k_raw, sv = kappa(W64)
    k_jac, _ = kappa(W64 / D)
    k_aug, sv_aug = kappa(np.vstack([W64, P]))
    k_aug_jac, _ = kappa(np.vstack([W64, P]) / np.linalg.norm(np.vstack([W64, P]), axis=0))
    out["kappa"] = dict(raw=k_raw, jacobi=k_jac, prior_aug=k_aug, prior_aug_jacobi=k_aug_jac,
                        col_norm_range=float(D.max() / D.min()))

    # 3. solves on the prior-augmented design
    A64, A32c = np.vstack([W64, P]), np.vstack([W32, P])
    ya = np.concatenate([y, np.zeros(L)])
    c64 = qr_solve(A64, ya)
    sols = {"fit_model": c_fit, "qr64_on_f64rows": c64,
            "qr64_on_f32rows": qr_solve(A32c, ya),
            "qr64_on_tf32rows": qr_solve(np.vstack([design(rtf, th)[0], P]), ya)}
    A32 = A32c.astype(np.float32)
    sols["qr32_on_f32rows"] = qr_solve(A32, ya.astype(np.float32)).astype(np.float64)
    Dj = np.linalg.norm(A32c, axis=0)
    sols["qr32_jacobi_on_f32rows"] = (qr_solve((A32c / Dj).astype(np.float32), ya.astype(np.float32))
                                      .astype(np.float64) / Dj)
    G = (A32.T @ A32)                                   # fp32 normal equations: kappa^2
    try:
        cf = sla.cho_factor(G)
        sols["chol32_normal_eq"] = sla.cho_solve(cf, A32.T @ ya.astype(np.float32)).astype(np.float64)
    except np.linalg.LinAlgError as e:
        out["chol32_normal_eq_error"] = str(e)
    # yardstick: the BLR posterior std of the mean predictions (Sigma = (A^T A)^-1 = R^-1 R^-T)
    R = np.linalg.qr(A64, mode="r")
    pstd = lambda Phi: np.linalg.norm(sla.solve_triangular(R, Phi.T, trans="T"), axis=0)
    live_v = r64["wV"] > 0
    out["posterior_std_rms"] = dict(E=1e3 * float(np.sqrt(np.mean((pstd(r64["E"]) / r64["nat"]) ** 2))),
                                    F=1e3 * float(np.sqrt(np.mean(pstd(r64["F"]) ** 2))),
                                    V=1e3 * float(np.sqrt(np.mean((pstd(r64["V"][live_v])
                                                                   / np.repeat(r64["nat"], 6)[live_v]) ** 2))))
    out["solves"] = {k: dict(train_rmse=metrics(r64, c) if np.isfinite(c).all() else None,
                             vs_qr64=diff(r64, c, c64) if np.isfinite(c).all() else None,
                             finite=bool(np.isfinite(c).all()))
                     for k, c in sols.items()}

    # 4. readout cancellation per site (X compact layout = [B | pair] for one species)
    X64, X32 = r64["X"], r32["X"]
    num = np.abs(X64 * c_fit).sum(1)
    e64 = X64 @ c_fit
    e32 = (X32 @ c_fit.astype(np.float32)).astype(np.float64)            # all-fp32 readout
    e32acc = X32.astype(np.float64) @ c_fit                              # fp32 X, fp64 accumulation
    live = num > 0                                                        # the isolated atom has no neighbours
    ratio = num[live] / np.abs(e64[live])
    out["cancellation"] = dict(
        ratio_median=float(np.median(ratio)), ratio_p99=float(np.quantile(ratio, 0.99)),
        ratio_max=float(ratio.max()),
        bound_meV_median=1e3 * EPS32 * float(np.median(num)), bound_meV_max=1e3 * EPS32 * float(num.max()),
        site_err_fp32_meV_rms=1e3 * float(np.sqrt(np.mean((e32 - e64) ** 2))),
        site_err_fp32_meV_max=1e3 * float(np.abs(e32 - e64).max()),
        site_err_fp32X_fp64acc_meV_rms=1e3 * float(np.sqrt(np.mean((e32acc - e64) ** 2))),
        sum_abs_B_median=float(np.median(np.abs(X64[:, :z["WB"].shape[0]] * c_fit[:z["WB"].shape[0]]).sum(1))),
        sum_abs_pair_median=float(np.median(np.abs(X64[:, z["WB"].shape[0]:] * c_fit[z["WB"].shape[0]:]).sum(1))),
        coef_absmax=float(np.abs(c_fit).max()), coef_norm=float(np.linalg.norm(c_fit)),
        site_energy_absmean=float(np.abs(e64).mean()))

    # 5. host merge vs GPU rows (per batch of rows)
    rows_per_batch = int(round(out["n_rows"] / r64["info"]["n_batches"]))
    out["merge"] = dict(rows_per_batch=rows_per_batch, t_merge_L_rows=merge_time(L, L),
                        t_merge_per_batch=merge_time(L, rows_per_batch))
    return out


if __name__ == "__main__":
    res = [analyse(m) for m in sys.argv[1:]]
    print(json.dumps(res, indent=1))


def lml_terms(r, th, z, dtype, rows_dtype=np.float64):
    """The data-dependent part of the linear-arm LML at theta, from a QR of the prior-augmented
    weighted design (objective.log_marginal_likelihood_qr's form): -1/2 |resid|^2 - log|det R|
    + log|det P| - sum log(sigma_t / w) - n/2 log 2 pi.  dtype: the QR's precision; rows_dtype:
    the rows' (f32 rows are exact in f64)."""
    W, y = design(r, th)
    W = W.astype(rows_dtype).astype(np.float64)
    pd = z["gamma"] / np.exp(th["log_sigma_c"])
    A = np.vstack([W, np.diag(pd)]).astype(dtype)
    ya = np.concatenate([y, np.zeros(len(pd))]).astype(dtype)
    Rf = np.linalg.qr(np.column_stack([A, ya]), mode="r")     # y as an extra column (stats._qr_merge):
    R, res2 = Rf[:-1, :-1], float(Rf[-1, -1]) ** 2           # the residual norm with no cancellation
    s = {k: np.exp(th["log_sigma_" + k]) for k in "EFV"}
    w = np.concatenate([r["wE"] / s["E"], r["wF"] / s["F"], r["wV"] / s["V"]])
    live = w > 0
    return (-0.5 * res2 - np.log(np.abs(np.diag(R).astype(np.float64))).sum() + np.log(pd).sum()
            + np.log(w[live]).sum() - 0.5 * live.sum() * np.log(2 * np.pi))


def lml_check(m, h=0.05):
    """LML and its central-difference gradient in the noise/prior hypers, for fp64 rows + fp64 QR
    (reference), fp32 rows + fp64 QR (the mixed plan) and fp32 rows + fp32 QR."""
    r64, r32 = load(m, "f64"), load(m, "f32")
    z = np.load(FIT / m / "model.npz")
    th = json.load(open(FIT / m / "theta_map.json"))
    keys = ["log_sigma_c", "log_sigma_E", "log_sigma_F", "log_sigma_V"]
    out = {}
    for tag, r, dt in (("f64rows_qr64", r64, np.float64), ("f32rows_qr64", r32, np.float64),
                       ("f32rows_qr32", r32, np.float32)):
        v0 = lml_terms(r, th, z, dt)
        g = {k: (lml_terms(r, th | {k: th[k] + h}, z, dt) - lml_terms(r, th | {k: th[k] - h}, z, dt)) / (2 * h)
             for k in keys}
        out[tag] = dict(lml=v0, grad=g)
    ref = out["f64rows_qr64"]
    for tag in ("f32rows_qr64", "f32rows_qr32"):
        out[tag]["dlml"] = out[tag]["lml"] - ref["lml"]
        out[tag]["dgrad"] = {k: out[tag]["grad"][k] - ref["grad"][k] for k in keys}
    return dict(model=m, **out)
