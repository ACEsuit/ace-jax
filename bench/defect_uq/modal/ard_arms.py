"""Ablation / sweep arms of the conformal force-sigma validation programme (fit_bench.py arm names).

Each arm is a dict of FitConfig overrides on top of the common bench365 ARD settings.  Importable
without fit_bench's side effects (the tests use it).
"""
# Every arm except ard_aniso and ard_default pins ard_force_shape="iso": the recorded rev-2 runs (2026-10-01..03) were iso
# (the library default became aniso on 2026-10-03).
ARD_ARMS = {
    "ard_legacy": {"_shape_variant": "legacy", "_score_source": "mixed", "ard_groups": "none", "ard_force_shape": "iso"},
    "ard_A":      {"_shape_variant": "legacy", "_score_source": "fit", "ard_force_shape": "iso"},
    "ard_AB":     {"ard_cluster_size": float("inf"), "ard_force_shape": "iso"},
    "ard_ABblk":  {"ard_force_shape": "iso"},                           # PRESS, ell = 3 r_cut, iso (the pre-2026-10-03 defaults, kept for the recorded ablation)
    "ard_aniso":  {"ard_force_shape": "aniso"},
    "ard_default": {"ard_force_shape": "aniso", "e0": "lsq"},   # the library defaults (joint E0, aniso, transfer exponent)
    **{f"ard_ell{k}": {"ard_cluster_size": float(k), "ard_force_shape": "iso"} for k in (2, 4, 6)},
    **{f"ard_f{int(f * 10)}": {"ard_val_frac": f, "ard_force_shape": "iso"} for f in (0.1, 0.3)},
}

# D4 of docs/dev/specs/2026-10-05-gp-discrepancy-design.md: noise-model sensitivity.  "noise" needs the
# feat/shared-noise source (ACEJAX_SRC), so these arms are kept out of ARD_ARMS (whose FitConfigs the bench tests
# build on main).  noise "shared" refuses ard_mode "joint" (that refits sigma_q by its own evidence), so both arms
# are sequential: the ARD posterior keeps the MAP's sigma_q.
D4_ARMS = {
    "ard_seq_pq":     {"ard_force_shape": "aniso", "e0": "lsq", "ard_mode": "sequential", "noise": "per-quantity"},
    "ard_seq_shared": {"ard_force_shape": "aniso", "e0": "lsq", "ard_mode": "sequential", "noise": "shared"},
}

# Phase 3 of the GP-discrepancy spec: --uq ard-gp on the gp_conv fit's settings (PCA-16, host-cache).  L-BFGS
# starts from gp_conv's converged theta (2026-10-06, logpost 540 855.5) with one start, so the arm costs a short
# MAP plus the ARD stage instead of gp_conv's 4-start search.
GP_CONV_THETA = {"log_ell": 0.6739701731311936, "log_A": 0.8986633725132979, "log_alpha": 2.3431424837222012,
                 "log_r0": 0.5522582965038343, "log_eps": -3.029046798886507, "log_rho": 4.332080370526107,
                 "log_sigma_c": 3.3429767866884825, "log_sigma_E": -4.0794906538949425,
                 "log_sigma_F": -2.720674755295538, "log_sigma_V": -2.28467628617451}
_GP_BASE = {"arm": "gp", "uq": "ard-gp", "m_per_species": 100, "density": "pca", "pca_d": 16, "lml": "host-cache",
            "map_restarts": 1, "map_steps": 300, "init": GP_CONV_THETA, "e0": "lsq", "ard_force_shape": "aniso"}
GP_ARMS = {
    "ard_gp_sw":  {**_GP_BASE, "ard_variance": "sandwich"},
    # ard_gp_dtc (--ard-variance dtc) ran 2026-10-06 and failed its acceptance; the option was removed (the arm
    # is at commit 097bed5; results/2026-10-07_ard_gp_acceptance.md)
}


def make_config(arm, common, cfg_cls=None):
    """FitConfig for a named arm of ARD_ARMS, D4_ARMS or GP_ARMS (None otherwise).  `common` are the shared
    FitConfig kwargs; the arm's overrides win (ard_val_frac 0.2 is the default of the f sweep's middle)."""
    arms = {**ARD_ARMS, **D4_ARMS, **GP_ARMS}
    if arm not in arms:
        return None
    if cfg_cls is None:
        from ace_jax.fit.pipeline import FitConfig as cfg_cls
    base = dict(common, arm="linear", uq="ard", ard_mode="joint", ard_val_frac=0.2, ard_laplace=True)
    if arm in GP_ARMS:                     # the arm's overrides set arm gp and uq ard-gp
        base["ard_laplace"] = False        # a finite-difference Laplace of the GP-block evidence adds nothing here
    return cfg_cls(**{**base, **arms[arm]})
