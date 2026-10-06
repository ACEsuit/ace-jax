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


def make_config(arm, common, cfg_cls=None):
    """FitConfig for a named arm of ARD_ARMS or D4_ARMS (None otherwise).  `common` are the shared
    FitConfig kwargs; the arm's overrides win (ard_val_frac 0.2 is the default of the f sweep's middle)."""
    arms = {**ARD_ARMS, **D4_ARMS}
    if arm not in arms:
        return None
    if cfg_cls is None:
        from ace_jax.fit.pipeline import FitConfig as cfg_cls
    base = dict(common, arm="linear", uq="ard", ard_mode="joint", ard_val_frac=0.2, ard_laplace=True)
    return cfg_cls(**{**base, **arms[arm]})
