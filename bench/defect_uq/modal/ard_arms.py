"""Ablation / sweep arms of the conformal force-sigma validation programme (fit_bench.py arm names).

Each arm is a dict of FitConfig overrides on top of the common bench365 ARD settings.  Importable
without fit_bench's side effects (the tests use it).
"""
ARD_ARMS = {
    "ard_legacy": {"_shape_variant": "legacy", "_score_source": "mixed", "ard_groups": "none"},
    "ard_A":      {"_shape_variant": "legacy", "_score_source": "fit"},
    "ard_AB":     {"ard_cluster_size": float("inf")},
    "ard_ABblk":  {"ard_force_shape": "iso"},                           # PRESS, ell = 3 r_cut, iso (the pre-2026-10-02 defaults, kept for the recorded ablation)
    "ard_aniso":  {"ard_force_shape": "aniso"},
    **{f"ard_ell{k}": {"ard_cluster_size": float(k)} for k in (2, 4, 6)},
    **{f"ard_f{int(f * 10)}": {"ard_val_frac": f} for f in (0.1, 0.3)},
}


def make_config(arm, common, cfg_cls=None):
    """FitConfig for a named ablation arm (None if `arm` is not in ARD_ARMS).  `common` are the shared
    FitConfig kwargs; the arm's overrides win (ard_val_frac 0.2 is the default of the f sweep's middle)."""
    if arm not in ARD_ARMS:
        return None
    if cfg_cls is None:
        from ace_jax.fit.pipeline import FitConfig as cfg_cls
    base = dict(common, arm="linear", uq="ard", ard_mode="joint", ard_val_frac=0.2, ard_laplace=True)
    return cfg_cls(**{**base, **ARD_ARMS[arm]})
