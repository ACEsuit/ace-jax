from dataclasses import dataclass, field


@dataclass
class FitConfig:
    """Every option of the fitting pipeline.  Defaults are run.py's; the CLI
    overrides the ones where it differs (see cli.py)."""
    model: str
    arm: str = "gp"                      # "linear" (M = 0) | "gp"
    # data
    energy_key: str = "energy"; force_key: str = "forces"; virial_key: str = "virial"
    ntrain: int = 800; ntest: int = 200; test_start: int | None = None
    seed: int = 0; batch: int = 4
    weights: dict | None = None          # ACEfit weights dict (per config type)
    factors: list | None = None          # ace_jax.fit.weights factor instances
    sigma_type: bool = False             # per-config-type noise block (diagnostic)
    route: dict | None = None            # ParamSet route overrides
    baseline: str | None = None          # dimer_mean.npz (mu_0 subtracted, added back)
    base_npz: str | None = None          # precomputed per-config mu_0 offsets
    e0: str = "lsq"                      # "lsq" (fit on train energies) | "model" (z["E0"])
    # GP
    m_per_species: int = 100
    kernel: str = "cosine"; bump: bool = True
    density: str = "none"                # "none" | "pair" | "pca"
    pca_d: int = 128
    warp: str = "none"
    embedding: str | None = None         # MACE table JSON: frozen coregionalization
    delta_s_floor_q: float | None = None
    fix_rho: str | None = None           # "auto" or a number (L-BFGS only)
    r0: float = 2.5
    # objective
    objective: str = "lml"               # "lml" | "loo"
    lml: str = "device"                  # "device" | "host-cache"
    lml_chunk: int = 64
    devices: int = 1
    # MAP
    opt: str = "lbfgs"                   # "lbfgs" | "adam"
    map_steps: int = 150; map_lr: float = 0.02
    map_restarts: int = 1
    init: dict | None = None             # Hypers field -> value
    # rungs
    rungs: tuple = ("map",)
    laplace: str = "fd"                  # "fd" (run_laplace_fd) | "svi" (run_laplace)
    n_draws: int = 64
    vi_steps: int = 1000
    nuts_warmup: int = 100; nuts_samples: int = 100; nuts_chains: int = 1
    pf_samples: int = 4; pf_maxiter: int = 10
    # prediction / UQ
    uq: str = "blr"                      # "blr" | "pops"
    deriv_dtc: bool = True
    predict_stats: str = "cached"        # "cached" (linear stats once; run.py) | "recompute" (per draw; CLI)
    predict_train: bool = True
    pops_posterior: str = "hypercube"; pops_leverage_pct: float = 0.0
    pops_ridge: object = "auto"          # "auto" | "blr" | float | {"E":..,"F":..,"V":..}
    pops_ridge_grid: tuple = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11,
                              1e-12, 1e-13, 1e-14)
    pops_val_frac: float = 0.2; pops_env_nf: int = 2000

    def validate(self):
        if self.arm not in ("linear", "gp"):
            raise ValueError(f"arm must be 'linear' or 'gp', got {self.arm!r}")
        if self.uq == "pops" and self.arm != "linear":
            raise ValueError("uq='pops' is the linear-arm misspecification predictive: use arm linear")
        if self.lml == "host-cache":
            if (self.arm != "gp" or self.density not in ("pair", "pca") or tuple(self.rungs) != ("map",)
                    or self.opt != "lbfgs"):
                raise ValueError("lml='host-cache' needs arm gp, density pair|pca, rungs ('map',) and "
                                 "opt lbfgs (the cached LML exposes value_and_grad for L-BFGS)")
            if self.devices > 1:
                raise ValueError("lml='host-cache' is single-device: set devices=1")
        if self.map_restarts < 1:
            raise ValueError(f"map_restarts must be >= 1, got {self.map_restarts}")
        if self.map_restarts > 1 and (self.opt != "lbfgs" or self.sigma_type):
            raise ValueError("map_restarts > 1 is the L-BFGS multi-start: set opt lbfgs "
                             "(and not sigma_type)")
        if self.predict_stats not in ("cached", "recompute"):
            raise ValueError(f"predict_stats must be 'cached' or 'recompute', got {self.predict_stats!r}")
        if self.fix_rho is not None and self.opt != "lbfgs":
            raise ValueError("fix_rho is implemented for opt lbfgs only")
        return self
