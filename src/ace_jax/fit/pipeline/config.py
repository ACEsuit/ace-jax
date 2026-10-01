from dataclasses import dataclass


@dataclass
class FitConfig:
    """Every option of the fitting pipeline.  Defaults are run.py's; the CLI
    overrides the ones where it differs (see cli.py)."""
    model: object                        # str | os.PathLike | basis.model.Basis | basis.model.BasisSpec
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
    r0: float | None = 2.5              # None: the basis's own mean radial length (FitData.r0)
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
    uq: str = "blr"                      # "blr" | "pops" | "ard"
    ard_mode: str = "joint"              # "joint" (sigma_q + ARD scales) | "sequential" (low memory)
    ard_variance: str = "sandwich"       # "sandwich" (configuration-clustered, spec addendum) | "kappa"
    ard_val_frac: float = 0.2            # train hold-out for the force-variance scale (lam, kappa)
    ard_cond_max: float = 1e14           # prior floor: cond(S) <= ard_cond_max
    ard_laplace: bool = False            # Laplace diagnostic of the hyperparameters (ard.json)
    deriv_dtc: bool = True
    predict_stats: str = "cached"        # "cached" (linear stats once; run.py) | "recompute" (per draw; CLI)
    predict_train: bool = True
    pops_posterior: str = "hypercube"; pops_leverage_pct: float = 0.0
    pops_ridge: object = "auto"          # "auto" | "blr" | float | {"E":..,"F":..,"V":..}
    pops_ridge_grid: tuple = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11,
                              1e-12, 1e-13, 1e-14)
    pops_val_frac: float = 0.2; pops_env_nf: int = 2000
    pops_rows: str = "auto"              # "auto" | "host" (rows cached in host RAM) | "device" (re-evaluated)
    learn_radial: bool = False           # learn the tensor radials (radial_learn.fit_radial) before the fit
    radial_n_q: int = 12                 # polynomial span after to_analytic widening
    radial_steps: int = 40               # L-BFGS steps per roughness weight
    radial_lam_grid: tuple = (0.0, 1e-2) # relative roughness weights; the gate picks among them and init
    radial_val_frac: float = 0.2         # train hold-out for the gate

    def validate(self):
        if self.arm not in ("linear", "gp"):
            raise ValueError(f"arm must be 'linear' or 'gp', got {self.arm!r}")
        if self.uq == "pops" and self.arm != "linear":
            raise ValueError("uq='pops' is the linear-arm misspecification predictive: use arm linear")
        if self.uq not in ("blr", "pops", "ard"):
            raise ValueError(f"uq must be 'blr', 'pops' or 'ard', got {self.uq!r}")
        if self.uq == "ard":
            if self.arm != "linear":
                raise ValueError("uq='ard' is the linear-arm ARD posterior: use arm linear (m_per_species 0)")
            if self.ard_mode not in ("joint", "sequential"):
                raise ValueError(f"ard_mode must be 'joint' or 'sequential', got {self.ard_mode!r}")
            if self.ard_variance not in ("sandwich", "kappa"):
                raise ValueError(f"ard_variance must be 'sandwich' or 'kappa', got {self.ard_variance!r}")
            if not 0.0 < self.ard_val_frac < 1.0:
                raise ValueError(f"ard_val_frac must be in (0, 1), got {self.ard_val_frac}")
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
        if self.pops_rows not in ("auto", "host", "device"):
            raise ValueError(f"pops_rows must be 'auto', 'host' or 'device', got {self.pops_rows!r}")
        if self.predict_stats not in ("cached", "recompute"):
            raise ValueError(f"predict_stats must be 'cached' or 'recompute', got {self.predict_stats!r}")
        if self.fix_rho is not None and self.opt != "lbfgs":
            raise ValueError("fix_rho is implemented for opt lbfgs only")
        if self.learn_radial:
            if not 0.0 < self.radial_val_frac < 1.0:
                raise ValueError(f"radial_val_frac must be in (0, 1), got {self.radial_val_frac}")
            if self.radial_n_q < 1:
                raise ValueError(f"radial_n_q must be >= 1, got {self.radial_n_q}")
            if self.radial_steps < 0:
                raise ValueError(f"radial_steps must be >= 0, got {self.radial_steps}")
            if not len(self.radial_lam_grid):
                raise ValueError("radial_lam_grid must hold at least one roughness weight")
        return self
