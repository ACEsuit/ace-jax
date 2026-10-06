from dataclasses import dataclass


@dataclass
class FitConfig:
    """Every option of the fitting pipeline.  Defaults are run.py's; the CLI
    overrides the ones where it differs (see cli.py)."""
    model: object                        # str | os.PathLike | basis.model.Basis | basis.model.BasisSpec
    arm: str = "gp"                      # "linear" (M = 0) | "gp"
    # data
    energy_key: str = "energy"; force_key: str = "forces"; virial_key: str = "virial"
    stress_key: str | None = None        # ASE/MACE stress label: virial = -stress * volume when no virial
    ntrain: int = 800; ntest: int = 200; test_start: int | None = None
    seed: int = 0; batch: int = 4
    batch_pack: str = "auto"             # "auto" | "on" | "off": size-aware batching (fit.data.build_dataset)
    weights: dict | None = None          # ACEfit weights dict (per config type)
    factors: list | None = None          # ace_jax.fit.weights factor instances
    sigma_type: bool = False             # per-config-type noise block (diagnostic)
    route: dict | None = None            # ParamSet route overrides
    baseline: str | None = None          # dimer_mean.npz (mu_0 subtracted, added back)
    base_npz: str | None = None          # precomputed per-config mu_0 offsets
    e0: str = "lsq"                      # "lsq" (fit jointly with the readout) | "prefit" (least squares on
                                         # the composition, then fixed) | "model" (z["E0"])
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
    lml_solver: str = "qr"               # "qr" | "cholesky": the linear arm's LML and posterior (GP arm: cholesky)
    devices: int = 1
    # MAP
    opt: str = "lbfgs"                   # "lbfgs" | "adam"
    map_steps: int = 150; map_lr: float = 0.02
    map_restarts: int = 1
    map_polish: str = "auto"             # Newton polish (exact Hessian, column-wise HVPs) of the L-BFGS MAP:
                                         # "auto" (linear arm, cached-Gram LML) | "on" (any arm) | "off"
    strict: bool = False                 # raise MapNotConverged (not a warning) when the MAP is not stationary
    init: dict | None = None             # Hypers field -> value
    noise: str = "per-quantity"          # "per-quantity" (sigma_E, sigma_F, sigma_V free) | "shared" (one
                                         # sigma for every weighted row, ACEfit's BLR: the weights set the balance)
    # rungs
    rungs: tuple = ("map",)
    solver: str = "evidence"             # "evidence" | "lstsq" (plain weighted least squares, no prior: teaching)
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
    # schema-3 force sigma (docs/dev/specs/2026-09-30-conformal-force-sigma-design.md, rev 2)
    ard_force_shape: str = "aniso"       # "iso" | "aniso" (Mahalanobis scores, forces_q_mahal)
    ard_shape_eps: float = 1e-3          # aniso ridge: V + eps tr(V)/3 I
    ard_coverage: float = 0.9            # 1 - alpha of the per-group conformal quantile q_g
    ard_groups: str = "distortion"       # "distortion" (8 groups: d bands x [z = z*]) | "none" (2)
    ard_cluster_size: float = 3.0        # sandwich block side, x r_cut (inf: whole configurations)
    ard_press: str = "exact"             # "exact" | "block" PRESS correction of the jackknife scores
    ard_shape_tau: float = 1.0           # fraction of sum sigma^2 kept in the shape factor R
    ard_transfer: str = "exponent"       # hold-out -> served scale: "exponent" (per-fit beta) | "sqrt" | "none"
    ard_n_min: int = 20                  # groups with fewer T_val configurations borrow a neighbour's scales
    ard_support: bool = True             # covariate-shift support flag (diagnostic)
    ard_support_features: str = "raw"    # "raw" phi | "normalised" phi/|phi| + log-norm channels (fit/support.py)
    ard_support_max_atoms: int = 50000
    _shape_variant: str = "press"        # bench-only ablation: "press" | "legacy" (#18 uncentred sandwich)
    _score_source: str = "fit"           # bench-only ablation: "fit" (P_fit) | "mixed" (#18 own-cluster-out)
    deriv_dtc: bool = True
    predict_stats: str = "cached"        # "cached" (the objective's; run.py) | "recompute" (per draw) |
                                         # "auto" (CLI: the objective's where bitwise a recompute's, the QR form)
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

    @property
    def joint_e0(self):
        """e0='lsq' fits E0 jointly (E0 columns in the linear model, a wide prior around the
        pre-fit value) under BLR and ARD (whose evidence keeps the E0 columns' fixed prior, outside
        the body-order groups). POPS keeps the pre-fit E0: it builds its own readout prior.
        (A learned-radial fit learns its radials with the pre-fit E0, then fits jointly.)"""
        return self.e0 == "lsq" and self.uq in ("blr", "ard")

    @property
    def pack_mode(self):
        """The build_dataset `pack` the pipeline uses: batch_pack, except that "auto" is "off" under the
        LOO objective, whose per-config (R, R) leverage blocks (R = 1 + 3 n_max + 6) are vmapped over a
        batch's C slots, so the larger C_eff of a packed layout would multiply that memory."""
        return "off" if (self.batch_pack == "auto" and self.objective == "loo") else self.batch_pack

    def validate(self):
        if self.batch_pack not in ("auto", "on", "off"):
            raise ValueError(f"batch_pack must be 'auto', 'on' or 'off', got {self.batch_pack!r}")
        if self.e0 not in ("lsq", "prefit", "model"):
            raise ValueError(f"e0 must be 'lsq', 'prefit' or 'model', got {self.e0!r}")
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
            for name, ok in (("ard_force_shape", ("iso", "aniso")), ("ard_groups", ("distortion", "none")),
                             ("ard_support_features", ("raw", "normalised")),
                             ("ard_press", ("exact", "block")), ("ard_transfer", ("exponent", "sqrt", "none")),
                             ("_shape_variant", ("press", "legacy")),
                             ("_score_source", ("fit", "mixed"))):
                if getattr(self, name) not in ok:
                    raise ValueError(f"{name} must be one of {ok}, got {getattr(self, name)!r}")
            if self._score_source == "mixed" and (self._shape_variant != "legacy" or self.ard_variance != "sandwich"):
                raise ValueError("_score_source='mixed' is the #18 own-cluster-out rule: it needs "
                                 "_shape_variant='legacy' and ard_variance='sandwich'")
            if not 0.0 < self.ard_coverage < 1.0:
                raise ValueError(f"ard_coverage must be in (0, 1), got {self.ard_coverage}")
            if self.ard_n_min < 1:
                raise ValueError(f"ard_n_min must be >= 1, got {self.ard_n_min}")
            if not self.ard_cluster_size > 0:
                raise ValueError(f"ard_cluster_size must be > 0 (inf: whole configurations), got {self.ard_cluster_size}")
            if not 0.0 < self.ard_shape_tau <= 1.0:
                raise ValueError(f"ard_shape_tau must be in (0, 1], got {self.ard_shape_tau}")
            if not self.ard_shape_eps >= 0:
                raise ValueError(f"ard_shape_eps must be >= 0, got {self.ard_shape_eps}")
            if self.ard_force_shape == "aniso" and not self.ard_shape_eps > 0:
                raise ValueError(f"ard_shape_eps must be > 0 with ard_force_shape='aniso' (the Mahalanobis "
                                 f"solve of V + eps tr(V)/3 I is singular for a rank-deficient V), got {self.ard_shape_eps}")
        if self.lml == "host-cache":
            if (self.arm != "gp" or self.density not in ("pair", "pca") or tuple(self.rungs) != ("map",)
                    or self.opt != "lbfgs"):
                raise ValueError("lml='host-cache' needs arm gp, density pair|pca, rungs ('map',) and "
                                 "opt lbfgs (the cached LML exposes value_and_grad for L-BFGS)")
            if self.devices > 1:
                raise ValueError("lml='host-cache' is single-device: set devices=1")
        if self.opt not in ("lbfgs", "adam"):
            raise ValueError(f"opt must be 'lbfgs' or 'adam', got {self.opt!r}")
        if self.map_polish not in ("auto", "on", "off"):
            raise ValueError(f"map_polish must be 'auto', 'on' or 'off', got {self.map_polish!r}")
        if self.map_polish == "on" and self.opt != "lbfgs":
            raise ValueError("map_polish 'on' polishes the L-BFGS MAP: set opt lbfgs")
        if self.map_polish == "on" and self.lml == "host-cache":
            raise ValueError("map_polish 'on' differentiates the LML twice (Hessian-vector products): "
                             "not with lml 'host-cache'")
        if self.map_restarts < 1:
            raise ValueError(f"map_restarts must be >= 1, got {self.map_restarts}")
        if self.map_restarts > 1 and (self.opt != "lbfgs" or self.sigma_type):
            raise ValueError("map_restarts > 1 is the L-BFGS multi-start: set opt lbfgs "
                             "(and not sigma_type)")
        from ..paramset import NOISE_MODES
        if self.noise not in NOISE_MODES:
            raise ValueError(f"noise must be one of {NOISE_MODES}, got {self.noise!r}")
        if self.noise == "shared":
            why = ("sigma_type (per-config-type noise ratios on per-quantity scales)" if self.sigma_type else
                   "uq ard with ard_mode joint (it refits sigma_E/F/V by its own evidence; use ard_mode "
                   "sequential)" if self.uq == "ard" and self.ard_mode == "joint" else
                   "solver lstsq (no noise hyperparameters)" if self.solver == "lstsq" else None)
            if why is not None:
                raise ValueError(f"noise 'shared' is not available with {why}")
        if self.pops_rows not in ("auto", "host", "device"):
            raise ValueError(f"pops_rows must be 'auto', 'host' or 'device', got {self.pops_rows!r}")
        if self.predict_stats not in ("cached", "recompute", "auto"):
            raise ValueError(f"predict_stats must be 'cached', 'recompute' or 'auto', got {self.predict_stats!r}")
        if self.lml_solver not in ("qr", "cholesky"):
            raise ValueError(f"lml_solver must be 'qr' or 'cholesky', got {self.lml_solver!r}")
        if self.solver not in ("evidence", "lstsq"):
            raise ValueError(f"solver must be 'evidence' or 'lstsq', got {self.solver!r}")
        if self.solver == "lstsq" and (self.arm != "linear" or self.uq != "blr" or self.learn_radial
                                       or tuple(self.rungs) != ("map",)):
            raise ValueError("solver lstsq is the plain linear least-squares fit: it needs arm linear, uq blr, "
                             "rungs ('map',) and no learn_radial")
        if self.fix_rho is not None and self.opt != "lbfgs":
            raise ValueError("fix_rho is implemented for opt lbfgs only")
        if self.learn_radial:
            if self.baseline is not None or self.base_npz is not None:
                raise ValueError("learn_radial with a baseline: the fit saves no model file, so the "
                                 "learned radials would be lost")
            if not 0.0 < self.radial_val_frac < 1.0:
                raise ValueError(f"radial_val_frac must be in (0, 1), got {self.radial_val_frac}")
            if self.radial_n_q < 1:
                raise ValueError(f"radial_n_q must be >= 1, got {self.radial_n_q}")
            if self.radial_steps < 0:
                raise ValueError(f"radial_steps must be >= 0, got {self.radial_steps}")
            if not len(self.radial_lam_grid):
                raise ValueError("radial_lam_grid must hold at least one roughness weight")
        return self
