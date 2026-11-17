"""Calibrated per-atom force uncertainty of a fitted linear model, served for one structure.

`ForceUQ` is what a calculator holds: the ARD posterior of a fit (`ard.ARDPosterior`, a `posterior.npz`)
checked against the model it belongs to, and the per-atom quantities it gives -- forces_std, forces_cov,
forces_q, forces_q_mahal, forces_group and the covariate-shift check forces_support.  It knows no model:
the rows come from a `design`, an object with

    n_columns, meta, coefficients()      the readout: its width, the model's meta, its coefficients
    one_config(atoms) -> (ds, b, live)   a one-structure dataset, its batch, the live atoms' mask
    has_slots(b), has_neighbours(b)      whether the batch has neighbour slots / any neighbour
    force_rows(b)                        the force design rows (N, 3, L) of the batch's atoms
    committee_coefs(R, dinv)             the shape factor in the form `committee_shape` takes
    committee_shape(b, coefs)            the per-atom shape V (N, 3, 3) by committee forces
    site_descriptors(ds, b)              the live atoms' descriptors the support reference uses

`ace_jax.fit.design.ForceDesign` is the one for an ACE potential."""
import numpy as np

from .ard import _NEED3, ARDPosterior

PROPS = ("forces_std", "forces_cov", "forces_q", "forces_q_mahal", "forces_group", "forces_support")


class ForceUQ:
    def __init__(self, posterior, design, shape_path="rows"):
        self.posterior, self.design, self.shape_path = posterior, design, shape_path
        self._committee = (design.committee_coefs(posterior.R, posterior.dinv)   # on the device once
                           if shape_path == "committee" else None)

    @classmethod
    def load(cls, path, design, shape_path="rows", shape_tau=1.0, shape_rank=None):
        """The posterior at `path` for the model of `design`, refused unless it is that model's: the same
        basis and elements, and its mean the model's coefficients.  shape_path "rows" (the shape from the
        force rows) or "committee" (from committee forces of the shape factor R); shape_tau / shape_rank
        truncate R (`jackknife.truncate_shape_factor`)."""
        import jax.numpy as jnp
        meta = design.meta
        post = ARDPosterior.load(path)
        if post.prior_root.M:
            raise ValueError(f"posterior {path} is from --uq ard-gp (a GP fit): serve it with "
                             "GPCalculator.from_file(gp_model.npz, posterior=...)")
        NZ = len(meta["elements"])
        L = design.n_columns
        # a joint-E0 posterior (meta e0_cols) carries NZ E0 columns after the readout: the model
        # file folds their shift into E0, and they are zero on every force row (ARDPosterior pads)
        if (len(post.mean) - post.n_e0 != L or post.meta.get("n_B") != meta["n_B"]
                or post.meta.get("NZ") != NZ):
            raise ValueError(f"posterior {path} does not match the model: "
                             f"basis {len(post.mean) - post.n_e0} vs {L}")
        p_els = [int(e) for e in post.meta.get("elements", [])]
        m_els = [int(e) for e in meta["elements"]]
        if p_els != m_els:                           # order-sensitive: columns are per species index
            raise ValueError(f"posterior {path} does not match the model: "
                             f"elements {p_els} vs {m_els}")
        # the posterior must belong to THIS model's readout: its mean is the fit's coefficients in the
        # design's column layout, bit-identical when one run wrote both files
        coef = design.coefficients()
        mean = np.asarray(post.mean, np.float64)[:len(post.mean) - post.n_e0]
        if coef.shape != mean.shape or not np.allclose(
                coef, mean, rtol=1e-10, atol=1e-14 * max(float(np.abs(mean).max(initial=0.0)), 1e-300)):
            raise ValueError(f"posterior {path} does not match the model: its mean is not the "
                             f"model's coefficients (a posterior from a different fit?)")
        # the L x L Cholesky factor and the sandwich factor: host->device once, not on every call
        if post.chol is not None:          # only kappa posteriors carry it (the sandwich shape serves from R)
            post = post._replace(chol=jnp.asarray(post.chol, jnp.float64))
        if post.Q is not None:
            post = post._replace(Q=jnp.asarray(post.Q, jnp.float64))
        if shape_path not in ("rows", "committee"):
            raise ValueError(f"shape_path must be 'rows' or 'committee', got {shape_path!r}")
        if post.R is not None:
            from .jackknife import truncate_shape_factor
            if not 0.0 < float(shape_tau) <= 1.0:
                raise ValueError(f"shape_tau must be in (0, 1], got {shape_tau!r}")
            R = truncate_shape_factor(post.R, float(shape_tau))
            if shape_rank is not None:
                if int(shape_rank) < 1:
                    raise ValueError(f"shape_rank must be >= 1, got {shape_rank!r}")
                R = R if int(shape_rank) >= R.shape[1] else truncate_shape_factor(R, int(shape_rank))
            post = post._replace(R=jnp.asarray(R, jnp.float64))
        elif shape_path == "committee" or shape_tau != 1.0 or shape_rank is not None:
            raise ValueError("shape_path='committee', shape_tau and shape_rank need a schema-3 jackknife shape factor R "
                             "(fit --uq ard with the default --ard-variance sandwich)")
        return cls(post, design, shape_path)

    def served_set(self):
        """The properties that share one pass over the force rows and the shape V: forces_std, plus
        forces_cov / forces_q / forces_group (schema 3) and forces_q_mahal (aniso).  Any one of them
        requested computes all of them."""
        post = self.posterior
        out = {"forces_std"}
        if post.group_table is not None:
            out |= {"forces_cov", "forces_q", "forces_group"}
            if post.force_shape == "aniso":
                out.add("forces_q_mahal")
        return out

    def quantities(self, atoms, which):
        """Per-atom properties of `atoms` as a dict: the requested ones (a subset of PROPS) together with
        every other property of `served_set` (one rows + shape pass serves them all): forces_std (N,),
        forces_cov (N,3,3), forces_q (N,), forces_q_mahal (N,; aniso only), forces_group (N,) int;
        forces_support (dict) only when requested (it is the expensive one).  Calibrated ARD std (lam x
        the shape, or kappa x the posterior std for --ard-variance kappa / schema 1) from the design rows."""
        post, design = self.posterior, self.design
        which = set(which)
        if which & {"forces_cov", "forces_q", "forces_q_mahal", "forces_group"} and post.group_table is None:
            raise ValueError(_NEED3)
        if "forces_q_mahal" in which and post.force_shape != "aniso":
            from ase.calculators.calculator import PropertyNotImplementedError
            raise PropertyNotImplementedError("forces_q_mahal needs an aniso posterior (--uq ard "
                                              "--force-shape aniso)")
        if "forces_support" in which and post.support is None:
            raise ValueError("this posterior has no support reference (fitted with --no-ard-support, or "
                             "predates it): refit with --uq ard without --no-ard-support")
        shared = self.served_set() if which - {"forces_support"} else set()
        n = len(atoms)
        ds, b, live = design.one_config(atoms)
        groups = (np.asarray(post.groups_of(b))[live].astype(np.int64)
                  if post.group_consts is not None else None)
        out = {}
        if "forces_support" in which:
            out["forces_support"] = self._support(ds, live, b)
        if not shared:
            return out
        if self.shape_path == "committee" and post.group_table is not None and design.has_slots(b):
            V = design.committee_shape(b, self._committee)[np.asarray(live)]
            out.update({k: np.asarray(v) for k, v in
                        post.served_from_V(V, groups, shared - {"forces_group"}).items()})
            out["forces_group"] = groups
            return out
        if not design.has_slots(b) or not design.has_neighbours(b):
            # no neighbours: forces are identically zero, so is every spread
            F = np.zeros((n, 3, len(np.asarray(post.mean))))
        else:
            # padding rows are zero; F[live] drops the padded nodes
            F = design.force_rows(b)
            if not bool(np.all(live)):
                F = F[live]
        if post.group_table is not None:
            out.update({k: np.asarray(v) for k, v in
                        post.served(F, groups, shared - {"forces_group"}).items()})
            out["forces_group"] = groups
        else:
            out["forces_std"] = np.asarray(post.forces_std(F, groups))
        return out

    def _support(self, ds, live, b):
        """Covariate-shift support of each atom against the posterior's reference (support.py), from
        the same descriptors the fit stage used (they depend on the model only)."""
        from .support import support_check
        post = self.posterior
        alpha = float(np.asarray(post.group_table["alpha"]))
        # alpha is the miscoverage (group_table); support_check takes it as such: q = Q_{1-alpha}
        return support_check(post.support, self.design.site_descriptors(ds, b), self.design.species(b)[live], alpha)

    def support_descriptors(self, atoms):
        """(X, Z) site descriptors and species indices of the live atoms; for `aj calibrate`."""
        ds, b, live = self.design.one_config(atoms)
        return self.design.site_descriptors(ds, b), self.design.species(b)[live]
