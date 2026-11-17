"""The design rows of a fitted ACE potential for one structure: what `ace_jax.uq.force.ForceUQ` asks of
a model (its docstring lists the interface).  Rows come from the FULL fit model (the lean evaluation form
is energy-only), under the highest matmul precision."""
import numpy as np

from ..eval import highest_precision


class ForceDesign:
    def __init__(self, model, meta, model_path=None):
        """model: the full ACEModel in float64; meta: its meta; model_path: the model.npz it was read from
        (the readout coefficients a posterior is checked against)."""
        from .gp.inducing import GPConfig
        self.model, self.meta, self.model_path = model, meta, model_path
        NZ = len(meta["elements"])
        self.cfg = GPConfig(r0=1.0, rcut=float(meta["rcut"]), n_B=meta["n_B"], n_pair=meta["n_pair"], NZ=NZ, C=1)
        self.n_columns = (meta["n_B"] + meta["n_pair"]) * NZ
        self.rows_fn = None                                # compiled once per cell shape (MD)

    def coefficients(self):
        """The readout in the `rows._place` layout (species-major WB blocks, then Wpair blocks)."""
        z = np.load(self.model_path)
        if "WB" not in z.files or "Wpair" not in z.files:
            raise ValueError(f"posterior= needs the model.npz written by the same fit; {self.model_path} "
                             f"has no WB/Wpair readout")
        return np.concatenate([np.asarray(z["WB"]).T.ravel(), np.asarray(z["Wpair"]).T.ravel()])

    def one_config(self, atoms):
        """(ds, b, live): the one-config dataset of atoms, its batch, and the live-node mask."""
        import jax

        from .data import Config, build_dataset
        c = Config(atoms.get_positions(), atoms.get_atomic_numbers(), atoms.get_cell().array, atoms.get_pbc(),
                   None, None, None, 1.0, 1.0, 1.0)
        ds = build_dataset([c], self.meta, np.zeros(len(self.meta["elements"])), 1)
        b = jax.tree.map(lambda a: a[0], ds)
        return ds, b, np.asarray(b.node_mask)

    @staticmethod
    def has_slots(b):
        return b.nbr.shape[1] > 0

    @staticmethod
    def has_neighbours(b):
        return bool(np.asarray(b.nbr_mask).any())

    @staticmethod
    def species(b):
        return np.asarray(b.node_z)

    def force_rows(self, b):
        """Force design rows (Ncap, 3, L) of batch b, node-chunked; padding rows are zero."""
        from .rows import chunked_rows_fn
        with highest_precision():
            if self.rows_fn is None:
                self.rows_fn = chunked_rows_fn(self.model, self.cfg)
            return self.rows_fn(b).F

    def committee_coefs(self, R, dinv):
        from .jackknife import committee_coefs
        return committee_coefs(R, dinv, self.cfg)

    def committee_shape(self, b, coefs):
        """The per-atom shape V (Ncap, 3, 3) = P P^T from the committee forces P of the shape factor."""
        import jax.numpy as jnp

        from .jackknife import committee_forces
        with highest_precision():
            P = committee_forces(self.model, self.cfg, b, *coefs)
            return np.asarray(jnp.einsum("nar,nbr->nab", P, P))

    def site_descriptors(self, ds, b):
        """Live-atom site descriptors (the ones the support reference uses) of one-config dataset ds."""
        from .gp.inducing import site_features
        live = np.asarray(b.node_mask)
        with highest_precision():
            return np.asarray(site_features(self.model, self.cfg, ds)[0])[0][live]
