"""User-facing entry points: load a model from a path, get descriptors without ASE.

Whole-dataset site descriptors are the common active-learning case, and they
should not pay per-structure calculator overhead or an ASE `Atoms` round-trip --
hence `site_descriptors` takes positions/cell/species directly.
"""

import functools
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from .io import load
from .model import highest_precision
from .nlist import sparse_graph


def _resolve(model, meta=None, dtype=None):
    """Accept a path, or an already-loaded (model, meta) pair."""
    if isinstance(model, (str, pathlib.Path)):
        m, mt, _ = load(model, dtype=dtype or _default_dtype())
        return m, mt
    if meta is None:
        raise ValueError("pass a path, or both a model and its meta")
    return model, meta


def _default_dtype():
    """f64 when the caller has enabled x64, else f32.  Never sets it here."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def species_indices(meta, numbers):
    z2i = {int(z): i for i, z in enumerate(meta["elements"])}
    try:
        return np.array([z2i[int(z)] for z in numbers], np.int32)
    except KeyError as e:
        raise ValueError(
            f"element Z={e.args[0]} not in model elements {meta['elements']}") from e


@functools.partial(jax.jit, static_argnums=(5,))
def _descriptors_padded(model, rij, zi, zj, send, n_nodes, node_z, mask):
    return model.site_descriptors(rij, zi, zj, send, n_nodes, node_z, mask)


def site_descriptors(model, positions, numbers, cell=None, pbc=False,
                     meta=None, cutoff=None, dtype=None, domain=None):
    """Site descriptors, (n_atoms, (n_B + n_pair) * n_species).

    Parity target is `ACEpotentials.site_descriptors`, which is marked in the
    ACEpotentials source as "RETIRING THIS FOR NOW BECAUSE IT IS HIGHLY INEFFICIENT"
    because it recomputes per site.  This takes the whole batch from one forward
    pass -- the same pass the energy uses -- so the port is genuinely faster
    here, not merely equivalent.

    `domain` restricts the returned rows (as ACEpotentials does); the forward pass
    still covers the whole structure, since a site's descriptor needs its
    neighbours regardless.
    """
    model, meta = _resolve(model, meta, dtype)
    rcut = float(cutoff if cutoff is not None else meta["rcut"])
    positions = np.ascontiguousarray(positions, float)
    if cell is None:
        cell = np.eye(3) * (np.ptp(positions, axis=0).max() + 2 * rcut + 1.0)
        pbc = False
    g = sparse_graph(positions, cell, np.broadcast_to(pbc, 3), rcut)
    node_z = np.asarray(species_indices(meta, numbers))
    # pad the edge list to a power-of-two bucket (masked): structures of similar size
    # share one compiled pass (run eagerly, every op dispatched one by one)
    n_e = len(g.senders); cap = 1 << max(n_e - 1, 0).bit_length()
    pad = lambda a, v: np.concatenate([np.asarray(a), np.full((cap - n_e,) + np.shape(a)[1:], v, np.asarray(a).dtype)])
    rij = pad(np.asarray(g.rij, dtype=dtype or _default_dtype()).reshape(n_e, 3), 0.5 * rcut)
    send, recv = pad(np.asarray(g.senders, np.int32), 0), pad(np.asarray(g.receivers, np.int32), 0)
    mask = np.arange(cap) < n_e
    with highest_precision():
        d = _descriptors_padded(model, jnp.asarray(rij), jnp.asarray(node_z[send]), jnp.asarray(node_z[recv]),
                                jnp.asarray(send), g.n_nodes, jnp.asarray(node_z), jnp.asarray(mask))
    d = np.asarray(d)
    return d if domain is None else d[np.asarray(domain)]
