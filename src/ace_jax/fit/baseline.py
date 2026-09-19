"""Apply the MACE-MH-1 pairwise mean mu_0 to configurations: the (E, F, V) of
sum_{i<j} V_zz'(r_ij), subtracted from the labels before fitting and added back
after prediction (a deterministic Delta-learning baseline; predictive variances
are unchanged).  E/F/V use the same edge-strain convention as
ACEModel.energy_forces_virial, which the report verified equals mace_virial.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from ..eval import sparse_graph


def load_mean(npz):
    z = np.load(npz)
    pairs = [tuple(sorted(p)) for p in z["pairs"]]
    idx = {p: k for k, p in enumerate(pairs)}
    return dict(elements=list(z["elements"]), r=jnp.asarray(z["r"]),
                V=jnp.asarray(z["V"]), idx=idx, rcut=float(z["rcut"]))


def _pair_type(zi, zj, mean):
    a, b = (zi, zj) if zi <= zj else (zj, zi)
    return mean["idx"][(int(a), int(b))]


def _baseline_padded(pos, send, recv, shifts, ptype, emask, npad, rgrid, Vtab):
    """mu_0 (E, F, V) for one config, edges/atoms padded to a fixed capacity and
    masked; jitted ONCE (static shapes) so 1000+ configs do not each recompile.
    npad rows of pos are dummy; masked edges (emask=0) contribute nothing."""
    def energy(p, eps):
        rij = p[recv] - p[send] + shifts
        rij = rij + rij @ (0.5 * (eps + eps.T))
        rij = jnp.where(emask[:, None] > 0, rij, 1.0)   # padded edges -> (1,1,1): finite gradient
        r = jnp.linalg.norm(rij, axis=-1)
        Vij = jax.vmap(lambda t, rr: jnp.interp(rr, rgrid, Vtab[t]))(ptype, r)
        return 0.5 * jnp.sum(jnp.where(emask > 0, Vij, 0.0))
    E, (gp, ge) = jax.value_and_grad(energy, argnums=(0, 1))(pos, jnp.zeros((3, 3)))
    return E, -gp, -ge


def subtract_baseline(configs, mean, ncap=None, ecap=None):
    """Return configs with (E,F,V) labels reduced by mu_0 and the per-config
    baselines (E, F, V) to add back.  Pads to fixed (ncap atoms, ecap edges) so
    the jitted kernel compiles once for the whole set."""
    graphs = [sparse_graph(c.positions, c.cell, c.pbc, mean["rcut"]) for c in configs]
    ncap = ncap or max(len(c.numbers) for c in configs)
    ecap = ecap or max(len(g.senders) for g in graphs)
    rgrid, Vtab = mean["r"], mean["V"]
    fn = jax.jit(_baseline_padded)
    out, base = [], []
    for c, g in zip(configs, graphs):
        n, e = len(c.numbers), len(g.senders)
        pos = np.zeros((ncap, 3)); pos[:n] = c.positions
        send = np.zeros(ecap, np.int32); send[:e] = g.senders
        recv = np.zeros(ecap, np.int32); recv[:e] = g.receivers
        shifts = np.zeros((ecap, 3)); shifts[:e] = g.shifts
        ptype = np.zeros(ecap, np.int32)
        ptype[:e] = [_pair_type(c.numbers[i], c.numbers[j], mean) for i, j in zip(g.senders, g.receivers)]
        emask = np.zeros(ecap); emask[:e] = 1.0
        E, F, V = fn(jnp.asarray(pos), jnp.asarray(send), jnp.asarray(recv), jnp.asarray(shifts),
                     jnp.asarray(ptype), jnp.asarray(emask), ncap - n, rgrid, Vtab)
        E0, F0, V0 = float(E), np.asarray(F)[:n], np.asarray(V)
        base.append((E0, F0, V0))
        out.append(c._replace(energy=None if c.energy is None else c.energy - E0,
                              forces=None if c.forces is None else c.forces - F0,
                              virial=None if c.virial is None else c.virial - V0))
    return out, base
