"""Typed hyperparameter blocks with a fitting route. The inner MAP optimises
the LML blocks, the outer VarOpt the VarOpt blocks; FIXED blocks never move.
Both optimisers see a flat vector of their route's blocks; `materialise` turns
the whole set back into the concrete objects stats/kernels/predict consume."""
from typing import NamedTuple, Optional
import jax.numpy as jnp

ROUTES = ("fixed", "lml", "varopt")

class ParamBlock(NamedTuple):
    name: str
    value: jnp.ndarray
    route: str
    prior: Optional[object] = None    # (mu, sigma) log-normal for LML blocks
    anchor: Optional[object] = None   # AnchorSpec for VarOpt blocks

class ParamSet(NamedTuple):
    blocks: tuple

    def block(self, name):
        return next(b for b in self.blocks if b.name == name)

    def _vec(self, route):
        parts = [b.value.reshape(-1) for b in self.blocks if b.route == route]
        return jnp.concatenate(parts) if parts else jnp.zeros((0,))

    def _set(self, route, x):
        out, i = [], 0
        for b in self.blocks:
            if b.route == route:
                n = b.value.size
                out.append(b._replace(value=x[i:i + n].reshape(b.value.shape)))
                i += n
            else:
                out.append(b)
        return ParamSet(tuple(out))

    def lml_vector(self):        return self._vec("lml")
    def set_lml_vector(self, x): return self._set("lml", x)
    def varopt_vector(self):     return self._vec("varopt")
    def set_varopt_vector(self, x): return self._set("varopt", x)

def sigma_type_block(n_types, init=None, prior_sigma=1.5):
    """An LML ParamBlock carrying the FREE per-config-type log-noise ratios:
    value shape (n_types-1, 3) (columns E, F, V).  The default type (row 0) is
    pinned to 0 by EXCLUSION -- it is never in the optimised vector; the full
    (n_types, 3) log_ratios is reconstructed with a leading zero row.  The prior
    is an independent N(0, prior_sigma) on each free entry (a weak log-normal on
    the ratio), stored as (mu, sigma) flat arrays for the LML MAP."""
    shape = (max(n_types - 1, 0), 3)
    val = jnp.zeros(shape) if init is None else jnp.asarray(init).reshape(shape)
    mu = jnp.zeros(val.size)
    sig = jnp.full(val.size, float(prior_sigma))
    return ParamBlock("sigma_type", val, "lml", prior=(mu, sig))


def from_hypers(hypers, prior, embed=None, hypers_route="lml", embed_route="fixed",
                n_types=1, sigma_type_init=None, sigma_type_prior_sigma=1.5):
    from .hypers import to_array
    blocks = [ParamBlock("hypers", to_array(hypers), hypers_route, prior=prior)]
    if embed is not None:
        blocks.append(ParamBlock("embed", jnp.asarray(embed), embed_route))
    if n_types > 1:
        blocks.append(sigma_type_block(n_types, sigma_type_init, sigma_type_prior_sigma))
    return ParamSet(tuple(blocks))

def materialise(self):
    from .hypers import from_array
    h = from_array(self.block("hypers").value)
    embed = None
    try:
        embed = self.block("embed").value
    except StopIteration:
        pass
    return h, embed
ParamSet.materialise = materialise

def sigma_type_ratios(self):
    """Full (n_types, 3) log_ratios with the pinned default row prepended, or
    None when there is no sigma_type block (single-type fit)."""
    try:
        free = self.block("sigma_type").value          # (n_types-1, 3)
    except StopIteration:
        return None
    return jnp.concatenate([jnp.zeros((1, 3)), free], axis=0)
ParamSet.sigma_type_ratios = sigma_type_ratios

def n_types(self):
    r = self.sigma_type_ratios()
    return 1 if r is None else int(r.shape[0])
ParamSet.n_types = n_types
