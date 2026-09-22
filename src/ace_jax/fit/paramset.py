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

def from_hypers(hypers, prior, embed=None, hypers_route="lml", embed_route="fixed"):
    from .hypers import to_array
    blocks = [ParamBlock("hypers", to_array(hypers), hypers_route, prior=prior)]
    if embed is not None:
        blocks.append(ParamBlock("embed", jnp.asarray(embed), embed_route))
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
