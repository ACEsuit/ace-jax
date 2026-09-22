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

def parse_route(spec):
    """Parse a `--route` argument into a validated {block_name: route} map.

    `spec` may be None/"" (-> {}), a JSON object string, or an already-decoded
    dict.  Every route must be one of ROUTES ('fixed'/'lml'/'varopt'); anything
    else, or a non-object JSON payload, raises ValueError so a bad CLI flag fails
    loudly rather than silently mis-routing a block."""
    import json
    if spec is None or spec == "":
        return {}
    d = spec if isinstance(spec, dict) else json.loads(spec)
    if not isinstance(d, dict):
        raise ValueError(f"--route must be a JSON object {{block: route}}, got {type(d).__name__}")
    bad = sorted({str(r) for r in d.values() if r not in ROUTES})
    if bad:
        raise ValueError(f"--route: unknown route(s) {bad}; routes are {list(ROUTES)}")
    return {str(k): str(v) for k, v in d.items()}


def apply_routes(ps, routes):
    """Return a ParamSet with each named block's route replaced by routes[name].
    A route naming a block that is absent from the set is ignored (a no-op)."""
    if not routes:
        return ps
    out = ParamSet(tuple(b._replace(route=routes.get(b.name, b.route)) for b in ps.blocks))
    _check_sigma_type_lml_layout(out)
    return out


def _check_sigma_type_lml_layout(ps):
    """`_sigma_type_decode` (objective.py) assumes the LML vector is exactly
    `[hypers | sigma_type free rows]`. If some OTHER block (e.g. `embed`) is
    also routed to "lml" while a `sigma_type` block is present, that block's
    values get interleaved into the LML vector ahead of the sigma_type rows
    and are silently mis-decoded as log-ratios -- a wrong fit with no error.
    Raise loudly instead."""
    names = [b.name for b in ps.blocks]
    if "sigma_type" not in names:
        return
    bad = sorted(b.name for b in ps.blocks
                 if b.route == "lml" and b.name not in ("hypers", "sigma_type"))
    if bad:
        raise ValueError(
            f"invalid route: block(s) {bad} routed to 'lml' alongside a 'sigma_type' "
            "block. The LML vector layout is [hypers | sigma_type free rows]; routing "
            "any other block to 'lml' would silently corrupt that decode. Route "
            f"{bad} to 'fixed' or 'varopt' instead.")


def build_fit_paramset(hypers, prior, *, embed=None, embed_route="fixed",
                       n_types=1, sigma_type=False, route=None):
    """Assemble the ParamSet a run.py fit optimises: an LML `hypers` block, an
    optional `embed` block, and -- only when `sigma_type` and n_types > 1 -- the
    per-config-type `sigma_type` LML block (Task 6), with any `--route` overrides
    applied last.  With `sigma_type=False`, no embed and no route (the default),
    this is exactly `from_hypers(hypers, prior)`: a single all-LML hypers block,
    whose `run_map_ps` reduces to the flat `run_map` (Task 3 equivalence), so a
    default-flags run stays numerically unchanged.

    Raises `ValueError` if `route` would route a block other than `hypers`/
    `sigma_type` to "lml" while a `sigma_type` block is present -- see
    `_check_sigma_type_lml_layout`."""
    nt = int(n_types) if sigma_type else 1
    ps = from_hypers(hypers, prior, embed=embed, embed_route=embed_route, n_types=nt)
    return apply_routes(ps, parse_route(route))


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
