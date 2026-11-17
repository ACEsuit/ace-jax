"""The A basis and its products, without a model.

The pooled one-particle basis A (per node, summed over neighbours) and the product basis AA
(products of A entries, one index array per correlation order) are what every model built on
the ACE basis shares, whatever it reads out: site energies (`EdgeSiteModel`), or equivariant
features for an operator.  Everything here takes arrays, not a model:

* pooling of edge features into nodes (`pool_sparse`, `pool_dense`);
* per-edge A rows from the two factors R and Y in two interchangeable forms (`edge_a`), the
  pooled A in the sparse and dense layouts (`pool_a_sparse`, `pool_a_dense`), and A pooled
  before the radial weights (`pool_first_dense`, `pool_first_sparse`);
* the product basis (`product_basis`, `product_basis_t`, `product_basis_dot`).

`estimate_a_bytes` and `with_edge_a_kind` read a model's A-basis fields (`aspec_r`, `aspec_y`,
`a_channels`, `edge_a_kind`, ...: the contract in `edge_model`'s docstring) and nothing else.
`EdgeSiteModel`'s methods of the same names call these with the model's own arrays.
"""
import dataclasses
import functools
import operator

import jax
import jax.numpy as jnp

EDGE_A_KINDS = ("gather", "matmul")


def pool_sparse(edge_feats, segment_ids, n_nodes, mask=None):
    """Sum edge features into nodes over a sparse edge list."""
    if mask is not None:
        edge_feats = jnp.where(mask[:, None], edge_feats, 0.0)
    return jax.ops.segment_sum(edge_feats, segment_ids, num_segments=n_nodes,
                               indices_are_sorted=False)


def pool_dense(edge_feats, mask):
    """Sum over a fixed neighbour axis: edge_feats (n_nodes, K, F), mask (n_nodes, K)."""
    return jnp.sum(jnp.where(mask[..., None], edge_feats, 0.0), axis=1)


# ------------------------------------------------------------------ product basis
def _mul_rows(At, s):
    return functools.reduce(operator.mul, [At[s[:, k]] for k in range(s.shape[1])])


def _product_basis_t_cpu(At, specs):
    return jnp.concatenate([_mul_rows(At, s) for s in specs], axis=0)


def _product_basis_t_default(At, specs):
    return jnp.concatenate([jnp.prod(At[s.T], axis=0) for s in specs], axis=0)


def _product_basis_cpu(A, specs):
    return _product_basis_t_cpu(A.T, specs).T


def _product_basis_default(A, specs):
    return jnp.concatenate([jnp.prod(A[:, g], axis=-1) for g in specs], axis=-1)


def product_basis_t(At, specs):
    """AA (n_AA, n) = prod_k At[spec[:, k]] from feature-major A (n_A, n), one
    (n_v, order) index array per correlation order in `specs`.

    The form is chosen per backend at lowering (`lax.platform_dependent`, so an
    exported bundle gets the form of the platform it is lowered for).  On the CPU,
    an explicit chain of whole-row multiplies: `jnp.prod`'s reverse mode lowers to
    pads, slices and transposing copies that XLA:CPU runs as strided scalar loops
    (14-18x the forward; 1.65x on the order-3 ACE SiGe step,
    docs/dev/cpu-gap-profile.md 6.2).  Elsewhere `jnp.prod`, the layout the GPU was
    tuned for (ace-vs-pace-gap.md 4.1).  Same multiply order, so the values agree
    bitwise; gradients to roundoff."""
    return jax.lax.platform_dependent(At, cpu=lambda At: _product_basis_t_cpu(At, specs),
                                      default=lambda At: _product_basis_t_default(At, specs))


def product_basis(A, specs):
    """`product_basis_t` for node-major A (n, n_A), returning (n, n_AA).  The CPU
    form multiplies feature-major rows (A.T) and transposes back."""
    return jax.lax.platform_dependent(A, cpu=lambda A: _product_basis_cpu(A, specs),
                                      default=lambda A: _product_basis_default(A, specs))


def _product_basis_dot_cpu(A, specs, C):
    return jnp.einsum("ai,ai->i", _product_basis_t_cpu(A.T, specs), C)


def _product_basis_dot_default(A, specs, C):
    return jnp.einsum("ia,ai->i", _product_basis_default(A, specs), C)


def product_basis_dot(A, specs, C):
    """sum_a AA[i, a] C[a, i] per node i: the product basis of node-major A (n, n_A)
    against per-node readout columns C (n_AA, n), without forming node-major AA on
    the CPU (its transpose back cost 6% of the order-2 ACE Cantor step).  Forms as
    `product_basis_t`."""
    return jax.lax.platform_dependent(A, C, cpu=lambda A, C: _product_basis_dot_cpu(A, specs, C),
                                      default=lambda A, C: _product_basis_dot_default(A, specs, C))


def one_hot_selector(idx, width, dtype):
    """One-hot (width, n) matrix S with X @ S == X[:, idx]."""
    return jnp.zeros((width, idx.shape[0]), dtype).at[idx, jnp.arange(idx.shape[0])].set(1)


def check_edge_a_kind(kind):
    if kind not in EDGE_A_KINDS:
        raise ValueError(f'edge_a_kind must be "gather" or "matmul", got {kind!r}')


# ------------------------------------------------------------------ A basis
def edge_a(R, Y, aspec_r, aspec_y, kind="gather", sel_r=None, sel_y=None):
    """Per-edge A-basis rows from radial columns R and harmonics Y.

    Two algebraically identical forms, selected by `kind`; they agree
    to bit-identity on values and (in f64) on gradients.

      "gather"  A = R[:, aspec_r] * Y[:, aspec_y]
      "matmul"  A = (R @ Sr) * (Y @ Sy),  Sr/Sy one-hot (`sel_r`, `sel_y`: `one_hot_selector`)

    They differ only in the reverse pass: the gather's adjoint is an axis-1
    scatter whose cost per slot grows with buffer length, while the matmul's
    adjoint is a matmul and is flat.  Neither wins everywhere -- on an M3 Pro
    the matmul wins throughout and by up to 11x, on a Xeon the gather wins
    below ~200k edge slots, and on an A4500 the gather's two scatters were
    66-84% of a PACE force call -- so this is a choice to calibrate, not a
    constant to hardcode.  See docs/findings/FINDINGS_apple_scaling.md and
    `edge_model.calibrate_edge_a`.
    """
    if kind == "matmul":
        # the one-hot products only select, so they must not round: at default
        # precision an f32 matmul on Ampere+ is TF32 (10-bit mantissa)
        hi = jax.lax.Precision.HIGHEST
        return jnp.matmul(R, sel_r, precision=hi) * jnp.matmul(Y, sel_y, precision=hi)
    return R[:, aspec_r] * Y[:, aspec_y]


def pool_a_sparse(rows, seg, channel, n_nodes, C=1):
    """A (n_nodes, C * n_A) from per-edge A rows (`edge_a`), summed per (node, channel); C
    neighbour-species channels.  Memory ~ E * n_A."""
    A = jax.ops.segment_sum(rows, seg * C + channel, num_segments=n_nodes * C)
    return A.reshape(n_nodes, C * rows.shape[1])


def pool_a_dense(cols, Y, channel, aspec_r, aspec_y, C=1):
    """A (n, C * n_A) from padded (n, K, .) factors, per node as a batched
    outer product sum_k cols_k (x) Y_k, then the used entries selected.  No
    per-edge column gather, so no scatter in the reverse pass (3-13x faster
    forces measured on A4500/A100/H100).  Memory ~ n * K * (C * n_cols + n_Y)
    + n * C * n_cols * n_Y, so it grows with channels and neighbour padding --
    which is why the sparse layout stays."""
    n, K, nc = cols.shape
    ny = Y.shape[-1]
    if C > 1:                                    # neighbour-species channel
        oh = jax.nn.one_hot(channel, C, dtype=cols.dtype)
        cols = (oh[..., :, None] * cols[..., None, :]).reshape(n, K, C * nc)
    A_full = jnp.einsum("nkr,nky->nry", cols, Y,
                        precision=jax.lax.Precision.HIGHEST).reshape(n, -1)
    mu = jnp.arange(C, dtype=aspec_r.dtype)[:, None]
    sel = ((mu * nc + aspec_r[None, :]) * ny + aspec_y[None, :]).reshape(-1)
    return A_full[:, sel]


def pool_first_dense(b, Y, zj, node_z, W, sel_y, C=1):
    """A_t (C*n_a, n) = sum_k W[z_i, z_j] b_k (x) Y_y, pooled BEFORE W.

    Every model here has a radial that is linear in per-pair coefficients
    over a fixed per-edge basis b (PACE: g_k with crad; ACE: polynomials,
    spline or factorised table with its weights), so the per-edge R_nl is
    never formed: pool b (x) Y per (node, neighbour-species channel), then
    apply W per node.  Feature-major output so the product-basis gathers read
    whole rows.  b (n, K, n_b), Y (n, K, n_Y), zj (n, K) neighbour channel, W (NZ, C, n_a, n_b)
    the per-pair weights, sel_y (n_Y, n_a) the one-hot harmonic selector.
    docs/dev/pace-performance-gap.md #4, #5."""
    n, K, nb = b.shape
    hi = jax.lax.Precision.HIGHEST
    if C > 1:
        oh = jax.nn.one_hot(zj, C, dtype=b.dtype)
        b = (oh[..., :, None] * b[..., None, :]).reshape(n, K, C * nb)
    Ag = jnp.einsum("nkg,nky->ngy", b, Y, precision=hi)                  # (n, C*nb, nY)
    Agy = jnp.matmul(Ag, sel_y, precision=hi)                            # (n, C*nb, n_a)
    Agy = Agy.reshape(n, C, nb, -1)
    At = jnp.einsum("nmka,nmak->man", Agy, W[node_z], precision=hi)     # (C, n_a, n)
    return At.reshape(C * At.shape[1], n)


def pool_first_sparse(b, Y, seg, zj, node_z, n_nodes, W, sel_y, C=1):
    """`pool_first_dense` for an edge list: pool b (x) Y per (node, channel)
    by segment_sum, then W per node.  Same output layout, (C*n_a, n_nodes)."""
    hi = jax.lax.Precision.HIGHEST
    nb = b.shape[1]
    outer = (b[:, :, None] * Y[:, None, :]).reshape(b.shape[0], nb * Y.shape[1])  # explicit: E may be 0
    Ag = jax.ops.segment_sum(outer, seg * C + (zj if C > 1 else 0),
                             num_segments=n_nodes * C)
    Ag = Ag.reshape(n_nodes, C, nb, -1)
    Agy = jnp.matmul(Ag, sel_y, precision=hi)                             # (n, C, nb, n_a)
    At = jnp.einsum("nmka,nmak->man", Agy, W[node_z], precision=hi)     # (C, n_a, n)
    return At.reshape(C * At.shape[1], n_nodes)


# ------------------------------------------------------------------ layouts
LAYOUTS = ("sparse", "dense")


def estimate_a_bytes(model, layout, n_nodes, n_edges, max_neighbours, itemsize):
    """Peak bytes of the A-basis stage (energy + forces; pool-first: the whole
    dense call), for choosing a layout.

    Fitted to measured peaks on A4500/A100/H100 (c_ace, 4k and 14k atoms):
    dense  ~ 2 n K (C n_cols + n_Y) + 3 n C n_cols n_Y   (factors + A_full)
    sparse ~ 2 E (n_cols + n_Y + n_A)                    (factors + per-edge rows)
    with C neighbour-species channels.  Dense grows with C and with the (n, K)
    padding; sparse only with the edge count.

    A pool-first model (`uses_edge_a` False; PACE) never forms per-edge columns
    or A rows: per edge it holds the fixed basis b (n_b wide, one-hot over the C
    channels) and Y; per node the pooled Ag = b (x) Y, A, the (n, C, n_b, n_a)
    Agy = Ag @ sel_y that W[z_i] contracts, and the product basis (n_P =
    `product_basis_width()`: the gathered A factors of every order and AA), so
    with n_b, n_Y = `pool_first_widths()`
    dense  ~ 2 n K (C n_b + n_Y) + per_node
    sparse ~ 2 E (n_b + n_Y + n_b n_Y) + per_node
    per_node = n C (n_b n_Y + n_a) + n C n_b n_a + n n_P / 4
    Refitted to XLA's compiled temp size of the dense E/F/V call on an A100
    (six PACE models x 1k/4k/8k atoms, float64,
    bench/perf/results/microbench_memory_float64.json): measured / estimate
    0.68-1.06, so the estimate errs high by at most 1.5x.  XLA's CPU backend
    keeps about twice as much live (fewer fusions), so on the CPU this is low
    by about 2x.
    """
    C, n_a = model.a_channels, int(model.aspec_r.shape[0])
    if not model.uses_edge_a:
        nb, ny = model.pool_first_widths()
        per_node = (n_nodes * C * (nb * ny + n_a) + n_nodes * C * nb * n_a
                    + n_nodes * model.product_basis_width() // 4)
        if layout == "dense":
            return itemsize * (2 * n_nodes * max_neighbours * (C * nb + ny) + per_node)
        if layout == "sparse":
            return itemsize * (2 * n_edges * (nb + ny + nb * ny) + per_node)
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    nc, ny = model.edge_a_widths()
    if layout == "dense":
        return itemsize * (2 * n_nodes * max_neighbours * (C * nc + ny) + 3 * n_nodes * C * nc * ny)
    if layout == "sparse":
        return itemsize * 2 * n_edges * (nc + ny + n_a)
    raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")


def with_edge_a_kind(model, kind):
    """Return `model` using the other A-basis form.  Values and gradients are
    unchanged (bit-identically, measured); only the reverse-pass cost differs."""
    check_edge_a_kind(kind)
    if kind == model.edge_a_kind:
        return model
    if kind == "gather":
        return dataclasses.replace(model, edge_a_kind="gather", a_sel_r=None, a_sel_y=None)
    n_r, n_y = model.edge_a_widths()
    dt = model.E0.dtype
    return dataclasses.replace(model, edge_a_kind="matmul",
                               a_sel_r=one_hot_selector(model.aspec_r, n_r, dt),
                               a_sel_y=one_hot_selector(model.aspec_y, n_y, dt))
