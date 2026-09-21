"""ACEpotentials' algebraic smoothness prior, ported from Julia.

Source: ACEpotentials.jl ``src/models/smoothness_priors.jl`` (v0.10.2, branch
``gp/hybrid-ace-gp``)::

    algebraic_smoothness_prior(model; p = 4, wl = 2/3, wn = 1.0) =
          smoothness_prior(model, bb -> sum((b.l/wl)^p + (b.n/wn)^p for b in bb))

``smoothness_prior`` (lines 105-123) evaluates the functional on the FULL
per-column nnll basis ``_nnll_basis`` (lines 69-85): for each species, the
tensor B-functions' bodies followed by the pair singletons, i.e.
(n_tensor + n_pair) * NZ columns -- the layout ``julia/smoothness_reference.jl``
dumps as ``nnll_flat``/``nnll_len``.  The result is ``Diagonal(gamma)``.

Two Julia branches are deliberately NOT ported and must be asserted away by the
caller:

* the embedded-model n-folding (lines 110-117) triggers only on
  ``meta["embedding"]``, which ``ace1_model`` never sets;
* the ``_coupling_scalings`` product is commented out in the Julia source
  (line 122); only the ACE1-compat variant (line 146, unused by the exporter)
  applies it.

The constants (p, wl, wn) are the PRIOR's defaults, fixed by the exporter's
argument-free call (export_model.jl:352) and INDEPENDENT of the basis-selection
level parameters (wL/NZ): basis selection only decides which nnll exist.

Bit-exactness: with wl = 2/3 every term is (3l)^4/16 or n^4 -- an integer over
2^4, exactly representable in float64 -- and the partial sums stay far below
2^53, so Julia's and Python's float paths agree bit-for-bit.
"""
import numpy as np

# The exporter's argument-free defaults (smoothness_priors.jl:125).
P, WL, WN = 4.0, 2.0 / 3.0, 1.0


def unflatten_nnll(nnll_flat, nnll_len):
    """(data, per-column lengths) npz layout -> per-column [(n, l), ...] lists."""
    out, off = [], 0
    for k in nnll_len:
        k = int(k)
        out.append([(int(nnll_flat[off + 2 * i]), int(nnll_flat[off + 2 * i + 1]))
                    for i in range(k)])
        off += 2 * k
    return out


def smoothness_prior(nnll, *, p=P, wl=WL, wn=WN):
    """gamma[i] = sum((l/wl)**p + (n/wn)**p for (n, l) in nnll[i]).

    nnll: per-column body lists over the FULL basis (tensor + pair, per
    species).  Returns (len(nnll),) float64, the diagonal of
    ``algebraic_smoothness_prior``."""
    gamma = np.zeros(len(nnll), dtype=np.float64)
    for i, bb in enumerate(nnll):
        s = 0.0
        for n, l in bb:                       # body order = Julia's generator order
            s += (l / wl) ** p + (n / wn) ** p
        gamma[i] = s
    return gamma
