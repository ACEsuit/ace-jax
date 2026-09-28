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

The embedded-model branch (lines 110-117) IS ported: when the export's
``meta["embedding"]`` is set (export_model.jl KIND=embedding), the tensor
block's radial index is unfolded to n' = (n-1) div d_max + 1 before the
functional is applied; the pair block is categorical and untouched.  The
``_coupling_scalings`` product is NOT ported: it is commented out in the Julia
source (line 122) and only the ACE1-compat variant (line 146, unused by the
exporter) applies it.

The constants (p, wl, wn) are the PRIOR's defaults, fixed by the exporter's
argument-free call (export_model.jl:352) and INDEPENDENT of the basis-selection
level parameters (wL/NZ): basis selection only decides which nnll exist.

Bit-exactness: with wl = 2/3 every term is (3l)^4/16 or n^4 -- an integer over
2^4, exactly representable in float64 -- and the partial sums stay far below
2^53, so Julia's and Python's float paths agree bit-for-bit.
"""
import json

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


def model_nnll(meta):
    """Per-column (n, l) bodies of an exported model, over the design
    layout ``[tensor x NZ | pair x NZ]`` -- the species-blocked order of
    ``fit/rows.py:_place`` (i.e. ``ACEModel.site_descriptors``).

    The tensor part is the exporter's own spec dump ``meta["nnll"]``
    (``get_nnll_spec(m.tensor)``, export_model.jl:331): n_B entries, shared
    across species.  Pair columns are the singletons (n, 0), n = 1..n_pair --
    confirmed against the oracle fixtures.  (Rebuilding the tensor spec from
    aspec_r/aa_specs would additionally need the Rnl table's per-l block
    layout, which the npz does not carry.)

    meta: the parsed ``meta_json`` of an export.  numpy only -- no jax import,
    usable by authoring scripts.
    """
    if "nnll" not in meta:
        raise ValueError(
            "export has no gamma and its meta has no 'nnll' (tensor spec dump), so "
            "the smoothness prior cannot be rebuilt: re-export the model with a "
            "julia/export_model.jl that writes meta[\"nnll\"]")
    n_B, n_pair, NZ = int(meta["n_B"]), int(meta["n_pair"]), len(meta["elements"])
    tensor = [[(int(n), int(l)) for n, l in bb] for bb in meta["nnll"]]
    if len(tensor) != n_B:
        raise ValueError(f"meta nnll has {len(tensor)} columns, n_B = {n_B}")
    if "len_basis" in meta and int(meta["len_basis"]) != NZ * (n_B + n_pair):
        raise ValueError(f"meta len_basis = {meta['len_basis']} but the "
                         f"[tensor x NZ | pair x NZ] layout gives {NZ * (n_B + n_pair)} "
                         f"(n_B={n_B}, n_pair={n_pair}, NZ={NZ})")
    d_max = embedding_d_max(meta)
    if d_max is not None:
        # Embedded model: the exporter's nnll carries the channel k folded into
        # the radial index, n = (n'-1)*d_max + k, and k is not a degree.  Julia's
        # smoothness_prior unfolds the TENSOR block to n' before applying the
        # functional; the pair block is categorical and is left alone.
        tensor = [[((n - 1) // d_max + 1, l) for n, l in bb] for bb in tensor]
    pair = [[(n, 0)] for n in range(1, n_pair + 1)]
    return tensor * NZ + pair * NZ


def embedding_d_max(meta):
    """``d_max`` of an embedded model's export, or None for a plain model.

    export_model.jl writes ``meta["embedding"]`` as the JSON of the Julia
    model's ``meta["embedding"]`` dict for KIND=embedding and as ``""``
    otherwise; older exports omit the key entirely.  Anything present that
    does not carry a usable ``d_max`` is an error naming the key and the
    expected shape of the value."""
    emb = meta.get("embedding")
    if not emb:
        return None
    if isinstance(emb, str):
        try:
            emb = json.loads(emb)
        except json.JSONDecodeError as e:
            raise ValueError(f"meta['embedding'] = {emb!r} is not valid JSON "
                             f"({e}) -- expected an object with a 'd_max' key") from e
    if not isinstance(emb, dict) or "d_max" not in emb:
        raise ValueError(f"meta['embedding'] = {emb!r} does not carry a "
                         "'d_max' key -- expected the KIND=embedding export's "
                         "embedding dict")
    return int(emb["d_max"])


def gamma_from_model(meta):
    """The model's prior diagonal, rebuilt from its exported structure."""
    return smoothness_prior(model_nnll(meta))


def prior_diagonal(z, meta, source=""):
    """The prior diagonal of a loaded export: ``z["gamma"]`` when the exporter
    stored it (authoritative, used verbatim), else rebuilt from ``meta`` with a
    log line naming ``source`` (the model path).  The one fallback both
    ``cli.py`` and ``bench/acegp_cantor/run.py`` go through.  Returns float64
    numpy; callers convert to jax."""
    if "gamma" in z.files:
        return np.asarray(z["gamma"], np.float64)
    print(f"gamma missing from {source} -- rebuilt via construct.prior "
          "(algebraic smoothness prior)", flush=True)
    return gamma_from_model(meta)
