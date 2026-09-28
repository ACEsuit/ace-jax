"""Frozen element embeddings: the reduction of a foundation-model element table to
the d channels an embedded ACE model uses.  Port of ACEpotentials
src/models/embeddings.jl (embedding_rows, _pca_reduce, _generic_frame,
embedding_widths); parity-tested against julia/embedding_reference.jl.

The embedded model replaces the categorical radial factor delta_{z', Z2} by a
frozen row emb[Z2, k]; with lossless per-order widths it is a reparameterisation
of the categorical basis, with d_max < lossless a compression of it.  Pure numpy.
"""
import json
from math import comb

import numpy as np

_MASK64 = (1 << 64) - 1


def read_embedding(path):
    """(Z, table (S, D), meta) from an embedding artefact: JSON with keys `Z` and
    `emb` plus provenance (as ACEpotentials `read_mace_embedding`)."""
    d = json.loads(open(path).read())
    if "emb" not in d or "Z" not in d:
        raise ValueError(f"{path} is not an embedding artefact: expected keys `emb` and `Z`")
    table = np.asarray(d["emb"], dtype=np.float64)
    Z = [int(z) for z in d["Z"]]
    if table.shape[0] != len(Z):
        raise ValueError(f"emb has {table.shape[0]} rows but Z has {len(Z)} entries")
    return Z, table, {k: v for k, v in d.items() if k not in ("emb", "Z")}


def embedding_widths(S, order, d_max=None):
    """Per-correlation-order channel widths d_nu = min(d_max, C(S+nu-1, nu)).
    d_max=None is lossless (every order at its full species-tensor dimension)."""
    dims = [comb(S + nu - 1, nu) for nu in range(1, order + 1)]
    return dims if d_max is None else [min(int(d_max), x) for x in dims]


def _generic_frame(r, d):
    """(r, d) with orthonormal rows, deterministic and generic: an xorshift64*
    stream seeded by (r, d), then the thin QR.  Bit-for-bit Julia's
    `_generic_frame` (the columns must be generic vectors so the channel-diagonal
    many-body basis spans Sym^nu; structured frames such as DCT rows do not)."""
    x = (0x9E3779B97F4A7C15 ^ (r << 32) ^ d) & _MASK64
    Mt = np.zeros((d, r))
    for j in range(r):                      # Julia: for j = 1:r, i = 1:d (i inner)
        for i in range(d):
            x ^= x >> 12
            x = (x ^ (x << 25)) & _MASK64
            x ^= x >> 27
            Mt[i, j] = 2.0 * ((((x * 0x2545F4914F6CDD1D) & _MASK64) >> 11) / 2.0 ** 53) - 1.0
    Q, _ = np.linalg.qr(Mt, mode="reduced")
    return Q[:, :r].T


def _fix_signs(U, rtol=1e-8):
    """Flip each column so that its FIRST entry within rtol of the column's
    largest |.| is positive.  Plain argmax|.| is not enough: for two elements the
    normalised U is +-(1, 1)/sqrt2, +-(1, -1)/sqrt2 -- an exact tie that round-off
    breaks differently on different LAPACKs."""
    A = np.abs(U)
    idx = np.argmax(A >= A.max(axis=0) * (1.0 - rtol), axis=0)             # first near-max entry
    return U * np.sign(U[idx, np.arange(U.shape[1])])


def _rank_tol(shape, s0):
    """Numerical-rank tolerance on singular values (as ACEpotentials _pca_reduce)."""
    return max(shape) * np.finfo(np.float64).eps * s0


def principal_frame(R, d):
    """Uncentred principal frame of the rows of R (n, D), truncated at
    min(d, numerical rank): returns (coordinates U S (n, k), frame V (D, k)) with
    coordinates = R @ V and, for d >= rank, their Gram exactly R R^T.  Signs are
    fixed on U (`_fix_signs`; V flipped to match), so the result does not depend on
    the LAPACK build.  Shared by the embedding reduction (`_pca_reduce`) and the GP
    feature map (inducing.build_pmap, density='pca')."""
    R = np.asarray(R, dtype=np.float64)
    U, sv, Vt = np.linalg.svd(R, full_matrices=False)
    k = min(int(d), int(np.sum(sv > _rank_tol(R.shape, sv[0] if sv.size else 0.0))))
    Uf = _fix_signs(U[:, :k])
    sgn = np.sign(np.sum(Uf * U[:, :k], axis=0))                          # +1 kept, -1 flipped
    return Uf * sv[:k], Vt[:k].T * sgn


def _pca_reduce(R, d):
    """Principal-frame reduction of an (S, D) block to (S, d): rows normalised
    (full-row similarity is the target), R = U S V^T, P = U S truncated at the
    numerical rank r; d <= r keeps the leading d columns, d > r mixes the r
    principal coordinates into d generic channels, P @ _generic_frame(r, d).

    Sign convention (`_fix_signs`): each singular vector is flipped so that its
    first near-largest-|.| entry is positive.  Julia's `_pca_reduce` leaves the sign to LAPACK, so for
    d > rank its rows (and hence the channel-diagonal many-body basis, which is
    NOT invariant to flipping a principal coordinate) depend on the BLAS build;
    fixing it makes the model deterministic.  The element Gram is unaffected, and
    lossless widths span the same space either way."""
    R = np.array(R, dtype=np.float64)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    P, _ = principal_frame(R, min(R.shape))                               # U S, all r, fixed signs
    r = P.shape[1]
    if d <= r:
        return P[:, :d]
    return P @ _generic_frame(r, d)


def embedding_rows(table, Z, zlist, d=None, reduction="pca", normalise=True):
    """The (len(zlist), d) embedding block for the atomic numbers zlist, in that
    order, reduced to d channels (ACEpotentials `embedding_rows`).

    reduction='pca' (default) keeps the elements' similarity (cosine Gram) exactly
    for d >= rank; 'truncate' takes the first d columns (kept for comparison, not
    a faithful reduction).  normalise=True scales each reduced row to unit norm --
    AFTER the reduction, which matters for the BLR prior (see the Julia comment)."""
    table = np.asarray(table, dtype=np.float64)
    d = table.shape[1] if d is None else int(d)
    if d < 1:
        raise ValueError(f"d = {d} must be positive")
    if reduction not in ("pca", "truncate"):
        raise ValueError(f"reduction = {reduction!r}; expected 'pca' or 'truncate'")
    if reduction == "truncate" and d > table.shape[1]:
        raise ValueError(f"d = {d} out of range for a width-{table.shape[1]} table")
    pos = {int(z): i for i, z in enumerate(Z)}
    missing = [int(z) for z in zlist if int(z) not in pos]
    if missing:
        raise ValueError(f"element Z = {missing[0]} is not in the embedding table")
    block = table[[pos[int(z)] for z in zlist]]
    rows = block[:, :d].copy() if reduction == "truncate" else _pca_reduce(block, d)
    if normalise:
        nrm = np.linalg.norm(rows, axis=1)
        for z, n in zip(zlist, nrm):
            if not n > 0:
                raise ValueError(f"element {z} has a zero embedding row at d = {d}")
        rows = rows / nrm[:, None]
    return rows
