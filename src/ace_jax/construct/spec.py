"""Integer basis specs (mb_spec / Rnl_spec / Ylm_spec) for the coupling shim,
import numpy as np
and the ported real-RPE admissibility filter.

The admissibility predicate is a direct port of ACEpotentials
`src/models/utils.jl:37` `_rpe_filter_real(L)` (L = 0 here): a body-tuple of
(n, l, m) channels is admissible iff some signed sum of the m's is <= L in
absolute value, sum(l) + L is even, and the L=0 single-channel special case
(one channel => l == 0) holds.
"""
from itertools import product


def _mm_filter(mm, L=0):
    """Exists signs s in {-1,+1}^len(mm) with |sum s_i m_i| <= L.  Order is small
    (<= correlation order, ~5), so the 2^n enumeration is fine."""
    if not mm:
        return True
    return any(abs(sum(s * m for s, m in zip(signs, mm))) <= L
               for signs in product((-1, 1), repeat=len(mm)))


def rpe_admissible(bb, L=0):
    """Port of _rpe_filter_real(L).  bb: sequence of (n, l, m) tuples/dicts."""
    if len(bb) == 0:
        return True
    ll = [b[1] if not isinstance(b, dict) else b["l"] for b in bb]
    mm = [b[2] if not isinstance(b, dict) else b["m"] for b in bb]
    mm_ok = _mm_filter(mm, L)
    parity_ok = (sum(ll) + L) % 2 == 0
    special = (ll[0] == 0) if (len(bb) == 1 and L == 0) else True
    return mm_ok and parity_ok and special


def ylm_spec(lmax):
    """Real spherical-harmonic (l, m) channels, l <= lmax, -l <= m <= l."""
    return [(l, m) for l in range(lmax + 1) for m in range(-l, l + 1)]


# NOTE (see docs/coupling-etshim-spec.md, "Known gap"): a complete from-scratch
# mb_spec that reproduces ACEpotentials' TotalDegree(NZ, 1/wL) enumeration AND its
# exact tie-break ordering is the ~400-600 LOC row-2 work and is deliberately NOT
# finished here -- this task is scoped to the *coupling bridge*. `spec_from_export`
# below reconstructs the specs from an existing export so the bridge and the layout
# contract can be parity-tested without first nailing the enumerator's ordering.
def spec_from_export(npz):
    """Reconstruct the A-basis spec (Rnl_idx, Ylm_idx) and per-order aa_spec from
    an exported model npz, as ground-truth inputs for the parity test."""
    import numpy as np
    z = np.load(npz)
    aspec = list(zip(np.asarray(z["aspec_r"]).tolist(), np.asarray(z["aspec_y"]).tolist()))
    order = 1
    aa = []
    while f"aa_spec_{order}" in z.files:
        aa.append(np.asarray(z[f"aa_spec_{order}"]))
        order += 1
    return aspec, aa


def _unflat(flat, lens, k):
    out, off = [], 0
    for n in lens:
        n = int(n)
        out.append([tuple(int(x) for x in flat[off + i * k:off + (i + 1) * k]) for i in range(n)])
        off += k * n
    return out


def spec_from_reference(npz):
    """Reconstruct the ET inputs (mb_spec bodies, Rnl_spec, Ylm_spec) and the dense
    reference A2B from a coupling_reference.jl oracle npz -- the ground truth for
    the numerical parity test."""
    import numpy as np
    z = np.load(npz)
    mb = _unflat(z["mb_flat"], z["mb_len"], 2)                      # list of list of (n, l)
    Rnl = [tuple(int(x) for x in r) for r in z["r_spec"]]          # (n, l)
    Ylm = [tuple(int(x) for x in y) for y in z["y_spec"]]          # (l, m)
    A2B_ref = np.zeros(tuple(int(v) for v in z["A2B_shape"]))
    A2B_ref[z["A2B_rows"], z["A2B_cols"]] = z["A2B_vals"]
    return mb, Rnl, Ylm, A2B_ref


def _level(n, l, NZ, wL):
    """ACEpotentials TotalDegree(NZ, 1/wL) level of a channel: n/NZ + l*wL."""
    return n / NZ + l * wL


def _couples_to_zero(ll):
    """Whether angular momenta ll can couple to total L=0: sum even AND the
    largest <= sum of the rest.  This is the L=0 admissibility ET enforces
    (the rpe m-filter alone is necessary but not sufficient: l=0 (x) l=2 passes
    the m-filter yet cannot reach L=0)."""
    if len(ll) == 0:
        return True
    s = sum(ll)
    return s % 2 == 0 and max(ll) <= s - max(ll)


def build_spec(NZ, order, totaldegree, wL=1.5, tol=1e-9):
    """From-scratch reproduction of ACEpotentials' ace1_model basis selection
    (Models.oneparticle_spec + sparse_AA_spec, src/models/{smoothness_priors,utils}.jl):
    the many-body mb_spec (list of (n,l) bodies), the one-particle Rnl_spec (n,l),
    and Ylm_spec (l,m), for TotalDegree(NZ, 1/wL) with max level = totaldegree.

    Returns (mb_spec, Rnl_spec, Ylm_spec).  Ordering need not match ACEpotentials'
    (the parity aligns basis functions by their invariant signatures); the SET is
    what must agree.  order = correlation order (body order - 1)."""
    import math
    md = totaldegree
    maxn1 = math.ceil(md * NZ)                                    # oneparticle_spec bounds
    maxl1 = math.ceil(md / wL)                                    # wl = 1/wL, maxl1 = ceil(md*wl)
    # one-particle (n,l), level <= md, sorted by (l, n)  [oneparticle_spec]
    Rnl = sorted([(n, l) for n in range(1, maxn1 + 1) for l in range(0, maxl1 + 1)
                  if _level(n, l, NZ, wL) <= md + tol], key=lambda b: (b[1], b[0]))
    # A-spec (n,l,m), stable-sorted by level (m does not change the level)
    A = [(n, l, m) for (n, l) in Rnl for m in range(-l, l + 1)]
    A.sort(key=lambda b: _level(b[0], b[1], NZ, wL))             # ascending level (stable)
    Alev = [_level(b[0], b[1], NZ, wL) for b in A]
    # AA/mb: non-decreasing A-index bodies up to `order`, pruned by level (DFS,
    # matching Polynomials4ML.gensparse), then rpe + couples-to-L=0 admissible.
    seen, mb = set(), []

    def emit(bb):
        if len(bb) == 0 or not rpe_admissible(bb) or not _couples_to_zero([b[1] for b in bb]):
            return
        nl = tuple((b[0], b[1]) for b in bb)
        if nl not in seen:
            seen.add(nl); mb.append([(b[0], b[1]) for b in bb])

    def dfs(start, bb, lvl):
        if bb:
            emit(bb)
        if len(bb) == order:
            return
        for i in range(start, len(A)):
            if lvl + Alev[i] > md + tol:                         # A sorted ascending -> prune tail
                break
            bb.append(A[i]); dfs(i, bb, lvl + Alev[i]); bb.pop()

    dfs(0, [], 0.0)
    return mb, Rnl, ylm_spec(maxl1)
