"""Integer basis specs (mb_spec / Rnl_spec / Ylm_spec) for the coupling shim,
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
