"""EquivariantTensors coupling via the compiled ace-jax-coupling library.

`couple(mb_spec, Rnl_spec, Ylm_spec)` returns the L = 0 real-basis coupling of
`EquivariantTensors.sparse_equivariant_tensor` in ace-jax's export layout (see
`Coupling`).  It calls `ace_jax_coupling.couple_raw`, a juliac-compiled build of
EquivariantTensors (no Julia at runtime), a core dependency on supported
platforms; the import is lazy, so a refit or a cache hit never loads it and an
unsupported platform raises `BasisUnavailable` only when a new shape is built.

`couple_cached(...)` wraps `couple` with a per-shape disk cache: the coupling
depends only on the three integer specs, so a new shape runs the shim once and
every later authoring of the same shape reconstructs the `Coupling` from the
cache without importing the library at all.
"""

import hashlib
import json
import os
import pathlib
import platform
import sys
import tempfile
import warnings

from typing import NamedTuple


class Coupling(NamedTuple):
    """Everything the ET call returns, in ace-jax's export layout.

    A2B: dense (n_B, n_AA) float64 coupling, one nonzero per column, its
        columns in the AA evaluation order of `aa_specs` (`align_columns`).
    aa_sig: per-column (n, l, m) tuples in A2B column order -- the column
        *identity*, from meta `𝔸spec`, permuted with the A2B columns.
    aspec: 0-based (Rnl_idx, Ylm_idx) per A function, in abasis.spec order --
        the index space A2B columns and aa_specs entries share.
    aa_specs: per-order (n_v, order) 0-based A-column index arrays, aabasis
        evaluation order (consumed as `A[:, g]` in eval).
    nnll_spec: per-B-row body lists from ET's `get_nnll_spec` -- exactly the
        exporter's meta["nnll"] source (per-row, not the input echo: the
        tensor meta also carries "mb_spec", which is the ET *input* echo).
    """

    A2B: object
    aa_sig: tuple
    aspec: tuple
    aa_specs: tuple
    nnll_spec: tuple


def subspace_residual(A, B):
    """Max-abs difference of the orthogonal projectors onto the row spaces of `A`
    and `B` (both shape (m, k)): ``||A⁺A_proj - B⁺B_proj||_max`` computed as
    ``|Qa Qaᵀ - Qb Qbᵀ|`` with Q an orthonormal basis of the row space (from a
    QR of the transpose).

    This is the invariant used to compare a DEGENERATE nnll coupling block across
    two basis constructions: at order >= 4 a block of multiplicity m has a
    symmetrisation basis defined only up to a within-block ORTHOGONAL rotation, so
    two enumerations give ``A`` and ``B = O A`` for some orthogonal ``O``.  The
    row-space projector is invariant to that rotation (and to row permutation and
    per-row sign), so a correct coupling gives ~0 here while the raw entries can
    differ substantially.  For multiplicity 1 it degenerates to the (sign- and
    scale-invariant) 1-D projector.  Pure numpy -- no Julia."""
    import numpy as np
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if A.shape != B.shape:
        raise ValueError(f"block shapes differ: {A.shape} vs {B.shape}")
    Qa = np.linalg.qr(A.T)[0]
    Qb = np.linalg.qr(B.T)[0]
    return float(np.abs(Qa @ Qa.T - Qb @ Qb.T).max())


COUPLING_LIB_VERSION = "0.2.0"   # == the core dependency pin (test_backend_id_matches_dependency_pin)


def backend_id():
    """Identity of the coupling backend, stored in cache entries so a library
    change invalidates them.  Computed WITHOUT importing the library (a cache
    hit must not need it installed)."""
    return f"ace-jax-coupling=={COUPLING_LIB_VERSION}"


class BasisUnavailable(RuntimeError):
    """Building a new basis needs the compiled coupling library, which is not
    installed (unsupported platform) or has no compiled payload (dev build)."""


def _lib():
    if os.environ.get("ACEJAX_COUPLING_CACHE_ONLY"):
        raise RuntimeError("ACEJAX_COUPLING_CACHE_ONLY is set but the coupling cache missed: "
                           "this basis shape has not been built before")
    try:
        import ace_jax_coupling
    except ModuleNotFoundError as e:
        raise BasisUnavailable(
            f"building a new basis is not available on this platform ({sys.platform} {platform.machine()}); "
            "fit from an existing basis with --model <file.npz> built elsewhere") from e
    try:
        ace_jax_coupling.build_info()
    except ace_jax_coupling.CouplingLibError as e:
        raise BasisUnavailable(f"the ace-jax-coupling install has no compiled library: {e}") from e
    if ace_jax_coupling.__version__ != COUPLING_LIB_VERSION:
        raise BasisUnavailable(f"ace-jax-coupling {ace_jax_coupling.__version__} is installed; this ace-jax "
                               f"needs =={COUPLING_LIB_VERSION}")
    return ace_jax_coupling


def couple(mb_spec, Rnl_spec, Ylm_spec):
    """mb_spec: list of list of (n, l); Rnl_spec: list of (n, l); Ylm_spec: list
    of (l, m).  Returns a `Coupling`.

    `aa_sig[j]` is the identity of A2B COLUMN j.  It is taken from the tensor's
    meta `𝔸spec` (the readable spec returned *alongside* the symmetrisation
    matrix), whose order is the library's A2B column order.  `SparseSymmProd`
    re-sorts its input by body length, so for an mb_spec that interleaves body
    orders (`build_spec`'s) the aabasis EVALUATION order (`aa_specs`) is a
    different permutation; `align_columns` permutes A2B and aa_sig into the
    evaluation order before returning, so column j of A2B multiplies AA
    product j."""
    import numpy as np
    raw = _lib().couple_raw(mb_spec, Rnl_spec, Ylm_spec)
    A2B = np.zeros(raw.A2B_shape, np.float64)
    A2B[raw.A2B_rows, raw.A2B_cols] = raw.A2B_vals
    so = raw.aa_sig_off
    aa_sig = tuple(tuple((int(n), int(l), int(m)) for n, l, m in raw.aa_sig[so[k]:so[k + 1]])
                   for k in range(len(so) - 1))
    aspec = tuple((int(r), int(y)) for r, y in raw.aspec)
    ao = raw.aa_off
    rows = [raw.aa_idx[ao[k]:ao[k + 1]] for k in range(len(ao) - 1)]
    max_ord = max(len(r) for r in rows)
    aa_specs = tuple(np.asarray([r for r in rows if len(r) == k], np.int64).reshape(-1, k)
                     for k in range(1, max_ord + 1))
    no = raw.nnll_off
    nnll_spec = tuple(tuple((int(n), int(l)) for n, l in raw.nnll[no[i]:no[i + 1]])
                      for i in range(len(no) - 1))
    return align_columns(Coupling(A2B=A2B, aa_sig=aa_sig, aspec=aspec, aa_specs=aa_specs,
                                  nnll_spec=nnll_spec), Rnl_spec, Ylm_spec)


def align_columns(cpl, Rnl_spec, Ylm_spec):
    """`cpl` with its A2B columns (and `aa_sig`) permuted into the AA
    evaluation order, the concatenated `aa_specs` rows that eval multiplies
    A2B against.  Idempotent.

    The library returns the A2B columns in ET's `𝔸spec` order (the body order
    of `mb_spec`, each body expanded into its m-realisations) but `aa_specs` in
    `SparseSymmProd` order, which stable-sorts the bodies by length.  Upstream
    ET multiplies its A2B against that sorted AA as is, so the two orders agree
    only for an `mb_spec` already grouped by body order (as ACEpotentials and
    `build_embedding_spec` pass).  `build_spec`'s DFS interleaves orders, and
    pairing the unpermuted columns with the AA products gave a model whose
    B functions were not rotation invariant.

    The permutation is found by identity, not assumed: each evaluation column's
    (n, l, m) signature, read through aspec -> (Rnl_spec, Ylm_spec), is looked up
    in `aa_sig`.  That lookup is unambiguous because the columns are the
    deduplicated A products, one per (n, l, m) multiset; the different coupling
    paths of one body are B rows, not columns.  `couple` and `couple_cached`
    (on a hit) both apply it, so an entry written before this fix is corrected
    on use."""
    import numpy as np
    key = lambda s: tuple(sorted(tuple(int(v) for v in t) for t in s))
    col = {key(s): j for j, s in enumerate(cpl.aa_sig)}
    rows = [r for g in cpl.aa_specs for r in np.asarray(g)]
    ev = [key((*Rnl_spec[cpl.aspec[a][0]], Ylm_spec[cpl.aspec[a][1]][1]) for a in r) for r in rows]
    if len(col) != len(cpl.aa_sig) or len(ev) != len(col) or set(ev) != set(col):
        raise ValueError("coupling: the AA evaluation columns (aa_specs) and the A2B column "
                         "signatures (aa_sig) are not the same set of products")
    perm = np.array([col[s] for s in ev], dtype=np.int64)
    if (perm == np.arange(len(perm))).all():
        return cpl
    A2B = np.asarray(cpl.A2B)[:, perm]
    return cpl._replace(A2B=A2B, aa_sig=tuple(cpl.aa_sig[j] for j in perm))


# --------------------------------------------------------------------- cache

_CACHE_SCHEMA = 1   # also hashed into coupling_key: bumping it moves every key; the "backend" stamp invalidates entries


def coupling_key(mb_spec, Rnl_spec, Ylm_spec):
    """sha256 over the canonical JSON of the three specs.

    Order matters and is preserved: mb_spec order IS the B-row order, and
    Rnl/Ylm order feeds the index spaces.  `sort_keys` only normalises dict
    key order inside the blob."""
    blob = json.dumps(
        {"schema": _CACHE_SCHEMA,
         "mb": [[list(b) for b in bb] for bb in mb_spec],
         "rnl": [list(s) for s in Rnl_spec],
         "ylm": [list(s) for s in Ylm_spec]},
        separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def default_cache_dir(env=None):
    """Cache root, or None to disable.  `ACEJAX_COUPLING_CACHE` wins (the
    literal value `none` disables); else `$XDG_CACHE_HOME/ace-jax/coupling`,
    defaulting to `~/.cache/...`."""
    env = os.environ if env is None else env
    v = env.get("ACEJAX_COUPLING_CACHE")
    if v:
        return None if v.lower() == "none" else pathlib.Path(v).expanduser()
    root = env.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return pathlib.Path(root) / "ace-jax" / "coupling"


def _entry_path(cache_dir, key):
    return pathlib.Path(cache_dir) / f"cpl-{key[:16]}.npz"


def _write_entry(path, cpl, key, mb_spec, Rnl_spec, Ylm_spec):
    """Atomically write one cache entry (tmp + os.replace, same dir)."""
    import numpy as np
    entry = {
        "A2B": np.asarray(cpl.A2B, np.float64),
        **{f"aa_spec_{k+1}": np.asarray(g, np.int64)
           for k, g in enumerate(cpl.aa_specs)},
        "meta_json": np.frombuffer(json.dumps({
            "schema": _CACHE_SCHEMA,
            "key": key,
            "mb": [[list(b) for b in bb] for bb in mb_spec],
            "rnl": [list(s) for s in Rnl_spec],
            "ylm": [list(s) for s in Ylm_spec],
            "aspec": [[r, y] for r, y in cpl.aspec],
            "aa_sig": [[list(t) for t in sig] for sig in cpl.aa_sig],
            "nnll_spec": [[list(b) for b in bb] for bb in cpl.nnll_spec],
            "backend": backend_id(),
        }).encode(), np.uint8),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # unique tmp per writer: two processes missing the cache on the same key
    # at once must not truncate each other's half-written file
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        # write through the fd: np.savez(path) appends '.npz' to names that do
        # not already end in it, which would leave tmp unreplaced
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **entry)
        # mkstemp creates the file 0600; publish with the default (umask-
        # adjusted) mode so a shared team cache dir stays readable by the team
        umask = os.umask(0o022)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)


def _read_entry(path, key):
    """(Coupling, valid) from a cache file; (None, False) on any problem.

    `valid` re-checks the stored specs against the requested key and the
    stored pin hash against the current one -- both recomputes, never wrong
    answers."""
    import numpy as np
    try:
        z = np.load(path)
        meta = json.loads(bytes(z["meta_json"]).decode())
        ok = (meta.get("schema") == _CACHE_SCHEMA and meta.get("key") == key
              and meta.get("backend") == backend_id())
        if not ok:
            return None, False
        n_orders = sum(1 for f in z.files if f.startswith("aa_spec_"))
        cpl = Coupling(
            A2B=np.asarray(z["A2B"], np.float64),
            aa_sig=tuple(tuple(tuple(int(v) for v in t) for t in sig)
                         for sig in meta["aa_sig"]),
            aspec=tuple((int(r), int(y)) for r, y in meta["aspec"]),
            aa_specs=tuple(np.asarray(z[f"aa_spec_{k+1}"], np.int64)
                           for k in range(n_orders)),
            nnll_spec=tuple(tuple((int(b[0]), int(b[1])) for b in bb)
                            for bb in meta["nnll_spec"]),
        )
    except Exception as e:
        # torn zip (BadZipFile), schema drift (KeyError/TypeError/ValueError),
        # unreadable file (OSError): all are misses, never errors.  The broad
        # net is the documented contract ("any unreadable entry is a miss"),
        # but the miss is announced so a permanently-broken entry is visible
        # in the log instead of silently recomputed on every authoring run.
        warnings.warn(f"coupling cache: unreadable entry {path.name} "  # noqa: B028 -- open: stacklevel
                      f"({type(e).__name__}: {e}); recomputing")
        return None, False
    return cpl, True


def couple_cached(mb_spec, Rnl_spec, Ylm_spec, cache_dir=None):
    """`couple` with a per-shape disk cache.  `cache_dir=None` uses
    `default_cache_dir()` (which honours `ACEJAX_COUPLING_CACHE`, `none` to
    disable); a miss calls `couple` and writes the entry best-effort -- an
    unwritable cache dir degrades to always-recompute, never an error."""
    if cache_dir is None:
        cache_dir = default_cache_dir()
    if cache_dir is None or (isinstance(cache_dir, str) and cache_dir.lower() == "none"):
        return couple(mb_spec, Rnl_spec, Ylm_spec)
    key = coupling_key(mb_spec, Rnl_spec, Ylm_spec)
    path = _entry_path(cache_dir, key)
    if path.exists():
        cpl, ok = _read_entry(path, key)
        if ok:
            return align_columns(cpl, Rnl_spec, Ylm_spec)   # an entry written before the fix is unaligned
    cpl = couple(mb_spec, Rnl_spec, Ylm_spec)
    try:
        _write_entry(path, cpl, key, mb_spec, Rnl_spec, Ylm_spec)
    except OSError:
        pass
    return cpl
