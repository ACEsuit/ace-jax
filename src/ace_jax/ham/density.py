"""The electron density as coefficients of atom-centred auxiliary functions,

    rho(r) = sum_i sum_(s, m) c[i, s, m] phi_s(|r - r_i|) Y_(l_s m)(r - r_i),

on top of `operators.centred`: the data and its file, the electron count, the predictor and the
metrics.  The model is a `CentredModel` with the one channel "rho" in an operator model file
(`operators.io`), whose `info["density"]` holds what the count needs: `charges` {"Z": [q, ...]},
the integral of each auxiliary function of an element (zero for l > 0), and `electrons`
{"Z": n}, the electrons an atom of each element brings.

Layout.  Atom order, shell order within an atom and the m order (m = -l..l, the real harmonics
of `eval.harmonics`) are those of an `OperatorData` block, so a density and the H and S of the
same structure index the same atoms the same way; `coefficients` is one vector (n_total,), atom
then shell then m (`operators.centred.offsets`), and `overlap` (n_total, n_total) is in the same
order.  Bringing a code's own order and signs to this one is the converter's work.

Neutral file: one `.npz` per structure (`read`, `write`, `names`):

    meta_json           schema_version = 1, kind = "density", and
        shells              {"Z": [l, l, ...]}: the auxiliary shells of each species
        m_order             "m=-l..l (ace-jax real harmonics)"
        nelectrons          electrons in the cell, or null
        info                the record's own dictionary; the keys with a meaning here:
            fit_metric          how the coefficients are defined from the density: "overlap"
                                (c = S^-1 w, w the projections on the auxiliary functions),
                                "coulomb", or "unknown"
            kpts, kpt_offset, smearing, width
                                what the density of a given Hamiltonian depends on
            self_consistent     true: the density of a converged calculation
            hamiltonian         path of the H this density produced or was built into (a record
                                with self_consistent false and a hamiltonian is a pair: density
                                in, H built from it)
            dN, corrected       of a prediction: the electron count error before `conserve`
    positions (N, 3), cell (3, 3; vectors in rows), numbers (N,), pbc (3,)
    coefficients        (n_total,) float64
    overlap             optional (n_total, n_total): the overlap of the auxiliary functions
    charges_<Z>         optional (n_aux(Z),): the integral of each auxiliary function
    radial_grid_<Z>, radial_values_<Z>
                        optional: the radial functions of the shells (`OrbitalBasis.radials`)

A density file may sit in a structure's DeepH directory as `density.npz`; `read` takes the
directory.  Units are the data's (the coefficients of the code that made them; lengths as the
positions).  Nothing here reads a DFT code's files."""
import glob
import json
import os
import warnings
from typing import NamedTuple

import numpy as np

from ..operators import centred as C
from ..operators.data import CentredData
from ..operators.orbitals import OrbitalBasis

SCHEMA_VERSION = 1
CHANNEL = "rho"
M_ORDER = "m=-l..l (ace-jax real harmonics)"
IN_DIRECTORY = "density.npz"                             # a density beside a structure's H and S (DeepH layout)


class DensityData(NamedTuple):
    """One structure with the expansion coefficients of its electron density.

    positions (N, 3); cell (3, 3), vectors in rows; numbers (N,); pbc; shells: the auxiliary
    `OrbitalBasis`; coefficients (n_total,): atom, then shell, then m; overlap (n_total, n_total)
    or None (or a function without arguments that returns it: `read(..., overlap="lazy")`);
    charges {z: (n_aux(z),)} or None: the integral of each auxiliary function (zero for l > 0);
    nelectrons: electrons in the cell, or None; info: see the module."""

    positions: np.ndarray
    cell: np.ndarray
    numbers: np.ndarray
    pbc: object
    shells: OrbitalBasis
    coefficients: np.ndarray
    overlap: object = None
    charges: dict = None
    nelectrons: float = None
    info: dict = None

    def S(self):
        """The overlap as an array (None when the record has none)."""
        return self.overlap() if callable(self.overlap) else self.overlap

    def centred(self):
        """The `operators.data.CentredData` of this record: what `operators.fit.fit_centred` reads."""
        return CentredData(self.cell, self.positions, self.numbers, self.shells, {CHANNEL: np.asarray(self.coefficients, float)}, self.pbc)


def write(path, data, compress=None):
    """Write `data` (a `DensityData`) to `path` (.npz).  compress: default only without an
    overlap (a dense float64 matrix does not compress, and the time is the matrix's)."""
    S = data.S()
    off = C.offsets(data.shells, data.numbers)
    if np.shape(data.coefficients) != (off[-1],):
        raise ValueError(f"density.write: coefficients {np.shape(data.coefficients)}, the atoms' shells hold ({off[-1]},)")
    out = dict(positions=np.asarray(data.positions, np.float64), cell=np.asarray(data.cell, np.float64), numbers=np.asarray(data.numbers, np.int64),
               pbc=np.broadcast_to(np.asarray(data.pbc, bool), (3,)).copy(), coefficients=np.asarray(data.coefficients, np.float64))
    if S is not None:
        if np.shape(S) != (off[-1],) * 2:
            raise ValueError(f"density.write: overlap {np.shape(S)}, coefficients ({off[-1]},)")
        out["overlap"] = np.asarray(S, np.float64)
    for z, q in (data.charges or {}).items():
        if np.shape(q) != (data.shells.n_orb(z),):
            raise ValueError(f"density.write: charges of Z = {z}: {np.shape(q)}, its shells hold ({data.shells.n_orb(z)},)")
        out[f"charges_{int(z)}"] = np.asarray(q, np.float64)
    if data.shells.radials is not None:
        for z, (g, v) in data.shells.radials.items():
            out[f"radial_grid_{z}"], out[f"radial_values_{z}"] = g, v
    meta = dict(schema_version=SCHEMA_VERSION, kind="density", shells={str(z): list(ls) for z, ls in data.shells.shells}, m_order=M_ORDER,
                nelectrons=None if data.nelectrons is None else float(data.nelectrons), info=dict(data.info or {}))
    out["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        (np.savez_compressed if (S is None if compress is None else compress) else np.savez)(f, **out)


def _file(path):
    path = os.fspath(path)
    return os.path.join(path, IN_DIRECTORY) if os.path.isdir(path) else path


def _overlap(path):
    with np.load(path) as z:
        return z["overlap"]


def read(path, overlap=True):
    """`DensityData` of a neutral file, or of a directory that holds `density.npz` (a
    structure's DeepH directory).  overlap: True, read it when the file has one; False, leave it
    out; "lazy": a function that reads it when called (the fit then holds one at a time)."""
    path = _file(path)
    with np.load(path) as z:
        if "meta_json" not in z:
            raise ValueError(f"{path}: not an ace-jax density file (no meta_json)")
        meta = json.loads(bytes(z["meta_json"]).decode())
        if meta.get("kind") != "density" or meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: not an ace-jax density file of schema {SCHEMA_VERSION} (kind {meta.get('kind')!r}, schema {meta.get('schema_version')!r})")
        if meta.get("m_order") != M_ORDER:
            raise ValueError(f"{path}: m order {meta.get('m_order')!r}, this version reads {M_ORDER!r}")
        sh = {int(k): v for k, v in meta["shells"].items()}
        rad = {k: (z[f"radial_grid_{k}"], z[f"radial_values_{k}"]) for k in sh} if sh and all(f"radial_grid_{k}" in z for k in sh) else None
        charges = {k: np.asarray(z[f"charges_{k}"]) for k in sh if f"charges_{k}" in z} or None
        S = None
        if "overlap" in z and overlap:
            S = (lambda p=path: _overlap(p)) if overlap == "lazy" else np.asarray(z["overlap"])
        pbc = np.asarray(z["pbc"], bool)
        return DensityData(np.asarray(z["positions"]), np.asarray(z["cell"]), np.asarray(z["numbers"]), bool(pbc.all()) if pbc.all() or not pbc.any() else tuple(bool(p) for p in pbc),
                           OrbitalBasis.of(sh, rad), np.asarray(z["coefficients"]), S, charges, meta.get("nelectrons"), meta.get("info") or {})


def names(directory):
    """The density files under `directory`, in name order: its `*.npz`, else its subdirectories
    that hold `density.npz` (DeepH directories with a density)."""
    directory = os.fspath(directory)
    if not os.path.isdir(directory):
        raise ValueError(f"{directory}: not a directory of density files")
    out = sorted(glob.glob(os.path.join(directory, "*.npz"))) or sorted(d for d in glob.glob(os.path.join(directory, "*")) if os.path.exists(os.path.join(d, IN_DIRECTORY)))
    if not out:
        raise ValueError(f"{directory}: no density files (*.npz) and no subdirectories with {IN_DIRECTORY}")
    return out


def projections(data, coefficients=None):
    """w = S c: the projections of the density on the auxiliary functions (local, and to first
    order independent of the basis, where c is not).  coefficients: other than the record's."""
    S = data.S()
    if S is None:
        raise ValueError("projections: the record has no overlap")
    return np.asarray(S) @ np.asarray(data.coefficients if coefficients is None else coefficients, float)


# ------------------------------------------------------------------ electron count
def _charge_vector(numbers, charges):
    try:
        return np.concatenate([np.asarray(charges[int(z)], float) for z in numbers])
    except KeyError as e:
        raise KeyError(f"no charges for atomic number {e.args[0]}") from None


def electron_count(c, numbers, charges):
    """N = sum_i q[z_i] . c_i."""
    q = _charge_vector(numbers, charges)
    if q.shape != np.shape(c):
        raise ValueError(f"electron_count: coefficients {np.shape(c)}, charges of these atoms ({len(q)},)")
    return float(q @ np.asarray(c, float))


def conserve(c, numbers, charges, nelectrons):
    """(c', dN): the coefficients with the electron count made exact by the smallest Euclidean
    change, c_i -= q[z_i] dN / sum_j |q[z_j]|^2, and dN = N(c) - nelectrons, the error before
    it.  dN is an out-of-distribution flag of its own (hundredths of an electron inside the
    training distribution, electrons outside): keep it with the prediction."""
    q = _charge_vector(numbers, charges)
    dN = electron_count(c, numbers, charges) - float(nelectrons)
    return np.asarray(c, float) - q * dN / (q @ q), dN


def fit_charges(datasets):
    """{z: (n_aux(z),)}: the charge of each l = 0 function by least squares on the electron
    counts of the records, sum over atoms and l = 0 shells of q c = nelectrons (minimum norm; zero
    for l > 0).  For data without measured charges only: with few structures the equations are
    as many as the structures and nearly dependent, and the charges are then not the functions'
    integrals (a warning says so).  Every record needs `nelectrons`."""
    sh = datasets[0].shells
    cols = [(z, s) for z in sh.species for s, l in enumerate(sh.ls(z)) if l == 0]
    A = np.zeros((len(datasets), len(cols)))
    for k, d in enumerate(datasets):
        if d.nelectrons is None:
            raise ValueError("fit_charges: a record has neither charges nor nelectrons")
        off, numbers = C.offsets(sh, d.numbers), np.asarray(d.numbers)
        for j, (z, s) in enumerate(cols):
            A[k, j] = np.asarray(d.coefficients)[off[:-1][numbers == z] + sh.shell_slice(z, s).start].sum()
    q, _, rank, sv = np.linalg.lstsq(A, np.asarray([d.nelectrons for d in datasets], float), rcond=1e-10)
    warnings.warn(f"density: the data has no charges of the auxiliary functions; fitted {len(cols)} of them to the electron counts of {len(datasets)} "
                  f"structures (rank {rank}, singular values {sv.max():.2e} to {sv.min():.2e}). They conserve the count on these structures and are "
                  "not the functions' integrals: give measured charges when you can.", stacklevel=2)
    out = {z: np.zeros(sh.n_orb(z)) for z in sh.species}
    for (z, s), v in zip(cols, q):
        out[z][sh.shell_slice(z, s).start] = v
    return out


def electrons_per_atom(datasets):
    """{z: electrons an atom of element z brings}, from the records' `nelectrons`: the atomic
    numbers when they give every count (or no record has one), else least squares over the
    elements' counts."""
    sh = datasets[0].shells
    have = [d for d in datasets if d.nelectrons is not None]
    if all(abs(float(d.nelectrons) - float(np.sum(d.numbers))) < 1e-6 for d in have):
        return {z: float(z) for z in sh.species}
    n = np.asarray([[np.sum(np.asarray(d.numbers) == z) for z in sh.species] for d in have], float)
    e, _, rank, _ = np.linalg.lstsq(n, np.asarray([d.nelectrons for d in have], float), rcond=1e-10)
    if rank < len(sh.species) or np.abs(n @ e - [d.nelectrons for d in have]).max() > 1e-6:
        warnings.warn("density: the electron counts of the training structures do not fix the electrons per atom of each element "
                      "(one composition, or charged cells); the minimum-norm solution is recorded", stacklevel=2)
    return {z: float(v) for z, v in zip(sh.species, e)}


# ------------------------------------------------------------------ prediction
class DensityPredictor:
    """Predicted densities of a model file (or an `operators.io.OperatorFile`, or a
    (`CentredModel`, info["density"]) pair): `predict(atoms)` gives a `DensityData` with the
    coefficients after `conserve` and, in its info, `dN`, the electron count error before it,
    and `corrected`.  A model without charges predicts uncorrected coefficients (dN None).
    The layout of `coefficients` is the module's and is what later consumers index."""

    def __init__(self, model):
        from ..operators import io
        if isinstance(model, (str, os.PathLike)):
            model = io.load(model)
        cen, rec = (model.centred, model.meta.get("info", {}).get("density", {})) if isinstance(model, io.OperatorFile) else model
        if cen is None or CHANNEL not in cen.channels:
            raise ValueError("DensityPredictor: the model file has no density model (a centred part with the channel 'rho')")
        self.model, self.predictor = cen, C.CentredPredictor(cen)
        self.charges = {int(z): np.asarray(q, float) for z, q in (rec or {}).get("charges", {}).items()} or None
        self.electrons = {int(z): float(n) for z, n in (rec or {}).get("electrons", {}).items()} or None

    def nelectrons(self, numbers):
        return float(sum((self.electrons or {}).get(int(z), float(z)) for z in numbers))

    def coefficients(self, positions, cell, numbers, pbc=True, nelectrons=None):
        """(c, dN): the corrected coefficients (n_total,) and the count error before the correction."""
        c = np.asarray(self.predictor(positions, cell, numbers, pbc)[CHANNEL])
        if self.charges is None:
            return c, None
        return conserve(c, numbers, self.charges, self.nelectrons(numbers) if nelectrons is None else nelectrons)

    def predict(self, atoms, nelectrons=None):
        """`DensityData` of an `ase.Atoms` (or anything with positions, cell, numbers and pbc)."""
        cell, numbers, pbc = np.asarray(getattr(atoms.cell, "array", atoms.cell), float), np.asarray(atoms.numbers), np.asarray(atoms.pbc, bool)
        n = self.nelectrons(numbers) if nelectrons is None else float(nelectrons)
        c, dN = self.coefficients(np.asarray(atoms.positions, float), cell, numbers, bool(pbc.all()) if pbc.all() or not pbc.any() else pbc, n)
        return DensityData(np.asarray(atoms.positions, float), cell, numbers, atoms.pbc, self.model.shells, c, None, self.charges, n,
                           dict(self_consistent=False, predicted=True, dN=dN, corrected=dN is not None))


def predict(model_file, atoms, nelectrons=None):
    """The predicted `DensityData` of `atoms` from a model file: corrected coefficients, dN in its info."""
    return DensityPredictor(model_file).predict(atoms, nelectrons)


# ------------------------------------------------------------------ metrics
def _spread(shells, numbers, c):
    """{l: (sum of squares of c about its mean, mask)}: for l = 0 about the mean over the
    structure's atoms of each species and shell; for l > 0 about zero (the mean of an
    equivariant vector over orientations)."""
    off, numbers = C.offsets(shells, numbers), np.asarray(numbers)
    dev, lam = np.array(c, float), np.zeros(len(c), np.int64)
    for z in shells.species:
        idx = off[:-1][numbers == z]
        for s, l in enumerate(shells.ls(z)):
            sl = shells.shell_slice(z, s)
            rows = idx[:, None] + np.arange(sl.start, sl.stop)
            lam[rows] = l
            if l == 0 and len(idx):
                dev[rows] -= dev[rows].mean()
    return dev, lam


def density_metrics(ref, pred, charges=None, nelectrons=None):
    """The error of predicted coefficients `pred` ((n_total,), or a `DensityData`) against the
    record `ref`, a dictionary:

        error, relative         sqrt(dc^T S dc), the L2 norm of the density error, and the same
                                over sqrt(c^T S c) (with the record's overlap; else absent)
        error_w, relative_w     |dw| and |dw| / |w| of the projections w = S c (with the overlap)
        rmse_l, rmse_w_l        {l: RMSE of the coefficients (projections) of angular momentum l
                                over the spread of the labels about their mean}: 1 is a model
                                no better than the mean coefficients (l = 0: the mean over the
                                structure's atoms of a species and shell; l > 0: zero)
        dN                      the electron count of `pred` minus `nelectrons` (charges and
                                nelectrons: the arguments, else the record's; else absent), and
                                dN_recorded, the error before `conserve` when `pred` is a
                                prediction that recorded it"""
    c1 = np.asarray(pred.coefficients if isinstance(pred, DensityData) else pred, float)
    c0 = np.asarray(ref.coefficients, float)
    if c1.shape != c0.shape:
        raise ValueError(f"density_metrics: predicted coefficients {c1.shape}, reference {c0.shape}")
    out, d = dict(n_atoms=len(ref.numbers)), c1 - c0
    S = ref.S()
    vecs = [("rmse_l", c0, d)]
    if S is not None:
        w0, dw = S @ c0, S @ d
        out.update(error=float(np.sqrt(max(d @ dw, 0.0))), relative=float(np.sqrt(max(d @ dw, 0.0) / (c0 @ w0))),
                   error_w=float(np.linalg.norm(dw)), relative_w=float(np.linalg.norm(dw) / np.linalg.norm(w0)))
        vecs.append(("rmse_w_l", w0, dw))
    for name, x0, dx in vecs:
        dev, lam = _spread(ref.shells, ref.numbers, x0)
        out[name] = {int(l): float(np.sqrt((dx[lam == l] ** 2).sum() / max((dev[lam == l] ** 2).sum(), 1e-300))) for l in np.unique(lam)}
    charges = charges if charges is not None else ref.charges
    nelectrons = nelectrons if nelectrons is not None else ref.nelectrons
    if charges is not None and nelectrons is not None:
        out["dN"] = electron_count(c1, ref.numbers, charges) - float(nelectrons)
    if isinstance(pred, DensityData) and (pred.info or {}).get("dN") is not None:
        out["dN_recorded"] = float(pred.info["dN"])
    return out


def summary(per):
    """A set of structures' `density_metrics` as one record: for each of error, relative, error_w
    and relative_w the median and the worst structure; the count error (the one before the
    correction when recorded) as root mean square and largest; the per-l ratios pooled as the
    root mean square over the structures."""
    out = dict(n=len(per))
    for k in ("error", "relative", "error_w", "relative_w"):
        v = np.asarray([m[k] for m in per if k in m], float)
        if len(v):
            out.update({f"{k}_median": float(np.median(v)), f"{k}_worst": float(v.max()), f"{k}_mean": float(v.mean())})
    dN = np.asarray([m.get("dN_recorded", m.get("dN")) for m in per if m.get("dN_recorded", m.get("dN")) is not None], float)
    if len(dN):
        out.update(dN_rms=float(np.sqrt(np.mean(dN ** 2))), dN_max=float(np.abs(dN).max()))
    for k in ("rmse_l", "rmse_w_l"):
        ls = sorted({l for m in per for l in m.get(k, {})})
        if ls:
            out[k] = {l: float(np.sqrt(np.mean([m[k][l] ** 2 for m in per if l in m.get(k, {})]))) for l in ls}
    return out


def format_summary(s):
    """Lines of text for a `summary`."""
    lines = []
    if "error_median" in s:
        lines.append(f"density error sqrt(dc S dc): median structure {s['error_median']:.5f}, worst {s['error_worst']:.5f}, mean {s['error_mean']:.5f}; "
                     f"relative to sqrt(c S c): median {s['relative_median']:.3e}, worst {s['relative_worst']:.3e}")
        lines.append(f"projections w = S c, |dw|: median {s['error_w_median']:.5f}, worst {s['error_w_worst']:.5f}; relative: median "
                     f"{s['relative_w_median']:.3e}, worst {s['relative_w_worst']:.3e}")
    for k, name in (("rmse_l", "coefficients"), ("rmse_w_l", "projections")):
        if k in s:
            lines.append(f"{name}, RMSE over the spread of the labels, per l: " + ", ".join(f"{l}: {v:.3f}" for l, v in s[k].items()))
    if "dN_rms" in s:
        lines.append(f"electron count error before the correction: RMS {s['dN_rms']:.4f}, largest {s['dN_max']:.4f}")
    return lines
