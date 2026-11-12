"""YE3T linear models (Goff and Thompson, arXiv:2609.31895) fitted with the authors' ye3t-methods, on a CPU.

    <ac-ye3t venv>/bin/python -m accuracy_cost.fit_ye3t FAMILY SIZE [SIZE ...] [--procs N]     (from bench/)
    <ac-ye3t venv>/bin/python -m accuracy_cost.fit_ye3t --count FAMILY                        (descriptors per entry)

Families (sweep.yaml), all with a fully symmetric scalar parent (lambda = (N), L = 0, even parity):

- `ye3t_sym_fit`: density-projected inputs (u = A), intermediates `young_kappa: symmetric_only`.
- `ye3t_mix_A`: the same catalogue with `young_kappa: all_valid` (mixed-permutation intermediates).  ye3t resolves
  these to the same space for a density source (ye3t/representation_config.py `count_fixed_content`; ye3t-methods
  config.py: "commutative density resolves all_valid Young blocks to physically trivial block sectors"), and the
  descriptor rows are identical bit for bit (`same_rows`).
- `ye3t_mix_T`: the tagged basis (u = T; ye3t-methods' tagged Cauchy image, shifted-Jacobi radials).

Protocol (the paper's Sec. IV.3 where it states one; fixed values where it tuned with DAKOTA):

- ridge, penalty 1e-8 (`model.fit.alpha`), by ye3t-methods' own normal-equation solve;
- cutoff RCUT; ChebExpCos radials with the decay of the authors' Si models (LAMBDA, from
  ye3t-lammps examples/PACKAGES/ye3t/cost_comparison/Si: tuned there together with a 4.757 A cutoff), taper 0.01 A;
- no ZBL reference (the authors fit the residual to a ZBL core tuned per element); E0_SI per atom is the fixed
  reference energy (`fit_E0: False`), so the saved model predicts total energies;
- weights: energy 1, forces 1 on every structure.  ye3t-methods has one global weight per quantity and no
  per-structure or per-config-type weight, so the GAP-18 sigmas are NOT applied (flagged in status.json);
- virials: not fitted.  The tool's stress target needs a stress on every structure (a third of the training set
  has none) and takes one global weight on stresses in eV/A^3; the authors' Si fits record no stress weight.

The descriptor list of each size is saved as `labels.json` in the run directory.

Speed and memory: ye3t-methods' density fit is one loop over the structures, calling `_linear_ace_geometry_row`
for each structure's energy and force rows on one thread (PyTorch autograd).  `fit` runs that function in worker
processes and hands the rows to the unchanged fit loop in order, so the arithmetic is ye3t-methods' own (the
coefficients are identical to a serial fit).  One 217-atom cell's rows take 9 GB at 205 descriptors and 30 GB at
490 (`row_gb`), so the worker count follows a memory budget (`_Rows`), and catalogues much past 1,000 descriptors
do not fit a 251 GB node at all.

The density rows are kept (`rows_dir`: RUNS/ye3t_rows/<key>, one .npz per structure; the key is the basis, the
representation and the structures, not the penalty or the weights), so a second solve of the same catalogue (another
penalty, a failed solve) reads them back in minutes: at 1,048 descriptors they took 5.5 hours to compute and are
3-4 GB.  `ACEJAX_AC_YE3T_ROWS=off` turns that off; any other value is the directory to use.
"""
import hashlib
import json
import os
import pathlib
import resource
import time

import numpy as np

from . import common
from .common import E0_SI, RCUT

FAMILIES = {"ye3t_sym_fit": ("density", "symmetric_only"), "ye3t_mix_A": ("density", "all_valid"),
            "ye3t_mix_T": ("tagged", "all_valid")}
LAMBDA, DCUT, ALPHA = 0.7798690236210712, 0.01, 1e-8
RUNTIME = {"evaluator": "torch", "neighbors": "ase", "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"}


def partitions(n, m=None):
    """The integer partitions of n, largest part first: every repeated-content pattern of a rank-n product."""
    m = m or n
    if n == 0:
        return [[]]
    return [[k] + p for k in range(min(n, m), 0, -1) for p in partitions(n - k, k)]


def config(family, e):
    """ye3t-methods' seven-section config for a sweep entry: `nmax` / `lmax` per rank 1..len(nmax); density
    catalogues take every source-block partition (the complete product basis at those caps)."""
    kind, kappa = FAMILIES[family]
    ranks = list(range(1, len(e["nmax"]) + 1))
    nm, lm = dict(zip(ranks, e["nmax"])), dict(zip(ranks, e["lmax"]))
    rep = {"group": "O3", "ranks": ranks, "parent": {"young_lambda": "(N)", "L": 0, "parity": "even"},
           "factorization": "cauchy", "subspace": "full",
           "uncoupled_factor_inputs": {"eta_count_per_rank": nm, "l_max_per_rank": lm},
           "intermediates": {"young_kappa": kappa, "block_rotation": {"policy": "all_valid"}}}
    cat = {"ranks": ranks, "nmax_per_rank": nm, "lmax_per_rank": lm,
           "source_block_partitions_by_rank": e.get("partitions") or {r: partitions(r) for r in ranks}}
    if kind == "density":
        radial = {"family": "pace_chebexp_cos", "cutoff_A": float(e.get("cutoff", RCUT)), "cutoff_width_A": DCUT,
                  "lambda": float(e.get("lambda", LAMBDA))}
        product = {"kind": "density"}
    else:
        radial = {"family": "shifted_jacobi", "cutoff_A": float(e.get("cutoff", RCUT))}
        product = {"kind": "tagged", "tag_counts_per_rank": e.get("tag_counts") or {r: [min(r, 2)] for r in ranks}}
        if e.get("angular_patterns"):
            cat["angular_patterns_by_rank"] = e["angular_patterns"]
    return {"metadata": {"schema": "ye3t_config_v1", "name": f"si_{family}_{e['size']}", "status": "stable"},
            "representation": rep,
            "basis": {"single_factors": {"species": ["Si"], "radial": radial, "chemical": {"kind": "explicit"}},
                      "tensor_product": product, "catalogue": cat},
            "runtime": dict(RUNTIME) if kind == "density" else {**RUNTIME, "evaluator": "native_cpu",
                                                                 "cache": {"mode": "auto"}},
            "model": {"kind": "linear",
                      "fit": {"solver": "ridge", "alpha": float(e.get("alpha", ALPHA)),
                              "weights": {"energy": 1.0, "forces": 1.0}},
                      "reference_energy": {"per_species_E0_eV": {"Si": E0_SI}, "fit_E0": False}},
            "targets": {"energy": "energy", "forces": "forces", "stress": None},
            "validation": {"checks": []}}


def basis_of(conf, max_fixed_contents=None):
    """The ye3t-methods Basis of a config.  `max_fixed_contents` (a sweep entry's key) raises the default of the
    catalogue preview's resource guard (`counts(maximum_fixed_contents=4096)`, ye3t_methods/config.py), which
    otherwise refuses the larger density catalogues before anything is compiled; it stays raised for the process,
    since the basis re-queries its counts when it is first evaluated."""
    from ye3t import YE3TRepresentation
    from ye3t_methods import Basis
    if max_fixed_contents:
        import ye3t_methods.config as C
        for cls in vars(C).values():
            f = getattr(cls, "counts", None) if isinstance(cls, type) else None
            if f is not None and "maximum_fixed_contents" in (f.__kwdefaults__ or {}):
                f.__kwdefaults__["maximum_fixed_contents"] = max(int(max_fixed_contents),
                                                                 f.__kwdefaults__["maximum_fixed_contents"])
    return Basis.from_config(conf["basis"], representation=YE3TRepresentation.from_config(conf["representation"]),
                             runtime=conf["runtime"])


def prepared(atoms):
    """The split's frames as ye3t-methods reads them: info["energy"] (total), arrays["forces"], nothing else."""
    out = []
    for a in atoms:
        b = a.copy(); b.info = {"energy": float(a.info["energy"])}
        for k in list(b.arrays):
            if k not in ("numbers", "positions"):
                del b.arrays[k]
        b.arrays["forces"] = np.asarray(a.arrays["forces"], float).copy()
        out.append(b)
    return out


_W = {}                                           # worker state: the basis and the row function (inherited by fork)


def _row(i):
    import torch
    torch.set_num_threads(1)
    d = _W["basis"]._descriptor
    r = _W["row"](_W["atoms"][i], evaluator=d.ace_descriptor.calculator.evaluator, descriptors=d.descriptor_specs,
                  cutoff=_W["basis"].cutoff, type_map=d.type_map, device="cpu", forces=True, stress=False,
                  chunk_size=None)
    return i, r["energy"].detach().cpu().numpy(), r["forces"].detach().cpu().numpy()


# Memory of one structure's rows in ye3t-methods' PyTorch density path (peak RSS, one thread, mnf148): 8.9 GB at
# 205 descriptors and 30.1 GB at 490 for the largest training cell (217 atoms); 1.9 and 5.4 GB for a 27-atom cell.
# `row_gb` is that fit (a power law in the descriptor count, linear in the atom count) with a quarter of headroom.
ROW_GB_217, ROW_F0, ROW_EXP, ROW_HEADROOM = 8.9, 205, 1.4, 1.25
SIZE_CLASSES = (32, 100, 10**9)                   # atoms: each class is computed on its own, never two at once


def row_gb(n_features, n_atoms):
    return ROW_HEADROOM * ROW_GB_217 * (n_features / ROW_F0) ** ROW_EXP * (0.18 + 0.82 * n_atoms / 217)


def mem_budget_gb():
    """$ACEJAX_AC_YE3T_MEM_GB, else 60% of the node's available memory."""
    if os.environ.get("ACEJAX_AC_YE3T_MEM_GB"):
        return float(os.environ["ACEJAX_AC_YE3T_MEM_GB"])
    avail = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    return 0.6 * avail / 1e6


def cap_process_memory(frac=0.85):
    """An address-space limit of `frac` of the node's memory on this process and its forks: an overrun fails the
    fit (MemoryError) rather than the node (2026-11-12: 24 workers at 490 descriptors took mnf148's 251 GB)."""
    total = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemTotal")) * 1024
    resource.setrlimit(resource.RLIMIT_AS, (int(frac * total), int(frac * total)))


def rows_dir(conf, atoms):
    """Where the density rows of `conf`'s catalogue on `atoms` are kept, or None ($ACEJAX_AC_YE3T_ROWS=off).  The
    key: basis, representation, dtype and every structure's numbers, cell and positions; nothing of the solve."""
    env = os.environ.get("ACEJAX_AC_YE3T_ROWS")
    if env == "off":
        return None
    h = hashlib.sha256(json.dumps([conf["basis"], conf["representation"], conf["runtime"]["dtype"]], sort_keys=True,
                                  default=str).encode())
    for a in atoms:
        for x in (a.numbers, np.asarray(a.cell), a.positions):
            h.update(np.ascontiguousarray(x, float).tobytes())
    return pathlib.Path(env) if env else common.RUNS / "ye3t_rows" / h.hexdigest()[:16]


def _stored(store, i, a):
    """Structure i's saved (energy row, force rows), or None: absent, unreadable, or not this structure's."""
    try:
        with np.load(store / f"{i}.npz") as z:
            return (z["e"], z["f"]) if np.array_equal(z["positions"], a.positions) else None
    except (OSError, ValueError, KeyError, EOFError):
        return None


def _store(store, i, a, e, f):
    tmp = store / f".{i}.{os.getpid()}.npz"                  # written whole, then renamed: a killed fit leaves no
    np.savez(tmp, e=e, f=f, positions=a.positions)           # half file under the final name
    os.replace(tmp, store / f"{i}.npz")


class _Rows:
    """Stands in for ye3t_methods.linear._linear_ace_geometry_row inside LinearModel.fit: the same rows, computed
    beforehand by forked workers and returned in the fit loop's order.  The structures are taken in size classes
    (SIZE_CLASSES), one class at a time, each with as many workers (at most `procs`) as fit `budget_gb` at
    `row_gb` of its largest cell; `plan` records the worker counts.  With `store` (a directory), rows found there
    are read, not computed, and each computed row is written as it arrives (`plan`: `stored` per class)."""

    def __init__(self, basis, atoms, procs, budget_gb=None, store=None):
        import multiprocessing as mp
        import ye3t_methods.linear as L
        _W.update(basis=basis, atoms=atoms, row=L._linear_ace_geometry_row)
        self.atoms, self.k, self.rows, self.plan = atoms, 0, {}, []
        budget, F, lo = budget_gb or mem_budget_gb(), len(basis.labels), 0
        if store is not None:
            store.mkdir(parents=True, exist_ok=True)
        for hi in SIZE_CLASSES:
            idx = [i for i, a in enumerate(atoms) if lo < len(a) <= hi]
            lo = hi
            if not idx:
                continue
            if store is not None:
                for i in idx:
                    r = _stored(store, i, atoms[i])
                    if r is not None and r[0].shape[-1] == F:
                        self.rows[i] = r
            n_all, idx = len(idx), [i for i in idx if i not in self.rows]
            if not idx:
                self.plan.append({"max_atoms": hi, "structures": n_all, "stored": n_all, "workers": 0})
                print("rows:", self.plan[-1], flush=True)
                continue
            need = row_gb(F, max(len(atoms[i]) for i in idx))
            n = max(1, min(procs, len(idx), int(budget / need)))
            self.plan.append({"max_atoms": hi, "structures": n_all, "stored": n_all - len(idx), "workers": n,
                              "row_gb": round(need, 1)})
            print("rows:", self.plan[-1], flush=True)
            with mp.get_context("fork").Pool(n, maxtasksperchild=8) as pool:   # a fresh process now and then: the
                for i, e, f in pool.imap_unordered(_row, idx, chunksize=1):    # allocator keeps a big cell's peak
                    self.rows[i] = (e, f)
                    if store is not None:
                        _store(store, i, atoms[i], e, f)

    def __call__(self, atoms, **kw):
        import torch
        mine = self.atoms[self.k]                            # the fit loop works on its own copies, in order
        if len(atoms) != len(mine) or not np.array_equal(atoms.positions, mine.positions) \
                or kw.get("stress") or not kw.get("forces"):
            raise RuntimeError("fit loop left the order or the targets the rows were computed for")
        e, f = self.rows.pop(self.k)
        self.k += 1
        return {"energy": torch.from_numpy(e), "forces": torch.from_numpy(f)}


def fit_model(conf, atoms, procs=1, max_fixed_contents=None):
    """ye3t-methods' LinearModel fitted to `atoms` (already `prepared`); density rows in `procs` processes."""
    import ye3t_methods.linear as L
    from ye3t_methods import LinearModel
    basis = basis_of(conf, max_fixed_contents)
    model = LinearModel(basis)
    store = rows_dir(conf, atoms) if conf["basis"]["tensor_product"]["kind"] == "density" else None
    if conf["basis"]["tensor_product"]["kind"] == "density" and (procs > 1 or store is not None):
        import torch
        threads = torch.get_num_threads()
        torch.set_num_threads(1)                             # before the fork: no thread pool to inherit
        basis.create(atoms[0])                               # compile once, in the parent
        rows, orig = _Rows(basis, atoms, procs, store=store), L._linear_ace_geometry_row
        L._linear_ace_geometry_row = rows
        if procs == 1:                                       # (a serial fit keeps its threads for the solve)
            torch.set_num_threads(threads)
        try:
            model.fit(atoms, config=conf)
        finally:
            L._linear_ace_geometry_row = orig
        model.row_plan, model.rows_dir = rows.plan, store and str(store)
    else:
        if procs > 1:                                        # tagged: ye3t-methods caches each structure's
            _prewarm(conf, atoms, procs, max_fixed_contents)  # target-free rows (runtime.cache "auto")
        model.fit(atoms, config=conf)
    return model


def _prewarm_chunk(args):
    """One worker: a fit of every `procs`-th structure, for its side effect -- ye3t-methods' geometry-row cache
    ($YE3T_ACE_CACHE_DIR/geometry_rows).  The chunk's own solve may fail (too few rows); the rows are cached first."""
    conf, atoms, cap = args
    from ye3t_methods import LinearModel
    try:
        LinearModel(basis_of(conf, cap)).fit(atoms, config=conf)
    except Exception as ex:
        return repr(ex)[-200:]
    return None


def _prewarm(conf, atoms, procs, max_fixed_contents=None):
    import multiprocessing as mp
    basis_of(conf, max_fixed_contents)                       # compile (and cache) the catalogue once, here
    with mp.get_context("spawn").Pool(procs) as pool:        # spawn: no native evaluator handle crosses a fork
        notes = pool.map(_prewarm_chunk, [(conf, atoms[k::procs], max_fixed_contents) for k in range(procs)])
    print("prewarm:", sum(n is None for n in notes), "of", procs, "chunk solves succeeded", flush=True)


def labels(basis):
    return [{"index": k, "label": str(l), **({"detail": l.as_dict()} if hasattr(l, "as_dict") else {})}
            for k, l in enumerate(basis.labels)]


def same_rows(e, atoms):
    """max |D(all_valid) - D(symmetric_only)| over `atoms` for the density catalogue of entry `e`: 0.0 where the
    two intermediate policies give the same descriptors."""
    A, S = (basis_of(config(f, e), e.get("max_fixed_contents")) for f in ("ye3t_mix_A", "ye3t_sym_fit"))
    return float(max(np.abs(A.create(a) - S.create(a)).max() for a in atoms))


def versions():
    import torch
    import ye3t
    import ye3t_methods
    v = lambda m: getattr(m, "__version__", "0.1.0")
    return {"ye3t": v(ye3t), "ye3t_methods": v(ye3t_methods), "torch": torch.__version__, "numpy": np.__version__,
            "ye3t_commit": os.environ.get("YE3T_COMMIT", "f8ff96d"),
            "ye3t_methods_commit": os.environ.get("YE3T_METHODS_COMMIT", "6b85c54")}


def fit(family, size, procs=1):
    from .split import ensure_split, read_atoms
    h = ensure_split()
    e, d = common.entry(family, size), common.run_dir(family, size)
    st = common.read_status(d)
    if st and st.get("status") == "ok" and pathlib.Path(st["model"]).exists():
        return st
    d.mkdir(parents=True, exist_ok=True)
    cap_process_memory()
    conf = config(family, e)
    (d / "config.json").write_text(json.dumps(conf, indent=1, default=str))
    atoms = prepared(read_atoms(common.DATA / "train.xyz"))
    if os.environ.get("ACEJAX_AC_YE3T_SUBSET"):              # a smoke run: every k-th frame (never a results row:
        atoms = atoms[::int(os.environ["ACEJAX_AC_YE3T_SUBSET"])]   # status.json records n_train)
    t0 = time.perf_counter()
    try:
        # `alpha_grid` (a sweep entry's key): the first penalty, in the order given, at which ye3t-methods' solve
        # succeeds -- its tagged solve is a Cholesky factorisation that stops with "not positive definite" when
        # the penalty is too small for the catalogue's conditioning.  No data enters the choice
        tried = []
        for alpha in e.get("alpha_grid") or [conf["model"]["fit"]["alpha"]]:
            conf["model"]["fit"]["alpha"] = float(alpha)
            try:
                model = fit_model(conf, atoms, procs if not tried else 1, e.get("max_fixed_contents"))
                break
            except np.linalg.LinAlgError as ex:
                tried.append({"alpha": float(alpha), "error": repr(ex)})
                print("solve failed at alpha", alpha, repr(ex), flush=True)
        else:
            raise RuntimeError(f"no penalty of {e.get('alpha_grid')} gave a solvable system")
        (d / "config.json").write_text(json.dumps(conf, indent=1, default=str))
        (d / "labels.json").write_text(json.dumps(labels(model.basis), indent=0, default=str))
        art = pathlib.Path(model.write(d / "model"))
        extra = {"n_params": len(model.basis.labels), "fit_wall_s": time.perf_counter() - t0, "procs": procs,
                 "row_plan": getattr(model, "row_plan", None), "rows_dir": getattr(model, "rows_dir", None)}
        try:                                                 # density: a strict PACE .yace; tagged: its JSON schema
            lmp = pathlib.Path(model.export_lammps(d / ("lammps.yace" if FAMILIES[family][0] == "density" else "lammps")))
            extra |= {"lammps_model": str(_lammps_file(lmp)), "pair_extra": _pair_extra(family)}
        except Exception as ex:
            extra |= {"lammps_export_error": repr(ex)[-1500:]}
        status = "ok"
    except Exception:
        import traceback
        status, art = "failed", d / "model"
        extra = {"error": traceback.format_exc()[-2500:], "fit_wall_s": time.perf_counter() - t0}
    common.write_status(d, family=family, size=size, status=status, model=str(art), split_hash=h,
                        e0_per_atom=E0_SI, n_train=len(atoms), entry=e,
                        protocol={"alpha": conf["model"]["fit"]["alpha"], "weights": conf["model"]["fit"]["weights"],
                                  "alphas_failed": locals().get("tried") or [],
                                  "per_config_type_weights": False, "virials": False,
                                  "radial": conf["basis"]["single_factors"]["radial"]},
                        fit_peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
                        versions=versions(), **extra)
    return common.read_status(d)


def _lammps_file(p):
    """The file pair_coeff names: export_lammps returns the .yace (density) or a directory / JSON (tagged)."""
    if p.is_file():
        return p
    for pat in ("*.yace", "model.ye3t.json", "*.ye3t.json"):
        hit = sorted(p.glob(pat))
        if hit:
            return hit[0]
    return p


def _pair_extra(family):
    return "model_family tagged_cauchy" if FAMILIES[family][0] == "tagged" else None


def _with_free_energy(calc):
    """ye3t-methods' tagged ASE calculator has no `free_energy` (ASE's cell filters and the relaxed QoIs ask for
    it; the density one has): the same object, reporting its energy under that name too."""
    if "free_energy" in calc.implemented_properties:
        return calc
    base = type(calc)

    class WithFreeEnergy(base):
        implemented_properties = [*base.implemented_properties, "free_energy"]

        def calculate(self, *a, **k):
            super().calculate(*a, **k)
            self.results["free_energy"] = self.results["energy"]
    calc.__class__ = WithFreeEnergy
    return calc


def calculator(path):
    """ye3t-methods' ASE calculator on a saved model (float64; E0_SI is in the model, so total energies): the
    native C++ evaluator when the model has one, else PyTorch."""
    from ye3t_methods import LinearModel
    m = LinearModel.read(str(path))
    try:
        c = m.ase_calculator(evaluator="native_cpu", neighbors="ase")
        from ase.build import bulk
        a = bulk("Si", "diamond", a=5.43); a.calc = c; a.get_potential_energy()
        return _with_free_energy(c)
    except Exception:
        return _with_free_energy(m.ase_calculator(evaluator="torch", neighbors="ase"))


def raw_ef(path, at):
    """E, F of the saved model: the timing gate's reference (LAMMPS adds the model's E0 too)."""
    a = at.copy(); a.calc = calculator(path)
    return float(a.get_potential_energy()), np.asarray(a.get_forces())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("family"); ap.add_argument("sizes", nargs="*")
    ap.add_argument("--procs", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--count", action="store_true")
    a = ap.parse_args()
    if a.count:
        for e in common.load_sweep()[a.family]:
            b = basis_of(config(a.family, e), e.get("max_fixed_contents"))
            print(e["size"], len(b.labels), e.get("n_params"), json.dumps(b.catalogue.counts().get("by_component", {}), default=str)[:200])
    else:
        for s in a.sizes:
            r = fit(a.family, s, a.procs)
            print(json.dumps({k: r.get(k) for k in ("family", "size", "status", "n_params", "fit_wall_s", "model",
                                                    "lammps_model", "lammps_export_error", "error")}))
