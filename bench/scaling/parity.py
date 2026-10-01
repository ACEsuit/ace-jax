"""Parity gates (spec thresholds), run per host before any timing.

- mlpace:  ace-jax PACEModel standalone vs ML-PACE on the same .yace
           (|dE|/atom <= 1e-6, |dF| <= 1e-5: the spline-grid gap)
- acejax:  ace-jax standalone vs ace-jax in LAMMPS via lammps-jax, f64
           (|dE|/atom <= 1e-10, |dF| <= 1e-9) -- periodic cells, so ghosts
- mace:    MACE standalone (PyTorch) vs MACE in LAMMPS via Symmetrix
           (|dE|/atom <= 1e-6)
- spline:  the learned-radial proxy (medium ACE) splined, as deployed
           (spline_tol="auto", 1e-10), vs kept analytic (None), standalone:
           |dE|/|E| <= 1e-9, max|dF| / max|F| <= 3e-8.  That is splining
           accuracy, not roundoff: docs/learned-radial-splining.md measured up
           to 9.4e-10 and 2.3e-8 over 20 learned-radial cases at 1e-10 (on
           the medium n_q=12 models 3.5e-11 and 9.6e-9, with 20% weight noise).
- acepot:   ACEpotentials.jl direct (the npz's splined ace1_model, rebuilt and
           asserted identical) vs ace-jax standalone on the npz, f64
           (|dE|/atom <= 1e-10, |dF| <= 1e-9): the same model, two codes
- trim:     the PR 309 trim library in LAMMPS vs Julia ETACE on the exact twin
           it compiles (|dE|/atom <= 1e-10, |dF| <= 1e-9)
- trim-ace: the trim library in LAMMPS vs ace-jax standalone: a sanity check at
           the spline error (|dE|/atom <= 1e-5, |dF| <= 1e-3; measured 8.6e-6 ..
           5.5e-4 eV/A): the library evaluates the exact polynomial radials,
           ace-jax and ACEpotentials.jl direct the npz's spline tables
The ACEpotentials.jl checks evaluate the extxyz-roundtripped structure on every
side (extxyz rounds positions to 1e-8 A), and are CPU only (`unsupported` on a
GPU host).  A failed acepot gate blocks the acepotentials (standalone) line.
The acejax gate also runs both learned-radial lines on the medium models: for
-learned both sides are splined identically, for -analytic both are exact.
Each LAMMPS side is `run 0` with a sorted force dump.  A failed gate blocks
the LAMMPS rows of that code on this host (`blocked`), and a failed spline gate
both modes of the -learned line; timing never runs on an unverified path.
"""
import pathlib
import subprocess

import numpy as np
from ase.io import write

from scaling.models import ACEPOT_CODES, LEARNED_CODES, planned_models
from scaling.run_lammps import (KOKKOS, export_bundle, finished, lammps_input, lammps_vars,
                                read_dump_forces, read_pe)
from scaling.structures import supercell

TOL = {"mlpace": (1e-6, 1e-5), "acejax": (1e-10, 1e-9), "mace": (1e-6, None),
       "spline": (1e-9, 3e-8),                      # relative: |dE|/|E|, max|dF| / max|F|
       "acepot": (1e-10, 1e-9), "trim": (1e-10, 1e-9), "trim-ace": (1e-5, 1e-3)}
# gates that check a standalone line: a failure blocks its standalone rows
STANDALONE_GATES = ("spline", "acepot")
LEARNED, ANALYTIC = LEARNED_CODES                     # "acejax-ace-learned", "acejax-ace-analytic"
N_ATOMS = 256


def error_summary(ex, head=200, tail=500):
    """Head and tail of repr(ex): some errors (NVRTC) put the message last."""
    r = repr(ex)
    return r if len(r) <= head + tail else r[:head] + " ... " + r[-tail:]


def compare(E_ref, F_ref, E, F, n_atoms, tol):
    dE = abs(float(E) - float(E_ref)) / n_atoms
    dF = float(np.abs(np.asarray(F) - np.asarray(F_ref)).max())
    ok = dE <= tol[0] and (tol[1] is None or dF <= tol[1])
    return {"status": "parity_ok" if ok else "parity_fail", "dE_per_atom": dE, "max_dF": dF}


def compare_rel(E_ref, F_ref, E, F, n_atoms, tol):
    """`compare` with relative thresholds: |dE|/|E_ref| <= tol[0] and
    max|dF| / max|F_ref| <= tol[1] (splining error scales with the model's
    energy and force magnitudes, not per atom).  The absolute dE/atom and
    max|dF| are kept for the tables."""
    out = compare(E_ref, F_ref, E, F, n_atoms, (np.inf, None))
    dE = abs(float(E) - float(E_ref)) / abs(float(E_ref))
    dF = out["max_dF"] / float(np.abs(np.asarray(F_ref)).max())
    ok = dE <= tol[0] and dF <= tol[1]
    return {**out, "status": "parity_ok" if ok else "parity_fail", "dE_rel": dE, "dF_rel": dF}


def blocked(rows):
    """(code, mode) lines a failed or erroring gate rules out.  'unsupported'
    (e.g. a MACE model Symmetrix cannot export) is per model, not per code.
    The spline gate checks the model itself (splined against exact), so it
    rules out the standalone line as well; the acepot gate checks the
    acepotentials line, which is standalone only."""
    bad = [r for r in rows if r["status"] in ("parity_fail", "error")]
    return ({(r["code"], "lammps") for r in bad if r.get("gate") != "acepot"}
            | {(r["code"], "standalone") for r in bad if r.get("gate") in STANDALONE_GATES})


def _lammps_ef(style, model_path, els, at, device, lmp, work, pjrt=None, aceplugin=None):
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    write(work / "x.data", at, format="lammps-data", specorder=els, masses=True)
    (work / "in.parity").write_text(lammps_input(style, model_path, els, work / "x.data", device,
                                                 0, dump=str(work / "f.dump"), warmup=0))
    cmd = [lmp, "-in", "in.parity", "-log", "log.lammps", "-nocite"]
    if device == "gpu":
        cmd += ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", *KOKKOS[style].split()]
    cmd += lammps_vars(style, pjrt, aceplugin)
    p = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=1800)
    log = (work / "log.lammps").read_text() if (work / "log.lammps").exists() else ""
    if not finished(p.returncode, log):
        raise RuntimeError(f"lammps exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}")
    return read_pe((work / "log.lammps").read_text()), read_dump_forces(work / "f.dump")


def _acejax_ef(path, at, **kw):
    """E, F from ACECalculator; kw (spline_tol) as the benchmark row passes it."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.calc.point import ACECalculator
    a = at.copy()
    a.calc = ACECalculator(path, **kw)
    return a.get_potential_energy(), a.get_forces()


def _julia_ef(env, m, at, work, which):
    """E, F from the ACEpotentials.jl drivers (`which`: "splined" or "etace") on
    `at` (already extxyz-roundtripped: written to work, re-read identically)."""
    from scaling import acepot
    xyz = pathlib.Path(work) / "x.extxyz"
    acepot.roundtrip(at, xyz)
    return acepot.julia_ef(acepot.julia_config(env), m["ace1"], xyz, which, pathlib.Path(work) / which)


def _mace_ef(path, at, device, head=None):
    from mace.calculators import mace_mp
    a = at.copy()
    a.calc = mace_mp(model=path, default_dtype="float64",
                     device="cuda" if device == "gpu" else "cpu",
                     **({"head": head} if head else {}))
    return a.get_potential_energy(), a.get_forces()


def _tol_kw(m):
    return {"spline_tol": m["spline_tol"]} if "spline_tol" in m else {}


BUNDLE_LAYOUTS = ("matrix", "dense", "sparse")


def gate_checks(small, system, medium=None):
    """(gate, model row, bundle layout) per check.  Every ace-jax bundle layout is
    gated: layout="auto" exports the neighbour matrix (or packed dense on an older
    lammps-jax), and run_lammps falls back to a sparse bundle when it runs out of
    memory, so each path (ghost atoms included) must be verified.
    The learned-radial lines exist at medium only, so they are gated on
    `medium` (the rows planned at that size): acejax for both, and the spline
    gate (learned against analytic, standalone) once."""
    checks = [("mlpace", small[("mlpace", system)], None)]
    for code in ("acejax-pace", "acejax-ace"):
        checks += [("acejax", small[(code, system)], lay) for lay in BUNDLE_LAYOUTS]
    for code in LEARNED_CODES if medium else ():
        checks += [("acejax", medium[(code, system)], lay) for lay in BUNDLE_LAYOUTS]
    if medium:
        checks.append(("spline", medium[(LEARNED, system)], None))
    checks += [("acepot", small[("acepotentials", system)], None),
               ("trim", small[("acepotentials-trim", system)], None),
               ("trim-ace", small[("acepotentials-trim", system)], None)]
    return checks + [("mace", small[("mace", system)], None)]


def gate(host, env, workroot="/tmp"):
    from scaling.sweep import HOSTS, lammps_supported
    device = HOSTS[host]["device"]
    lmp, pjrt = env["lmp"], env.get("pjrt")
    rows = []
    small = {(m["code"], m["system"]): m for m in planned_models() if m["size"] == "small"}
    medium = {(m["code"], m["system"]): m for m in planned_models() if m["size"] == "medium"}
    for system in ("SiGe", "Cantor"):
        at = supercell(system, N_ATOMS)
        work = pathlib.Path(workroot) / f"parity_{host}_{system}"
        for gate_name, m, layout in gate_checks(small, system, medium):
            row = {"code": m["code"], "mode": "parity", "gate": gate_name, "system": system,
                   "model": m["name"], "n_atoms": N_ATOMS, "device": device, "host": host}
            if layout:
                row["bundle_layout"] = layout
            if m["code"] in ACEPOT_CODES and device != "cpu":     # CPU-only lines
                rows.append({**row, "status": "unsupported"})
                continue
            try:
                if gate_name in ("acepot", "trim", "trim-ace"):
                    from scaling import acepot
                    w = work / gate_name
                    w.mkdir(parents=True, exist_ok=True)
                    at_rt = acepot.roundtrip(at, w / "x.extxyz")       # what every side reads
                    if gate_name == "acepot":
                        E0, F0 = _acejax_ef(m["path"], at_rt)
                        E1, F1 = _julia_ef(env, m, at_rt, w, "splined")
                    else:
                        E0, F0 = (_julia_ef(env, m, at_rt, w, "etace") if gate_name == "trim"
                                  else _acejax_ef(m["path"], at_rt))
                        E1, F1 = _lammps_ef("acepotentials-trim", m["trim_lib"], m["elements"], at_rt,
                                            device, env.get("lmp_ace", lmp), w, aceplugin=env.get("ace_plugin"))
                        row["trim_lib"] = m["trim_lib"]
                elif gate_name == "mlpace":
                    E0, F0 = _acejax_ef(m["path"], at)
                    E1, F1 = _lammps_ef("mlpace", m["path"], m["elements"], at, device, lmp,
                                        work / "mlpace")
                elif gate_name == "acejax":
                    if not lammps_supported(m["code"], device):     # lammps-jax: GPU only
                        rows.append({**row, "status": "unsupported"})
                        continue
                    E0, F0 = _acejax_ef(m["path"], at, **_tol_kw(m))
                    info = {}
                    bundle, used, _ = export_bundle(m, at, "float64", work / f"{m['code']}_{layout}",
                                                    layout=layout, info=info)
                    row["layout"] = used
                    if "spline_tol" in m:
                        row["spline_tol"], row["splined"] = m["spline_tol"], info.get("splined")
                    E1, F1 = _lammps_ef("acejax", bundle, m["elements"], at, device,
                                        env.get("lmp_jax", lmp), work / f"{m['code']}_{layout}", pjrt)
                elif gate_name == "spline":                 # exact reference: the analytic line
                    ref = medium[(ANALYTIC, system)]
                    E0, F0 = _acejax_ef(ref["path"], at, **_tol_kw(ref))
                    E1, F1 = _acejax_ef(m["path"], at, **_tol_kw(m))
                    row["spline_tol"] = m["spline_tol"]
                else:
                    if not pathlib.Path(m["symmetrix"]).exists():
                        rows.append({**row, "status": "unsupported"})
                        continue
                    E0, F0 = _mace_ef(m["path"], at, device, m.get("head"))
                    E1, F1 = _lammps_ef("mace", m["symmetrix"], m["elements"], at, device, lmp,
                                        work / "mace")
                cmp = compare_rel if gate_name == "spline" else compare
                row.update(cmp(E0, F0, E1, F1, N_ATOMS, TOL[gate_name]))
            except Exception as ex:                       # a missing style is a failed gate
                row.update({"status": "error", "error": error_summary(ex)})
            rows.append(row)
    return rows
