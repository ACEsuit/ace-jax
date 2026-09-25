"""Parity gates (spec thresholds), run per host before any timing.

- mlpace:  ace-jax PACEModel standalone vs ML-PACE on the same .yace
           (|dE|/atom <= 1e-6, |dF| <= 1e-5: the spline-grid gap)
- acejax:  ace-jax standalone vs ace-jax in LAMMPS via lammps-jax, f64
           (|dE|/atom <= 1e-10, |dF| <= 1e-9) -- periodic cells, so ghosts
- mace:    MACE standalone (PyTorch) vs MACE in LAMMPS via Symmetrix
           (|dE|/atom <= 1e-6)
Each LAMMPS side is `run 0` with a sorted force dump.  A failed gate blocks
the LAMMPS rows of that code on this host (`blocked`); timing never runs on an
unverified path.
"""
import pathlib
import subprocess

import numpy as np
from ase.io import write

from scaling.models import planned_models
from scaling.run_lammps import KOKKOS, export_bundle, finished, lammps_input, read_dump_forces, read_pe
from scaling.structures import supercell

TOL = {"mlpace": (1e-6, 1e-5), "acejax": (1e-10, 1e-9), "mace": (1e-6, None)}
N_ATOMS = 256


def compare(E_ref, F_ref, E, F, n_atoms, tol):
    dE = abs(float(E) - float(E_ref)) / n_atoms
    dF = float(np.abs(np.asarray(F) - np.asarray(F_ref)).max())
    ok = dE <= tol[0] and (tol[1] is None or dF <= tol[1])
    return {"status": "parity_ok" if ok else "parity_fail", "dE_per_atom": dE, "max_dF": dF}


def blocked(rows):
    """(code, mode) lines a failed or erroring gate rules out.  'unsupported'
    (e.g. a MACE model Symmetrix cannot export) is per model, not per code."""
    return {(r["code"], "lammps") for r in rows if r["status"] in ("parity_fail", "error")}


def _lammps_ef(style, model_path, els, at, device, lmp, work, pjrt=None):
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    write(work / "x.data", at, format="lammps-data", specorder=els, masses=True)
    (work / "in.parity").write_text(lammps_input(style, model_path, els, work / "x.data", device,
                                                 0, dump=str(work / "f.dump"), warmup=0))
    cmd = [lmp, "-in", "in.parity", "-log", "log.lammps", "-nocite"]
    if device == "gpu":
        cmd += ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", *KOKKOS[style].split()]
    if pjrt:
        cmd += ["-var", "pjrt", pjrt]
    p = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=1800)
    log = (work / "log.lammps").read_text() if (work / "log.lammps").exists() else ""
    if not finished(p.returncode, log):
        raise RuntimeError(f"lammps exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}")
    return read_pe((work / "log.lammps").read_text()), read_dump_forces(work / "f.dump")


def _acejax_ef(path, at):
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.calc.point import ACECalculator
    a = at.copy()
    a.calc = ACECalculator(path)
    return a.get_potential_energy(), a.get_forces()


def _mace_ef(path, at, device, head=None):
    from mace.calculators import mace_mp
    a = at.copy()
    a.calc = mace_mp(model=path, default_dtype="float64",
                     device="cuda" if device == "gpu" else "cpu",
                     **({"head": head} if head else {}))
    return a.get_potential_energy(), a.get_forces()


def gate(host, env, workroot="/tmp"):
    from scaling.sweep import HOSTS, lammps_supported
    device = HOSTS[host]["device"]
    lmp, pjrt = env["lmp"], env.get("pjrt")
    rows = []
    small = {(m["code"], m["system"]): m for m in planned_models() if m["size"] == "small"}
    for system in ("SiGe", "Cantor"):
        at = supercell(system, N_ATOMS)
        work = pathlib.Path(workroot) / f"parity_{host}_{system}"
        checks = [("mlpace", small[("mlpace", system)]),
                  ("acejax", small[("acejax-pace", system)]),
                  ("acejax", small[("acejax-ace", system)]),
                  ("mace", small[("mace", system)])]
        for gate_name, m in checks:
            row = {"code": m["code"], "mode": "parity", "gate": gate_name, "system": system,
                   "model": m["name"], "n_atoms": N_ATOMS, "device": device, "host": host}
            try:
                if gate_name == "mlpace":
                    E0, F0 = _acejax_ef(m["path"], at)
                    E1, F1 = _lammps_ef("mlpace", m["path"], m["elements"], at, device, lmp,
                                        work / "mlpace")
                elif gate_name == "acejax":
                    if not lammps_supported(m["code"], device):     # lammps-jax: GPU only
                        rows.append({**row, "status": "unsupported"})
                        continue
                    E0, F0 = _acejax_ef(m["path"], at)
                    bundle, layout, _ = export_bundle(m, at, "float64", work / m["code"])
                    row["layout"] = layout
                    E1, F1 = _lammps_ef("acejax", bundle, m["elements"], at, device,
                                        env.get("lmp_jax", lmp), work / m["code"], pjrt)
                else:
                    if not pathlib.Path(m["symmetrix"]).exists():
                        rows.append({**row, "status": "unsupported"})
                        continue
                    E0, F0 = _mace_ef(m["path"], at, device, m.get("head"))
                    E1, F1 = _lammps_ef("mace", m["symmetrix"], m["elements"], at, device, lmp,
                                        work / "mace")
                row.update(compare(E0, F0, E1, F1, N_ATOMS, TOL[gate_name]))
            except Exception as ex:                       # a missing style is a failed gate
                row.update({"status": "error", "error": repr(ex)[:300]})
            rows.append(row)
    return rows
