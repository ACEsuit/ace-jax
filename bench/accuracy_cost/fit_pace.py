"""PACE via one pacemaker ladder fit (how a PACE practitioner reaches a small
accurate model): each size is a ladder checkpoint, converted to .yace.  The
pacemaker defaults stand, apart from the shared cutoff, E0 and GAP-18 weights.
Run under the pacemaker venv (python 3.9, python-ace 0.2.8, tensorpotential).

Installed-version facts (see data.md): pyace 0.2.8 has no `data.reference_energy`
key, so E0 enters as the dataframe's `energy_corrected` column and is written
into the .yace's E0 line; the tensorpot loss has energy and
force terms only, so PACE fits no virials; `interim_potential_ladder_step_<k>`
counts ladder steps, not functions."""
import json
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np

from . import common
from .common import DATA, E0_SI, RCUT


def input_yaml(e):
    return {
        "cutoff": RCUT, "seed": 0,
        "data": {"filename": "train.pckl.gzip"},    # energy_corrected = energy - n E0_SI (dataframes)
        "potential": {"deltaSplineBins": 0.001, "elements": ["Si"],
                      "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled",
                                             "fs_parameters": [1, 1, 1, 0.5], "ndensity": 2}},
                      "bonds": {"ALL": {"radbase": "SBessel", "radparameters": [5.25], "rcut": RCUT,
                                        "dcut": 0.01, "NameOfCutoffFunction": "cos"}},
                      "functions": {"number_of_functions_per_element": e["max_functions"],
                                    "ALL": {"nradmax_by_orders": [15, 6, 4, 3, 2, 2],
                                            "lmax_by_orders": [0, 4, 3, 2, 1, 0]}}},
        "fit": {"loss": {"kappa": "auto", "L1_coeffs": 1e-8, "L2_coeffs": 1e-8},
                "weighting": {"type": "ExternalWeightingPolicy", "filename": "weights.pckl.gzip"},
                "optimizer": "BFGS", "maxiter": e.get("maxiter", 1500),
                "ladder_step": e["ladder_step"], "ladder_type": "power_order"},
        "backend": {"evaluator": "tensorpot", "batch_size": 50, "display_step": 50},
    }


def reduced(a):
    """A copy with a Minkowski-reduced, right-handed cell: the same lattice and Cartesian positions.
    pyace's neighbour builder rejects left-handed cells (the 10 `decohesion` frames with a_3 along
    -z raise "Extreme shape of unit cell").  Forces and E are unchanged; no virials reach PACE."""
    from ase.geometry import minkowski_reduce
    b = a.copy()
    cell, _ = minkowski_reduce(a.cell, a.pbc)
    cell = np.array(cell)
    if np.linalg.det(cell) < 0:
        cell[2] = -cell[2]                        # same lattice, handedness fixed
    b.set_cell(cell, scale_atoms=False)           # Cartesian positions kept; images are unchanged
    b.info.update(a.info); b.arrays["forces"] = a.arrays["forces"].copy()
    return b


def dataframes(atoms, wts):
    import pandas as pd
    atoms = [reduced(a) for a in atoms]
    nat = [len(a) for a in atoms]
    en = [a.info["energy"] for a in atoms]
    data = pd.DataFrame({"ase_atoms": atoms, "energy": en,
                         "energy_corrected": [x - n * E0_SI for x, n in zip(en, nat)],
                         "forces": [a.arrays["forces"] for a in atoms]})
    w = pd.DataFrame({"w_energy": [x[0] for x in wts], "w_forces": [x[1] for x in wts]}, index=data.index)
    return data, w


def n_functions(path):
    """Basis functions in a pacemaker potential yaml (all species blocks)."""
    import yaml
    t = yaml.safe_load(pathlib.Path(path).read_text())
    return sum(len(b.get("nbody") or b.get("funcspecs") or []) for b in t["species"])


def checkpoint_for(d, n):
    """The first ladder checkpoint with >= n functions per element, and its count.  Files are
    named by ladder step, so each one is opened and counted."""
    for p, c in checkpoints(d):
        if c >= n:
            return p, c
    raise FileNotFoundError(f"{d}: no ladder checkpoint with >= {n} functions "
                            f"(have {[c for _, c in checkpoints(d)]}; see {d}/fit.log)")


def checkpoints(d):
    """[(path, function count)] over the ladder dir and its resume<k>/ segments, by count (counts
    grow along the ladder; a resumed segment starts above its initial potential)."""
    d = pathlib.Path(d)
    segs = [d] + sorted(d.glob("resume*"), key=lambda p: int(p.name[6:]))
    steps = [p for s in segs for p in sorted(s.glob("interim_potential_ladder_step_*.yaml"),
                                             key=lambda p: int(p.stem.rsplit("_", 1)[1]))]
    out = []
    for p in steps:
        try:
            out.append((p, n_functions(p)))
        except Exception as e:                    # half-written (mid-commit/download): skip, don't abort
            print(f"warning: skipping unreadable checkpoint {p}: {type(e).__name__}: {e}", flush=True)
    return sorted(out, key=lambda pc: pc[1])


SETTINGS_KEYS = ("max_functions", "ladder_step", "n_train", "maxiter")


def ladder_settings(e):
    """What makes ladder checkpoints reusable (ladder_settings.json beside them)."""
    return {k: e.get(k) for k in SETTINGS_KEYS}


def check_settings(d, e):
    """Before resuming: checkpoints from other settings are moved to ladder.stale<k>/ (refit from
    scratch); none recorded (an older run) is resumed with a warning.  Returns True if kept."""
    d = pathlib.Path(d); f = d / "ladder_settings.json"
    if not f.exists():
        print(f"warning: {d} has no ladder_settings.json; resuming its checkpoints as they are", flush=True)
        return True
    try:
        same = json.loads(f.read_text()) == ladder_settings(e)
    except ValueError:
        same = False
    if same:
        return True
    k = 1 + len(list(d.parent.glob(d.name + ".stale*")))
    d.rename(d.parent / f"{d.name}.stale{k}"); d.mkdir()
    print(f"pace: {d} was fitted with other settings; moved to {d.name}.stale{k}, refitting", flush=True)
    return False


def segment_wall(d):
    """Summed fit wall time over the ladder's segments: each runs from its cmd.txt (written at
    start) to its fit.log's last write, so a segment killed by a timeout still counts."""
    d = pathlib.Path(d)
    return sum(s.joinpath("fit.log").stat().st_mtime - s.joinpath("cmd.txt").stat().st_mtime
               for s in [d, *d.glob("resume*")] if s.joinpath("fit.log").exists() and s.joinpath("cmd.txt").exists())


def with_e0(path, e0=E0_SI):
    """Set the .yace's per-species E0 (pace_yaml2yace writes 0: the fit saw energy_corrected)."""
    p = pathlib.Path(path); t = p.read_text()
    t, k = re.subn(r"(?m)^E0: \[[^\]]*\]", f"E0: [{e0!r}]", t)
    if k != 1:
        raise ValueError(f"{p}: expected one E0 line, found {k}")
    p.write_text(t)


def _tool(name):
    return str(pathlib.Path(sys.executable).parent / name)


def main(overrides=None):
    import yaml
    from .split import XML, ensure_split, read_atoms
    from .weights import command_line, gap_sigmas, pacemaker
    h = ensure_split()
    e = {**common.entry("pace", "ladder"), **(overrides or {})}
    d = common.run_dir("pace", "ladder"); d.mkdir(parents=True, exist_ok=True)
    if (common.read_status(d) or {}).get("status") != "ok" and checkpoints(d):
        check_settings(d, e)
    (d / "ladder_settings.json").write_text(json.dumps(ladder_settings(e)))
    ats = [a for a in read_atoms(DATA / "train.xyz") if a.info["config_type"] != "isolated_atom"]  # the E0 frame: no neighbours, pyace drops it
    if e.get("n_train"):
        ats = ats[::max(1, len(ats) // e["n_train"])][:e["n_train"]]        # strided: the file is sorted by config_type
    data, w = dataframes(ats, pacemaker(*gap_sigmas(command_line(DATA / XML)), ats))
    data.to_pickle(d / "train.pckl.gzip", compression="gzip"); w.to_pickle(d / "weights.pckl.gzip", compression="gzip")
    (d / "input.yaml").write_text(yaml.safe_dump(input_yaml(e)))
    st = common.read_status(d) or {}
    if st.get("status") != "ok" and checkpoints(d):     # an interrupted ladder (timeout, preemption): resume it
        st = resume(e, d, h)
    else:
        st = common.fit_once("pace", "ladder", "output_potential.yaml", [_tool("pacemaker"), "input.yaml"], split_hash=h)
    if st["status"] != "ok":
        raise SystemExit(st.get("error"))
    convert(e, h, d, st["fit_wall_s"])


def resume_input(e, d, prior):
    """input.yaml for a resumed segment: the same fit, started from the last ladder checkpoint
    (pacemaker's ladder extends an initial potential towards the same target, rung by rung)."""
    inp = input_yaml(e)
    inp["potential"]["initial_potential"] = str(prior)
    inp["data"]["filename"] = str(pathlib.Path(d) / "train.pckl.gzip")
    inp["fit"]["weighting"]["filename"] = str(pathlib.Path(d) / "weights.pckl.gzip")
    return inp


def resume(e, d, h):
    """Run the rest of the ladder in d/resume<k>/ from the largest checkpoint so far; the final
    potential is copied to d/output_potential.yaml and fit_wall_s sums every segment."""
    import yaml
    prior, cnt = checkpoints(d)[-1]
    k = 1 + len(list(d.glob("resume*")))
    seg = d / f"resume{k}"; seg.mkdir()
    (seg / "input.yaml").write_text(yaml.safe_dump(resume_input(e, d, prior)))
    print(f"pace: resuming the ladder from {prior} ({cnt} functions) in {seg}", flush=True)
    rc, _, peak = common.run_logged([_tool("pacemaker"), "input.yaml"], seg)
    ok = rc == 0 and (seg / "output_potential.yaml").exists()
    if ok:
        shutil.copy(seg / "output_potential.yaml", d / "output_potential.yaml")
    common.write_status(d, family="pace", size="ladder", status="ok" if ok else "failed",
                        model=str(d / "output_potential.yaml"), fit_wall_s=segment_wall(d), fit_peak_rss_gb=peak,
                        split_hash=h, segments=k + 1, **({} if ok else {"returncode": rc, "error": common.tail(seg / "fit.log")}))
    return common.read_status(d)


def plan(take, d, max_functions):
    """[(requested n, checkpoint path, actual function count)], one row per distinct count: a
    `take` value that lands on an already-taken checkpoint is dropped (logged), not written twice."""
    d, rows, seen = pathlib.Path(d), [], {}
    for n in take:
        src, cnt = (d / "output_potential.yaml", n_functions(d / "output_potential.yaml")) \
            if n == max_functions else checkpoint_for(d, n)
        print(f"pace take {n}: checkpoint {src.name} holds {cnt} functions" + (f" (duplicate of take {seen[cnt]}, skipped)" if cnt in seen else ""), flush=True)
        if cnt not in seen:
            seen[cnt] = n; rows.append((n, src, cnt))
    return rows


def convert(e, h, d, fit_wall_s):
    """Ladder checkpoints -> runs/pace/f<N>/{model.yaml,model.yace,status.json}; re-runnable alone."""
    for n, src, cnt in plan(e["take"], d, e["max_functions"]):
        out = common.run_dir("pace", f"f{n}"); out.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, out / "model.yaml")
        subprocess.run([_tool("pace_yaml2yace"), str(out / "model.yaml"), "-o", str(out / "model.yace")], check=True)
        with_e0(out / "model.yace")
        common.write_status(out, family="pace", size=f"f{n}", status="ok", model=str(out / "model.yace"),
                            n_params=cnt, requested=n, split_hash=h, fit_wall_s=fit_wall_s, ladder=str(d),
                            checkpoint=str(src.relative_to(d)))


def convert_only():
    from .split import ensure_split
    e, d = common.entry("pace", "ladder"), common.run_dir("pace", "ladder")
    st = common.read_status(d)
    if not st or st.get("status") != "ok":
        raise SystemExit(f"{d}: no finished ladder fit")
    convert(e, ensure_split(), d, st["fit_wall_s"])


if __name__ == "__main__":
    if sys.argv[1:] == ["--convert-only"]:
        convert_only()
    else:
        main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else None)
