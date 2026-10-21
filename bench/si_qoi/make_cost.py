"""Cost JSON for plot_cost.py --cost from a timing results file (accuracy_cost.timing rows):
{QoI model name: microseconds per atom-step}, taking each model's fastest timed code (a LAMMPS row only if
it passed its parity gate; standalone rows have none) (ACE: the ace-jax LAMMPS plugin; PACE: ace-jax or ML-PACE; MACE: float64 LAMMPS or float32
standalone).  Models with no ok row are left out (plot_cost then drops them from the cost axis).

    python bench/si_qoi/make_cost.py bench/accuracy_cost/results/sulis-gpu.jsonl [more.jsonl ...] [--out cost.json]

Several files are pooled (e.g. sulis-gpu-ace-splined.jsonl: linear ACE with its analytic radials splined
at 1e-10, as learned radials are by default); each row's gate is looked up in its own file.
"""
import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from plot_cost import COST_NAME, FAM, REDUCED, ROOT, SEEDED, cost_name, family_of, seed_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("timings", nargs="+")
    ap.add_argument("--dtype", help="only rows at this precision (float64: like-for-like, e.g. MACE without float32); "
                                    "the reduced-precision lines (plot_cost.REDUCED) are kept, under their own names")
    ap.add_argument("--prefer", action="append", default=[],
                    help="a timings file whose rows supersede every other file's for the models it times "
                         "(e.g. a re-timed ladder); repeatable")
    ap.add_argument("--allow", action="append", default=[], metavar="CODE:DTYPE",
                    help="also keep this code at this precision past --dtype (nep-gpumd:mixed: GPUMD's NEP is mixed "
                         "precision only; plot_cost marks it)")
    ap.add_argument("--source", action="append", default=[], metavar="CODE=STEM",
                    help="for the models a timing file with stem STEM covers, take this code's rows only from it (acejax-ace=sulis-gpu-acejax-auto: "
                         "each ace-jax model in the product-basis form aa_dag='auto' picks, not the fastest of every run)")
    ap.add_argument("--out", default=str(ROOT / "bench/si_qoi/results/cost_sulis_gpu.json"))
    a = ap.parse_args()
    allow = {tuple(s.split(":", 1)) for s in a.allow}
    source = dict(s.split("=", 1) for s in a.source)
    rows, gates = [], set()
    for f in a.timings:
        rs = [json.loads(l) | {"_file": pathlib.Path(f).stem} for l in open(f)]
        gates |= {(f, r["code"], r["family"], r["size"]) for r in rs if r.get("mode") == "parity" and r.get("status") == "parity_ok"}
        rows += [r | {"_path": f} for r in rs]
    # models a --source file covers: their rows of that code come from it alone (others, e.g. the wL ladder, as before)
    sourced = {(r["code"], r["family"], r["size"]) for r in rows if source.get(r["code"]) == r["_file"]}
    best, code, fam_of = {}, {}, {}
    prefer = [str(pathlib.Path(p)) for p in a.prefer]
    preferred = {}                       # cost name -> the --prefer file that times it
    for r in rows:
        if str(pathlib.Path(r["_path"])) in prefer and r.get("mode") != "parity" and family_of(r) in FAM:
            preferred.setdefault(cost_name(family_of(r), r["size"]), r["_path"])
    for r in rows:
        fam = family_of(r)                                       # plot_cost's split families (MACE by L, wL)
        if r.get("mode") == "parity" or r.get("status") != "ok" or fam not in FAM or (
                a.dtype and r["dtype"] != a.dtype and fam not in REDUCED and (r["code"], r["dtype"]) not in allow):
            continue
        if preferred.get(cost_name(fam, r["size"]), r["_path"]) != r["_path"]:   # superseded by a --prefer file
            continue
        if r.get("mode") in ("lammps", "gpumd") and (r["_path"], r["code"], r["family"], r["size"]) not in gates:   # standalone: no gate
            continue
        # a line under its own cost name (float32, the FFI kernel) is not the sourced line: kept from any file
        if (r["code"], r["family"], r["size"]) in sourced and r["_file"] != source[r["code"]] and fam not in COST_NAME:
            continue
        name = cost_name(fam, r["size"])                      # Symmetrix-XL under its own name (MACExl_)
        us = 1e6 / r["atom_steps_per_s"]
        if us < best.get(name, float("inf")):
            best[name], fam_of[name], code[name] = us, fam, (f'{r["code"]} {r["dtype"]} {r["mode"]}' + (f' dag:{r["aa_dag"]}' if r.get("aa_dag") else "")
                                          + f' [{r["_file"]}]')
    for n in list(best):                    # seeded families (plot_cost.SEEDED): seed 0 is timed for its size
        base, k = seed_split(n)             # only theirs: GAP's sizes (s9000, ...) read as seed suffixes too
        if fam_of[n] in SEEDED and k is not None and base not in best:
            best[base], code[base] = best[n], code[n]
    pathlib.Path(a.out).write_text(json.dumps(best, indent=1, sort_keys=True) + "\n")
    pathlib.Path(a.out).with_suffix(".codes.json").write_text(json.dumps(code, indent=1, sort_keys=True) + "\n")
    for n in sorted(best):
        print(f"{n:22s} {best[n]:8.3f} us/atom-step  ({code[n]})")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
