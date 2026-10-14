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
from plot_cost import FAM, ROOT, cost_name, family_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("timings", nargs="+")
    ap.add_argument("--dtype", help="only rows at this precision (float64: like-for-like, e.g. MACE without float32)")
    ap.add_argument("--out", default=str(ROOT / "bench/si_qoi/results/cost_sulis_gpu.json"))
    a = ap.parse_args()
    rows, gates = [], set()
    for f in a.timings:
        rs = [json.loads(l) | {"_file": pathlib.Path(f).stem} for l in open(f)]
        gates |= {(f, r["code"], r["family"], r["size"]) for r in rs if r.get("mode") == "parity" and r.get("status") == "parity_ok"}
        rows += [r | {"_path": f} for r in rs]
    best, code = {}, {}
    for r in rows:
        fam = family_of(r)                                       # plot_cost's split families (MACE by L, wL)
        if r.get("mode") == "parity" or r.get("status") != "ok" or fam not in FAM or (a.dtype and r["dtype"] != a.dtype):
            continue
        if r.get("mode") == "lammps" and (r["_path"], r["code"], r["family"], r["size"]) not in gates:   # standalone: no gate
            continue
        name = cost_name(fam, r["size"])                      # Symmetrix-XL under its own name (MACExl_)
        us = 1e6 / r["atom_steps_per_s"]
        if us < best.get(name, float("inf")):
            best[name], code[name] = us, f'{r["code"]} {r["dtype"]} {r["mode"]} [{r["_file"]}]'
    pathlib.Path(a.out).write_text(json.dumps(best, indent=1, sort_keys=True))
    pathlib.Path(a.out).with_suffix(".codes.json").write_text(json.dumps(code, indent=1, sort_keys=True))
    for n in sorted(best):
        print(f"{n:22s} {best[n]:8.3f} us/atom-step  ({code[n]})")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
