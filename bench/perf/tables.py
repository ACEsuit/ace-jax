"""Markdown tables for docs/pace-performance-gap.md from bench/perf/results/bigsweep.json.

    python bench/perf/tables.py bench/perf/results/bigsweep.json
"""
import collections
import json
import sys


def k(x):
    return "-" if x is None else f"{x / 1e3:,.0f}k"


def main(path):
    rows = json.load(open(path))
    gpu = sorted({r.get("gpu") for r in rows if r.get("gpu")})
    print(f"GPU: {', '.join(gpu)}\n")
    by = collections.defaultdict(dict)
    for r in rows:
        key = (r["model"], int(r["n"]), r["dtype"])
        if r["code"] == "mlpace":
            by[key]["mlpace"] = r.get("atom_steps_per_s")
            continue
        n = int(r["n"])
        if r.get("variant") == "baseline":
            a = r.get("ase_calc") or {}
            s = r.get("skin_calc") or {}
            if a.get("call_s"):
                by[key]["ase_e2e"] = n / a["call_s"]
                by[key]["ase_model"] = n / a["model_s"]
                by[key]["ase_nlist_ms"] = 1e3 * a["nlist_s"]
                by[key]["ase_call_ms"] = 1e3 * a["call_s"]
                by[key]["ase_model_ms"] = 1e3 * a["model_s"]
            else:
                by[key]["ase_e2e"] = None
                by[key]["ase_status"] = a.get("status")
            by[key]["skin"] = n / s["call_s"] if s.get("call_s") else None
            by[key]["skin_status"] = s.get("status")
            by[key]["rebuild_ms"] = 1e3 * s["rebuild_s"] if s.get("rebuild_s") else None
            by[key]["parity"] = r.get("parity_skin_vs_ase")
        else:
            s = r.get("skin_calc") or {}
            by[key]["fast"] = n / s["call_s"] if s.get("call_s") else None
            by[key]["fast_status"] = s.get("status") or (r.get("error", "")[:40] if r.get("error") else None)
    print("| model | N | dtype | ML-PACE (LAMMPS) | ace-jax ASE calc e2e | ace-jax model call only "
          "| gap e2e | gap model | skin calc | skin calc + variant | gap (skin+variant) |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(by, key=lambda t: (t[0].split("_")[0], ["small", "medium", "large"].index(
            t[0].split("_")[1]), t[2], t[1])):
        d = by[key]
        ml = d.get("mlpace")
        g = lambda x: f"{ml / x:.1f}x" if (ml and x) else "-"  # noqa: B023 (used this iteration)
        print(f"| {key[0]} | {key[1]} | {key[2][-2:]} | {k(ml)} | {k(d.get('ase_e2e')) if d.get('ase_e2e') else d.get('ase_status') or '-'} "
              f"| {k(d.get('ase_model'))} | {g(d.get('ase_e2e'))} | {g(d.get('ase_model'))} "
              f"| {k(d.get('skin')) if d.get('skin') else d.get('skin_status') or '-'} "
              f"| {k(d.get('fast')) if d.get('fast') else d.get('fast_status') or '-'} | {g(d.get('fast'))} |")
    print("\nASE calculator split (ms per call): call = neighbour list + transfers + model + rest")
    print("| model | N | call | nlist_s (list + layout + H2D) | model call | skin rebuild |")
    print("|---|---:|---:|---:|---:|---:|")
    for key in sorted(by):
        d = by[key]
        if d.get("ase_call_ms") and key[2] == "float64":
            print(f"| {key[0]} | {key[1]} | {d['ase_call_ms']:.1f} | {d['ase_nlist_ms']:.1f} "
                  f"| {d['ase_model_ms']:.1f} | {d.get('rebuild_ms') or 0:.1f} |")
    print("\nparity (skin calc vs ASE calc, same model):",
          max((d["parity"]["max_dF"] for d in by.values() if d.get("parity")), default=None))


if __name__ == "__main__":
    main(sys.argv[1])
