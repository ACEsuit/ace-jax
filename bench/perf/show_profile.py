"""Pretty-print a bench/perf/results/acejax_*.json profile.

    python bench/perf/show_profile.py bench/perf/results/acejax_Cantor_medium_8192_float64_dense_baseline.json
"""
import json
import sys


def show(path, top=40):
    d = json.load(open(path))
    k = d.pop("kernels", None)
    print(json.dumps(d, indent=1))
    if not k or "rows" not in k:
        print(k)
        return
    print({kk: v for kk, v in k.items() if kk != "rows"})
    for r in k["rows"][:top]:
        ops = [o[-80:] for o in (r.get("op_names") or [])][:2]
        print(f"{r['us_per_call']:8.1f}us {r['frac'] * 100:5.1f}% x{r['launches_per_call']:.0f} "
              f"{str(r['GBps']):>7s}GB/s bwd={r['bwd']} {r['hlo_op']} {r['out_shape']} "
              f"| {r['kernel'][:40]} | {r['srcs']} | {ops}")


def brief(path):
    """One line per profile (a file holding one JSON object or a list, or a
    JSON line as the last line of a log)."""
    txt = open(path).read().strip()
    try:
        d = json.loads(txt)
    except json.JSONDecodeError:
        d = json.loads(txt.splitlines()[-1])
    for r in (d if isinstance(d, list) else [d]):
        m = r.get("model", {})
        c = r.get("calc", {})
        print(f"{r.get('yace', '').split('pace_')[-1]:18s} n={r.get('n')} {r.get('variant')} "
              f"efv={m.get('efv_s')} rev={m.get('efv_rev_s')} E-only={m.get('energy_only_s')} "
              f"calc={c.get('call_s')} nl={c.get('nlist_s')} "
              f"par={r.get('parity_vs_baseline')} par_rev={r.get('parity_rev_vs_baseline')} "
              f"peak={r.get('peak_bytes')} err={str(r.get('error', ''))[:200]}")


if __name__ == "__main__":
    if sys.argv[1] == "--brief":
        for p in sys.argv[2:]:
            brief(p)
    else:
        show(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
