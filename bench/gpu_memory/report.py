"""Markdown tables from the harness's rows (`gpu_memory.run`): per case, per variant, bisection, checks.

    cd bench && python -m gpu_memory.report gpu_memory/results/sulis-a100.jsonl
"""
import json
import sys

import numpy as np


def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def cases_table(rs, variant="baseline"):
    out = ["| cell | atoms | mode | K | fill | peak GB | program GB | temp GB | KB/atom (program) | us/atom-step |",
           "|---|--:|---|--:|--:|--:|--:|--:|--:|--:|"]
    order = {c: i for i, c in enumerate(("bulk", "slab", "crack"))}
    rs = sorted((r for r in rs if r.get("variant") == variant and not r.get("bisect") and "n" in r
                 and "error" not in r and "mode" in r),
                key=lambda r: (order.get(r["cell"], 9), r["n"], r["mode"] != "ef"))
    for r in rs:
        out.append(f"| {r['cell']} | {r['n']} | {r['mode']} | {r['K']} | {r['fill']:.2f} | {r['peak_bytes'] / 1e9:.2f} | "
                   f"{r['program_peak'] / 1e9:.2f} | {r['temp_bytes'] / 1e9:.2f} | {r['program_peak'] / r['n'] / 1e3:.1f} | "
                   f"{r['us_per_atom_step']:.2f} |")
    return "\n".join(out)


def fit(rs, variant, cell="bulk", mode="ef", n_min=32768):
    """(fixed bytes, marginal bytes/atom) of the program peak: least squares over n >= n_min."""
    pts = [(r["n"], r["program_peak"]) for r in rs if r.get("variant") == variant and r.get("cell") == cell
           and r.get("mode") == mode and "error" not in r and r.get("n", 0) >= n_min and r.get("program_peak")]
    if len(pts) < 2:
        return None
    n, b = np.array(pts, float).T
    slope, icpt = np.polyfit(n, b, 1)
    return icpt, slope


def variants_table(rs, n_ref=32768):
    out = ["| variant | program KB/atom at 32k | fixed GB + KB/atom (fit) | largest run | us/atom-step at 32k | parity max rel |",
           "|---|--:|--:|--:|--:|--:|"]
    seen = []
    for r in rs:
        v = r.get("variant")
        if v and v not in seen:
            seen.append(v)
    for v in seen:
        ref = [r for r in rs if r.get("variant") == v and r.get("cell") == "bulk" and r.get("mode") == "ef"
               and r.get("n") == n_ref and "error" not in r]
        ran = [r["n"] for r in rs if r.get("variant") == v and r.get("mode") == "ef" and "error" not in r and "n" in r]
        bis = [r for r in rs if r.get("variant") == v and r.get("bisect_result")]
        chk = [r for r in rs if r.get("variant") == v and r.get("check")]
        f = fit(rs, v)
        if not (ref or ran or chk):
            continue
        largest = (f"{bis[-1]['max_atoms']:,}" + ("" if bis[-1]["first_fail_atoms"] else "+")) if bis else (
            f"{max(ran):,} (not bisected)" if ran else "-")
        out.append(f"| {v} | {ref[-1]['program_peak'] / n_ref / 1e3:.1f} | " if ref else f"| {v} | - | ")
        out[-1] += (f"{f[0] / 1e9:.2f} + {f[1] / 1e3:.2f} | " if f else "- | ") + f"{largest} | "
        out[-1] += (f"{ref[-1]['us_per_atom_step']:.2f} | " if ref else "- | ")
        out[-1] += (f"{chk[-1]['parity_max_rel']:.1e} |" if chk else "- |")
    return "\n".join(out)


def checks(rs):
    out = []
    for r in rs:
        if not r.get("check"):
            continue
        s = f"- {r['variant']}: parity max rel {r['parity_max_rel']:.1e}, FD rel {r['fd_rel_err']:.1e}"
        for k in ("nve", "nve_baseline"):
            if k in r:
                d = r[k]
                s += (f"; {k} {d['steps']} steps x {d['n']} atoms: drift {d['drift_eV_atom']:.2e} eV/atom, "
                      f"max dev {d['max_dev_eV_atom']:.2e}")
        out.append(s)
    return "\n".join(out)


def main(argv=None):
    rs = rows((argv or sys.argv[1:])[0])
    print(cases_table(rs), "\n")
    print(variants_table(rs), "\n")
    print(checks(rs))


if __name__ == "__main__":
    main()
