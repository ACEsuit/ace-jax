"""Markdown tables for the YE3T / polymlp section of docs/dev/accuracy-cost-results.md, from the committed result
files (results/metrics.jsonl, results/sulis/sulis-cpu-*.jsonl), the QoI results (bench/si_qoi) and, where present,
the runs' status.json (design rank, alpha: on /storage, optional).

    python -m accuracy_cost.ye3t_polymlp_report          (from bench/)
"""
import json
import pathlib
import sys

from . import common
from .qoi_models import NAMES

SULIS = common.HERE / "results" / "sulis"
FIT_FAMILIES = ("polymlp", "ye3t_sym_fit", "ye3t_sym_wfit", "ye3t_mix_A", "ye3t_mix_T")
REFERENCE = ("ace", "ace_wl")                     # overlaid ladders (the "ACE saturates" question)
CHEB = ("o3d12", "o4d12", "o4d14", "o4d16", "o4d20")


def rows(path):
    p = pathlib.Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def us(path, code, family, size):
    """µs per atom-step of an ok, gated row, else None."""
    rs = rows(path)
    gated = any(r["mode"] == "parity" and r["status"] == "parity_ok" and (r["code"], r["family"], r["size"]) ==
                (code, family, size) for r in rs)
    for r in rs:
        if (r["code"], r["family"], r["size"]) == (code, family, size) and r.get("status") == "ok" \
                and r["mode"] != "parity" and gated:
            return 1e6 / r["atom_steps_per_s"]
    return None


def qoi(name):
    sys.path.insert(0, str(common.HERE.parent / "si_qoi"))
    from compare import qois
    from score import score
    s = score(name, qois("CASTEP"))
    return s["gm_norm"], s["n_qoi"], s["n_total"]


def fmt(v, f="{:.2f}"):
    return "–" if v is None else f.format(v)


def accuracy():
    met = {(r["family"], r["size"]): r for r in rows(common.RESULTS / "metrics.jsonl")}
    sweep = common.load_sweep()
    out = ["| family | size | coefficients | design rank | alpha | E RMSE (meV/atom) | F RMSE (eV/Å) | QoI error | CPU node µs/atom-step |",
           "|---|---|---|---|---|---|---|---|---|"]
    for fam in FIT_FAMILIES:
        code = "polymlp" if fam == "polymlp" else "ye3t"
        tfile = SULIS / f"sulis-cpu-128x1-{'polymlp' if fam == 'polymlp' else 'ye3t'}.jsonl"
        for e in sorted(sweep.get(fam) or [], key=lambda e: e.get("n_params") or 0):
            r = met.get((fam, e["size"]))
            if not r:
                continue
            st = common.read_status(common.run_dir(fam, e["size"])) or {}
            g, n, tot = qoi(NAMES[fam].format(s=e["size"]))
            out.append(f"| {fam} | {e['size']} | {r.get('n_params')} | {fmt((st.get('design_rank') or {}).get('rank'), '{}')} | "
                       f"{fmt(st.get('alpha') or (st.get('protocol') or {}).get('alpha'), '{:.0e}')} | "
                       f"{r['rmse_E'] * 1e3:.1f} | {r['rmse_F']:.4f} | "
                       f"{fmt(g) if n == tot else fmt(g) + f' ({n}/{tot})'} | {fmt(us(tfile, code, fam, e['size']))} |")
    return "\n".join(out)


def reference():
    met = [r for r in rows(common.RESULTS / "metrics.jsonl") if r["family"] in REFERENCE]
    out = ["| family | size | coefficients | E RMSE (meV/atom) | F RMSE (eV/Å) | QoI error |", "|---|---|---|---|---|---|"]
    for r in sorted(met, key=lambda r: (r["family"], r.get("n_params") or 0)):
        g, n, tot = qoi(("ACE_{s}" if r["family"] == "ace" else "ACEwl_{s}").format(s=r["size"]))
        out.append(f"| {r['family']} | {r['size']} | {r.get('n_params')} | {r['rmse_E'] * 1e3:.1f} | {r['rmse_F']:.4f} | "
                   f"{fmt(g) if n == tot else '–'} |")
    return "\n".join(out)


def evaluators(layout):
    """YE3T-symmetric against the other linear-ACE codes at one layout: µs per atom-step, and ratios to ye3t."""
    cols = [("ye3t-sym", "ace_cheb", f"sulis-cpu-{layout}-ye3t.jsonl", "YE3T (`pair_style ye3t`)"),
            ("mlpace-product", "ace_cheb", f"sulis-cpu-{layout}-ye3t.jsonl", "ML-PACE product"),
            ("mlpace", "ace_cheb", f"sulis-cpu-{layout}-ye3t.jsonl", "ML-PACE recursive"),
            # stock ML-PACE on the exact Tabulated file: the plotted 128x1 rows are the ace_tab5 run (earlier job)
            ("mlpace", "ace_tab", f"sulis-cpu-{layout}-ace_tab{'5' if layout == '128x1' else ''}.jsonl",
             "ML-PACE recursive, tabulated file"),
            ("mlpace", "ace_tab", f"sulis-cpu-{layout}-ace_tabfork.jsonl", "patched ML-PACE, tabulated file"),
            ("acejax-ace", "ace", f"sulis-cpu-{layout}-ffi.jsonl", "ace-jax C++ kernel")]
    cols = [c for c in cols if (SULIS / c[2]).exists()]
    out = ["| fit | " + " | ".join(c[3] for c in cols) + " | product / YE3T | recursive / YE3T | recursive (tabulated) / YE3T |",
           "|---|" + "---|" * (len(cols) + 3)]
    for s in CHEB:
        v = [us(SULIS / f, code, fam, s) for code, fam, f, _ in cols]
        y = v[0]
        ratio = lambda k, v=v, y=y: fmt(v[k] / y if y and len(v) > k and v[k] else None)
        out.append(f"| {s} | " + " | ".join(fmt(x) for x in v) + f" | {ratio(1)} | {ratio(2)} | {ratio(3)} |")
    return "\n".join(out)


def polymlp_timing():
    sweep = common.load_sweep().get("polymlp") or []
    out = ["| size | coefficients | 1 rank × 128 threads | 1 rank × 1 thread |", "|---|---|---|---|"]
    for e in sorted(sweep, key=lambda e: e["n_params"]):
        a, b = (us(SULIS / f"sulis-cpu-{l}-polymlp.jsonl", "polymlp", "polymlp", e["size"]) for l in ("128x1", "1x1"))
        out.append(f"| {e['size']} | {e['n_params']} | {fmt(a)} | {fmt(b)} |")
    return "\n".join(out)


def gates():
    out = ["| file | code | family | size | max dE (eV/atom) | max dF (eV/Å) | status |", "|---|---|---|---|---|---|---|"]
    for p in sorted(SULIS.glob("sulis-cpu-*ye3t.jsonl")) + sorted(SULIS.glob("sulis-cpu-*polymlp.jsonl")):
        for r in rows(p):
            if r["mode"] == "parity":
                out.append(f"| {p.stem} | {r['code']} | {r['family']} | {r['size']} | {fmt(r.get('max_dE_per_atom'), '{:.1e}')} | "
                           f"{fmt(r.get('max_dF'), '{:.1e}')} | {r['status']} |")
    return "\n".join(out)


if __name__ == "__main__":
    for title, f in (("Fitted families", accuracy), ("Reference ladders", reference),
                     ("Evaluators, 128 ranks x 1 thread (µs/atom-step)", lambda: evaluators("128x1")),
                     ("Evaluators, 1 rank x 1 thread (µs/atom-step)", lambda: evaluators("1x1")),
                     ("polymlp timing (µs/atom-step)", polymlp_timing), ("Gates", gates)):
        print(f"\n### {title}\n\n{f()}")
