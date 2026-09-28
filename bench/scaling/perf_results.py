"""Tables for docs/perf-optimisation-results.md, from the committed rows.

    python bench/scaling/perf_results.py ['bench/scaling/results/*.jsonl'] [docs/perf-optimisation-results.md]

The prose lives in `perf_results_template.md`; its `{{name}}` placeholders are
filled with tables computed here from:
- the scaling rows (after: the pattern; before: `before-perf/` beside it);
  both with their repeat runs from `repeats/` (`plot.load`; a repeated case
  counts at its median);
- the Task 10 micro-benchmarks, `bench/perf/results/microbench_{before,after}.json`.
A host whose ace-jax rows are still being re-run is listed as pending.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scaling.plot import (ACEJAX, CODES, MODES, aggregate, before_after_hosts,  # noqa: E402
                          before_pattern, load, parity_table, si, spread_pct, target_n,
                          throughput)

HERE = pathlib.Path(__file__).resolve().parent
MICRO = HERE.parent / "perf" / "results"
HOST_ORDER = ("modal-a100", "moriarty-gpu", "moriarty-cpu")
SYSTEMS, SIZES = ("SiGe", "Cantor"), ("small", "medium", "large")

# the spec's success criteria (docs/perf-optimisation-spec.md): Cantor medium,
# float64, A100, 8,192 atoms
CRIT_HOST, CRIT_SYSTEM, CRIT_N = "modal-a100", "Cantor", 8192
MODEL_MS_TARGET, E2E_TARGET, LAMMPS_TARGET, LAMMPS_SIZES = 4.5, 1.1e6, 1.0e6, (32768, 131072)


def _ok(rows, host, system, size, code, mode, dtype="float64"):
    """{n: row} of a line's ok rows (one per case, at the median of its runs),
    and the line's largest ok N."""
    got = {r["n_atoms"]: r for r in aggregate(rows)
           if (r.get("host"), r.get("system"), r.get("size"), r.get("code"), r.get("mode"),
               r.get("dtype"), r.get("status")) == (host, system, size, code, mode, dtype, "ok")
           and throughput(r)}
    return got, (max(got) if got else None)


def summary_table(after, before, host):
    """before / after at the host's target N (float64), per system, size, code
    and mode, with ML-PACE in LAMMPS and each line's largest N."""
    n0 = target_n(host)
    out = ["| system | size | code | mode | before | after | speed-up | ML-PACE in LAMMPS "
           "| largest N before | largest N after |", "|---|---|---|---|---|---|---|---|---|---|"]
    for system in SYSTEMS:
        for size in SIZES:
            ml, _ = _ok(after, host, system, size, "mlpace", "lammps")
            ml_s = si(throughput(ml[n0])) if n0 in ml else "—"
            for code in ACEJAX:
                for mode in MODES:
                    b, bmax = _ok(before, host, system, size, code, mode)
                    a, amax = _ok(after, host, system, size, code, mode)
                    if not (a or b):
                        continue
                    tb = throughput(b[n0]) if n0 in b else None
                    ta = throughput(a[n0]) if n0 in a else None
                    up = f"{ta / tb:.1f}×" if ta and tb else "—"
                    cb = si(tb) + spread_pct(b[n0]) if tb else "—"
                    ca = si(ta) + spread_pct(a[n0]) if ta else "—"
                    out.append(f"| {system} | {size} | {CODES[code][0]} | {mode} "
                               f"| {cb} | {ca} | {up} "
                               f"| {ml_s} | {bmax or '—'} | {amax or '—'} |")
    return "\n".join(out)


def summaries(after, before):
    done, pending = before_after_hosts(after, before)
    parts = []
    for h in sorted(done, key=lambda h: HOST_ORDER.index(h) if h in HOST_ORDER else 99):
        parts += [f"### {h} (N = {target_n(h):,})", "", summary_table(after, before, h), ""]
    for h in pending:
        parts += [f"### {h}", "", "Pending: the ace-jax rows are being re-run.", ""]
    return "\n".join(parts).strip()


def _micro(which):
    p = MICRO / f"microbench_{which}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _micro_key(r):
    return (pathlib.Path(r["model"]).stem, r["n"], r["dtype"])


def micro_table():
    before, after = _micro("before"), _micro("after")
    if not (before and after):
        return "Micro-benchmark results not found."
    b = {_micro_key(r): r for r in before["rows"] if r.get("status") == "ok"}
    out = [f"{after['gpu']}, one container for both runs (before at `{before['src_commit'][:7]}`, "
           f"after at `{after['src_commit'][:7]}`); median of repeated MD-like calls.", "",
           "| model | N | model call before (ms) | compiled step after (ms) "
           "| model call after, calculator-level (ms) | end to end before | end to end after "
           "| speed-up |", "|---|---|---|---|---|---|---|---|"]
    for r in after["rows"]:
        k = _micro_key(r)
        if r.get("status") != "ok" or k not in b:
            continue
        rb = b[k]
        step = f"{r['step_s'] * 1e3:.2f}" if r.get("step_s") else "—"
        up = r["call_atom_steps_per_s"] / rb["call_atom_steps_per_s"]
        out.append(f"| {k[0]} | {k[1]} | {rb['model_s'] * 1e3:.2f} | {step} "
                   f"| {r['model_s'] * 1e3:.2f} | {si(rb['call_atom_steps_per_s'])} "
                   f"| {si(r['call_atom_steps_per_s'])} | {up:.1f}× |")
    return "\n".join(out)


def _verdict(ok):
    return "**met**" if ok else "**missed**"


def _ran(r):
    return f"ran ({r.get('layout')}), {si(throughput(r))}" if r else "did not run"


def criteria_table(after, before):
    """Each success criterion, met or missed, with the measured numbers."""
    out = ["| criterion | target | before | after | verdict |", "|---|---|---|---|---|"]
    mb, ma = _micro("before"), _micro("after")
    key = (f"pace_{CRIT_SYSTEM}_medium", CRIT_N, "float64")
    rb = next((r for r in (mb or {}).get("rows", []) if _micro_key(r) == key), None)
    ra = next((r for r in (ma or {}).get("rows", []) if _micro_key(r) == key), None)
    if rb and ra:
        step, call = ra["step_s"] * 1e3, ra["model_s"] * 1e3
        out.append(f"| model call, like for like (micro-benchmark) | ≤ {MODEL_MS_TARGET} ms "
                   f"| {rb['model_s'] * 1e3:.2f} ms | {step:.2f} ms (compiled step) "
                   f"| {_verdict(step <= MODEL_MS_TARGET)} |")
        out.append(f"| model call, calculator-level incl. host-device traffic | ≤ {MODEL_MS_TARGET} ms "
                   f"| {rb['model_s'] * 1e3:.2f} ms | {call:.2f} ms "
                   f"| {_verdict(call <= MODEL_MS_TARGET)} |")
        eb, ea = rb["call_atom_steps_per_s"], ra["call_atom_steps_per_s"]
        out.append(f"| end to end (micro-benchmark) | ≥ {si(E2E_TARGET)} atom-steps/s | {si(eb)} "
                   f"| {si(ea)} | {_verdict(ea >= E2E_TARGET)} |")
    pts = {}
    for label, rows in (("before", before), ("after", after)):
        for mode in MODES:
            got, _ = _ok(rows, CRIT_HOST, CRIT_SYSTEM, "medium", "acejax-pace", mode)
            pts[(label, mode)] = got
    if pts[("after", "standalone")]:
        tb = pts[("before", "standalone")].get(CRIT_N)
        ta = pts[("after", "standalone")].get(CRIT_N)
        if ta:
            out.append(f"| end to end (scaling suite, standalone) | ≥ {si(E2E_TARGET)} atom-steps/s "
                       f"| {si(throughput(tb)) if tb else '—'} | {si(throughput(ta))} "
                       f"| {_verdict(throughput(ta) >= E2E_TARGET)} |")
    if pts[("after", "lammps")]:
        tb = pts[("before", "lammps")].get(CRIT_N)
        ta = pts[("after", "lammps")].get(CRIT_N)
        if ta:
            out.append(f"| LAMMPS throughput | ≥ {si(LAMMPS_TARGET)} atom-steps/s "
                       f"| {si(throughput(tb)) if tb else '—'} | {si(throughput(ta))} "
                       f"| {_verdict(throughput(ta) >= LAMMPS_TARGET)} |")
        for n in LAMMPS_SIZES:
            rb_, ra_ = pts[("before", "lammps")].get(n), pts[("after", "lammps")].get(n)
            out.append(f"| LAMMPS runs at {n:,} atoms | runs | {_ran(rb_)} | {_ran(ra_)} "
                       f"| {_verdict(bool(ra_))} |")
        _, bmax = _ok(before, CRIT_HOST, CRIT_SYSTEM, "medium", "acejax-pace", "lammps")
        _, amax = _ok(after, CRIT_HOST, CRIT_SYSTEM, "medium", "acejax-pace", "lammps")
        out.append(f"| LAMMPS largest N (for information) | — | {bmax or '—'} | {amax or '—'} | — |")
    return "\n".join(out)


def render(pattern="bench/scaling/results/*.jsonl", doc="docs/perf-optimisation-results.md"):
    after, before = load(pattern), load(before_pattern(pattern))
    text = (HERE / "perf_results_template.md").read_text()
    for name, fill in (("microbench", micro_table()), ("criteria", criteria_table(after, before)),
                       ("summary", summaries(after, before)), ("parity", parity_table(after))):
        text = text.replace("{{" + name + "}}", fill)
    pathlib.Path(doc).write_text(text)
    return doc


if __name__ == "__main__":
    print(render(*sys.argv[1:3]))
