"""Attribute an XLA program's peak device memory to the ace-jax stage that made each buffer.

Reads one module's `--xla_dump_to` text dump (`--xla_dump_hlo_as_text`):
  `*-live-range.txt`          the flattened schedule (while bodies included) and
                              the values live at its peak ("Live ranges at T (peak)");
  `*-buffer-assignment.txt`   where each value sits: XLA:GPU packs every temporary
                              into one preallocated-temp allocation, at offsets
                              that values with disjoint lifetimes share;
  `*after_optimizations.txt`  each instruction's op_name and stack frame (the
                              file/function/line tables printed at its top).
The peak-live values are deduplicated by address (in-place updates alias), the
largest first, and each charged to the innermost ace-jax frame of its
instruction, mapped to a stage by `CATEGORIES`, and to a role from its op_name:
"residual" (recomputed forward values under jax.checkpoint: `rematted_computation`),
"cotangent" (other reverse-mode values: `transpose(`), else "forward".  Temp
bytes not live at the peak are "slack" (heap-simulation fragmentation).
Arguments split into the skin list ("arrays") and the model's parameters.

    python -m gpu_memory.attribute <dump dir> [--n atoms]
"""
import argparse
import collections
import glob
import os
import re

# (file basename, function) -> stage; a function absent here falls back to its file's "*" entry
CATEGORIES = {
    ("skin.py", "*"): "pair list",                          # skin list, compaction, force assembly
    ("edge_model.py", "site_energies_dense_blocked"): "pair list",   # block copies of rij/zi/zj/mask
    ("edge_model.py", "blocks"): "pair list",
    ("edge_model.py", "energy_forces_virial_dense"): "pair list",    # g_r, strain
    ("edge_model.py", "total"): "pair list",
    ("edge_model.py", "pool_a_dense"): "A basis",
    ("edge_model.py", "_mul_rows"): "product basis",
    ("edge_model.py", "_product_basis_t_default"): "product basis",
    ("edge_model.py", "_product_basis_default"): "product basis",
    ("edge_model.py", "product_basis_dot"): "product basis",
    ("edge_model.py", "_product_basis_dot_default"): "product basis",
    ("edge_model.py", "*"): "other",
    ("model.py", "_blocked_a_onehot"): "A basis",
    ("model.py", "_blocked_a_scatter"): "A basis",
    ("model.py", "_readout_folded"): "readout",
    ("model.py", "*"): "per-pair radial/harmonic",          # _blocked_radial, radial, angular, R masking
    ("radial.py", "*"): "per-pair radial/harmonic",
    ("harmonics.py", "*"): "per-pair radial/harmonic",
    ("splinify.py", "*"): "per-pair radial/harmonic",
    ("aa_dag.py", "*"): "product basis",
    ("nlist.py", "*"): "pair list",
}
STAGES = ("per-pair radial/harmonic", "A basis", "product basis", "readout", "pair list", "other")
ROLES = ("forward", "residual", "cotangent")


def _module_files(dump, module_re):
    ba = [p for p in glob.glob(os.path.join(dump, "*after_optimizations-buffer-assignment.txt"))
          if re.search(module_re, os.path.basename(p))]
    if not ba:
        raise FileNotFoundError(f"no buffer assignment matching {module_re!r} in {dump}")
    # the last module compiled (module_NNNN): the case's step, not a warm-up compiled before it
    ba = max(ba, key=lambda p: int(re.search(r"module_(\d+)", os.path.basename(p))[1]))
    stem = ba[:-len("-buffer-assignment.txt")]
    return ba, stem + ".txt", stem + "-live-range.txt"


def _frames(hlo_lines):
    """chain(stack frame id) -> [(file, function, line), innermost first], from the dump's tables."""
    tables, cur = collections.defaultdict(dict), None
    for line in hlo_lines:
        s = line.strip()
        if s in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
            cur = s
            continue
        m = re.match(r"(\d+) (.*)", s) if cur else None
        if m:
            tables[cur][int(m[1])] = m[2]
        else:
            cur = None                                     # a blank line ends the tables
    files = {k: v.strip('"') for k, v in tables["FileNames"].items()}
    funcs = {k: v.strip('"') for k, v in tables["FunctionNames"].items()}
    locs, frames = {}, {}
    for k, v in tables["FileLocations"].items():
        d = dict(re.findall(r"(\w+)=(\d+)", v))
        locs[k] = (files.get(int(d["file_name_id"]), "?"), funcs.get(int(d["function_name_id"]), "?"),
                   int(d["line"]))
    for k, v in tables["StackFrames"].items():
        d = dict(re.findall(r"(\w+)=(\d+)", v))
        frames[k] = (int(d["file_location_id"]), int(d["parent_frame_id"]))

    def chain(fid):
        out, seen = [], set()
        while fid in frames and fid not in seen:
            seen.add(fid)
            loc, parent = frames[fid]
            out.append(locs.get(loc, ("?", "?", 0)))
            fid = parent
        return out
    return chain


def _instructions(hlo_lines):
    """instruction name -> (op_name, stack_frame_id)."""
    out = {}
    pat = re.compile(r"^\s*(?:ROOT )?%([\w.\-]+) = .*?metadata=\{([^}]*)\}")
    for line in hlo_lines:
        m = pat.match(line)
        if m:
            op = re.search(r'op_name="([^"]*)"', m[2])
            sf = re.search(r"stack_frame_id=(\d+)", m[2])
            out[m[1]] = (op[1] if op else "", int(sf[1]) if sf else None)
    return out


def _allocations(ba_lines):
    """[{size, kind, values: [(name with {index}, size, shape, offset)]}]."""
    allocs, cur = [], None
    for line in ba_lines:
        m = re.match(r"allocation (\d+): size (\d+)(.*)", line)
        if m:
            kind = ("argument" if "parameter" in m[3] else "constant" if "constant" in m[3]
                    else "output" if "live-out" in m[3] else "temp")
            cur = {"id": int(m[1]), "size": int(m[2]), "kind": kind, "values": []}
            allocs.append(cur)
            continue
        m = re.match(r"\s+value: <\d+ ([\w.\-]+(?:\{[\d,]*\})?) @\d+> \(size=(\d+),offset=(\d+)\): (\S+)", line)
        if m and cur is not None:
            cur["values"].append((m[1].removesuffix("{}"), int(m[2]), m[4], int(m[3])))
        elif not line.startswith((" ", "allocation")):
            cur = None
    return allocs


def _peak_live(lr_lines):
    """(peak schedule time, {value name with {index}}) from the live-range dump."""
    T = next(int(m[1]) for m in (re.search(r"Live ranges at (\d+) \(peak\)", s) for s in lr_lines) if m)
    live, on = set(), False
    for s in lr_lines:
        if s.strip() == "BufferLiveRange:":
            on = True
            continue
        if on:
            m = re.match(r"\s+(\S+):(\d+)-(\d+)$", s)
            if not m:
                on = False
                continue
            if int(m[2]) <= T <= int(m[3]):
                live.add(m[1].removesuffix("{}"))          # "x{}" there is plain "x" in the assignment
    return T, live


def _paint(values):
    """[(value, bytes)]: the union of the values' address ranges, each byte charged to the
    largest value covering it (aliasing values -- in-place updates -- are counted once)."""
    painted, out = [], []                     # disjoint sorted [lo, hi) ranges
    for v in sorted(values, key=lambda v: -v[1]):
        lo, hi = v[3], v[3] + v[1]
        free, cur = 0, lo
        for a, b in painted:
            if b <= cur or a >= hi:
                continue
            free += max(0, a - cur)
            cur = max(cur, b)
        free += max(0, hi - cur)
        if free:
            painted = sorted(painted + [(lo, hi)])
            merged = [painted[0]]
            for a, b in painted[1:]:
                if a <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], b))
                else:
                    merged.append((a, b))
            painted = merged
            out.append((v, free))
    return out


def stage_of(chain):
    """(stage, "file:function:line") of the innermost ace_jax frame of a stack (innermost first)."""
    for f, fn, ln in chain:
        if "/ace_jax/" not in f:
            continue
        base, fn = os.path.basename(f), fn.rsplit(".", 1)[-1]        # "ACEModel.radial" -> "radial"
        cat = CATEGORIES.get((base, fn)) or CATEGORIES.get((base, "*"))
        if cat:
            return cat, f"{base}:{fn}:{ln}"
    return "other", (f"{os.path.basename(chain[0][0])}:{chain[0][1]}:{chain[0][2]}" if chain else "?")


def role_of(op):
    return "residual" if "rematted_computation" in op else "cotangent" if "transpose(" in op else "forward"


def attribute(dump, module_re=r"jit_packed|jit_energy", top=25):
    """{"temp", "argument", "output": allocation bytes; "peak_live": bytes live at the peak;
    "stages": {stage: {role: bytes}} of the peak-live temp; "arguments": {"skin list", "model"};
    "slack": temp not live at the peak; "top": the largest peak-live values}."""
    ba, hlo, lr = _module_files(dump, module_re)
    hlo_lines = open(hlo).read().splitlines()
    chain, ins = _frames(hlo_lines), _instructions(hlo_lines)
    allocs = _allocations(open(ba).read().splitlines())
    T, live = _peak_live(open(lr).read().splitlines())
    totals = collections.Counter()
    stages = {s: dict.fromkeys(ROLES, 0) for s in STAGES}
    args = collections.Counter()
    sites = collections.Counter()
    rows = []
    for a in allocs:
        totals[a["kind"]] += a["size"]
        if a["kind"] == "argument":
            name = a["values"][0][0] if a["values"] else ""
            args["skin list" if name.startswith(("arrays", "u.")) else "model"] += a["size"]
            continue
        if a["kind"] != "temp":
            continue
        for (name, _, shape, _), size in _paint([v for v in a["values"] if v[0] in live]):
            base = re.sub(r"\{[\d,]*\}$", "", name)
            op, sf = ins.get(base, ("", None))
            stage, site = stage_of(chain(sf) if sf is not None else [])
            role = role_of(op)
            stages[stage][role] += size
            sites[(stage, role, site)] += size
            rows.append({"size": size, "stage": stage, "role": role, "site": site, "shape": shape})
    rows.sort(key=lambda r: -r["size"])
    live_temp = sum(sum(d.values()) for d in stages.values())
    return {"module": os.path.basename(ba), "peak_time": T, **dict(totals), "arguments": dict(args),
            "peak_live": live_temp + totals["argument"] + totals["output"], "slack": totals["temp"] - live_temp,
            "stages": stages,
            "sites": [{"stage": s, "role": r, "site": site, "bytes": b}
                      for (s, r, site), b in sites.most_common(top)],
            "top": rows[:top]}


def table(att, n):
    """Markdown: bytes per atom live at the peak, by stage and role."""
    lines = ["| at the peak | forward | residual | cotangent | total |", "|---|--:|--:|--:|--:|"]
    for s, d in att["stages"].items():
        if sum(d.values()):
            lines.append(f"| {s} | " + " | ".join(f"{d[r] / n:,.0f}" for r in ROLES)
                         + f" | {sum(d.values()) / n:,.0f} |")
    for k, b in att["arguments"].items():
        lines.append(f"| argument: {k} | | | | {b / n:,.0f} |")
    lines.append(f"| outputs | | | | {att.get('output', 0) / n:,.0f} |")
    lines.append(f"| temp slack (not live at the peak) | | | | {att['slack'] / n:,.0f} |")
    lines.append(f"| **program total** | | | | **{(att.get('temp', 0) + att.get('argument', 0) + att.get('output', 0)) / n:,.0f}** |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--module", default=r"jit_packed|jit_energy")
    a = p.parse_args(argv)
    att = attribute(a.dump, a.module)
    print(table(att, a.n))
    for r in att["top"][:12]:
        print(f"{r['size'] / a.n:10,.0f} B/atom  {r['stage']:26s} {r['role']:9s} {r['site']:45s} {r['shape']}")


if __name__ == "__main__":
    main()
