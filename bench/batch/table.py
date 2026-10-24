"""Markdown table of bench/batch/run.py results: per (dtype, cell size, path), the ACECalculator loop's µs per
atom, the batched µs per atom at each batch size, and the best speed-up.

    python bench/batch/table.py results.jsonl [more.jsonl ...]"""
import json
import sys
from collections import defaultdict


def main(paths):
    loop, batch = {}, defaultdict(dict)
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            path = ("kernel" if r.get("ffi") else "JAX") + ("" if r["kind"] == "loop" else f" / {r['builder']}")
            key = (r["dtype"], r["n"], "kernel" if r.get("ffi") else "JAX")
            if r["kind"] == "loop":
                loop[key] = r["us_per_atom"]
            else:
                batch[key + (path,)][r["cells"]] = r["us_per_atom"]
    sizes = sorted({c for v in batch.values() for c in v})
    print("| dtype | atoms | model / builder | loop | " + " | ".join(f"B={c}" for c in sizes) + " | best speed-up |")
    print("|---|--:|---|--:|" + "--:|" * len(sizes) + "--:|")
    for k in sorted(batch):
        lp = loop.get(k[:3])
        row = [f"{batch[k][c]:.2f}" if c in batch[k] else "" for c in sizes]
        best = min(batch[k].values())
        print(f"| {k[0]} | {k[1]} | {k[3]} | {lp:.1f} | " + " | ".join(row) + f" | {lp / best:.1f}x |")


if __name__ == "__main__":
    main(sys.argv[1:])
