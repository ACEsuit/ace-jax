"""Print a fusion instruction of an optimised HLO dump and the computation it
calls (with op_name metadata shortened), to see what a hot kernel computes.

    python bench/perf/hlo_fusion.py <after_optimizations.txt> <instr_name> [max_lines]
"""
import re
import sys


def main(path, name, max_lines=80):
    txt = open(path).read()
    m = re.search(rf"^\s*(?:ROOT )?%{re.escape(name)} = .*$", txt, re.M)
    if not m:
        print("not found")
        return
    line = m.group(0)
    print(re.sub(r"metadata=\{[^}]*\}", "", line)[:1500])
    c = re.search(r"calls=%([\w.\-]+)", line)
    if not c:
        return
    body = re.search(rf"^%{re.escape(c.group(1))} .*?^\}}", txt, re.M | re.S)
    lines = body.group(0).splitlines()
    for ln in lines[:max_lines]:
        op = re.search(r'op_name="([^"]*)"', ln)
        src = re.search(r'source_line=(\d+)', ln)
        short = re.sub(r",?\s*metadata=\{[^}]*\}", "", ln).strip()
        print(f"  {short[:180]}   # {(op.group(1)[-60:] if op else '')} L{src.group(1) if src else ''}")
    if len(lines) > max_lines:
        print(f"  ... {len(lines) - max_lines} more lines")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 80)
