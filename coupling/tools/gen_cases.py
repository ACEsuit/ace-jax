"""Write coupling/python/tests/data/cases.json: the specs every coupling
backend test runs.  Run from the repo root with the ace-jax env:
    uv run python coupling/tools/gen_cases.py"""
import json
import pathlib

from ace_jax.construct.spec import build_spec, spec_from_reference, ylm_spec

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "coupling/python/tests/data/cases.json"


def _case(mb, R, Y):
    return {"mb": [[[int(n), int(l)] for n, l in bb] for bb in mb],
            "R": [[int(n), int(l)] for n, l in R],
            "Y": [[int(l), int(m)] for l, m in Y]}


def main():
    cases = {"tiny": _case([[(1, 0)], [(1, 1), (1, 1)]], [(1, 0), (1, 1), (2, 0)], ylm_spec(1))}
    for f in ["SiGe_o2d6", "CrMnFe_o2d5", "CrMnFe_o3d6", "SiGe_o4d5w05"]:
        mb, R, Y, _ = spec_from_reference(str(ROOT / f"fixtures/coupling_ref_{f}.npz"))
        cases[f] = _case(mb, R, Y)
    cases["build_NZ2_o5_d8"] = _case(*build_spec(2, 5, 8, 1.5))
    cases["build_NZ3_o4_d10"] = _case(*build_spec(3, 4, 10, 1.5))
    # reach the order-8 table bound: largest totaldegree with <= 3000 bodies
    for td in (3.0, 2.5, 2.0):
        mb, R, Y = build_spec(4, 8, td, 1.5)
        if len(mb) <= 3000 and max(map(len, mb)) == 8:
            cases["build_NZ4_o8"] = _case(mb, R, Y)
            break
    else:
        raise SystemExit("no order-8 case with <= 3000 bodies; lower NZ or totaldegree")
    for k, v in cases.items():
        print(f"{k:18s} n_mb={len(v['mb']):5d} max_order={max(map(len, v['mb']))}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, separators=(",", ":")))


if __name__ == "__main__":
    main()
