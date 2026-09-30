"""Tabulate qoi.py results against a reference (MACE): per-quantity deviations.

    python bench/learn_radial/qoi_compare.py --dir runs/qoi --ref mace
Files: <dir>/<ref>_<system>.json and every other <model>_<system>*.json.
"""
import argparse
import json
import pathlib

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--dir", required=True); p.add_argument("--ref", default="mace")
a = p.parse_args()
d = pathlib.Path(a.dir)


def load(f):
    return json.loads(f.read_text())


def pct(x, r):
    return 100.0 * (x - r) / abs(r)


for system in ("sige", "cantor"):
    ref = load(d / f"{a.ref}_{system}.json")
    models = sorted(f for f in d.glob(f"*_{system}*.json") if not f.name.startswith(a.ref))
    print(f"\n===== {system}: deviation from {a.ref} (lattice/moduli in %, vacancy in eV)")
    if system == "sige":
        cols = [(ph, q) for ph in ("Si", "Ge", "SiGe") for q in ("a0_cubic", "B_GPa", "C11_GPa", "C12_GPa", "C44_GPa")]
        print(f"{'model':32s}" + "".join(f"{ph + ' ' + q.split('_')[0]:>11s}" for ph, q in cols)
              + f"{'Si vac':>9s}{'Ge vac':>9s}{'|%| mean':>10s}")
        print(f"{a.ref + ' (absolute)':32s}" + "".join(f"{ref[ph][q]:11.3f}" for ph, q in cols)
              + f"{ref['Si']['vacancy_eV']:9.3f}{ref['Ge']['vacancy_eV']:9.3f}")
        for f in models:
            m = load(f)
            dev = [pct(m[ph][q], ref[ph][q]) for ph, q in cols]
            dv = [m[e]["vacancy_eV"] - ref[e]["vacancy_eV"] for e in ("Si", "Ge")]
            print(f"{f.stem:32s}" + "".join(f"{x:+11.2f}" for x in dev) + "".join(f"{x:+9.3f}" for x in dv)
                  + f"{np.mean(np.abs(dev)):10.2f}")
    else:
        def derived(dr):
            return {"a0": dr["a0_cubic"], "B": dr["B_GPa"], "C'": 0.5 * (dr["C11_GPa"] - dr["C12_GPa"]),
                    "C44": dr["C44_GPa"], "C11": dr["C11_GPa"], "C12": dr["C12_GPa"]}
        qs = ["a0", "B", "C'", "C44", "C11", "C12"]
        rd = [derived(x) for x in ref["draws"]]
        print(f"{'model':32s}" + "".join(f"{'d' + str(k) + ' ' + q:>10s}" for k in range(3) for q in qs[:4])
              + f"{'vac MAE':>9s}{'|%| mean':>10s}")
        print(f"{a.ref + ' (absolute)':32s}" + "".join(f"{rd[k][q]:10.2f}" for k in range(3) for q in qs[:4]))
        rv = ref["draws"][0]["vacancy_eV"]
        for f in models:
            m = load(f)
            md = [derived(x) for x in m["draws"]]
            dev = [pct(md[k][q], rd[k][q]) for k in range(3) for q in qs[:4]]
            mv = m["draws"][0]["vacancy_eV"]
            vmae = np.mean([abs(x - y) for el in rv for x, y in zip(mv[el], rv[el])])
            print(f"{f.stem:32s}" + "".join(f"{x:+10.1f}" for x in dev) + f"{vmae:9.3f}{np.mean(np.abs(dev)):10.2f}")
