"""ML-PACE (with 0001-tabulated-radials.patch) against ace-jax on a linear ACE model exported by write_yace.

    python compare.py <tab_eval> <model.npz> [--delta 0.001] [--configs 6] [--out m.yace]

Builds rattled, mixed-species diamond cells over the model's elements, writes the .yace, runs tab_eval
(ML-PACE's C-tilde evaluator) on them and reports the E/atom and force differences against ACECalculator
on the source .npz (the whole export: tables, products, ML-PACE's look-up) and on the .yace (the
evaluators alone, the same tables)."""
import argparse
import subprocess
import tempfile

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from ase.build import bulk  # noqa: E402
from ase.data import atomic_numbers  # noqa: E402

import ace_jax as aj  # noqa: E402
from ace_jax.calc.point import ACECalculator  # noqa: E402
from ace_jax.eval import write_yace  # noqa: E402
from ace_jax.eval.pace_io import load_tree  # noqa: E402


def cells(z, n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for c in range(n):
        at = bulk("Si", "diamond", 5.43 * (0.97 + 0.06 * rng.random()), cubic=True).repeat(2)
        at.numbers[:] = rng.choice(z, len(at))                    # every element present in the set
        at.numbers[:len(z)] = z
        at.rattle(0.15, seed=int(rng.integers(1 << 30)))
        if c % 2:
            at.set_cell(at.cell @ (np.eye(3) + 0.03 * rng.standard_normal((3, 3))), scale_atoms=True)
        out.append(at)
    return out


def tab_eval(exe, yace, ats):
    idx = {e: i for i, e in enumerate(load_tree(yace)["elements"])}
    inp = [str(len(ats))]
    for a in ats:
        inp += [str(len(a)), " ".join(map("{:.17g}".format, np.asarray(a.cell).ravel())),
                " ".join(str(int(p)) for p in a.pbc)]
        inp += [f"{idx[s]} " + " ".join(map("{:.17g}".format, p)) for s, p in zip(a.get_chemical_symbols(), a.positions)]
    r = subprocess.run([exe, str(yace)], input="\n".join(inp), capture_output=True, text=True, check=True)
    lines, k, res = r.stdout.split("\n"), 0, []
    for a in ats:
        res.append((float(lines[k]), np.array([list(map(float, lines[k + 1 + i].split())) for i in range(len(a))])))
        k += 1 + len(a)
    return res


def diff(res, ats, ref):
    calc = ACECalculator(ref, skin=0)
    dE, dF, F = [], [], []
    for (E1, F1), a in zip(res, ats):
        b = a.copy()
        b.calc = calc
        dE.append((E1 - b.get_potential_energy()) / len(a))
        F0 = b.get_forces()
        dF.append(F1 - F0)
        F.append(F0)
    dE, dF, F = np.abs(dE), np.concatenate(dF).ravel(), np.concatenate(F).ravel()
    return dE.max(), np.abs(dF).max(), np.sqrt(np.mean(dF ** 2)), np.abs(F).max()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tab_eval")
    p.add_argument("model")
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--configs", type=int, default=6)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    model, _, _ = aj.load(a.model)
    yace = a.out or tempfile.mkstemp(suffix=".yace")[1]
    info = write_yace(model, yace, delta=a.delta)
    ats = cells([atomic_numbers[e] for e in load_tree(yace)["elements"]], a.configs)
    res = tab_eval(a.tab_eval, yace, ats)
    print(f"{a.model}: elements {load_tree(yace)['elements']}, {info['functions']} functions, "
          f"{info['terms']} terms, delta {a.delta}, {len(ats)} configurations of {len(ats[0])} atoms")
    for name, ref in (("vs .npz ", a.model), ("vs .yace", yace)):
        e, fm, fr, F = diff(res, ats, ref)
        print(f"  ML-PACE {name}: max dE/atom {e:.1e} eV, max dF {fm:.1e} eV/A, rms dF {fr:.1e} (max |F| {F:.1f})")


if __name__ == "__main__":
    main()
