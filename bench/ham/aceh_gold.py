"""ACEhamiltonians.jl on the MACE-H paper's bulk gold data, for the cross-code comparison
(`gold_pareto.py`): the data in the form `aceh_gold.jl` reads, and its predictions back into the
DeepH layout that `deeph_fit.py pred=` scores.
    python bench/ham/aceh_gold.py export <gold dir> <out.h5> [n_train=32] [bonds=600] [rcut=7.0] [flip=1]
    python bench/ham/aceh_gold.py collect <predictions.h5> <gold dir> <pred dir> [flip=1]
export: the first n_train training structures of the seeded order every code uses (`deeph_fit.train_names`),
all their on-site blocks and a random `bonds` of each one's off-site blocks; every block of the
test structures.  Blocks in FHI-aims' own phases (flip=1; what ACEhamiltonians.jl is written
for), neighbour vectors within rcut for its on-site states."""
import os
import sys

import h5py
import numpy as np

import aitb
from deeph_fit import onsite, train_names

from ace_jax.eval.nlist import sparse_graph
from ace_jax.ham import deeph
from ace_jax.operators.data import OperatorData


def signs(d, flip):
    s = aitb.atom_signs(d.basis.ls(int(d.numbers[0]))) if flip else np.ones(d.blocks["H"][(0, 0, 0, 0, 0)].shape[0])
    return s[:, None] * s[None, :]                               # +-1: its own inverse


def export(gold, out, n_train=32, bonds=600, rcut=7.0, flip=1):
    with h5py.File(out, "w") as f:
        for tag, names in (("train", train_names(gold)[:n_train]), ("test", sorted(os.listdir(os.path.join(gold, "test"))))):
            for k, name in enumerate(names):
                d = deeph.read(os.path.join(gold, tag, name), operators=("H",))
                ss = signs(d, flip)
                keys = sorted(key for key in d.blocks["H"] if not onsite(key))
                if tag == "train":
                    keys = [keys[i] for i in np.sort(np.random.default_rng(k).choice(len(keys), size=min(bonds, len(keys)), replace=False))]
                g = f.create_group(f"{tag}/{k:03d}")
                g.attrs["name"] = name
                g["positions"], g["cell"], g["numbers"] = d.positions, d.cell, d.numbers
                g["bonds"] = np.asarray(keys, np.int64)             # (n, 5): Rx, Ry, Rz, i, j (0-based)
                g["H"] = np.stack([d.blocks["H"][key] * ss for key in keys])
                g["onsite"] = np.stack([d.blocks["H"][(0, 0, 0, i, i)] * ss for i in range(len(d.numbers))])
                gr = sparse_graph(d.positions, d.cell, True, rcut)
                order = np.argsort(gr.senders, kind="stable")
                g["vectors"] = np.asarray(gr.rij)[order]
                g["offsets"] = np.searchsorted(np.asarray(gr.senders)[order], np.arange(len(d.numbers) + 1))
        f["shells"] = np.asarray(d.basis.ls(int(d.numbers[0])))
        f.attrs["rcut"] = rcut
    print(f"{out}: {n_train} training structures ({bonds} bonds each), test structures with every block")


def collect(pred, gold, out, flip=1):
    """Predictions of `aceh_gold.jl` (shell pairs s1 <= s2 of each block) as full blocks in
    ace-jax's phases: the other shell pairs of a bond are the transpose of the reversed bond's."""
    with h5py.File(pred, "r") as f:
        for k in sorted(f["test"]):
            g = f[f"test/{k}"]
            name = g.attrs["name"]
            d = deeph.read(os.path.join(gold, "test", name), operators=("H",))
            ss = signs(d, flip)
            up = np.triu(np.ones_like(ss))                         # the s1 <= s2 part, by orbitals
            o = np.cumsum([0] + [2 * l + 1 for l in d.basis.ls(int(d.numbers[0]))])
            for a in range(len(o) - 1):
                up[o[a]:o[a + 1], o[a]:o[a + 1]] = 1.0
            keys = [tuple(int(v) for v in key) for key in g["bonds"][()]]
            P = dict(zip(keys, g["pred_off"][()]))
            H = {key: v * up + (P[(-key[0], -key[1], -key[2], key[4], key[3])] * up).T * (1 - up) for key, v in P.items()}
            for i, v in enumerate(g["pred_on"][()]):
                H[(0, 0, 0, i, i)] = v * up + (v * up).T * (1 - up)
            H = {key: v * ss for key, v in H.items()}
            deeph.write(os.path.join(out, name), OperatorData(d.cell, d.positions, d.numbers, d.basis, {"H": H}), files={"H": "hamiltonians_pred.h5"})
            err = np.concatenate([(H[key] - v).ravel() for key, v in d.blocks["H"].items()])
            print(f"{name}: H MAE {1e3 * np.abs(err).mean():.3f} meV, largest {1e3 * np.abs(err).max():.1f}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    pos = [x for x in a[1:] if "=" not in x]
    kw = {k: (float(v) if k == "rcut" else int(v)) for k, v in (x.split("=") for x in a[1:] if "=" in x)}
    {"export": export, "collect": collect}[a[0]](*pos, **kw)
