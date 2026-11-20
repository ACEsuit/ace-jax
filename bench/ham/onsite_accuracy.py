"""Accuracy against cost of the on-site Hamiltonian model on the aluminium data of the
ACEhamiltonians paper, for comparison with ACEhamiltonians.jl (bench/ham/aceh_onsite_accuracy.jl).

Protocol (the paper's on-site fits): train on the odd-numbered atoms (1-based) of BCC
SK-supercell-001 and FCC SK-supercell-002 (365 atoms each, 500 K MD), correlation order 2,
environment cutoff 9 angstrom; test on every third atom of FCC SK-supercell-001 and BCC
SK-supercell-002 (243 atoms each).  Blocks are divided by 27 (the files hold the sum over a
3 x 3 x 3 k-grid: the on-site overlap is 27).
    uv run python bench/ham/onsite_accuracy.py export <out.h5>          # environments and blocks for Julia
    uv run python bench/ham/onsite_accuracy.py fit <totaldegree> [wL]   # one ace-jax model
"""
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aitb

from ace_jax.eval.nlist import sparse_graph
from ace_jax.operators.data import OperatorData
from ace_jax.operators.features import build_site_basis
from ace_jax.operators.fit import fit_onsite
from ace_jax.operators.onsite import OnsitePredictor, _features, _readout, shell_pairs, site_graph

DATA = os.environ.get("ACEH_DATA", "/storage/eng/essswb/ACEhamiltonians-data/training_data")
TRAIN = [("BCC/SK-supercell-001.h5", np.arange(0, 729, 2)), ("FCC/SK-supercell-002.h5", np.arange(0, 729, 2))]
TEST = [("FCC/SK-supercell-001.h5", np.arange(1, 729, 3)), ("BCC/SK-supercell-002.h5", np.arange(1, 729, 3))]
RCUT, KGRID = 9.0, 27.0


def dataset(name, raw=False):
    d = aitb.load(os.path.join(DATA, name), operators=("H",), raw=raw)
    blocks = {"H": {(0, 0, 0, i, i): d["H"][i] / KGRID for i in range(len(d["positions"]))}}
    return OperatorData(d["cell"], d["positions"], d["numbers"], aitb.AL, blocks)


def export(out):
    """Neighbour vectors and on-site blocks in FHI-aims' own phases, for ACEhamiltonians.jl."""
    import h5py
    with h5py.File(out, "w") as f:
        for tag, sets in (("train", TRAIN), ("test", TEST)):
            vecs, off, blocks = [], [0], []
            for name, atoms in sets:
                d = dataset(name, raw=True)
                g = sparse_graph(d.positions, d.cell, True, RCUT)
                order = np.argsort(g.senders, kind="stable")
                start = np.searchsorted(g.senders[order], np.arange(len(d.positions) + 1))
                for i in atoms:
                    vecs.append(g.rij[order[start[i]:start[i + 1]]]); off.append(off[-1] + len(vecs[-1]))
                    blocks.append(d.blocks["H"][(0, 0, 0, int(i), int(i))])
            f[f"{tag}/vectors"], f[f"{tag}/offsets"] = np.concatenate(vecs), np.asarray(off)
            f[f"{tag}/blocks"] = np.asarray(blocks)                       # (n, 14, 14), row-major
            print(tag, len(blocks), "atoms,", off[-1] / len(blocks), "neighbours")
        f["shells"] = np.asarray(aitb.AL.ls(13))


def best(f, reps=3):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)


def fit(totaldegree, wL=1.0, order=None):
    order = int(os.environ.get("HAM_ORDER", 2)) if order is None else order
    extra = {k: float(os.environ[v]) for k, v in (("r0", "HAM_R0"), ("n_q_factor", "HAM_NQ")) if v in os.environ}
    if "HAM_RADIAL" in os.environ:
        extra["rnl_basis"] = os.environ["HAM_RADIAL"]
    for k, v in (("nmax_by_order", "HAM_NMAX"), ("lmax_by_order", "HAM_LMAX")):
        if v in os.environ:
            extra[k] = tuple(int(x) for x in os.environ[v].split(","))
    rcut = float(os.environ.get("HAM_RCUT", RCUT))
    basis = build_site_basis([13], order=order, totaldegree=totaldegree, lmax_feat=4, rcut=rcut, wL=wL, **extra)
    print("  settings:", {"rcut": rcut, **extra})
    train = [dataset(n) for n, _ in TRAIN]
    t = time.perf_counter()
    model, report = fit_onsite(basis, train, channels=("H",), atoms=[a for _, a in TRAIN], ridge=1e-10)
    t_fit = time.perf_counter() - t
    fast = OnsitePredictor(model)
    ncoef = sum(np.asarray(c).shape[1] for c in model.coef.values())
    err = {}
    for tag, sets in (("train", TRAIN), ("test", TEST)):
        sq, n = 0.0, 0
        per = {}
        for name, atoms in sets:
            d = dataset(name)
            _, (blocks,) = fast.dense(d.positions, d.cell, d.numbers)
            diff = np.asarray(blocks)[atoms, 0] - np.stack([d.blocks["H"][(0, 0, 0, int(i), int(i))] for i in atoms])
            sq += (diff ** 2).sum(); n += diff.size
            for s1, s2, l1, l2 in shell_pairs(aitb.AL, 13):
                e = diff[:, aitb.AL.shell_slice(13, s1), aitb.AL.shell_slice(13, s2)]
                a = per.setdefault("spd"[l1] + "spd"[l2], [0.0, 0])
                a[0] += (e ** 2).sum(); a[1] += e.size
        upper = np.sqrt(sum(v[0] for v in per.values()) / sum(v[1] for v in per.values()))   # shell pairs s1 <= s2 once
        err[tag] = (upper, {k: np.sqrt(v[0] / v[1]) for k, v in per.items()})
    d = dataset(TEST[0][0])
    N = len(d.positions)
    rij, zi, zj, seg, n = site_graph(basis, d.positions, d.cell, d.numbers)
    at = (np.arange(N, dtype=np.int32),)
    F = _features(basis, rij, zi, zj, seg, n)
    t_nl = best(lambda: site_graph(basis, d.positions, d.cell, d.numbers))
    t_f = best(lambda: jax.block_until_ready(_features(basis, rij, zi, zj, seg, n)))
    t_r = best(lambda: jax.block_until_ready(_readout(fast.readout, F, at)))
    us = lambda x: 1e6 * x / N
    n_aa = sum(int(g.shape[0]) for g in basis.aa_specs)
    print(f"ace-jax order {order} totaldegree {totaldegree} wL {wL}: n_AA {n_aa}, features {sum(basis.n_B)}, "
          f"coefficients {ncoef}; fit {t_fit:.1f} s; RMSE over shell pairs s1 <= s2: train {1e3 * err['train'][0]:.3f} test {1e3 * err['test'][0]:.3f} meV; "
          f"per atom ({len(rij) / N:.0f} neighbours, {len(os.sched_getaffinity(0))} thread): neighbour list {us(t_nl):.1f}, "
          f"features {us(t_f):.1f}, readout {us(t_r):.1f}, total {us(t_nl + t_f + t_r):.1f} us")
    print("   test RMSE by block type (meV):", {k: round(1e3 * v, 3) for k, v in err["test"][1].items()})


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(sys.argv[2])
    else:
        fit(int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 1.0)
