"""H and S of a folder of npz caches in a smaller basis, without rerunning the DFT: the functions of
the smaller basis are on-site linear combinations of those of the larger one (FHI-aims orthonormalises
the functions of each l on an atom, so removing a function is a combination, not a deletion), and
the blocks transform as T_i^T M T_j.

    JAX_PLATFORMS=cpu uv run python bench/ham/reduce_basis.py <npz dir> <out dir> <large.npz> <small.npz>

<large.npz>, <small.npz>: one structure computed in both bases.  For every element and every l whose
number of shells differs, the map T (large shells x small shells) is fitted to the overlaps with
the unchanged shells of the neighbours; the fit residual and |T^T T - 1| are printed (both vanish when
the small basis is a subset of the span of the large one).  The mapped H is the large basis'
self-consistent Kohn-Sham operator in the small basis: not what a rerun converges to
(docs/dev/ham-results.md)."""
import glob
import os
import sys

import numpy as np

from ace_jax.ham import cache


def shells(orb, z, l):
    return [s for s, ll in enumerate(orb.ls(z)) if ll == l]


def fit_maps(a, b):
    """{Z: (n_orb large, n_orb small)} from the overlaps of one structure in both bases."""
    oa, ob, maps = a.basis, b.basis, {}
    for z in sorted({int(v) for v in a.numbers}):
        changed = [l for l in sorted(set(oa.ls(z))) if len(shells(oa, z, l)) != len(shells(ob, z, l))]
        same = lambda orb, zj: np.concatenate([np.arange(orb.n_orb(zj))[orb.shell_slice(zj, s)] for s, ll in enumerate(orb.ls(zj))
                                               if len(shells(oa, zj, ll)) == len(shells(ob, zj, ll))])
        M = np.zeros((oa.n_orb(z), ob.n_orb(z)))
        for l in sorted(set(oa.ls(z))):
            sa, sb = shells(oa, z, l), shells(ob, z, l)
            t = np.eye(len(sa))
            if l in changed:
                X, Y = [], []
                for k, blk in a.blocks["S"].items():
                    i, j = k[3], k[4]
                    if int(a.numbers[i]) != z or (i == j and k[:3] == (0, 0, 0)) or k not in b.blocks["S"]:
                        continue
                    zj = int(a.numbers[j])
                    X.append(np.stack([blk[oa.shell_slice(z, s)][:, same(oa, zj)] for s in sa], -1).reshape(-1, len(sa)))
                    Y.append(np.stack([b.blocks["S"][k][ob.shell_slice(z, s)][:, same(ob, zj)] for s in sb], -1).reshape(-1, len(sb)))
                X, Y = np.concatenate(X), np.concatenate(Y)
                t = np.linalg.lstsq(X, Y, rcond=None)[0]
                print(f"Z = {z}, l = {l}: {len(sa)} -> {len(sb)} shells; fit residual {np.abs(X @ t - Y).max():.1e}, |T^T T - 1| {np.abs(t.T @ t - np.eye(len(sb))).max():.1e}", flush=True)
            for ia, s in enumerate(sa):
                for ib, q in enumerate(sb):
                    M[oa.shell_slice(z, s), ob.shell_slice(z, q)] = t[ia, ib] * np.eye(2 * l + 1)
        maps[z] = M
    return maps


def reduce(d, maps, basis):
    z = [int(v) for v in d.numbers]
    return d._replace(basis=basis, blocks={c: {k: maps[z[k[3]]].T @ v @ maps[z[k[4]]] for k, v in blk.items()} for c, blk in d.blocks.items()})


def main(src, dst, large, small):
    a, b = cache.load(large)[0], cache.load(small)[0]
    maps = fit_maps(a, b)
    os.makedirs(dst, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(src, "*.npz"))):
        d, meta = cache.load(f)
        cache.save(os.path.join(dst, os.path.basename(f)), reduce(d, maps, b.basis),
                   meta={**(meta.get("source") or {}), "reduced_basis": {"from": os.path.abspath(large), "to": os.path.abspath(small)}})
    print(f"{len(glob.glob(os.path.join(dst, '*.npz')))} structures written to {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
