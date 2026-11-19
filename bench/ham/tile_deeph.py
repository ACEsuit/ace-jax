"""Make a DeepH-layout supercell from a periodic structure's blocks (by translation symmetry),
keeping atom pairs closer than a cutoff: test input of realistic size for timing models that
read the DeepH layout (bench/ham/maceh_time.py).
    uv run --extra ham python bench/ham/tile_deeph.py <deeph dir> <out dir> <n | n1,n2,n3> <cutoff angstrom>"""
import itertools
import sys

import numpy as np

from ace_jax.ham import deeph


def tile(d, n, cutoff):
    N = len(d.numbers)
    n = np.array([n] * 3 if np.isscalar(n) else n, int)
    cells = list(itertools.product(*(range(v) for v in n)))
    index = {c: k for k, c in enumerate(cells)}
    pos = np.concatenate([d.positions + np.array(c) @ d.cell for c in cells])
    blocks = {}
    for name, bl in d.blocks.items():
        out = {}
        for key, B in bl.items():
            if np.linalg.norm(d.bond(key)) >= cutoff:
                continue
            R, i, j = np.array(key[:3]), key[3], key[4]
            for c in cells:
                t = np.array(c) + R
                cj, Rs = tuple(int(v) for v in t % n), tuple(int(v) for v in t // n)
                out[(*Rs, i + N * index[c], j + N * index[cj])] = B
        blocks[name] = out
    return deeph.OperatorData(n[:, None] * d.cell, pos, np.tile(d.numbers, len(cells)), d.basis, blocks)


if __name__ == "__main__":
    src, out, cutoff = sys.argv[1], sys.argv[2], float(sys.argv[4])
    t = tile(deeph.read(src), [int(v) for v in sys.argv[3].split(",")] * (1 if "," in sys.argv[3] else 3), cutoff)
    deeph.write(out, t)
    print(len(t.numbers), "atoms,", len(t.blocks["H"]) / len(t.numbers), "blocks per atom ->", out)
