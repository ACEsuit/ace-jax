"""End-to-end check of an ingested collinear-spin FHI-aims calculation (`ingest.jl fhi-aims-spin`,
`deeph.read_spin`): the eigenvalues of H_up and H_dn with the overlap against the ones aims.out
lists for its first k-point, and the electron count on the calculation's k-grid with its chemical
potential and Gaussian smearing.
    python bench/ham/spin_check.py <deeph spin dir> <aims.out> <nk> <gaussian width, eV>"""
import re
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.special import erfc

from ace_jax.ham import deeph

d = deeph.read_spin(sys.argv[1])
lines = open(sys.argv[2]).read().splitlines()
nk, width = int(sys.argv[3]), float(sys.argv[4])
last = lambda pat: float([m.group(1) for ln in lines if (m := re.search(pat, ln))][-1])
mu = last(r"Chemical potential \(Fermi level\):\s+(-?[\d.]+) eV")
moment = last(r"N = N_up - N_down \(sum over all k points\):\s+(-?[\d.]+)")
offsets = np.concatenate([[0], np.cumsum([d.basis.n_orb(int(z)) for z in d.numbers])])
n = offsets[-1]


def bloch(blocks, k):
    M = np.zeros((n, n), complex)
    for (Rx, Ry, Rz, i, j), b in blocks.items():
        M[offsets[i]:offsets[i + 1], offsets[j]:offsets[j + 1]] += np.exp(2j * np.pi * np.dot(k, (Rx, Ry, Rz))) * b
    return M


def listed(tag):
    """(k, eigenvalues in eV) of the last table in aims.out headed `tag`."""
    i = max(k for k, ln in enumerate(lines) if tag in ln)
    k = [float(x) for x in re.findall(r"-?\d+\.\d+", next(ln for ln in lines[i:i + 4] if "K-point" in ln))[:3]]
    out = []
    for ln in lines[i:]:
        p = ln.split()
        if len(p) == 4 and p[0].isdigit():
            out.append(float(p[3]))
        elif out:
            break
    return k, np.array(out)


print(f"{len(d.numbers)} atoms, {n} orbitals, {len(d.blocks['S'])} blocks")
worst = 0.0
for tag, ch in (("Spin-up eigenvalues", "H_up"), ("Spin-down eigenvalues", "H_dn")):
    k, ref = listed(tag)
    e = eigh(bloch(d.blocks[ch], k), bloch(d.blocks["S"], k), eigvals_only=True)[:len(ref)]
    worst = max(worst, np.abs(e - ref).max())
    print(f"  {ch} at k = {k}: {len(ref)} eigenvalues listed by FHI-aims, largest difference {np.abs(e - ref).max():.1e} eV")
N = {"H_up": 0.0, "H_dn": 0.0}
ks = np.array([[a, b, c] for a in range(nk) for b in range(nk) for c in range(nk)]) / nk
for k in ks:
    S = bloch(d.blocks["S"], k)
    for ch in N:
        N[ch] += 0.5 * erfc((eigh(bloch(d.blocks[ch], k), S, eigvals_only=True) - mu) / width).sum() / len(ks)
print(f"  electrons at FHI-aims' chemical potential: {N['H_up'] + N['H_dn']:.5f} (nuclear charge {int(d.numbers.sum())}); "
      f"moment {N['H_up'] - N['H_dn']:.4f} (FHI-aims prints {moment:.4f})")
assert worst < 2e-5 and abs(N["H_up"] + N["H_dn"] - d.numbers.sum()) < 1e-3
