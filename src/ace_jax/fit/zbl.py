"""Tabulated ZBL universal repulsive pair potential as the prior mean mu_0.

Guaranteed monotonic-repulsive, confined to the core by a smootherstep switch to
zero over [r_i, r_o], so it is inert at bulk distances (avoids the fitted-pair-
mean force blow-up) while giving correct short-range repulsion for close
approaches.  Emits the same npz schema as baseline.load_mean consumes.

    python -m ace_jax.fit.zbl zbl_mean.npz --elements 24,25,26,27,28
"""
import argparse
import numpy as np

KE = 14.399645          # e^2 / 4 pi eps0  [eV.Angstrom]


def zbl_pair(r, zi, zj):
    a = 0.46850 / (zi ** 0.23 + zj ** 0.23)
    x = r / a
    phi = (0.18175 * np.exp(-3.19980 * x) + 0.50986 * np.exp(-0.94229 * x)
           + 0.28022 * np.exp(-0.40290 * x) + 0.02817 * np.exp(-0.20162 * x))
    return KE * zi * zj / r * phi


def _switch(r, r_i, r_o):
    t = np.clip((r - r_i) / (r_o - r_i), 0.0, 1.0)
    return 1.0 - (6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3)


def tabulate(elements, r_i=1.0, r_o=2.0, rmin=0.5, rcut=6.25, nr=1200):
    """ZBL-with-switch tabulated for all unordered element pairs.  Returns the
    dict for np.savez (elements, r, V, pairs, E_atom, rcut)."""
    elements = [int(z) for z in elements]
    r = np.linspace(rmin, rcut, nr)
    pairs = [(a, b) for i, a in enumerate(elements) for b in elements[i:]]
    V = np.array([zbl_pair(r, a, b) * _switch(r, r_i, r_o) for a, b in pairs])
    return dict(elements=np.array(elements), r=r, V=V, pairs=np.array(pairs),
                E_atom=np.zeros(len(elements)), rcut=rcut)


def main(argv=None):
    p = argparse.ArgumentParser(description="write a ZBL mu_0 pair-potential npz")
    p.add_argument("out")
    p.add_argument("--elements", default="24,25,26,27,28", help="comma-separated Z")
    p.add_argument("--r-i", type=float, default=1.0); p.add_argument("--r-o", type=float, default=2.0)
    a = p.parse_args(argv)
    d = tabulate([int(z) for z in a.elements.split(",")], a.r_i, a.r_o)
    np.savez(a.out, **d)
    print(f"wrote {a.out}  ({len(d['pairs'])} pairs, switch [{a.r_i}, {a.r_o}] A)")


if __name__ == "__main__":
    main()
