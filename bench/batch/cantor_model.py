"""A 5-species (Cr Mn Fe Co Ni) linear ACE with the analytic radial and random coefficients: a timing
model for the multi-species radial forms (docs/dev/batched-evaluation-results.md).  The coefficients are
not fitted; only the shapes matter.

    python bench/batch/cantor_model.py --out cantor.npz [--order 3 --max-degree 10]"""
import argparse

import jax

jax.config.update("jax_enable_x64", True)
import equinox as eqx
import numpy as np

from ace_jax.basis.export import save_npz
from ace_jax.basis.model import BasisSpec, build_basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--max-degree", type=int, default=10)
    ap.add_argument("--rcut", type=float, default=5.0)
    a = ap.parse_args()
    b = build_basis(BasisSpec(elements=("Cr", "Mn", "Fe", "Co", "Ni"), order=a.order, max_degree=a.max_degree, rcut=a.rcut))
    rng = np.random.default_rng(0)
    m = b.model
    m = eqx.tree_at(lambda m: (m.WB, m.Wpair), m, (1e-3 * rng.standard_normal(m.WB.shape),
                                                    1e-3 * rng.standard_normal(m.Wpair.shape)))
    save_npz(a.out, b._replace(model=m))
    print(a.out, m.radial_kind, "Wnlq", m.rnl_Wnlq.shape, "WB", m.WB.shape)


if __name__ == "__main__":
    main()
