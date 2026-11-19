"""Figures of the user page on Hamiltonian models, on the MACE-H paper's bulk gold data: the
bands and density of states of an ace-jax model over the DFT reference's (as `aj ham bands`
draws them), and the band errors of ace-jax, MACE-H and DeepH-E3 along the same path.
    python bench/ham/gold_bands.py <model.npz> <test structure (DeepH dir)> <out dir> [<code>=<its hamiltonians_pred.h5> ...]
Writes <out dir>/gold_bands.png and gold_band_errors.png.  Band errors: each spectrum from its own
Fermi level (of a 4x4x4 grid, 11 electrons per atom), reference states from -10 to 3 eV."""
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ace_jax.ham import cli, deeph
from ace_jax.ham.calculator import HamiltonianCalculator
from ace_jax.ham.operator import BlockSum, block_bands, smeared_occupations

model, src, out = sys.argv[1:4]
others = dict(a.split("=") for a in sys.argv[4:])
WIN, COL = (-10.0, 3.0), {"ace-jax": "#2a78d6", "MACE-H": "#eb6834", "DeepH-E3": "#1baf7a"}
name, at, d = cli.structures(src)[0]
calc = HamiltonianCalculator(model, kpts=(4, 4, 4), nelectrons={79: 11}, width=0.1)
calc.overlap = "S" if "S" in calc.predictor.channels else None
bs = BlockSum(sorted(d.blocks["H"]), d.numbers, d.basis)
mu = lambda e: float(smeared_occupations(jnp.asarray(e), jnp.asarray(calc.weights), calc.electrons(d.numbers), calc.width, calc.smearing, 2.0)[1])
bp = at.cell.bandpath(None, npoints=160, pbc=True)
x, ticks, labels = bp.get_linear_kpoint_axis()


def record(H):
    r = dict(x=x, ticks=ticks, labels=np.asarray(labels), grid_weights=calc.weights, degeneracy=2.0)
    r["bands"], r["grid_bands"] = (np.asarray(block_bands(bs, H, d.blocks["S"], ks)) for ks in (bp.kpts, calc.kpts))
    r["fermi_level"] = mu(r["grid_bands"])
    return r


ref = record(d.blocks["H"])
pred = calc._blocks(at)["H"]
recs = {"ace-jax": record({k: pred[k] for k in d.blocks["H"]})}
for code, path in others.items():
    recs[code] = record(deeph.read(src, operators=(("H", os.path.abspath(path)),)).blocks["H"])
for r in recs.values():
    r.update(bands_ref=ref["bands"], grid_bands_ref=ref["grid_bands"], fermi_level_ref=ref["fermi_level"])
os.makedirs(out, exist_ok=True)
rmse = {}
for code, r in recs.items():
    w = (ref["grid_bands"] > ref["fermi_level"] + WIN[0]) & (ref["grid_bands"] < ref["fermi_level"] + WIN[1])
    rmse[code] = 1e3 * np.sqrt(np.mean((((r["grid_bands"] - r["fermi_level"]) - (ref["grid_bands"] - ref["fermi_level"]))[w]) ** 2))
cli.plot_bands(recs["ace-jax"], os.path.join(out, "gold_bands.png"), (-10.0, 4.0), 0.1,
               f"Bulk gold, 64 atoms, a test structure: band RMSE {rmse['ace-jax']:.1f} meV from own Fermi levels", ("DFT (FHI-aims)", "ace-jax"))
f, axes = plt.subplots(1, len(recs), figsize=(3.4 * len(recs), 3.6), sharey=True, gridspec_kw=dict(wspace=0.06))
cut = np.r_[0, np.flatnonzero(np.diff(x) == 0) + 1, len(x)]
for ax, (code, r) in zip(np.atleast_1d(axes), recs.items()):
    err = 1e3 * ((r["bands"] - r["fermi_level"]) - (ref["bands"] - ref["fermi_level"]))
    keep = ((ref["bands"] > ref["fermi_level"] + WIN[0]) & (ref["bands"] < ref["fermi_level"] + WIN[1])).all(0)       # whole bands inside the window
    for lo, hi in zip(cut[:-1], cut[1:]):
        ax.plot(x[lo:hi], err[lo:hi][:, keep], color=COL.get(code, "#52514e"), lw=0.4, alpha=0.35)
    for t in ticks:
        ax.axvline(t, color="#8a8f98", lw=0.5)
    ax.axhline(0.0, color="#0b0b0b", lw=0.6)
    names = {}
    for t, s in zip(ticks, labels):
        names.setdefault(float(t), []).append("Γ" if s == "G" else s)
    ax.set_xticks(list(names)); ax.set_xticklabels(["|".join(v) for v in names.values()]); ax.set_xlim(x[0], x[-1])
    ax.set_title(f"{code}: RMSE {rmse[code]:.1f} meV", fontsize=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
lim = max(abs(v) for ax in np.atleast_1d(axes) for v in ax.get_ylim())
np.atleast_1d(axes)[0].set_ylim(-min(lim, 40), min(lim, 40)); np.atleast_1d(axes)[0].set_ylabel("Band error against DFT / meV")
f.savefig(os.path.join(out, "gold_band_errors.png"), dpi=200, bbox_inches="tight")
print(name, {k: round(float(v), 2) for k, v in rmse.items()})
