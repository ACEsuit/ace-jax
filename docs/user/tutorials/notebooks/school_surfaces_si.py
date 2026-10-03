# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ace-jax>=0.1.1",
#     "marimo>=0.25",
#     "matplotlib>=3.8",
# ]
# ///
"""Tutorial 6: surfaces -- coverage, repair, and letting the atoms move.

Adapted from the MLIP-school-2026 notebook E2 (ACEsuit/MLIP-school-2026). Run it with
`uvx marimo edit --sandbox school_surfaces_si.py`, or as a plain script
(`python school_surfaces_si.py`), which is how CI smoke-tests it.
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 6: surfaces

    [Tutorial 4](https://acesuit.github.io/ace-jax/tutorials/dataset-and-properties/)
    ended with a question: would a bulk-only silicon model get a surface
    energy right? This tutorial answers it, measures why not in descriptor
    space, repairs the data set, and then lets the surface atoms move, which
    exposes the next gap.

    **Goals**

    1. Build silicon slabs for three faces and compute their surface energies
       with the labeller and with the bulk-only model.
    2. See in descriptor space how far the surface atoms sit from the data.
    3. Repair the data set with a few slabs, and check the surface energies.
    4. Relax the slabs with the repaired model, find what still breaks, and
       fix that too.

    It is adapted from notebook E2 of the
    [MLIP school 2026](https://github.com/ACEsuit/MLIP-school-2026).

    **Run this notebook**: with [uv](https://docs.astral.sh/uv/) installed, one command
    opens it in your browser (no account needed):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/notebooks/school_surfaces_si.py
    ```

    or open it in [molab](https://molab.marimo.io/github/ACEsuit/ace-jax/blob/main/docs/user/tutorials/notebooks/school_surfaces_si.py), marimo's hosted service (free to
    preview; sign in to run). The documentation website shows a static copy,
    run on a CPU when the site was built. The whole notebook runs in about 2 minutes.
    """)
    return


@app.cell
def _():
    import pathlib
    import time

    import jax

    jax.config.update("jax_enable_x64", True)  # fitting needs float64

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from ace_jax import ACECalculator
    from ace_jax.tutorials import campaign as C
    from ace_jax.tutorials import labels as L
    from ace_jax.tutorials import structures as T

    return ACECalculator, C, L, T, mo, np, pathlib, plt, time


@app.cell
def _(L, mo, pathlib):
    BASE = "https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/school/"


    def shipped(rel):
        """A shipped data file: the checkout's copy when run from one, else the GitHub one."""
        _p = (mo.notebook_dir() / "../data/school" / rel) if mo.notebook_dir() else None
        return _p if _p is not None and _p.exists() else BASE + rel


    caches = [L.LabelCache.from_file(shipped(f)) for f in ("e1/labels-mpa-0.xyz", "e2/labels-mpa-0.xyz",
                                                             "c/labels-mpa-0.xyz")]


    class _Caches:          # one lookup over the three shipped caches
        def get(self, atoms, model):
            return next((h for c in caches if (h := c.get(atoms, model)) is not None), None)


    cache = _Caches()


    def label(xs):
        return L.label(xs, model="mpa-0", cache=cache)


    work = pathlib.Path("ace_jax_tutorial_6")
    work.mkdir(exist_ok=True)
    return label, shipped, work


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: the bulk model

    The starting point is Tutorial 4's data set at its defaults: ten strained
    and rattled 8-atom diamond cells (strain range 0.08, rattle 0.02 Å),
    labelled by MACE-MPA-0. Fit it with the evidence fit at the size used
    throughout this tutorial: correlation order 3, total degree 10, cutoff 5.5 Å.
    """)
    return


@app.cell
def _(T, label, mo, time, work):
    from ace_jax.basis.model import BasisSpec
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data, save_model


    def fit_model(train, out):
        """The evidence fit of every step here: order 3, degree 10, rcut 5.5, E0 fitted."""
        _cfg = FitConfig(model=BasisSpec(order=3, max_degree=10, rcut=5.5, elements=("Si",)), arm="linear",
                         m_per_species=0, e0="lsq", opt="lbfgs", r0=None, rungs=("map",),
                         predict_stats="recompute", predict_train=False,
                         energy_key="energy", force_key="forces", virial_key="virial").validate()
        _res = fit(_cfg, load_fit_data(_cfg, train=list(train), log=lambda *a: None), log=lambda *a: None)
        return str(save_model(_res, work / out, log=lambda *a: None))


    bulk_training = label(T.e1_cells(0.08, 0.02))
    _t = time.time()
    bulk_model = fit_model(bulk_training, "fit_bulk")
    mo.md(f"Fitted the bulk model on {len(bulk_training)} cells in {time.time() - _t:.0f} s.")
    return bulk_model, bulk_training, fit_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: build the slabs

    A slab is a crystal cut on one plane, a few layers thick, with vacuum
    above and below so that it does not see its own periodic image. Build the
    (100), (110) and (111) faces with `ase.build.surface`. Two details matter:

    - **Vacuum.** The labeller sees about 6 Å, and its message passing twice
      that; the gap between a slab and its image must be at least 12 Å. The
      vacuum slider sets the vacuum on each side.
    - **Which (111).** Diamond has two (111) cleavage planes: the wide
      "shuffle" plane, which breaks one bond per surface atom, and the
      narrow "glide" pair, which breaks three. `surface` takes the glide pair
      by default; `T.slab` shifts the crystal by a quarter of the cubic
      diagonal first, so the cut is on the shuffle plane. A surface atom with
      a single neighbour is the sign of a glide cut.

    The sliders snap to the settings whose labels ship.
    """)
    return


@app.cell
def _(T, mo):
    layers = mo.ui.slider(steps=list(T.E2_LAYERS), value=6, label="layers", show_value=True)
    vacuum = mo.ui.slider(steps=list(T.E2_VACUA), value=8.0, label="vacuum per side (Å)", show_value=True)
    mo.hstack([layers, vacuum], justify="start")
    return layers, vacuum


@app.cell
def _(T, layers, mo, vacuum):
    slabs = T.e2_slabs(layers.value, vacuum.value)
    gap, coord = T.min_vacuum_gap(slabs), T.min_coordination(slabs)
    mo.md("| face | atoms |\n|---|---|\n" + "\n".join(f"| ({s.info['miller']}) | {len(s)} |" for s in slabs)
          + f"\n\nSmallest gap between a slab and its image: **{gap:.1f} Å**; fewest neighbours of any "
          f"atom: **{coord}**.")
    return coord, gap, slabs


@app.cell(hide_code=True)
def _(coord, gap, mo):
    _ok = gap >= 12.0 - 1e-6 and coord >= 2
    mo.callout(
        mo.md("**Checkpoint 1 passed:** a gap of at least 12 Å, and no atom with a single neighbour.")
        if _ok else mo.md(f"**Checkpoint 1:** the gap is {gap:.1f} Å (at least 12 needed: more vacuum) "
                          f"or an atom has {coord} neighbour (a glide cut). The cells below wait for a "
                          "valid slab."),
        kind="success" if _ok else "danger",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: surface energies, true and predicted

    $$\gamma = \frac{E_\text{slab} - N_\text{slab}\,E_\text{bulk}/N_\text{bulk}}{2A}$$

    with $A$ the slab's in-plane area. The bulk reference is the 8-atom cubic
    cell at 5.43 Å.
    """)
    return


@app.cell
def _(ACECalculator, T, coord, gap, label, mo, slabs):
    mo.stop(gap < 12.0 - 1e-6 or coord < 2, mo.md("Build a valid slab first (checkpoint 1)."))
    ref_bulk, *labelled_slabs = label([T.c_structures()["bulk"], *slabs])


    def gammas(e_bulk, e_slabs, structures=slabs):
        return [T.surface_energy(e_bulk, len(ref_bulk), e, s) for s, e in zip(structures, e_slabs)]


    def model_energies(model_file, xs):
        _calc = ACECalculator(model_file, skin=0)
        _out = []
        for _a in xs:
            _b = _a.copy(); _b.calc = _calc
            _out.append(_b.get_potential_energy())
        return _out


    def model_gammas(model_file, structures=slabs):
        _e = model_energies(model_file, [ref_bulk, *structures])
        return gammas(_e[0], _e[1:], structures)


    truth = gammas(ref_bulk.info["energy"], [a.info["energy"] for a in labelled_slabs])
    return gammas, model_energies, model_gammas, ref_bulk, truth


@app.cell
def _(bulk_model, model_gammas, np, plt, slabs, truth):
    gamma_bulk = model_gammas(bulk_model)
    _x = np.arange(len(slabs))
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _ax.bar(_x - 0.2, truth, 0.4, label="MACE-MPA-0")
    _ax.bar(_x + 0.2, gamma_bulk, 0.4, label="bulk-only ACE")
    _ax.set_xticks(_x, [f"({s.info['miller']})" for s in slabs]); _ax.set_ylabel("γ (eV/Å²)")
    _ax.legend(frameon=False); _ax.set_title("surface energies")
    _fig.tight_layout()
    _fig
    return (gamma_bulk,)


@app.cell
def _(gamma_bulk, mo, slabs, truth):
    mo.md("| face | MACE-MPA-0 | bulk-only ACE | error |\n|---|---|---|---|\n"
          + "\n".join(f"| ({s.info['miller']}) | {t:.4f} | {g:.4f} | {g - t:+.4f} |"
                      for s, t, g in zip(slabs, truth, gamma_bulk))
          + "\n\nγ in eV/Å². The bulk-only model is wrong by several times the surface energy itself.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: how far outside the data?

    Describe every atom by its ACE descriptor in a fixed reference basis
    (order 3, degree 10: 120 numbers per atom), and compare the slab atoms
    with the bulk training atoms. A principal-component plot shows the gap;
    one number measures it: the median distance from a slab atom to its
    nearest training atom, divided by the training atoms' own typical
    spacing (the median distance from each to its nearest neighbour).
    """)
    return


@app.cell
def _(C, bulk_training, np, plt, slabs):
    _B = C.reference_basis()
    Xb = np.concatenate(C.atom_descriptors(bulk_training, _B))
    Xs = np.concatenate(C.atom_descriptors(slabs, _B))
    novelty = C.nn_ratio(Xb, Xs)
    _mu = Xb.mean(0)
    _, _, _Vt = np.linalg.svd(Xb - _mu, full_matrices=False)
    _pb, _ps = (Xb - _mu) @ _Vt[:2].T, (Xs - _mu) @ _Vt[:2].T
    _fig, _ax = plt.subplots(figsize=(5, 4))
    _ax.plot(_pb[:, 0], _pb[:, 1], ".", ms=4, alpha=0.6, label="bulk training atoms")
    _ax.plot(_ps[:, 0], _ps[:, 1], "x", ms=5, label="slab atoms")
    _ax.set_xlabel("PC 1"); _ax.set_ylabel("PC 2"); _ax.legend(frameon=False)
    _ax.set_title(f"slab atoms sit {novelty:.1f}× out")
    _fig.tight_layout()
    _fig
    return (novelty,)


@app.cell(hide_code=True)
def _(mo, novelty):
    _ok = novelty > 5
    mo.callout(
        mo.md(f"**Checkpoint 2 passed:** the median slab atom is {novelty:.1f} training spacings from the "
              "nearest training atom: the model is extrapolating, which is why its surface energies "
              "are wrong.")
        if _ok else mo.md(f"**Checkpoint 2:** the slab atoms sit only {novelty:.1f}× out."),
        kind="success" if _ok else "warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5: repair the data set

    Add slabs of the same three faces to the training set, at a thickness the
    test slabs do not have (4 layers, or 6 when the test slabs are 4): the
    test slabs stay held out. Three structures, three labels.
    """)
    return


@app.cell
def _(T, bulk_training, fit_model, label, layers, model_gammas, slabs, truth, vacuum):
    repair = label(T.e2_slabs(T.e2_repair_layers(layers.value), vacuum.value))
    from ace_jax.tutorials.labels import structure_key
    assert not ({structure_key(a) for a in repair} & {structure_key(s) for s in slabs}), "a test slab is in training"
    repaired_model = fit_model([*bulk_training, *repair], "fit_repaired")
    gamma_repaired = model_gammas(repaired_model)
    repaired_errors = [abs(g - t) for g, t in zip(gamma_repaired, truth)]
    return repair, repaired_errors, repaired_model


@app.cell
def _(gamma_bulk, mo, repaired_errors, slabs, truth):
    mo.md("| face | bulk-only error | repaired error |\n|---|---|---|\n"
          + "\n".join(f"| ({s.info['miller']}) | {abs(g - t):.4f} | {e:.5f} |"
                      for s, g, t, e in zip(slabs, gamma_bulk, truth, repaired_errors))
          + "\n\n|γ error| in eV/Å².")
    return


@app.cell(hide_code=True)
def _(gamma_bulk, mo, repaired_errors, truth):
    _bulk_err = max(abs(g - t) for g, t in zip(gamma_bulk, truth))
    _ok = max(repaired_errors) < 0.01 and max(repaired_errors) < 0.1 * _bulk_err
    mo.callout(
        mo.md(f"**Checkpoint 3 passed:** three slabs bring every surface energy to within "
              f"{max(repaired_errors):.5f} eV/Å² of the labeller, from {_bulk_err:.2f}.")
        if _ok else mo.md(f"**Checkpoint 3:** the repaired errors are up to {max(repaired_errors):.4f} eV/Å²."),
        kind="success" if _ok else "warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6: let the atoms move

    The surface energies above are for ideal cleaves, frozen in place. A
    real surface relaxes. Relax each test slab with the repaired model
    (BFGS, until every force is below 0.03 eV/Å, at most 200 steps) and look
    at what happens.
    """)
    return


@app.cell
def _(ACECalculator, np, slabs):
    from ase.optimize import BFGS


    def relax(model_file, structures):
        """BFGS to fmax 0.03 eV/A (at most 200 steps); per slab: converged, energy change, final
        largest force, and the relaxed copy."""
        _out = []
        for _s in structures:
            _a = _s.copy(); _a.calc = ACECalculator(model_file, skin=0)
            _e0 = _a.get_potential_energy()
            _conv = BFGS(_a, logfile=None).run(fmax=0.03, steps=200)
            _out.append(dict(converged=bool(_conv), dE=_a.get_potential_energy() - _e0,
                             fmax=float(np.abs(_a.get_forces()).max()), atoms=_a))
        return _out


    def relax_table(results):
        return ("| face | converged | energy change (eV) | largest force (eV/Å) |\n|---|---|---|---|\n"
                + "\n".join(f"| ({s.info['miller']}) | {r['converged']} | {r['dE']:+.2f} | {r['fmax']:.3f} |"
                            for s, r in zip(slabs, results)))

    return relax, relax_table


@app.cell
def _(mo, relax, relax_table, repaired_model, slabs):
    relaxed_repaired = relax(repaired_model, slabs)
    mo.md(relax_table(relaxed_repaired))
    return (relaxed_repaired,)


@app.cell(hide_code=True)
def _(mo, relaxed_repaired):
    _bad = [r for r in relaxed_repaired if not r["converged"] or abs(r["dE"]) > 50]
    mo.callout(
        mo.md(f"**The repaired model falls apart when the atoms move**: {len(_bad)} of 3 relaxations "
              "release tens to hundreds of eV, or never converge. The model was taught surface energies "
              "at the ideal cleave and nothing else about surfaces: three structures, each with zero "
              "forces by symmetry. Away from that one geometry it has no information, and the evidence "
              "fit's prior is not enough to stop a relaxation from finding a spurious minimum.")
        if _bad else mo.md("The repaired model relaxes these slabs without collapsing."),
        kind="warn" if _bad else "info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 7: displaced slabs

    Teach the model what a surface does when its atoms move: two rattled
    copies of each repair slab (0.05 and 0.12 Å), six more labels. Refit,
    relax again, and compare the relaxed surface energies with the
    labeller's own: the same slabs relaxed by MACE-MPA-0 with the same
    protocol (those energies ship with the tutorial).
    """)
    return


@app.cell
def _(L, T, bulk_training, fit_model, label, repair, shipped):
    displaced = label(T.e2_displaced(repair))
    stable_model = fit_model([*bulk_training, *repair, *displaced], "fit_displaced")
    from ace_jax.fit.xyz import read_extxyz
    relaxed_truth_energy = {str(f.info["from_key"]): float(f.info["energy"])
                            for f in read_extxyz(str(L.fetch(shipped("e2/relaxed-mpa-0.xyz"))))}
    return relaxed_truth_energy, stable_model


@app.cell
def _(gammas, mo, model_energies, ref_bulk, relax, relax_table, relaxed_truth_energy, slabs, stable_model):
    from ace_jax.tutorials.labels import structure_key as _key

    relaxed_stable = relax(stable_model, slabs)
    _eb = model_energies(stable_model, [ref_bulk])[0]
    gamma_relaxed = gammas(_eb, [r["atoms"].get_potential_energy() for r in relaxed_stable])
    gamma_relaxed_truth = gammas(ref_bulk.info["energy"], [relaxed_truth_energy[_key(s)] for s in slabs])
    relaxed_errors = [abs(a - b) for a, b in zip(gamma_relaxed, gamma_relaxed_truth)]
    mo.md(relax_table(relaxed_stable) + "\n\n| face | relaxed γ, ACE | relaxed γ, MACE-MPA-0 | error |\n|---|---|---|---|\n"
          + "\n".join(f"| ({s.info['miller']}) | {a:.4f} | {b:.4f} | {abs(a - b):.4f} |"
                      for s, a, b in zip(slabs, gamma_relaxed, gamma_relaxed_truth)))
    return relaxed_errors, relaxed_stable


@app.cell(hide_code=True)
def _(mo, relaxed_errors, relaxed_stable):
    _ok = (all(r["converged"] and abs(r["dE"]) < 5 for r in relaxed_stable) and max(relaxed_errors) < 0.01)
    mo.callout(
        mo.md(f"**Checkpoint 4 passed:** every relaxation converges gently, and the relaxed surface "
              f"energies agree with the labeller's own relaxed values to {max(relaxed_errors):.4f} eV/Å².")
        if _ok else mo.md(f"**Checkpoint 4:** a relaxation still misbehaves, or the relaxed surface energies "
                          f"differ from the labeller's by up to {max(relaxed_errors):.4f} eV/Å²."),
        kind="success" if _ok else "warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reflection

    Twelve structures turned a model that could not see a surface into one
    that relaxes three of them correctly. Would this model now predict the
    Si(100) surface you would see in an experiment? What is still missing
    from the data? Think before you open the answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"A model answer": mo.md(
        "No. Real silicon surfaces reconstruct: (100) forms rows of dimers (2×1, and c(4×2) at "
        "low temperature) and (111) the 7×7 reconstruction, with adatoms and stacking faults. "
        "None of those bonding environments is in the data, and the surfaces here were relaxed "
        "only locally from the ideal cleave, at zero temperature. Steps, vacancies, adatoms and "
        "finite-temperature motion are missing too. Each new environment is another coverage "
        "gap: finding them one at a time by hand does not scale, which is what Tutorial 7 "
        "automates.")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. **Thickness.** Change the layers to 4 and to 12. Do the bulk-only
       errors and the novelty ratio change? (The repair slabs switch to 6
       layers when the test slabs are 4.)
    2. **Vacuum.** Set the vacuum to 4 Å per side. Which check fails, and
       why would a too-small gap make the labeller's surface energy wrong?
    3. **Fewer repairs.** Repair with the (111) slab only (`repair[2:]`).
       Which surface energies come back, and which do not?
    4. **The glide cut.** Build a glide-plane (111) slab with
       `ase.build.surface(bulk("Si", "diamond", a=5.43, cubic=True), (1, 1, 1), 6, vacuum=8.0)`
       and count its atoms' neighbours with `T.min_coordination`.

    ## Summary

    - A bulk-only model extrapolates on surfaces, and a descriptor-space
      distance shows it before any label is spent on the test.
    - A few targeted structures repair a property; a property measured at one
      geometry says nothing about the forces around it.
    - Train on what the simulation will do: displaced, relaxed and moving
      configurations, not only the ideal one.

    Next: [Tutorial 7](https://acesuit.github.io/ace-jax/tutorials/curation/)
    lets molecular dynamics and a selection rule find the missing structures.
    """)
    return


if __name__ == "__main__":
    app.run()
