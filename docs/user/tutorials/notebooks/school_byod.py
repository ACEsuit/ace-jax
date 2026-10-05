# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ace-jax>=0.1.1",
#     "marimo>=0.25",
#     "matplotlib>=3.8",
# ]
# ///
"""Tutorial 9: bring your own data.

Adapted from the MLIP-school-2026 notebook D (https://mlipschool.uk/d/d_bring_your_own_data). Run it with
`uvx marimo edit --sandbox school_byod.py`, or as a plain script
(`python school_byod.py`), which is how CI smoke-tests it.
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 9: bring your own data

    The earlier tutorials worked on silicon. This one is a template for your
    own system: declare the property you need, build a training set aimed at
    it, fit, measure the property, check where the property's structures sit
    in descriptor space, and repair the data set once. It runs as is on a
    demo, zincblende GaAs and its (100) surface energy, and takes your own
    structure in Step 2.

    **Goals**

    1. State the target property and the tolerance it needs, before fitting.
    2. Label reference structures, and isolated atoms for the reference
       energies.
    3. Fit a two-element model and measure the property against the labeller.
    4. Use a coverage check to see why the first fit misses, and repair it.

    It uses the steps of Tutorials
    [4](https://acesuit.github.io/ace-jax/tutorials/dataset-and-properties/) and
    [6](https://acesuit.github.io/ace-jax/tutorials/surfaces/), and is adapted from
    notebook D of the [MLIP School 2026](https://mlipschool.uk/d/d_bring_your_own_data).

    **Run this notebook**: with [uv](https://docs.astral.sh/uv/) installed, one command
    opens it in your browser (no account needed):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/notebooks/school_byod.py
    ```

    or open it in [molab](https://molab.marimo.io/github/ACEsuit/ace-jax/blob/main/docs/user/tutorials/notebooks/school_byod.py), marimo's hosted service (free to
    preview; sign in to run). The documentation website shows a static copy,
    run on a CPU when the site was built. The whole notebook runs in about 2 minutes.
    """)
    return


@app.cell
def _():
    import io
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

    return ACECalculator, C, L, T, io, mo, np, pathlib, plt, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: declare the target

    A potential is fitted for a purpose. Write down what you will compute
    with it, how, and how accurately, before choosing any data: the target
    decides the data set, and the tolerance decides when the work is done.
    For the demo the target is the unrelaxed (100) surface energy of GaAs,

    $$\gamma = \frac{E_\text{slab} - N_\text{slab}\,E_\text{bulk}/N_\text{bulk}}{2A},$$

    to within 0.01 eV/Å² of the labeller.
    """)
    return


@app.cell
def _():
    target = dict(name="GaAs(100) surface energy, unrelaxed", quantity="gamma", units="eV/Å²",
                  tolerance=0.01, method="(E_slab - N_slab * E_bulk / N_bulk) / (2 A)")
    return (target,)


@app.cell(hide_code=True)
def _(mo, target):
    _ok = all(k in target for k in ("name", "quantity", "units", "tolerance", "method")) and target["tolerance"] > 0
    mo.callout(
        mo.md(f"**Checkpoint 1 passed:** the target is *{target['name']}*, to within "
              f"{target['tolerance']} {target['units']}.")
        if _ok else mo.md("**Checkpoint 1:** the target needs a name, quantity, units, method and a positive "
                          "tolerance."),
        kind="success" if _ok else "warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: your system

    Upload one periodic structure (`.xyz`, `.extxyz`, `.cif` or `.vasp`; a
    bare `POSCAR` must be renamed `POSCAR.vasp`) with one or two elements and
    at most 64 atoms. Without an upload the notebook uses the demo: an 8-atom
    cubic cell of zincblende GaAs.

    The labels for the demo ship with the tutorial. Your own system needs the
    labeller installed: in a marimo sandbox add `mace-torch` to the packages
    panel; elsewhere,

    ```bash
    pip install mace-torch --extra-index-url https://download.pytorch.org/whl/cpu
    ```

    MACE-MPA-0 covers most of the periodic table, but it is a foundation
    model: check its answer for your system against the literature in Step 3.
    """)
    return


@app.cell
def _(mo):
    upload = mo.ui.file(filetypes=[".xyz", ".extxyz", ".cif", ".vasp"], label="your structure")
    upload
    return (upload,)


@app.cell
def _(T, io, mo, upload):
    from ase.io import read

    if upload.value:
        _f = upload.value[0]
        _fmt = {"xyz": "extxyz", "extxyz": "extxyz", "cif": "cif", "vasp": "vasp"}[_f.name.rsplit(".", 1)[-1].lower()]
        system = read(io.StringIO(_f.contents.decode()), format=_fmt, index=-1)
        demo = False
    else:
        system, demo = T.d_system(), True
    species = tuple(sorted(set(system.get_chemical_symbols())))
    system_ok = (len(system) <= 64 and system.cell.rank == 3 and bool(system.pbc.all()) and 1 <= len(species) <= 2)
    mo.md(f"System: **{system.get_chemical_formula()}**, {len(system)} atoms, species {', '.join(species)}"
          + (" (the demo)." if demo else "."))
    return demo, species, system, system_ok


@app.cell(hide_code=True)
def _(mo, system_ok):
    _ok = system_ok
    mo.callout(
        mo.md("**Checkpoint 2 passed:** a periodic cell with one or two elements and at most 64 atoms.")
        if _ok else mo.md("**Checkpoint 2:** this notebook takes a periodic cell (three cell vectors, "
                          "periodic in all directions) with one or two elements and at most 64 atoms."),
        kind="success" if _ok else "danger",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: references and the truth

    Two kinds of reference: an **isolated atom** of each element, a single
    atom in a 15 Å box, whose energy is the element's reference energy $E_0$
    (an isolated atom in the training set pins it, Tutorial 1); and the
    **target structures**, the bulk cell and a 4-layer (100) slab with 8 Å
    of vacuum, whose labelled energies give the true value of the target.
    The target structures are never trained on.
    """)
    return


@app.cell
def _(L, T, demo, mo, pathlib, species, system, system_ok, target):
    mo.stop(not system_ok, mo.md("Step 2's checkpoint must pass first."))
    URL = "https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/school/d/labels-mpa-0.xyz"
    _here = (mo.notebook_dir() / "../data/school/d/labels-mpa-0.xyz") if mo.notebook_dir() else None
    cache = L.LabelCache.from_file(_here if _here is not None and _here.exists() else URL) if demo else None
    work = pathlib.Path("ace_jax_tutorial_9")
    work.mkdir(exist_ok=True)


    def label(xs):
        return L.label(xs, model="mpa-0", cache=cache)


    try:
        isolated = label(T.d_isolated(species))
        target_structures = label(T.d_targets(system))
    except L.LabelsUnavailable as _e:          # your own system without the labeller installed
        mo.stop(True, mo.callout(mo.md(f"**Labels needed.** {_e}"), kind="warn"))


    def gamma(e_bulk, e_slab):
        b, s = target_structures
        return T.surface_energy(e_bulk, len(b), e_slab, s)


    truth = gamma(target_structures[0].info["energy"], target_structures[1].info["energy"])
    mo.md("| reference | energy (eV) |\n|---|---|\n"
          + "\n".join(f"| isolated {a.get_chemical_symbols()[0]} | {a.info['energy']:.4f} |" for a in isolated)
          + f"\n\nThe labeller's {target['name']}: **{truth:.4f} {target['units']}** "
          f"({truth * 16.0218:.2f} J/m²). Labels used so far: {L.labels_used()}.")
    return gamma, isolated, label, target_structures, truth, work


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: a training set and a first fit

    The recipe of Tutorial 4, applied to your cell: copies with the cell
    scaled over $1 \pm s$ and the atoms rattled, plus the isolated atoms.
    The sliders snap to the settings whose demo labels ship.

    The basis is categorical in the elements (Tutorial 3), order 3 and total
    degree 8. The school's version of this notebook needed more observations
    than basis functions, because its least-squares fit has no prior; the
    evidence fit's prior regularises an underdetermined basis, so no such
    rule is needed here, though more data still helps.
    """)
    return


@app.cell
def _(T, mo):
    strain = mo.ui.slider(steps=list(T.D_STRAINS), value=0.06, label="strain range s", show_value=True)
    rattle = mo.ui.slider(steps=list(T.D_RATTLES), value=0.03, label="rattle σ (Å)", show_value=True)
    n_train = mo.ui.slider(steps=list(T.D_NTRAIN), value=40, label="training cells", show_value=True)
    mo.hstack([strain, rattle, n_train], justify="start")
    return n_train, rattle, strain


@app.cell
def _(L, T, isolated, label, mo, n_train, rattle, species, strain, system, target_structures, time, work):
    from ace_jax.basis.model import BasisSpec, build_basis
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data, save_model

    training = label(T.d_training(system, strain.value, rattle.value, n_train.value))
    _held_out = {T.structure_fingerprint(a) for a in target_structures}
    assert not any(T.structure_fingerprint(a) in _held_out for a in training), "a target structure is in the training set"
    basis = build_basis(BasisSpec(order=3, max_degree=8, rcut=5.5, elements=species))


    def fit_model(train, out):
        _cfg = FitConfig(model=basis, arm="linear", m_per_species=0, e0="lsq", opt="lbfgs", r0=None,
                         rungs=("map",), predict_stats="recompute", predict_train=False,
                         energy_key="energy", force_key="forces", virial_key="virial").validate()
        _res = fit(_cfg, load_fit_data(_cfg, train=list(train), log=lambda *a: None), log=lambda *a: None)
        return str(save_model(_res, work / out, log=lambda *a: None))


    _t = time.time()
    model_v1 = fit_model([*isolated, *training], "fit_v1")
    v1_seconds = time.time() - _t
    mo.md(f"{len(training)} training cells + {len(isolated)} isolated atoms; basis of "
          f"**{basis.meta['len_basis']}** functions; fitted in {v1_seconds:.0f} s. "
          f"Labels used so far: {L.labels_used()}.")
    return basis, fit_model, model_v1, training


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5: measure the target
    """)
    return


@app.cell
def _(ACECalculator, gamma, target_structures):
    def model_gamma(model_file):
        """The target computed with a fitted model, on the labeller's target structures."""
        _calc = ACECalculator(model_file, skin=0)
        _E = []
        for _a in target_structures:
            _b = _a.copy(); _b.calc = _calc
            _E.append(_b.get_potential_energy())
        return gamma(*_E)

    return (model_gamma,)


@app.cell
def _(mo, model_gamma, model_v1, target, truth):
    gamma_v1 = model_gamma(model_v1)
    err_v1 = abs(gamma_v1 - truth)
    mo.md(f"| | γ ({target['units']}) |\n|---|---|\n| labeller | {truth:.4f} |\n| model v1 | {gamma_v1:.4f} |\n"
          f"| error | {err_v1:.4f} (tolerance {target['tolerance']}) |")
    return err_v1, gamma_v1


@app.cell(hide_code=True)
def _(err_v1, mo, target):
    import math as _math

    _ok = err_v1 > target["tolerance"]
    mo.callout(
        mo.md("**Checkpoint 3 failed:** the first fit's surface energy is not finite, so the fit "
              "failed numerically. Its numbers below are meaningless.")
        if not _math.isfinite(err_v1) else
        mo.md(f"**Checkpoint 3 passed:** the bulk-only model misses the target by {err_v1:.3f} "
              f"{target['units']}, beyond the tolerance. The next step asks why.")
        if _ok else mo.md(f"**Checkpoint 3:** the first fit already meets the tolerance ({err_v1:.3f}). "
                          "With your own system and target that can happen: the coverage check below "
                          "still says how far the target sits from the data."),
        kind="danger" if not _math.isfinite(err_v1) else "success" if _ok else "info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6: coverage

    Compute every atom's descriptor in the model's own basis, for the
    training cells and for the target structures, and measure how far the
    slab's most exposed atom sits from the nearest training atom, in units of
    the training atoms' own spacing. Tutorial 6 used the median atom instead
    (silicon's slab atoms sat well over 10× out); half of a thin slab's atoms are
    bulk-like, so here the worst one tells more. A ratio near 1 means the
    target is inside the data.
    """)
    return


@app.cell
def _(C, mo, model_v1, np, plt, target_structures, training):
    Xt = np.concatenate(C.atom_descriptors(training, model_v1))
    Xq = np.concatenate(C.atom_descriptors(target_structures[1:], model_v1))      # the slab's atoms
    ratio_v1 = C.nn_ratio(Xt, Xq, q=1.0)               # the slab's most exposed atom
    _mu = Xt.mean(0)
    _, _, _Vt = np.linalg.svd(Xt - _mu, full_matrices=False)
    _pt, _pq = (Xt - _mu) @ _Vt[:2].T, (Xq - _mu) @ _Vt[:2].T
    _fig, _ax = plt.subplots(figsize=(5, 4))
    _ax.plot(_pt[:, 0], _pt[:, 1], ".", ms=4, alpha=0.5, label="training atoms")
    _ax.plot(_pq[:, 0], _pq[:, 1], "x", ms=6, label="slab atoms")
    _ax.set_xlabel("PC 1"); _ax.set_ylabel("PC 2"); _ax.legend(frameon=False)
    _ax.set_title(f"the slab sits {ratio_v1:.1f}× out")
    _fig.tight_layout()
    _fig
    return (ratio_v1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 7: one repair round

    The target's atoms are surface atoms; the training set has none. Add
    thin (100) slabs, 3 and 5 layers (never the 4-layer target), each with
    15 rattles from 0 to 0.12 Å, and refit with exactly the same basis, so
    the change is down to the data alone.
    """)
    return


@app.cell
def _(C, L, T, fit_model, isolated, label, mo, model_gamma, np, system, target, target_structures, training, truth,
      work):
    repair = label(T.d_repair(system))
    model_v2 = fit_model([*isolated, *training, *repair], "fit_v2")
    gamma_v2 = model_gamma(model_v2)
    err_v2 = abs(gamma_v2 - truth)
    _Xt = np.concatenate(C.atom_descriptors([*training, *repair], model_v2))
    ratio_v2 = C.nn_ratio(_Xt, np.concatenate(C.atom_descriptors(target_structures[1:], model_v2)), q=1.0)
    from ase.io import write as _write
    _write(str(work / "train.xyz"), [*isolated, *training, *repair])
    mo.md(f"| model | training structures | γ error ({target['units']}) | coverage ratio |\n|---|---|---|---|\n"
          f"| v1 | {len(training) + len(isolated)} | see Step 5 | see Step 6 |\n"
          f"| v2 | {len(training) + len(isolated) + len(repair)} | {err_v2:.4f} | {ratio_v2:.1f} |\n\n"
          f"Labels used in all: {L.labels_used()}.")
    return err_v2, model_v2, ratio_v2


@app.cell(hide_code=True)
def _(err_v1, err_v2, mo, ratio_v1, ratio_v2, target):
    import math as _math

    _finite = all(_math.isfinite(x) for x in (err_v1, err_v2, ratio_v1, ratio_v2))
    _ok = _finite and err_v2 < target["tolerance"] and err_v2 < err_v1 and ratio_v2 < ratio_v1
    mo.callout(
        mo.md("**Checkpoint 4 failed:** a fit returned a non-finite surface energy or coverage ratio, "
              "so the comparison cannot be made.")
        if not _finite else
        mo.md(f"**Checkpoint 4 passed:** with the repair set the error falls from {err_v1:.4f} to "
              f"{err_v2:.1e} {target['units']}, within the tolerance, and the slab's most exposed atom "
              f"sits {ratio_v2:.2g}× out instead of {ratio_v1:.0f}×: the repair slabs, 3 and 5 layers "
              "thick, contain the target's surface environments almost exactly, though not the "
              "4-layer target itself.")
        if _ok else mo.md(f"**Checkpoint 4:** after the repair the error is {err_v2:.4f} {target['units']} "
                          f"(tolerance {target['tolerance']}), and the coverage ratio {ratio_v2:.1f}. For your "
                          "own system, look at which environments the target has that the data still lacks."),
        kind="success" if _ok else "warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Take it home

    Both models, and the labelled training set (`train.xyz`, written in
    Step 7), are in `ace_jax_tutorial_9/`. The same fit from the command line:

    ```bash
    aj fit --order 3 --max-degree 8 --rcut 5.5 --train train.xyz \
        --e0 lsq --m-per-species 0 --opt lbfgs --out fit_v2
    ```

    From here: Tutorial [7](https://acesuit.github.io/ace-jax/tutorials/curation/)
    automates the next rounds, choosing structures from MD by novelty or by
    the model's own uncertainty.

    ## Exercises

    1. **The recipe.** Move the sliders: which matters most for the
       bulk-only error, the strain range, the rattle or the number of cells?
       Does any setting meet the tolerance without slabs?
    2. **Your system.** Upload your own structure and run the notebook
       through. Compare the labeller's value with the literature first: the
       labeller is a model too (Tutorial 8).
    3. **Another target.** Make the target the (110) surface: in Step 3,
       replace the slab with `surface(system, (1, 1, 0), 4, vacuum=8.0)`
       (from `ase.build`; it needs the labeller installed) and repeat. Does
       the (100) repair set help?
    """)
    return


if __name__ == "__main__":
    app.run()
