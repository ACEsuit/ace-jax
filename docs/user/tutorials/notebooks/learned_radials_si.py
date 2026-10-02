# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ace-jax>=0.1.0",
#     "marimo>=0.25",
#     "matplotlib>=3.8",
# ]
# ///
"""Tutorial 2: learn the radial basis of an ACE model.

Run it with `uvx marimo edit --sandbox learned_radials_si.py`, or as a plain
script (`python learned_radials_si.py`), which is how CI smoke-tests it.
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 2: learned radials

    An ACE basis function multiplies radial functions $R_{nl}(r)$ of the neighbour
    distances with spherical harmonics. The radial functions are mixtures of
    a fixed polynomial basis,

    $$R_{nl}(r) = \mathrm{env}(x)\sum_q W_{nq}\,P_q(x), \qquad x = x(r),$$

    and `aj fit` keeps the mixing weights W frozen at whatever the basis was
    built with. This notebook **learns W from the data** and shows what that
    buys on a small silicon dataset. It runs on a CPU in a few minutes.

    **Goals**

    1. Fit a baseline with a frozen basis of seeded random radials (`radial_mode="glorot_normal"`).
    2. Learn the radial weights by variable projection, with a held-out gate.
    3. Refit with the learned basis and compare the test errors.
    4. Deploy the learned model: it is splined automatically and runs through `ACECalculator`.

    Work through [Tutorial 1](https://acesuit.github.io/ace-jax/tutorials/first-fit/)
    first; this one reuses its data split. On the command line,
    `aj fit --learn-radial` runs the same learning step in one fit.

    **Run this notebook**: with [uv](https://docs.astral.sh/uv/) installed, one command
    opens it in your browser (no account needed):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/notebooks/learned_radials_si.py
    ```

    or open it in [molab](https://molab.marimo.io/github/ACEsuit/ace-jax/blob/main/docs/user/tutorials/notebooks/learned_radials_si.py), marimo's hosted service (free to
    preview; sign in to run). The documentation website shows a static copy,
    run on a CPU when the site was built: there the interactive controls show
    their default values. The notebook runs in about three minutes, two of them for learning.
    """)
    return


@app.cell
def _():
    import pathlib
    import time
    import urllib.request

    import jax

    jax.config.update("jax_enable_x64", True)  # fitting and radial learning need float64

    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.io import read, write

    from ace_jax import ACECalculator

    return ACECalculator, jnp, mo, np, pathlib, plt, read, time, urllib, write


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: data and splits

    The same silicon data as Tutorial 1: 39 training and 13 test
    configurations. Radial learning needs a second, **validation** split
    inside the training set: the learner optimises W on the *fit* split and a
    gate compares the learned and the initial radials on the *validation*
    split. The test set is never seen until the end.
    """)
    return


@app.cell
def _(mo, pathlib, read, urllib, write):
    work = pathlib.Path("ace_jax_tutorial_2")
    work.mkdir(exist_ok=True)
    URL = "https://raw.githubusercontent.com/ACEsuit/ace-jax/main/fixtures/si_tiny_train.xyz"
    _local = mo.notebook_dir() / "../../../../fixtures/si_tiny_train.xyz" if mo.notebook_dir() else None
    _src = work / "si_tiny_train.xyz"
    if _local is not None and _local.exists():
        _src.write_bytes(_local.read_bytes())
    elif not _src.exists():
        urllib.request.urlretrieve(URL, _src)
    _bulk = [_a for _a in read(_src, ":") if _a.info["config_type"] != "isolated_atom"]
    _train = [_a for _i, _a in enumerate(_bulk) if _i % 4]
    files = {k: work / f"{k}.xyz" for k in ("train", "test", "fit", "val")}
    write(files["test"], _bulk[::4])                                 # 13 configs, as in Tutorial 1
    write(files["train"], _train)                                    # 39 configs
    write(files["val"], _train[::3])                                 # 13 of the training configs
    write(files["fit"], [_a for _i, _a in enumerate(_train) if _i % 3])  # the other 26
    keys = dict(energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
    mo.md(" · ".join(f"{k}: **{len(read(v, ':'))}** configs" for k, v in files.items()))
    return files, keys, work


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: the frozen baseline

    Build a basis whose radial weights W are seeded random mixtures
    (`radial_mode="glorot_normal"`, or `aj basis ... --radial-mode
    glorot_normal`). That is a deliberately poor start, so the effect of
    learning is easy to see; the default `onehot` basis is a good frozen basis
    (exercise 2). Save it to a file, since
    the learner writes its result into a copy of that file, then fit it as in
    Tutorial 1 and record the test errors.
    """)
    return


@app.cell
def _(files, keys, work):
    from ace_jax.basis.export import save_npz
    from ace_jax.basis.model import BasisSpec, build_basis
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data, save_model

    basis_file = work / "si_basis.npz"
    # seeded random radial mixtures (seed 0): a poor frozen basis, a clear start for learning
    save_npz(basis_file, build_basis(BasisSpec(order=3, max_degree=10, elements=("Si",),
                                               radial_mode="glorot_normal")))

    def config(model_file):
        """The linear fit used throughout: evidence-maximised, E0 by least squares."""
        return FitConfig(model=str(model_file), arm="linear", m_per_species=0, e0="lsq",
                         opt="lbfgs", r0=2.35, rungs=("map",), predict_stats="recompute",
                         predict_train=False, **keys).validate()

    def fit_and_test(model_file, out):
        """Fit on the training set, return (test metrics, fitted model path)."""
        _cfg = config(model_file)
        _res = fit(_cfg, load_fit_data(_cfg, train=str(files["train"]), test=str(files["test"])))
        return _res.preds.metrics["test/map"], save_model(_res, work / out)

    frozen_metrics, frozen_model = fit_and_test(basis_file, "fit_frozen")
    return basis_file, config, fit_and_test, frozen_metrics, frozen_model, load_fit_data


@app.cell(hide_code=True)
def _(frozen_metrics, mo):
    mo.md(
        f"Frozen seeded radials: test E RMSE **{frozen_metrics['E']['rmse']:.1f} meV/atom**, "
        f"F RMSE **{frozen_metrics['F']['rmse']:.3f} eV/Å**."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: learn the radials

    For a given W the best readout is a closed-form ridge regression, so the
    learner minimises the *projected* residual over W alone (variable
    projection, VarPro), with L-BFGS. Every `reprofile_every` steps the
    evidence hyperparameters are re-fitted at the current W.

    Then the **gate** fits a readout for each candidate, the initial W and
    the learned W, on the fit split and scores it on the validation split. It
    keeps the better one, and keeps the initial radials on a tie, so learning
    can never make the selected model worse on held-out data.

    The steps are:

    - `load_fit_data` + `build_problem` set up the linear problem on the fit split,
      with the validation split as its test set;
    - `to_analytic(model, n_q)` widens the radial polynomial span to `n_q` terms,
      which leaves every radial unchanged but lets the learner reach higher degrees;
    - `fit_radial` learns and gates; `save_result` writes the selected radials
      into a copy of the basis file.

    On the command line, `aj fit --learn-radial` runs these steps (the hold-out,
    the gate and the final fit on the whole training set) in one fit.
    """)
    return


@app.cell
def _(mo):
    steps = mo.ui.slider(0, 80, step=10, value=40, label="L-BFGS steps", show_value=True)
    steps
    return (steps,)


@app.cell
def _(basis_file, config, files, load_fit_data, np, steps, time, work):
    from ace_jax.fit.pipeline.problem import build_problem
    from ace_jax.fit.radial_learn import fit_radial, save_result
    from ace_jax.fit.radial_model import rnl_degrees, to_analytic

    # The radial-learning step. Everything it needs is the basis file and the
    # fit/val split; everything after it reads only learned_dir / "model.npz".
    _t = time.time()
    _cfg = config(basis_file)
    _d = load_fit_data(_cfg, train=str(files["fit"]), test=str(files["val"]))
    _prob = build_problem(_cfg, _d).prob
    start_model, _ = to_analytic(_d.model, 12)          # n_q = 12 polynomials per radial
    _prob = _prob._replace(model=start_model)
    learn_log = []
    W, learn_info = fit_radial(
        _prob, _d.ds_train, _d.ds_test, start_model.rnl_Wnlq,
        lam_grid=(0.0,),                                # roughness penalty weights to try
        rough_weights=1.0 / (1.0 + rnl_degrees(_d.meta)) ** 2,
        steps=int(steps.value), reprofile_every=20, log=learn_log.append)
    learned_dir = work / "learned"
    save_result(learned_dir, W, learn_info, src_npz=str(basis_file), model=start_model)
    learn_seconds = time.time() - _t
    gate_scores = {k: float(v) for k, v in learn_info["scores"].items()}
    return W, gate_scores, learn_info, learn_seconds, learned_dir, start_model


@app.cell(hide_code=True)
def _(gate_scores, learn_info, learn_seconds, mo):
    mo.md(
        f"Learned in {learn_seconds:.0f} s. Gate scores on the validation split "
        "(lower is better):\n\n| candidate | score |\n|---|---|\n"
        + "\n".join(f"| `{k}` | {v:.1f} |" for k, v in gate_scores.items())
        + f"\n\nSelected: **`{learn_info['selected']}`**"
    )
    return


@app.cell(hide_code=True)
def _(learn_info, mo):
    _ok = learn_info["selected"] != "init"
    mo.callout(
        mo.md("**Checkpoint 1 passed:** the learned radials beat the initial ones on the validation split.")
        if _ok else
        mo.md("**Checkpoint 1:** the gate kept the initial radials. With 0 steps that is expected "
              "(nothing was learned); otherwise try more steps."),
        kind="success" if _ok else "warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: what changed?

    Plot the first few radial functions $R_n(r)$, before and after learning.
    Each curve is scaled to unit root-mean-square over the plotted range, so
    only their shapes are compared. The grey histogram shows where the training data
    has neighbour pairs: the radials can only be learned where there is data.
    """)
    return


@app.cell
def _(W, files, jnp, keys, np, plt, start_model):
    from ace_jax.eval import sparse_graph
    from ace_jax.fit.data import load_configs
    from ace_jax.fit.radial_model import poly_env

    _r = np.linspace(1.8, 5.5, 300)
    _z = jnp.zeros(len(_r), dtype=int)
    _P = np.asarray(poly_env(start_model, jnp.asarray(_r), _z, _z))         # (n_r, n_q)
    _R0 = _P @ np.asarray(start_model.rnl_Wnlq[0, 0]).T                     # (n_r, n_radials)
    _R1 = _P @ np.asarray(W[0, 0]).T
    _d = np.concatenate([np.linalg.norm(sparse_graph(c.positions, c.cell, c.pbc, 5.5).rij, axis=1)
                         for c in load_configs(str(files["fit"]), **keys)])
    _fig, _axes = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True)
    for _n, _ax in enumerate(_axes.flat):
        _ax.hist(_d, bins=60, density=True, color="0.85")
        _tw = _ax.twinx()
        for _R, _lab in ((_R0, "initial"), (_R1, "learned")):
            _y = _R[:, _n] / np.sqrt(np.mean(_R[:, _n] ** 2))
            _tw.plot(_r, _y, label=_lab)
        _tw.set_yticks([]); _ax.set_yticks([])
        _ax.set_title(f"radial {_n}")
    _axes[0, 0].figure.legend(*_tw.get_legend_handles_labels(), loc="upper right")
    for _ax in _axes[1]:
        _ax.set_xlabel("r (Å)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5: refit and test

    The learner's `model.npz` holds the selected radials with a readout fitted
    on the fit split only. Refit it on the whole training set with exactly
    the same settings as the baseline, so the radial basis is the only
    difference, and compare on the test set.
    """)
    return


@app.cell
def _(fit_and_test, frozen_metrics, learned_dir, mo):
    learned_metrics, learned_model = fit_and_test(learned_dir / "model.npz", "fit_learned")
    _rows = [("frozen seeded radials", frozen_metrics), ("learned radials", learned_metrics)]
    mo.md(
        "| basis | test E RMSE (meV/atom) | test F RMSE (eV/Å) | test V RMSE (eV) |\n|---|---|---|---|\n"
        + "\n".join(f"| {n} | {m['E']['rmse']:.1f} | {m['F']['rmse']:.3f} | {m['V']['rmse']:.3f} |"
                    for n, m in _rows)
    )
    return learned_metrics, learned_model


@app.cell(hide_code=True)
def _(frozen_metrics, learned_metrics, mo):
    _ok = learned_metrics["F"]["rmse"] < frozen_metrics["F"]["rmse"]
    mo.callout(
        mo.md("**Checkpoint 2 passed:** the learned radial basis gives smaller test force errors.")
        if _ok else mo.md("**Checkpoint 2:** the learned basis did not reduce the test force error."),
        kind="success" if _ok else "warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6: deploy

    A learned radial is a polynomial mixture, slower to evaluate than the
    cubic splines of a stock model. The model file records that its radials
    were learned, and `ACECalculator` (like `export_lammps`) then converts them
    to splines at a relative tolerance of 10⁻¹⁰ before evaluating
    (`spline_tol="auto"`). `calc.splined` reports what was converted.
    Compare it with an exact, unsplined evaluation (`spline_tol=None`).
    """)
    return


@app.cell
def _(ACECalculator, learned_model, mo, np):
    from ase.build import bulk

    _atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat(2)
    _atoms.rattle(0.05, seed=1)
    calc_spline = ACECalculator(str(learned_model), skin=0)
    _exact = ACECalculator(str(learned_model), skin=0, spline_tol=None)
    _atoms.calc = calc_spline
    _E1, _F1 = _atoms.get_potential_energy(), _atoms.get_forces()
    _atoms.calc = _exact
    _E2, _F2 = _atoms.get_potential_energy(), _atoms.get_forces()
    deploy_dE = abs(_E1 - _E2) / abs(_E2)
    deploy_dF = np.abs(_F1 - _F2).max() / np.abs(_F2).max()
    mo.md(
        f"`calc.splined` = `{calc_spline.splined}`\n\n"
        f"Splined vs exact on a rattled 64-atom cell: relative energy difference "
        f"{deploy_dE:.1e}, largest force difference {deploy_dF:.1e} of max|F|."
    )
    return calc_spline, deploy_dE, deploy_dF


@app.cell(hide_code=True)
def _(calc_spline, deploy_dE, deploy_dF, mo):
    _ok = calc_spline.splined is not None and deploy_dE < 1e-7 and deploy_dF < 1e-6
    mo.callout(
        mo.md("**Checkpoint 3 passed:** the learned model was splined and agrees with the exact evaluation.")
        if _ok else mo.md("**Checkpoint 3:** the calculator did not spline the model, or disagrees with "
                          "the exact evaluation (did the gate keep `init`?)."),
        kind="success" if _ok else "warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. **Step budget.** Move the *L-BFGS steps* slider to 0, 10 and 80. With 0
       steps nothing is learned and the gate keeps `init`; how quickly does
       the validation score improve with steps?
    2. **A better start.** In Step 2, build the basis with
       the default `radial_mode="onehot"` (Tutorial 1's basis). Does learning still help,
       and in which of energy, forces and virials? The gate keeps the starting
       radials whenever learning does not improve the held-out score.
    3. **Smoothness.** Pass `lam_grid=(0.0, 1e-2)` to `fit_radial`. A positive
       weight penalises rough radials; the gate picks the best of all
       candidates. When might a smoother radial generalise better?
    4. **Command line.** `aj fit ... --learn-radial` runs the same steps in one
       fit (the hold-out split, the gate and the final fit on the whole training
       set); see the [how-to guide](https://acesuit.github.io/ace-jax/howto/learned-radials/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - The radial basis of a frozen ACE model is a choice made when the basis
      is built. Learning it by VarPro optimises that choice on the data,
      with a held-out gate that keeps the initial radials unless the learned
      ones predict better.
    - The learned model is an ordinary `model.npz`: fit it, evaluate it and
      export it like any other. At deployment it is splined to 10⁻¹⁰, so it
      runs as fast as a stock model.

    On production-sized data the gain is larger and survives a good start:
    see the learned-radial results in the ace-jax repository
    (`docs/dev/learn-radial-results.md`).

    Next: [Tutorial 3](https://acesuit.github.io/ace-jax/tutorials/multi-element/)
    fits a five-element alloy and compares two ways of describing the elements.
    """)
    return


if __name__ == "__main__":
    app.run()
