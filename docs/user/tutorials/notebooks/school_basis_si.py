# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ace-jax>=0.1.0",
#     "marimo>=0.25",
#     "matplotlib>=3.8",
# ]
# ///
"""Tutorial 4: basis size, overfitting and the evidence.

Adapted from the MLIP-school-2026 notebook E1x (ACEsuit/MLIP-school-2026). Run it with
`uvx marimo edit --sandbox school_basis_si.py`, or as a plain script
(`python school_basis_si.py`), which is how CI smoke-tests it.
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 4: basis size, overfitting and the evidence

    In [Tutorial 3](https://acesuit.github.io/ace-jax/tutorials/dataset-and-properties/)
    a near-perfect fit got a vacancy wrong because the data did not cover the
    question. This time the data covers the question, and the risk is the model:
    how large a basis can 200 silicon structures support?

    **Goals**

    1. Hold out a test set, and fit the same data with a growing basis.
    2. Watch plain least squares fit its training set better and better while
       the gap to the test set opens.
    3. Fit the same bases with the evidence (Bayesian) fit, and use its
       log-evidence to choose the basis size without looking at the test set.

    It is adapted from notebook E1x of the
    [MLIP school 2026](https://github.com/ACEsuit/MLIP-school-2026).

    **Run this notebook**: with [uv](https://docs.astral.sh/uv/) installed, one command
    opens it in your browser (no account needed):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/notebooks/school_basis_si.py
    ```

    or open it in [molab](https://molab.marimo.io/github/ACEsuit/ace-jax/blob/main/docs/user/tutorials/notebooks/school_basis_si.py), marimo's hosted service (free to
    preview; sign in to run). The documentation website shows a static copy,
    run on a CPU when the site was built. The sweep in Step 2 fits eight models
    and takes about 10 minutes on a CPU, half of it the largest basis.
    """)
    return


@app.cell
def _():
    import pathlib
    import time
    import urllib.request

    import jax

    jax.config.update("jax_enable_x64", True)  # fitting needs float64

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.io import read

    return mo, np, pathlib, plt, read, time, urllib


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: the data and a test set

    `reference.xyz` holds 250 rattled and strained bulk silicon cells (8 and
    16 atoms) from the MLIP school, labelled with MACE-MPA-0 (see Tutorial 3)
    under the keys `energy`, `forces` and `virial`. Nothing here labels
    anything new.

    A test set is the structures the fit never sees. Shuffle before splitting,
    since the file is ordered by cell size, and keep 80% for training.
    """)
    return


@app.cell
def _(mo, np, pathlib, read, urllib):
    URL = ("https://raw.githubusercontent.com/ACEsuit/ace-jax/main/"
           "docs/user/tutorials/data/school/e1x/reference.xyz")
    work = pathlib.Path("ace_jax_tutorial_4")
    work.mkdir(exist_ok=True)
    _local = (mo.notebook_dir() / "../data/school/e1x/reference.xyz") if mo.notebook_dir() else None
    data_file = work / "reference.xyz"
    if _local is not None and _local.exists():
        data_file.write_bytes(_local.read_bytes())
    elif not data_file.exists():
        urllib.request.urlretrieve(URL, data_file)
    frames = read(data_file, ":")
    _order = np.random.default_rng(11).permutation(len(frames))
    _cut = int(0.8 * len(frames))
    train_frames = [frames[i] for i in _order[:_cut]]
    test_frames = [frames[i] for i in _order[_cut:]]
    mo.md(f"**{len(frames)}** structures: **{len(train_frames)}** for training, "
          f"**{len(test_frames)}** held out.")
    return test_frames, train_frames, work


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: two fits per basis

    The sweep holds everything fixed (correlation order 3, cutoff 5.5 Å) except
    the maximum total degree, so any change is down to the basis size. Each
    basis is fitted twice:

    - **least squares** (`solver="lstsq"`): minimise the weighted squared
      error of energies, forces and virials, with weights 30, 1 and 1 (the
      ACEpotentials defaults) and nothing else. This is what the school's
      workbench did.
    - **the evidence fit** (the default, as in Tutorials 1 and 3): Bayesian
      linear regression with a smoothness prior on the coefficients, its noise
      levels and prior scale chosen by maximising the evidence $p(\text{data}
      \mid \text{basis})$.

    On the command line the two are

    ```bash
    aj fit --order 3 --max-degree 12 --train train.xyz --test test.xyz \
        --m-per-species 0 --e0 lsq --weights '{"default": {"E": 30, "F": 1, "V": 1}}' \
        --solver lstsq --out lstsq12
    aj fit --order 3 --max-degree 12 --train train.xyz --test test.xyz \
        --m-per-species 0 --e0 lsq --weights '{"default": {"E": 30, "F": 1, "V": 1}}' \
        --opt lbfgs --out evidence12
    ```
    """)
    return


@app.cell
def _(mo, test_frames, time, train_frames):
    from ace_jax.basis.model import BasisSpec, build_basis
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data

    DEGREES = (8, 12, 14, 16)
    WEIGHTS = {"default": {"E": 30.0, "F": 1.0, "V": 1.0}}
    _t0 = time.time()
    sweep = []
    with mo.status.progress_bar(total=2 * len(DEGREES), title="Fitting") as _bar:
        for _d in DEGREES:
            _basis = build_basis(BasisSpec(order=3, max_degree=_d, rcut=5.5, elements=("Si",)))
            _row = {"degree": _d, "basis": int(_basis.meta["len_basis"])}
            for _solver in ("lstsq", "evidence"):
                _cfg = FitConfig(model=_basis, arm="linear", m_per_species=0, e0="lsq", opt="lbfgs", r0=None,
                                 rungs=("map",), solver=_solver, weights=WEIGHTS, predict_train=True,
                                 predict_stats="recompute",
                                 energy_key="energy", force_key="forces", virial_key="virial").validate()
                _res = fit(_cfg, load_fit_data(_cfg, train=train_frames, test=test_frames, log=lambda *a: None),
                           log=lambda *a: None)
                _k = "lstsq" if _solver == "lstsq" else "map"
                _m = _res.preds.metrics
                _row[_solver] = {"train": _m[f"train/{_k}"]["E"]["rmse"], "test": _m[f"test/{_k}"]["E"]["rmse"]}
                if _solver == "evidence":
                    _row["log_evidence"] = _res.map.log_evidence
                _bar.update(subtitle=f"degree {_d}, {_solver}")
            sweep.append(_row)
    sweep_seconds = time.time() - _t0
    mo.md("| degree | basis | lstsq train | lstsq test | evidence train | evidence test | log-evidence |\n"
          "|---|---|---|---|---|---|---|\n"
          + "\n".join(f"| {r['degree']} | {r['basis']} | {r['lstsq']['train']:.2f} | {r['lstsq']['test']:.2f} | "
                      f"{r['evidence']['train']:.2f} | {r['evidence']['test']:.2f} | {r['log_evidence']:.0f} |"
                      for r in sweep)
          + f"\n\nEnergy RMSE in meV/atom. {2 * len(DEGREES)} fits in {sweep_seconds:.0f} s.")
    return DEGREES, sweep


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: read the curves

    The left panel is the school's picture. The right panel is what the
    evidence fit adds: a score for each basis computed from the training data
    alone.
    """)
    return


@app.cell
def _(np, plt, sweep):
    n = np.array([r["basis"] for r in sweep])
    ls_train, ls_test = (np.array([r["lstsq"][k] for r in sweep]) for k in ("train", "test"))
    ev_train, ev_test = (np.array([r["evidence"][k] for r in sweep]) for k in ("train", "test"))
    log_ev = np.array([r["log_evidence"] for r in sweep])
    by_evidence = int(np.argmax(log_ev))                   # the training data's choice
    by_test = int(np.nanargmin(ls_test))                   # what the held-out set says, for least squares
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 4))
    _a1.loglog(n, ls_train, "o--", c="C0", label="least squares, train")
    _a1.loglog(n, ls_test, "o-", c="C0", label="least squares, test")
    _a1.loglog(n, ev_train, "s--", c="C1", label="evidence, train")
    _a1.loglog(n, ev_test, "s-", c="C1", label="evidence, test")
    _a1.set_xlabel("basis size"); _a1.set_ylabel("energy RMSE (meV/atom)"); _a1.legend(frameon=False, fontsize=8)
    _a2.plot(n, log_ev, "s-", c="C1")
    _a2.plot(n[by_evidence], log_ev[by_evidence], "*", ms=16, c="C3", label="largest evidence")
    _a2.set_xscale("log"); _a2.set_xlabel("basis size"); _a2.set_ylabel("log-evidence"); _a2.legend(frameon=False)
    _fig.tight_layout()
    _fig
    return by_evidence, by_test, ev_test, ls_test, ls_train, n


@app.cell(hide_code=True)
def _(by_test, ls_test, ls_train, mo, n, np):
    _ratio = ls_test[-1] / ls_train[-1]
    _falls = bool(np.all(np.diff(ls_train) < 0))
    _ok1 = _falls and _ratio > 3
    mo.callout(
        mo.md(f"**Checkpoint 1 passed:** least squares fits its training set better with every basis "
              f"(down to {ls_train[-1]:.2f} meV/atom), but at {n[-1]} functions its test error is "
              f"{_ratio:.1f}× its training error, and the test error is lowest at {n[by_test]} functions: "
              "the largest basis has started fitting noise.")
        if _ok1 else mo.md(f"**Checkpoint 1:** least squares does not overfit here (test/train ratio "
                           f"{_ratio:.1f} at the largest basis)."),
        kind="success" if _ok1 else "warn")
    return


@app.cell(hide_code=True)
def _(by_evidence, by_test, ls_test, mo, n, sweep):
    _ok2 = by_evidence < len(sweep) - 1 and abs(ls_test[by_evidence] - ls_test[by_test]) <= 0.1 * ls_test[by_test]
    mo.callout(
        mo.md(f"**Checkpoint 2 passed:** the log-evidence is largest at {n[by_evidence]} functions "
              f"(degree {sweep[by_evidence]['degree']}), the basis the held-out test set also prefers, "
              "found without looking at the test set.")
        if _ok2 else mo.md(f"**Checkpoint 2:** the evidence chose {n[by_evidence]} functions and the test "
                           f"set {n[by_test]}."),
        kind="success" if _ok2 else "warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the evidence is doing

    The log-evidence $\log p(\text{data} \mid \text{basis})$ is the
    probability of the training data under the model, averaged over every
    coefficient vector the prior allows rather than taken at the best one. A
    larger basis can always fit the data better, but it spreads its prior
    over many more coefficient vectors that fit badly, and the average pays
    for that (the "Occam factor"). So the evidence rises while extra
    functions explain real structure, and falls once they only explain
    noise.

    Two things to notice in the table:

    - The evidence fit's own test error stays nearly flat as the basis grows.
      It does not chase the training set (its training error is higher than
      least squares'), because its noise levels are fitted too: it is told how
      much of the data is noise.
    - Below the largest basis, least squares has the lower test error on this
      data set, for forces as well as energies (exercise 2). With 200 smooth
      bulk structures there is little noise to guard against, and the
      evidence chooses an almost vanishing prior. What it adds here is not a
      better fit at a given basis but a test error that does not turn up as
      the basis grows, and a way to choose the basis.

    The test set is still the final judge; the evidence lets you choose
    without spending one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. **Other knobs.** The basis has two more: the correlation order (`order`,
       2 to 4) and `wL` in `BasisSpec`, which trades angular against radial
       resolution. Compare them at matched basis size, not matched degree:
       otherwise you only learn that bigger models fit better. Does the
       log-evidence agree with the test set?
    2. **Forces.** Add the test force RMSE (`metrics["test/map"]["F"]["rmse"]`,
       and `"test/lstsq"` for least squares) to the table. Which fit is better on
       forces?
    3. **Less data.** Train on the first 40 of `train_frames`. Where does least
       squares start to overfit now? How noisy are the test errors on so small
       a set, and does the evidence still pick a sensible basis?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Hint for exercise 3": mo.md(
            "With 40 frames the least-squares test error is no longer smooth in the basis size: "
            "it can fall again at a larger basis by chance, because 50 test frames and 40 training "
            "frames give noisy estimates. The log-evidence still peaks at a moderate basis (degree "
            "12 in our run) and the evidence fit's test error stays smoother."),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - `FitConfig(solver="lstsq")` (`aj fit --solver lstsq`) is plain weighted
      least squares: no prior, no uncertainty. A large enough basis overfits.
    - The evidence fit reports `res.map.log_evidence`. It is comparable across
      bases fitted to the same data, and its maximum picks a basis size from
      the training data alone.

    Next: [Tutorial 7](https://acesuit.github.io/ace-jax/tutorials/truth-about-the-truth/)
    asks where the reference labels themselves come from.
    """)
    return


if __name__ == "__main__":
    app.run()
