# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ace-jax>=0.1.0",
#     "marimo>=0.25",
#     "matplotlib>=3.8",
# ]
# ///
"""Tutorial 5: basis size, overfitting and the evidence.

Adapted from the MLIP-school-2026 notebook E1x (https://mlipschool.uk/e1x/e1x_basis_and_overfitting). Run it with
`uvx marimo edit --sandbox school_basis_si.py`, or as a plain script
(`python school_basis_si.py`), which is how CI smoke-tests it.
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 5: basis size, overfitting and the evidence

    In [Tutorial 4](https://acesuit.github.io/ace-jax/tutorials/dataset-and-properties/)
    a near-perfect fit got a vacancy wrong because the data did not cover the
    question. This time the data covers the question, and the risk is the model:
    how large a basis can 40 silicon structures support?

    **Goals**

    1. Hold out a test set, and fit the same data with a growing basis.
    2. Watch plain least squares fit its training set better and better while
       the gap to the test set opens.
    3. Fit the same bases with the evidence (Bayesian) fit, and use its
       log-evidence to choose the basis size without looking at the test set.

    It is adapted from notebook E1x of the
    [MLIP School 2026](https://mlipschool.uk/e1x/e1x_basis_and_overfitting).

    **Run this notebook**: with [uv](https://docs.astral.sh/uv/) installed, one command
    opens it in your browser (no account needed):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/notebooks/school_basis_si.py
    ```

    or open it in [molab](https://molab.marimo.io/github/ACEsuit/ace-jax/blob/main/docs/user/tutorials/notebooks/school_basis_si.py), marimo's hosted service (free to
    preview; sign in to run). The documentation website shows a static copy,
    run on a CPU when the site was built. The sweep in Step 2 fits eight models
    and takes about 4 minutes on a CPU.
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
    16 atoms) from the MLIP school, labelled with MACE-MPA-0 (see Tutorial 4)
    under the keys `energy`, `forces` and `virial`. Nothing here labels
    anything new.

    A test set is the structures the fit never sees. Shuffle before splitting,
    since the file is ordered by cell size, and hold out 20%. Of the other 200,
    the fits below train on only `N_TRAIN = 40`: few enough data that a large
    basis can overfit them (exercise 3 trains on all 200).
    """)
    return


@app.cell
def _(mo, np, pathlib, read, urllib):
    URL = ("https://raw.githubusercontent.com/ACEsuit/ace-jax/main/"
           "docs/user/tutorials/data/school/e1x/reference.xyz")
    work = pathlib.Path("ace_jax_tutorial_5")
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
    N_TRAIN = 40                                       # of the 200; exercise 3 sets it to 200
    train_frames = [frames[i] for i in _order[:_cut]][:N_TRAIN]
    test_frames = [frames[i] for i in _order[_cut:]]
    mo.md(f"**{len(frames)}** structures: **{len(train_frames)}** for training (of {_cut}), "
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
    - **the evidence fit** (the default, as in Tutorials 1 and 4): Bayesian
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
    by_test_ev = int(np.nanargmin(ev_test))                # ... and for the evidence fit, the basis it chooses for
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
    return by_evidence, by_test, by_test_ev, ev_test, ls_test, ls_train, n


@app.cell(hide_code=True)
def _(by_test, ls_test, ls_train, mo, n, np):
    _ratio = ls_test[-1] / ls_train[-1]
    _falls = bool(np.all(np.diff(ls_train) < 0))
    _ok1 = _falls and _ratio > 3 and by_test < len(n) - 1
    mo.callout(
        mo.md(f"**Checkpoint 1 passed:** least squares fits its training set better with every basis "
              f"(down to {ls_train[-1]:.2f} meV/atom), but its test error is lowest at {n[by_test]} "
              f"functions and rises beyond; at {n[-1]} functions it is {_ratio:.1f}× the training error: "
              "the larger bases fit noise.")
        if _ok1 else mo.md(f"**Checkpoint 1:** least squares does not overfit here (test/train ratio "
                           f"{_ratio:.1f} at the largest basis)."),
        kind="success" if _ok1 else "warn")
    return


@app.cell(hide_code=True)
def _(by_evidence, by_test_ev, ev_test, mo, n, sweep):
    # the evidence picks a basis for the evidence fit, so it is judged by that fit's test error
    _ok2 = abs(ev_test[by_evidence] - ev_test[by_test_ev]) <= 0.1 * ev_test[by_test_ev]
    mo.callout(
        mo.md(f"**Checkpoint 2 passed:** the log-evidence is largest at {n[by_evidence]} functions "
              f"(degree {sweep[by_evidence]['degree']}), "
              + (f"where the evidence fit also tests best ({ev_test[by_evidence]:.2f} meV/atom)"
                 if by_evidence == by_test_ev else
                 f"where the evidence fit tests at {ev_test[by_evidence]:.2f} meV/atom, against its "
                 f"best of {ev_test[by_test_ev]:.2f}")
              + ": a basis chosen without looking at the test set.")
        if _ok2 else mo.md(f"**Checkpoint 2:** the evidence chose {n[by_evidence]} functions; the evidence "
                           f"fit tests best at {n[by_test_ev]}."),
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
    """)
    return


@app.cell(hide_code=True)
def _(by_evidence, by_test, ev_test, ls_test, mo, n):
    mo.md(f"""
    Two things to notice in the table:

    - **Tuned on the test set, least squares does best:** {ls_test[by_test]:.2f} meV/atom at
      {n[by_test]} functions. But that choice spent the test set. The evidence picks
      {n[by_evidence]} functions from the training data alone, where the evidence fit tests at
      {ev_test[by_evidence]:.2f} meV/atom.
    - **Past the best size, the evidence fit degrades more gently.** At {n[-1]} functions
      least squares tests at {ls_test[-1]:.2f} meV/atom, {ls_test[-1] / ls_test[by_test]:.1f}× its
      best; the evidence fit at {ev_test[-1]:.2f}, {ev_test[-1] / ev_test.min():.1f}× its best. Its
      noise levels are fitted too, so it is told how much of the data is noise and does not chase
      it (exercise 2 compares the forces).

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
    3. **More data.** Set `N_TRAIN = 200` in Step 1, five times the data. Does
       least squares still overfit within this sweep? Where does the
       log-evidence peak now?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Hint for exercise 3": mo.md(
            "With 200 frames least squares no longer overfits in this sweep: its test error falls "
            "with every basis, to about 0.9 meV/atom at 684 functions, and the log-evidence rises "
            "all the way to the largest basis too. Five times the data supports every basis here; "
            "to see the evidence turn over you would need a larger one (degree 18 has over 1,000 "
            "functions)."),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - `FitConfig(solver="lstsq")` (`aj fit --solver lstsq`) is plain weighted
      least squares: no prior, no uncertainty. A basis too large for the data overfits.
    - The evidence fit reports `res.map.log_evidence`. It is comparable across
      bases fitted to the same data, and its maximum picks a basis size from
      the training data alone.

    Next: [Tutorial 6](https://acesuit.github.io/ace-jax/tutorials/surfaces/)
    finds what a bulk data set is missing for a surface, and repairs it.
    """)
    return


if __name__ == "__main__":
    app.run()
