# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ace-jax>=0.1.1",
#     "marimo>=0.25",
#     "matplotlib>=3.8",
# ]
# ///
"""Tutorial 7: automating curation -- sampling against selection.

Adapted from the MLIP-school-2026 notebook E3 (https://mlipschool.uk/e3/e3_automating_curation). Run it with
`uvx marimo edit --sandbox school_curation_si.py`, or as a plain script
(`python school_curation_si.py`), which is how CI smoke-tests it.
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 7: automating curation

    [Tutorial 6](https://acesuit.github.io/ace-jax/tutorials/surfaces/) found the
    missing surface data by hand. Here a loop finds it: run molecular
    dynamics with the current model, pick a few frames, label them, refit,
    and repeat. The question is how to pick. Three rules compete at the same
    label budget, and each is scored on one target: the Si(111) surface energy.

    **Goals**

    1. Write two selection rules: random sampling, and descriptor novelty.
    2. Run the loop with each, and with ace-jax's own uncertainty, and compare
       the error of the target against the labels spent.
    3. Read where each rule spent its labels.

    It builds on Tutorials 4 and 6 and is adapted from notebook E3 of the
    [MLIP School 2026](https://mlipschool.uk/e3/e3_automating_curation).

    **Run this notebook**: with [uv](https://docs.astral.sh/uv/) installed, one command
    opens it in your browser (no account needed):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/notebooks/school_curation_si.py
    ```

    or open it in [molab](https://molab.marimo.io/github/ACEsuit/ace-jax/blob/main/docs/user/tutorials/notebooks/school_curation_si.py), marimo's hosted service (free to
    preview; sign in to run). The documentation website shows a static copy,
    run on a CPU when the site was built. The three campaigns take about 5 minutes.
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

    from ace_jax.tutorials import campaign as C
    from ace_jax.tutorials import curation as K
    from ace_jax.tutorials import labels as L
    from ace_jax.tutorials import structures as T

    return C, K, L, T, mo, np, pathlib, plt, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: the campaign

    - **Start:** the bulk model of Tutorial 6 (ten strained and rattled
      cells), and a separate seed set of eight bulk cells that every refit
      starts from. The uncertainty rule needs an ARD fit (Step 2), so its
      start is the same data fitted with ARD, which predicts γ(111) a little
      differently: the three curves do not start at the same point.
    - **Pool:** each round runs 60 steps of 400 K Langevin dynamics with the
      current model from three starting points, in this order: bulk, a (110)
      slab (a distractor: the target does not depend on it) and the target's
      (111) slab, keeping every fourth frame after the start (the starting
      structures are the test structures, so they are never candidates).
      Frames with two atoms closer than 1.7 Å are dropped.
    - **Selection:** each round picks 4 frames, labels them, and refits the
      seed set plus everything picked so far. Two rounds: 8 labels.
    - **Score:** the error of the model's γ(111) against the labeller's.

    The MD pools of the run on this page, and labels for every frame in them,
    ship with the tutorial, so the loop replays them instead of rerunning MD
    (which differs from machine to machine). Running MD live needs the
    labeller (exercise 3).
    """)
    return


@app.cell
def _(L, mo, pathlib):
    BASE = "https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/school/"


    def shipped(rel):
        _p = (mo.notebook_dir() / "../data/school" / rel) if mo.notebook_dir() else None
        return _p if _p is not None and _p.exists() else BASE + rel


    _caches = [L.LabelCache.from_file(shipped(f)) for f in ("e1/labels-mpa-0.xyz", "c/labels-mpa-0.xyz",
                                                              "e3/labels-mpa-0.xyz")]


    class _Caches:
        def get(self, atoms, model):
            return next((h for c in _caches if (h := c.get(atoms, model)) is not None), None)


    _cache = _Caches()


    def label(xs):
        return L.label(xs, model="mpa-0", cache=_cache)


    from ace_jax.fit.xyz import read_extxyz
    from ase import Atoms

    shipped_pools = {}
    for _f in read_extxyz(str(L.fetch(shipped("e3/pools.xyz")))):
        _a = Atoms(numbers=_f.numbers, positions=_f.positions, cell=_f.cell, pbc=_f.pbc)
        _a.info.update({k: _f.info[k] for k in ("config_type", "miller") if k in _f.info})
        shipped_pools.setdefault(str(_f.info["driver"]), {}).setdefault(int(_f.info["round"]), []).append(_a)
    work = pathlib.Path("ace_jax_tutorial_7")
    work.mkdir(exist_ok=True)
    return label, shipped_pools, work


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: two selection rules

    Each rule gets the pool's descriptors (one array per frame, a row per
    atom, in the reference basis of Tutorial 6), the training set's, the
    number of frames to pick, and a random generator, and returns the indices
    of its picks.

    - `pick_random`: any `count` distinct frames, uniformly at random.
    - `pick_novel`: score each frame by its most unfamiliar atom, the largest
      over its atoms of the distance to the nearest training atom, and take
      the `count` highest scores.
    """)
    return


@app.cell
def _(np):
    def pick_random(pool_descriptors, training_descriptors, count, rng):
        return [int(i) for i in rng.choice(len(pool_descriptors), size=count, replace=False)]


    def pick_novel(pool_descriptors, training_descriptors, count, rng):
        T = np.concatenate([np.asarray(r, float) for r in training_descriptors])
        scores = [float(np.linalg.norm(np.asarray(r, float)[:, None, :] - T[None], axis=2).min(1).max())
                  for r in pool_descriptors]
        return [int(i) for i in np.argsort(-np.asarray(scores), kind="stable")[:count]]

    return pick_novel, pick_random


@app.cell(hide_code=True)
def _(np, pick_novel, pick_random):
    from ace_jax.tutorials import campaign as _C
    _rng = np.random.default_rng(0)
    _train = [np.zeros((4, 6)), np.full((4, 6), 0.01), np.full((4, 6), 0.02), np.full((1, 6), 25.0)]
    _pool = [np.full((3, 6), x) for x in (0.02, 3.0, 25.05, 12.0)]
    _r = pick_random(_pool, _train, 2, _rng)
    _n = pick_novel(_pool, _train, 2, _rng)
    _ok = (len(set(_r)) == 2 and all(0 <= i < 4 for i in _r) and _n[0] == 3 and set(_n) == {1, 3}
           and _n == [int(i) for i in np.argsort(_C.score_novelty(_pool, _train))[::-1][:2]])
    import marimo as _mo
    _mo.callout(
        _mo.md("**Checkpoint 1 passed:** on a test pool, `pick_random` returns two distinct frames and "
               "`pick_novel` picks the frame farthest from every training atom first (frame 3, though "
               "frame 2 sits next to one lone outlier).")
        if _ok else _mo.md("**Checkpoint 1:** a rule returned the wrong frames on the test pool: "
                           f"random {_r}, novel {_n} (expected 3 then 1)."),
        kind="success" if _ok else "danger",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The third rule is ace-jax's own: fit with ARD (`uq="ard"`, see the
    [ASE calculator how-to](https://acesuit.github.io/ace-jax/howto/ase/)) and
    pick the frames whose largest per-atom force uncertainty, `forces_std`,
    is highest. It asks the model where it is unsure, rather than where the
    data are thin.
    """)
    return


@app.cell
def _(C, K, label, pick_novel, pick_random, shipped_pools, time, work):
    def as_campaign_pick(rule):
        """Adapt a Step 2 rule to the loop: descriptors in the reference basis."""
        _B = C.reference_basis()

        def pick(pool, train, count, rng, model_file, posterior):
            return rule(C.atom_descriptors(pool, _B), C.atom_descriptors(train, _B), count, rng)
        return pick


    rules = {"random": as_campaign_pick(pick_random), "novelty": as_campaign_pick(pick_novel),
             "uncertainty": None}          # None: the loop's ARD rule
    _t = time.time()
    runs = {d: K.run_campaign(d, label, work / d, pools=shipped_pools[d], pick=rules[d]) for d in rules}
    campaign_seconds = time.time() - _t
    return campaign_seconds, runs


@app.cell
def _(campaign_seconds, mo, plt, runs):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    for _d, _r in runs.items():
        _h = _r["history"]
        _ax.semilogy([x["labels"] for x in _h], [x["err"] for x in _h], "o-", label=_d)
    _ax.set_xlabel("labels spent"); _ax.set_ylabel("|γ(111) error| (eV/Å²)"); _ax.legend(frameon=False)
    _ax.set_title(f"three campaigns, {campaign_seconds:.0f} s")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, runs):
    mo.md("| rule | start | after 4 labels | after 8 labels | frames picked |\n|---|---|---|---|---|\n"
          + "\n".join(f"| {d} | {r['history'][0]['err']:.2e} | {r['history'][1]['err']:.2e} | "
                      f"{r['history'][2]['err']:.2e} | {r['history'][1]['picks']}, {r['history'][2]['picks']} |"
                      for d, r in runs.items())
          + "\n\n|γ(111) error| in eV/Å². Frames are numbered within each round's pool: 0-14 from the "
          "bulk run, 15-29 from the (110) slab, 30-44 from the (111) slab (a live run may drop "
          "unphysical frames and shift these).")
    return


@app.cell(hide_code=True)
def _(mo, runs):
    _e = {d: r["history"][1]["err"] for d, r in runs.items()}
    _ok = _e["novelty"] < 0.1 * _e["random"]
    mo.callout(
        mo.md(f"**Checkpoint 2 passed:** after 4 labels, novelty selection is {_e['random'] / _e['novelty']:.0f}× "
              "closer to the target than random sampling.")
        if _ok else mo.md(f"**Checkpoint 2:** novelty ({_e['novelty']:.2e}) did not beat random "
                          f"({_e['random']:.2e}) by 10×."),
        kind="success" if _ok else "warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: where did the labels go?

    The pool is ordered: bulk frames first, then the (110) distractor, then
    the (111) target. So the picks say what each rule looked at.

    - **Random** spreads its labels over all three runs; only some land on
      the target's surface.
    - **Novelty** goes straight to the (111) slab's MD frames: their surface
      atoms are the farthest from the bulk-only training set in descriptor
      space.
    - **Uncertainty** spends its first round on the (110) distractor: that is
      where the model's forces are least certain, and it is right to be
      unsure there. But the target is (111), and the uncertainty rule does not
      know what the labels are for.

    On this problem novelty wins, by luck as much as by design: the target's
    surface happens to be the most novel thing in the pool. Neither
    novelty nor uncertainty knows the target. Point the rule at what matters
    (for example, only MD from the target's own structures), or label more
    per round, and the difference closes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reflection

    Is "most novel" the same as "most useful"? When would novelty selection
    waste labels? Think before opening the answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"A model answer": mo.md(
        "No. Novelty is distance from the data in descriptor space, which says nothing about "
        "whether a structure matters for the property you need. An unphysical frame from a "
        "badly behaved MD run, a high-energy collision or a configuration the simulation will "
        "never visit can be the most novel thing in the pool, and novelty selection will label "
        "it first. Descriptor distance is also only a proxy for what the model does not know; "
        "a committee or posterior uncertainty (the third rule here) measures that directly, "
        "but is just as blind to the target. A useful campaign aims the sampling at the "
        "simulations you will run, and filters the candidates (here, the 1.7 Å distance check) "
        "before any rule ranks them.")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. **Distance to what?** `pick_novel` scores against the whole training
       set. Score against only the last eight training structures (the
       school's version): do the picks change?
    2. **Budget.** Run one round of 8 picks instead of two of 4
       (`K.run_campaign(..., rounds=1, per_round=8)`: it replays the shipped
       first-round pool, every frame of which is labelled). Is one large round
       better or worse than two small ones?
    3. **Live MD.** With the labeller installed (`pip install mace-torch
       --extra-index-url https://download.pytorch.org/whl/cpu`), drop
       `pools=` from `run_campaign` to run the MD yourself, and try other
       seeds (`seed=1`, `seed=2`). Does the ranking of the rules hold?
    4. **Uncertainty-driven MD.** The `ase_uhal` package biases MD towards
       uncertain configurations (hyperactive learning), instead of filtering
       an ordinary MD run afterwards: see its
       [documentation](https://kermodegroup.github.io/ase_uhal/).

    ## Summary

    - A curation loop (MD, select, label, refit) automates the search of
      Tutorial 6.
    - Selection rules spend the same budget very differently; measure them on
      the property you need.
    - Novelty and uncertainty are blind to the target: aim the sampling.

    Next: [Tutorial 8](https://acesuit.github.io/ace-jax/tutorials/truth-about-the-truth/)
    asks where the labels themselves come from.
    """)
    return


if __name__ == "__main__":
    app.run()
