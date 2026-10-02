# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     # TODO(pypi): replace this git requirement with "ace-jax" once ace-jax is on PyPI.
#     "ace-jax @ git+https://github.com/ACEsuit/ace-jax@main",
#     "marimo>=0.25",
#     "matplotlib>=3.8",
# ]
# ///
"""Tutorial 3: multi-element fits, the categorical and the species-embedding basis.

Run it with `uvx marimo edit --sandbox multi_element.py`, or as a plain script
(`python multi_element.py`), which is how CI smoke-tests it.
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 3: multi-element fits, categorical and embedded species

    An ACE basis for one element is a set of functions of the neighbour
    positions. With several elements it must also say *which* element each
    neighbour is, and there are two ways to do that. The **categorical** basis
    gives every element its own channel, so its size grows quickly with the
    number of elements. The **species-embedding** basis describes each element
    by a few numbers and lets all elements share the same channels. This
    notebook fits both to the five-element CrMnFeCoNi (Cantor) alloy and
    compares them.

    **Goals**

    1. Read a multi-element dataset and check its composition.
    2. Build a categorical basis and see how its size grows with the number of elements.
    3. Build species-embedding bases: a one-hot embedding that keeps every element
       distinct, and a compressed one that describes the elements by two channels.
    4. Fit all three with the evidence fit, and compare basis size, fit time,
       test errors and parity plots, in bulk and around a vacancy.
    5. Refit with a quarter of the data, and decide when each basis wins.

    Work through [Tutorial 1](https://acesuit.github.io/ace-jax/tutorials/first-fit/)
    first; this one uses the same fit and explains only what is new for several elements.

    **Run this notebook**: with [uv](https://docs.astral.sh/uv/) installed, one command
    opens it in your browser (no account needed):

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/notebooks/multi_element.py
    ```

    or open it in [molab](https://molab.marimo.io/github/ACEsuit/ace-jax/blob/main/docs/user/tutorials/notebooks/multi_element.py), marimo's hosted service (free to
    preview; sign in to run). The documentation website shows a static copy,
    run on a CPU when the site was built: there the interactive controls show
    their default values. The notebook fits six models and runs in about
    three minutes on a CPU, most of it the fits.
    """)
    return


@app.cell
def _():
    import json
    import pathlib
    import time
    import urllib.request
    from collections import Counter

    import jax

    jax.config.update("jax_enable_x64", True)  # fitting needs float64

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from ase.io import read, write

    from ace_jax import ACECalculator
    from ace_jax.basis.model import BasisSpec, build_basis

    return (ACECalculator, BasisSpec, Counter, build_basis, json, mo, np, pathlib, plt, read, time,
            urllib, write)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: the data

    The Cantor alloy is a random face-centred-cubic solid solution of Cr, Mn,
    Fe, Co and Ni. The data are small subsets shipped with ace-jax
    ([data README](https://github.com/ACEsuit/ace-jax/blob/main/docs/user/tutorials/data/cantor/README.md)),
    labelled by the MACE-MH-1 foundation model and stored under `mace_energy`,
    `mace_force` and `mace_virial`:

    - **training set**: 40 strained and rattled bulk cells of 32 atoms;
    - **test set**: 30 more bulk cells from the same distribution (`config_type=bulk`),
      plus 30 cells of 47 atoms, each a 48-atom cell with one atom removed and not relaxed
      (`config_type=vacancy`). The training set has no vacancies.

    The cell below uses the copies in your ace-jax checkout if this notebook is
    run from one, and otherwise downloads them from GitHub.
    """)
    return


@app.cell
def _(mo, pathlib, read, urllib, write):
    work = pathlib.Path("ace_jax_tutorial_3")
    work.mkdir(exist_ok=True)
    URL = "https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/data/cantor/"
    _here = mo.notebook_dir() / "../data/cantor" if mo.notebook_dir() else None
    for _name in ("cantor_train.xyz", "cantor_test.xyz", "cantor_vacancy.xyz"):
        _local = _here / _name if _here is not None else None
        if _local is not None and _local.exists():
            (work / _name).write_bytes(_local.read_bytes())
        elif not (work / _name).exists():
            urllib.request.urlretrieve(URL + _name, work / _name)
    train_frames = read(work / "cantor_train.xyz", ":")
    _bulk, _vac = read(work / "cantor_test.xyz", ":"), read(work / "cantor_vacancy.xyz", ":")
    for _a in train_frames + _bulk:
        _a.info["config_type"] = "bulk"
    for _a in _vac:
        _a.info["config_type"] = "vacancy"
    test_frames = _bulk + _vac
    train_file, test_file = work / "train.xyz", work / "test.xyz"
    write(train_file, train_frames)
    write(test_file, test_frames)
    keys = dict(energy_key="mace_energy", force_key="mace_force", virial_key="mace_virial")
    return keys, test_file, test_frames, train_file, train_frames, work


@app.cell
def _(Counter, mo, np, train_frames):
    elements = ("Cr", "Mn", "Fe", "Co", "Ni")
    _counts = np.array([[Counter(a.get_chemical_symbols())[e] for e in elements] for a in train_frames])
    _frac = _counts / _counts.sum(axis=1, keepdims=True)
    composition = {e: (_counts[:, i].sum(), _frac[:, i].min(), _frac[:, i].max())
                   for i, e in enumerate(elements)}
    mo.md(
        "| element | atoms in the training set | share of the atoms | share per cell (min to max) |\n"
        "|---|---|---|---|\n"
        + "\n".join(f"| {e} | {n} | {n / _counts.sum():.1%} | {lo:.0%} to {hi:.0%} |"
                    for e, (n, lo, hi) in composition.items())
        + f"\n\n**{len(train_frames)}** training cells, **{_counts.sum()}** atoms, "
        f"about {_counts.sum() // len(elements)} atoms of each element."
    )
    return composition, elements


@app.cell(hide_code=True)
def _(composition, mo, test_frames, train_frames):
    _total = sum(n for n, _, _ in composition.values())
    _types = sorted({a.info["config_type"] for a in test_frames})
    _ok = (len(composition) == 5 and all(0.15 < n / _total < 0.25 for n, _, _ in composition.values())
           and _types == ["bulk", "vacancy"] and len(train_frames) == 40)
    mo.callout(
        mo.md("**Checkpoint 1 passed:** five elements in near-equal amounts in 40 training cells, "
              "and a test set of bulk and vacancy cells.")
        if _ok else mo.md("**Checkpoint 1:** the data are not as described: check the download."),
        kind="success" if _ok else "danger",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every cell is close to equiatomic: each element makes up 20% of the atoms,
    and no cell strays far from that. Keep this in mind for Step 6: these
    data can say how well a basis interpolates between near-equiatomic cells,
    but not how it transfers to compositions it has not seen.

    ## Step 2: the categorical basis

    An ACE basis function couples the radial and angular functions of up to
    `order` neighbours. In the **categorical** basis the radial functions
    $R_{nl}(r, Z_i, Z_j)$ carry the element of the centre atom, $Z_i$, and of
    the neighbour, $Z_j$, so each neighbour slot of a basis function also picks
    an element. A function of order $\nu$ then comes in a copy for every
    unordered choice of $\nu$ neighbour elements, and every centre element has
    its own coefficients. With $S$ elements the number of coefficients grows
    roughly like $S^{\nu+1}$.

    This is the default: `aj fit --order 2 --max-degree 4` with five elements in
    the data builds it. The table counts the coefficients (many-body plus pair,
    for every centre element) as elements are added, at the settings used in
    this notebook (order 2, degree 4) and at order 3.
    """)
    return


@app.cell
def _(BasisSpec, build_basis, elements, mo):
    growth = {}
    for _order in (2, 3):
        for _n in range(1, 6):
            _b = build_basis(BasisSpec(order=_order, max_degree=4, elements=elements[:_n], rcut=5.5))
            growth[_order, _n] = _b.meta["len_basis"]
    mo.md(
        "| elements | " + " | ".join(",".join(elements[:n]) for n in range(1, 6)) + " |\n"
        "|---|" + "---|" * 5 + "\n"
        + "\n".join(f"| order {o}, degree 4 | " + " | ".join(str(growth[o, n]) for n in range(1, 6)) + " |"
                    for o in (2, 3))
    )
    return (growth,)


@app.cell(hide_code=True)
def _(growth, mo):
    _r2, _r3 = growth[2, 5] / growth[2, 1], growth[3, 5] / growth[3, 1]
    _ok = _r2 > 10 and _r3 > _r2
    mo.callout(
        mo.md(f"**Checkpoint 2 passed:** five elements need {_r2:.0f} times the coefficients of one "
              f"at order 2, and {_r3:.0f} times at order 3: far faster than the number of elements.")
        if _ok else mo.md(f"**Checkpoint 2:** the categorical basis grew by only {_r2:.1f} times."),
        kind="success" if _ok else "warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: species embeddings

    The species-embedding basis replaces the element label of a neighbour by
    a short vector, its **embedding** $e_k(Z_j)$, $k = 1, \dots, d$. The
    radial functions become products,

    $$R_{nkl}(r, Z_j) = Q_{nl}(r)\, e_k(Z_j),$$

    so a basis function sees $d$ **species channels** in place of $S$
    elements. Two elements with similar embeddings look similar to every basis
    function, and the coefficients of a channel are shared by all elements.
    The centre element still has its own coefficients. The embedding is fixed
    when the basis is built (it is not fitted), so the model stays linear.

    With $d$ as large as all the element combinations need (`d_max=None`, the
    default) the embedding loses nothing. An identity table, one row per
    element (`embedding="identity"`), keeps every element distinct, so this
    **one-hot embedding** basis holds the same species information as the
    categorical basis, in a different construction. With a small `d_max` and
    a table that says how the elements resemble each other, the embedding
    **compresses** the elements into a few channels.

    The table can come from anywhere: the element embedding of a foundation
    model such as MACE is one choice. Here it is three columns everyone can
    check: a constant (every neighbour is a 3d transition metal), the number
    of valence electrons and the Pauling electronegativity, each shifted and
    scaled to mean 0 and spread 1 over the five elements. ace-jax normalises
    each row and keeps its `d_max` leading principal components. The table is
    a JSON file with the atomic numbers `Z` and the rows `emb`.
    """)
    return


@app.cell
def _(elements, json, mo, np, work):
    from ase.data import atomic_numbers

    _valence = [6, 7, 8, 9, 10]                       # 3d + 4s electrons
    _pauling = [1.66, 1.55, 1.83, 1.88, 1.91]         # Pauling electronegativity
    _props = np.array([_valence, _pauling], float).T
    _std = (_props - _props.mean(axis=0)) / _props.std(axis=0)
    table = np.hstack([np.ones((5, 1)), _std])
    Z = [atomic_numbers[e] for e in elements]
    table_file = work / "elements.json"
    table_file.write_text(json.dumps({"Z": Z, "emb": table.round(6).tolist()}))
    mo.md("| element | constant | valence electrons (scaled) | electronegativity (scaled) |\n|---|---|---|---|\n"
          + "\n".join(f"| {e} | {r[0]:.0f} | {r[1]:+.2f} | {r[2]:+.2f} |" for e, r in zip(elements, table))
          + f"\n\nWritten to `{table_file}`.")
    return Z, table, table_file


@app.cell
def _(Z, elements, np, plt, table):
    from ace_jax.basis.embedding import embedding_rows

    _rows = embedding_rows(table, Z, Z, d=2)             # what d_max=2 keeps, rows normalised
    _fig, _ax = plt.subplots(figsize=(4.2, 4))
    _t = np.linspace(0, 2 * np.pi, 200)
    _ax.plot(np.cos(_t), np.sin(_t), "-", c="0.85", lw=0.8)
    _offset = {"Cr": (8, 6), "Mn": (-22, -14)}           # Cr and Mn nearly coincide
    for _e, (_x, _y) in zip(elements, _rows):
        _ax.plot(_x, _y, "o", ms=8)
        _ax.annotate(_e, (_x, _y), textcoords="offset points", xytext=_offset.get(_e, (6, 4)))
    _ax.set_aspect("equal"); _ax.set_xlabel("channel 1"); _ax.set_ylabel("channel 2")
    _ax.set_title("the elements in two channels")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In two channels the five elements sit on a circle (each row has unit
    length): Cr and Mn almost on top of each other, Co and Ni close together on
    the other side, and Fe in between. Every basis function now sees a
    neighbour's element as a point on this circle, so it barely tells Cr from
    Mn.
    Choose the number of channels of the compressed basis below (exercise 1
    tries the others), then build the three bases. All use correlation order 2,
    degree 4 and the same cutoff, 5.5 Å.
    """)
    return


@app.cell
def _(mo):
    d_max = mo.ui.dropdown(options=["1", "2", "3"], value="2", label="channels d_max")
    d_max
    return (d_max,)


@app.cell
def _(BasisSpec, build_basis, d_max, elements, mo, table_file):
    specs = {
        "categorical": BasisSpec(order=2, max_degree=4, elements=elements, rcut=5.5),
        "one-hot embedding": BasisSpec(order=2, max_degree=4, elements=elements, rcut=5.5,
                                       embedding="identity"),
        "compressed embedding": BasisSpec(order=2, max_degree=4, elements=elements, rcut=5.5,
                                          embedding=str(table_file), d_max=int(d_max.value)),
    }
    bases = {k: build_basis(s) for k, s in specs.items()}
    mo.md("| basis | many-body functions per element | pair functions per element | coefficients |\n"
          "|---|---|---|---|\n"
          + "\n".join(f"| {k} | {b.meta['n_B']} | {b.meta['n_pair']} | **{b.meta['len_basis']}** |"
                      for k, b in bases.items()))
    return (bases,)


@app.cell(hide_code=True)
def _(bases, mo):
    _n = {k: b.meta["len_basis"] for k, b in bases.items()}
    _ok = (_n["compressed embedding"] < 0.5 * _n["categorical"]
           and abs(_n["one-hot embedding"] / _n["categorical"] - 1) < 0.25)
    mo.callout(
        mo.md(f"**Checkpoint 3 passed:** the one-hot embedding is about as large as the categorical "
              f"basis ({_n['one-hot embedding']} against {_n['categorical']} coefficients); the "
              f"compressed one has {_n['compressed embedding'] / _n['categorical']:.0%} of them.")
        if _ok else mo.md("**Checkpoint 3:** the basis sizes are not as expected."),
        kind="success" if _ok else "warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The two embedding bases differ from the categorical one in more than the
    species: they are built like the ACE1 models of ACEpotentials, with
    different radial polynomials, and with a pair potential that resolves the
    neighbour's element (note the pair functions per element in the table).
    The one-hot embedding is the control: it has the species information of
    the categorical basis and the construction of the compressed one, so
    comparing the three separates the effect of the construction from the
    effect of compressing the elements.

    ## Step 4: fit all three

    The fit is the evidence fit of Tutorial 1, the same for every basis:
    energies, forces and virials, noise levels and prior chosen by maximising
    the evidence, reference energies E0 by least squares. The test set holds
    both config types, so the fit reports its errors per type.

    The command-line equivalents, with the files this notebook writes to
    `ace_jax_tutorial_3/`, are

    ```bash
    aj fit --order 2 --max-degree 4 --rcut 5.5 --train train.xyz --test test.xyz \
        --energy-key mace_energy --force-key mace_force --virial-key mace_virial \
        --e0 lsq --m-per-species 0 --opt lbfgs --out fit_categorical

    aj fit --order 2 --max-degree 4 --rcut 5.5 --basis-embedding identity \
        --train train.xyz --test test.xyz \
        --energy-key mace_energy --force-key mace_force --virial-key mace_virial \
        --e0 lsq --m-per-species 0 --opt lbfgs --out fit_onehot

    aj fit --order 2 --max-degree 4 --rcut 5.5 --basis-embedding elements.json --d-max 2 \
        --train train.xyz --test test.xyz \
        --energy-key mace_energy --force-key mace_force --virial-key mace_virial \
        --e0 lsq --m-per-species 0 --opt lbfgs --out fit_compressed
    ```

    and `aj basis` saves a basis on its own, for example the compressed one:

    ```bash
    aj basis --elements Cr,Mn,Fe,Co,Ni --order 2 --max-degree 4 --rcut 5.5 \
        --embedding elements.json --d-max 2 --out compressed_basis.npz
    ```
    """)
    return


@app.cell
def _(keys, test_file, time, work):
    from ace_jax.fit.pipeline import FitConfig, fit, load_fit_data, save_model
    from ace_jax.fit.report import rmse_by_type

    def fit_basis(basis, train, out):
        """The evidence fit of Tutorial 1 on `train`, scored on the bulk + vacancy test set."""
        _t = time.time()
        _cfg = FitConfig(model=basis, arm="linear", m_per_species=0, e0="lsq", opt="lbfgs", r0=None,
                         rungs=("map",), predict_stats="recompute", predict_train=False,
                         **keys).validate()
        _log = []
        _res = fit(_cfg, load_fit_data(_cfg, train=str(train), test=str(test_file), log=_log.append),
                   log=_log.append)
        _a = _res.preds.arrays["test/map"]
        _types = [c.config_type for c in _res.data.test_o]
        return dict(
            seconds=time.time() - _t, arrays=_a, types=_types, n=basis.meta["len_basis"],
            log_sigma_E=float(_res.theta.log_sigma_E),        # the evidence's energy noise level
            rmse=rmse_by_type(_types, _a["nat"], _a["E"], _a["E_mean"], _a["F"], _a["F_mean"],
                              _a["V"], _a["V_mean"]),
            table=next(s for s in _log if isinstance(s, str) and s.startswith("RMSE, test")),
            model_file=save_model(_res, work / out, log=_log.append))

    return (fit_basis,)


@app.cell
def _(bases, fit_basis, train_file):
    fits = {k: fit_basis(b, train_file, "fit_" + k.split()[0].replace("-", "")) for k, b in bases.items()}
    return (fits,)


@app.cell
def _(fits, mo):
    def comparison(fits):
        """A Markdown table of basis size, fit time and the per-type test errors."""
        return ("| basis | coefficients | fit time (s) | bulk E (meV/atom) | bulk F (eV/Å) "
                "| vacancy E (meV/atom) | vacancy F (eV/Å) |\n|---|---|---|---|---|---|---|\n"
                + "\n".join(f"| {k} | {f['n']} | {f['seconds']:.0f} | {f['rmse']['bulk']['E']:.1f} "
                            f"| {f['rmse']['bulk']['F']:.3f} | {f['rmse']['vacancy']['E']:.1f} "
                            f"| {f['rmse']['vacancy']['F']:.3f} |" for k, f in fits.items()))

    mo.md(comparison(fits))
    return (comparison,)


@app.cell(hide_code=True)
def _(fits, mo):
    mo.md("The fit's own error table for the compressed basis, as `aj fit` prints it:\n\n"
          f"```text\n{fits['compressed embedding']['table']}\n```")
    return


@app.cell(hide_code=True)
def _(fits, mo):
    _b = {k: f["rmse"]["bulk"] for k, f in fits.items()}
    _bestE, _bestF = min(r["E"] for r in _b.values()), min(r["F"] for r in _b.values())
    _c = _b["compressed embedding"]
    _ok = (all(r["F"] < 0.25 for r in _b.values())
           and _c["E"] < 1.5 * _bestE and _c["F"] < 1.25 * _bestF)
    mo.callout(
        mo.md(f"**Checkpoint 4 passed:** every fit has a bulk force error below 0.25 eV/Å, and the "
              f"compressed basis is within {_c['E'] / _bestE - 1:.0%} of the best energy error and "
              f"{_c['F'] / _bestF - 1:.0%} of the best force error.")
        if _ok else mo.md("**Checkpoint 4:** the compressed basis is well behind the best fit, or a fit "
                          "failed. With one channel that is the point of exercise 1."),
        kind="success" if _ok else "warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5: parity plots

    The fit's own predictions for the test set, one column per basis, bulk
    and vacancy cells in different colours.
    """)
    return


@app.cell
def _(fits, np, plt):
    _fig, _axes = plt.subplots(2, 3, figsize=(11, 7))
    for _j, (_k, _f) in enumerate(fits.items()):
        _a, _t = _f["arrays"], np.array(_f["types"])
        _at = np.repeat(_t, _a["nat"])
        for _row, _ref, _fit, _sel in ((0, _a["E"] / _a["nat"], _a["E_mean"] / _a["nat"], _t),
                                       (1, _a["F"].ravel(), _a["F_mean"].ravel(), np.repeat(_at, 3))):
            _ax = _axes[_row, _j]
            for _ct in ("bulk", "vacancy"):
                _m = _sel == _ct
                _ax.plot(_ref[_m], _fit[_m], "o", ms=5 - 2 * _row, alpha=0.6, label=_ct)
            _lo, _hi = _ref.min(), _ref.max()
            _ax.plot([_lo, _hi], [_lo, _hi], "k-", lw=0.8)
        _axes[0, _j].set_title(_k)
        _axes[0, _j].set_xlabel("MACE energy (eV/atom)"); _axes[1, _j].set_xlabel("MACE force (eV/Å)")
    _axes[0, 0].set_ylabel("ACE energy (eV/atom)"); _axes[1, 0].set_ylabel("ACE force (eV/Å)")
    _axes[0, 0].legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The fitted models are ordinary `model.npz` files. As a check, evaluate
    the compressed model with `ACECalculator` on the first test cell and
    compare with the fit's prediction.
    """)
    return


@app.cell
def _(ACECalculator, fits, test_frames):
    _f = fits["compressed embedding"]
    _atoms = test_frames[0].copy()
    _atoms.calc = ACECalculator(str(_f["model_file"]), skin=0)
    calc_gap = abs(_atoms.get_potential_energy() - _f["arrays"]["E_mean"][0]) / len(_atoms)
    print(f"calculator vs fit, first test cell: {calc_gap:.1e} eV/atom")
    return (calc_gap,)


@app.cell(hide_code=True)
def _(calc_gap, mo):
    _ok = calc_gap < 1e-6
    mo.callout(
        mo.md("**Checkpoint 5 passed:** the calculator reproduces the fit's prediction.")
        if _ok else mo.md(f"**Checkpoint 5:** the calculator differs from the fit by {calc_gap:.1e} eV/atom."),
        kind="success" if _ok else "danger",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6: less data per element

    A compressed basis has fewer coefficients to determine, so it should need
    less data. Refit all three bases on the first few training cells only
    (a quarter of them by default), with the same test set.
    """)
    return


@app.cell
def _(mo):
    n_small = mo.ui.dropdown(options=["5", "10", "20"], value="10", label="training cells")
    n_small
    return (n_small,)


@app.cell
def _(bases, fit_basis, n_small, train_frames, work, write):
    small_file = work / "train_small.xyz"
    write(small_file, train_frames[:int(n_small.value)])
    small_fits = {k: fit_basis(b, small_file, "fit_small_" + k.split()[0].replace("-", ""))
                  for k, b in bases.items()}
    return (small_fits,)


@app.cell
def _(comparison, mo, n_small, small_fits):
    mo.md(f"Trained on **{n_small.value}** cells (about {int(n_small.value) * 32 // 5} atoms of each "
          "element):\n\n" + comparison(small_fits))
    return


@app.cell(hide_code=True)
def _(fits, mo, small_fits):
    _E = lambda f, k: f[k]["rmse"]["bulk"]["E"]
    _c, _o = "compressed embedding", "one-hot embedding"
    _ok = (_E(small_fits, _c) < 0.7 * _E(small_fits, _o) and _E(small_fits, _c) < _E(small_fits, "categorical")
           and _E(small_fits, _c) / _E(fits, _c) < _E(small_fits, _o) / _E(fits, _o))
    mo.callout(
        mo.md(f"**Checkpoint 6 passed:** with less data the compressed basis keeps its energy error "
              f"({_E(fits, _c):.1f} to {_E(small_fits, _c):.1f} meV/atom), while the one-hot embedding's "
              f"grows from {_E(fits, _o):.1f} to {_E(small_fits, _o):.1f} meV/atom.")
        if _ok else mo.md("**Checkpoint 6:** the compressed basis did not hold its energy error better "
                          "than the one-hot embedding with less data."),
        kind="success" if _ok else "warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 7: when does each one win?

    The two tables answer most of this. The numbers below are this run's; the
    sentences are written for the default settings (two channels, 10 cells in
    Step 6).
    """)
    return


@app.cell(hide_code=True)
def _(fits, growth, mo, n_small, small_fits):
    _r = lambda f, k, t="bulk": f[k]["rmse"][t]
    _cat, _one, _cmp = "categorical", "one-hot embedding", "compressed embedding"
    _vac = {k: _r(fits, k, "vacancy") for k in fits}
    mo.md(f"""
- **The construction matters as much as the species.** The one-hot embedding
  has the species information of the categorical basis and a similar size
  ({fits[_one]['n']} against {fits[_cat]['n']} coefficients), yet its bulk energy error
  is {_r(fits, _one)['E']:.1f} against {_r(fits, _cat)['E']:.1f} meV/atom, and its force error
  {_r(fits, _one)['F']:.3f} against {_r(fits, _cat)['F']:.3f} eV/Å. That gap comes from the rest
  of the construction (the radial basis and the species-resolved pair
  potential), not from how the species are treated. Compare bases that differ
  in one thing before you credit a result to it.
- **With enough data, compression costs little.** On 40 cells the compressed
  basis, with {fits[_cmp]['n'] / fits[_one]['n']:.0%} of the one-hot embedding's coefficients, has
  errors of {_r(fits, _cmp)['E']:.1f} meV/atom and {_r(fits, _cmp)['F']:.3f} eV/Å, and fits in
  {fits[_cmp]['seconds']:.0f} s against {fits[_one]['seconds']:.0f} s.
- **With little data per element, compression wins in energy.** On
  {n_small.value} cells the compressed basis has a bulk energy error of
  {_r(small_fits, _cmp)['E']:.1f} meV/atom, against {_r(small_fits, _one)['E']:.1f} (one-hot embedding)
  and {_r(small_fits, _cat)['E']:.1f} (categorical). The forces, 96 labels per cell against
  one energy, are closer: {_r(small_fits, _cmp)['F']:.3f}, {_r(small_fits, _one)['F']:.3f} and
  {_r(small_fits, _cat)['F']:.3f} eV/Å. Exercise 2 finds where this stops.
- **Around a vacancy no basis is clearly ahead.** The vacancy cells, absent
  from the training set, have energy errors of {_vac[_cat]['E']:.1f}, {_vac[_one]['E']:.1f} and
  {_vac[_cmp]['E']:.1f} meV/atom (categorical, one-hot, compressed) and force errors of
  {_vac[_cat]['F']:.3f}, {_vac[_one]['F']:.3f} and {_vac[_cmp]['F']:.3f} eV/Å, all larger than in bulk.
  The parity plots show the two embedding fits placing the vacancy cells too
  low in energy. The training set has no vacancies; adding some is the remedy,
  whichever the basis.
- **More elements favour embeddings.** From one to five elements the
  categorical basis grows {growth[2, 5] / growth[2, 1]:.0f} times at order 2 and
  {growth[3, 5] / growth[3, 1]:.0f} times at order 3. A compressed embedding grows with its
  channels, not with the element combinations, and needs data in proportion.
- **Unseen compositions: not tested here.** Every cell is close to
  equiatomic, so these data cannot say how either basis transfers to, say, a
  Ni-rich alloy or a binary. What is certain is structural: a categorical
  coefficient for an element combination that never occurs in the training
  data is set by the prior alone, while an embedded basis shares its channels
  between elements. Whether that sharing predicts well depends on the table,
  and only a test on held-out compositions can say.
""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. **Channels.** Set *channels d_max* to 3, then to 1. With 3 the
       embedding keeps all of the table; with 1 it keeps only the side of
       the circle each element sits on. How do the errors change, and what
       is the energy noise level the evidence chose,
       `fits["compressed embedding"]["log_sigma_E"]` (the log of an energy in eV)?
    2. **Data per element.** Set *training cells* to 5 and to 20. Does the
       compressed basis keep its lead in energy at every size?
    3. **Another table.** Drop the electronegativity column from the table in
       Step 3 (keep the constant and the valence electrons) and refit. Does
       the compressed basis care which properties describe the elements?
    4. **Order 3.** Build the order-3 bases of Step 2 with an embedding
       (`BasisSpec(order=3, ...)`) and compare their sizes with the
       categorical ones. The 40 training cells hold 4120 labels (one energy,
       96 force components and 6 virial components each): how close does
       each order-3 basis come to that, and at what degree would the
       categorical one pass it?
    5. **Compressed cells.** Download `cantor_compressed.xyz` (the
       [data README](https://github.com/ACEsuit/ace-jax/blob/main/docs/user/tutorials/data/cantor/README.md)
       describes it) into `ace_jax_tutorial_3/` and score each model on it from there with
       `aj eval --model fit_categorical/model.npz --data cantor_compressed.xyz --energy-key mace_energy --force-key mace_force --virial-key mace_virial`.
       Which basis extrapolates to shorter bonds best?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "Hint for exercise 1": mo.md(
            "Three channels (265 coefficients) fit about as well as two: bulk errors of about "
            "10.9 meV/atom and 0.144 eV/Å. One channel (155 coefficients) keeps the forces "
            "(0.143 eV/Å) but loses the energies: the bulk energy error is about 1190 meV/atom. "
            "The evidence has given up on them: `log_sigma_E` is about +2.0 (an energy noise of "
            "about 7 eV per cell) against -3.1 with two channels, so the fit treats the energies "
            "as noise and fits the forces."),
        "Hint for exercise 2": mo.md(
            "Bulk energy errors (meV/atom) for categorical, one-hot and compressed: 56, 63 and 72 "
            "on 5 cells; 46, 30 and 14 on 10; 26, 16 and 10 on 20; 16, 9.6 and 10 on 40. On 5 cells "
            "the compressed basis loses its lead in energy, though its force error is still the "
            "smallest (0.21 eV/Å, against 0.23 and 0.34): five cells are too few to fit energies "
            "with any of these bases."),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    - `aj fit` builds a **categorical** basis for every element in the data;
      its size grows much faster than the number of elements.
    - `BasisSpec(embedding=...)`, or `--basis-embedding` on the command line,
      builds a **species-embedding** basis: `identity` keeps every element
      distinct, and a table of element properties with a small `d_max`
      compresses the elements into a few shared channels.
    - Compare bases that differ in one thing at a time: the one-hot
      embedding separates the construction from the compression.
    - Compression pays off when there is little data per element and many
      elements; whether it transfers to new compositions has to be tested on
      them.

    The [multi-element how-to](https://acesuit.github.io/ace-jax/howto/multi-element/)
    has the command-line recipe for this dataset, including out-of-distribution
    test sets. Next: [Tutorial 4](https://acesuit.github.io/ace-jax/tutorials/dataset-and-properties/)
    builds a dataset of your own and tests a property it does not contain.
    """)
    return


if __name__ == "__main__":
    app.run()
