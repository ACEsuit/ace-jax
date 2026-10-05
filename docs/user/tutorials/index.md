# Tutorials

The tutorials are [marimo](https://marimo.io) notebooks. A marimo notebook
is a plain Python file. You can run it as an interactive notebook, as an app
or as a script.

- Each tutorial page on this site is the notebook, run on a CPU when the
  site was built. Thus you can read it without running it.
- The top of each page gives the command that opens the notebook
  interactively.
- Each notebook lists its dependencies (PEP 723 script metadata). Thus a
  sandboxed run installs only what the notebook needs.
- Each notebook runs on a laptop CPU in a few minutes. It does not need a
  GPU or a cloud account.

| | Tutorial | You will | Time |
|---|---|---|---|
| 1 | [First fit: silicon](first-fit.md) | build a basis, fit linear ACE, check it with a parity plot, an equation of state and NVE molecular dynamics | ~1 min |
| 2 | [Learned radials](learned-radials.md) | learn the radial basis by variable projection, compare with the frozen basis, deploy the splined model | ~3 min |
| 3 | [Multi-element fits](multi-element.md) | fit a five-element alloy with the categorical basis and with species-embedding bases, and compare size, fit time and accuracy with plenty and with little data | ~3 min |
| 4 | [Your data, your property](dataset-and-properties.md) | build and label a strain/rattle dataset, fit it, score it with R², and find what a vacancy reveals that the RMSE hides | ~1 min |
| 5 | [Basis size and the evidence](basis-and-evidence.md) | sweep the basis size with plain least squares and with the evidence fit, watch least squares overfit, and let the log-evidence choose the basis | ~10 min |
| 6 | [Surfaces: coverage and repair](surfaces.md) | find that a bulk model cannot see a surface, measure it in descriptor space, repair the data, and make relaxations stable | ~2 min |
| 7 | [Automating curation](curation.md) | run an MD-select-label-refit loop with random, novelty and uncertainty selection, and compare them at the same label budget | ~5 min |
| 8 | [The truth about the truth](truth-about-the-truth.md) | compute a Si(111) surface energy with two foundation-model labellers, and see an ACE fit follow whichever one taught it | ~1 min |
| 9 | [Bring your own data](bring-your-own-data.md) | declare a target property, label references, fit a two-element model, check coverage and repair the data, on GaAs or your own structure | ~2 min |

Tutorials 4 to 9 come from the [MLIP School 2026](https://mlipschool.uk/)
notebooks. Their reference labels come from MACE foundation models with the
MIT licence. The labels are included with the tutorials. Thus, with the
default settings, you do not need a labeller.

All tutorials have the same structure:

- **goals** at the top;
- numbered **steps**;
- a **checkpoint** after each step, which tells you if the step worked;
- **exercises** at the end. Each exercise changes one thing.

!!! note "Coming next"
    A tutorial on calibrated uncertainty for ACE models is planned.

## Running a notebook

=== "One command (recommended)"

    1. Install [uv](https://docs.astral.sh/uv/). On Linux and macOS, use
       `curl -LsSf https://astral.sh/uv/install.sh | sh`.
    2. Run a tutorial directly from GitHub:

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/notebooks/first_fit_si.py
    ```

    marimo downloads the notebook. `--sandbox` builds a temporary
    environment from the dependency list of the notebook. The first run
    takes approximately 1 minute. Then the notebook opens in your browser.
    You do not need an account.

    Your changes go to a temporary copy. To keep your changes:

    1. Download the file:
       `curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/notebooks/first_fit_si.py`.
    2. Run `uvx marimo edit --sandbox first_fit_si.py`.

=== "molab (in the browser)"

    Each tutorial page has a link to the notebook on
    [molab](https://molab.marimo.io), the marimo hosted service. The preview
    is free. To run the notebook, sign in to molab (free) with a GitHub or
    Google account.

=== "Locally, in your environment"

    In an environment with ace-jax installed:

    ```bash
    pip install marimo matplotlib
    marimo edit first_fit_si.py       # interactive
    python first_fit_si.py            # or run it top to bottom as a script
    ```

The notebooks write their outputs (data splits, fitted models) to the
directory `ace_jax_tutorial_<n>/`, in the directory where you start them.
