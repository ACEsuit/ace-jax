# Tutorials

The tutorials are [marimo](https://marimo.io) notebooks: plain Python files
that run as interactive notebooks, as apps or as scripts. Each tutorial page on
this site is the notebook itself, run on a CPU when the site was built, so it
can be read without running anything; the top of each page has the one command
that opens it interactively. Each one lists its
own dependencies (PEP 723 script metadata), so a sandboxed run installs
exactly what it needs. Each runs on a laptop CPU in a few minutes and needs
no GPU or cloud account.

| | Tutorial | You will | Time |
|---|---|---|---|
| 1 | [First fit: silicon](first-fit.md) | build a basis, fit linear ACE, check it with a parity plot, an equation of state and NVE molecular dynamics | ~1 min |
| 2 | [Learned radials](learned-radials.md) | learn the radial basis by variable projection, compare with the frozen basis, deploy the splined model | ~3 min |
| 3 | [Your data, your property](dataset-and-properties.md) | build and label a strain/rattle dataset, fit it, score it with R², and find what a vacancy reveals that the RMSE hides | ~1 min |
| 4 | [Basis size and the evidence](basis-and-evidence.md) | sweep the basis size with plain least squares and with the evidence fit, watch least squares overfit, and let the log-evidence choose the basis | ~10 min |
| 7 | [The truth about the truth](truth-about-the-truth.md) | compute a Si(111) surface energy with two foundation-model labellers, and see an ACE fit follow whichever one taught it | ~1 min |

Tutorials 3, 4 and 7 are adapted from the [MLIP school 2026](https://github.com/ACEsuit/MLIP-school-2026) notebooks; tutorials 5, 6 and 8 will follow. Their reference labels come from MIT-licensed MACE foundation models and ship with the tutorials, so the default settings need no labeller.

Every tutorial has the same shape: **goals** at the top, numbered **steps**,
a **checkpoint** after each step that says whether it worked, and
**exercises** at the end that change one thing at a time.

!!! note "Coming next"
    A tutorial on calibrated uncertainty for ACE models is planned.

## Running a notebook

=== "One command (recommended)"

    With [uv](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`
    on Linux and macOS), run a tutorial straight from GitHub:

    ```bash
    uvx marimo edit --sandbox https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/notebooks/first_fit_si.py
    ```

    marimo downloads the notebook, `--sandbox` builds a throwaway environment
    from the notebook's own dependency list (the first run takes a minute
    or so), and the notebook opens in your browser. No account is needed.
    Edits go to a temporary copy: to keep them, download the file first
    (`curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/notebooks/first_fit_si.py`) and run `uvx marimo edit --sandbox first_fit_si.py`.

=== "molab (in the browser)"

    Each tutorial page links to it on [molab](https://molab.marimo.io),
    marimo's hosted service. The preview is free; running the notebook needs
    a (free) molab sign-in, with a GitHub or Google account.

=== "Locally, in your environment"

    In an environment with ace-jax installed:

    ```bash
    pip install marimo matplotlib
    marimo edit first_fit_si.py       # interactive
    python first_fit_si.py            # or run it top to bottom as a script
    ```

The notebooks write their outputs (data splits, fitted models) to a
directory next to where they are started, `ace_jax_tutorial_<n>/`.
