# Tutorials

The tutorials are [marimo](https://marimo.io) notebooks: plain Python files
that run as interactive notebooks, as apps or as scripts. Each one lists its
own dependencies (PEP 723 script metadata), so a sandboxed run installs
exactly what it needs. Each runs on a laptop CPU in a few minutes and needs
no GPU or cloud account.

| | Tutorial | You will | Time |
|---|---|---|---|
| 1 | [First fit: silicon](first-fit.md) | build a basis, fit linear ACE, check it with a parity plot, an equation of state and NVE molecular dynamics | ~1 min |
| 2 | [Learned radials](learned-radials.md) | learn the radial basis by variable projection, compare with the frozen basis, deploy the splined model | ~3 min |

Every tutorial has the same shape: **goals** at the top, numbered **steps**,
a **checkpoint** after each step that says whether it worked, and
**exercises** at the end that change one thing at a time.

!!! note "Coming next"
    A tutorial on calibrated uncertainty for ACE models is planned.

## Running a notebook

=== "molab (in the browser)"

    Click the **Open in molab** badge on a tutorial's page. It opens a
    preview of the notebook on [molab](https://molab.marimo.io), marimo's
    hosted service; from there you can run it in the cloud, which installs its
    dependencies (molab asks you to sign in to run notebooks).

=== "Locally, sandboxed"

    With [uv](https://docs.astral.sh/uv/) installed:

    ```bash
    curl -LO https://raw.githubusercontent.com/ACEsuit/ace-jax/main/docs/user/tutorials/notebooks/first_fit_si.py
    uvx marimo edit --sandbox first_fit_si.py
    ```

    `--sandbox` creates a throwaway environment from the notebook's own
    dependency list.

=== "Locally, in your environment"

    In an environment with ace-jax installed:

    ```bash
    pip install marimo matplotlib
    marimo edit first_fit_si.py       # interactive
    python first_fit_si.py            # or run it top to bottom as a script
    ```

The notebooks write their outputs (data splits, fitted models) to a
directory next to where they are started, `ace_jax_tutorial_<n>/`.
