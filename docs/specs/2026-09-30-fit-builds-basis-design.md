# `aj fit` builds the basis; one run file; no visible backend — design

Status: approved in conversation 2026-09-30. Stacked on PR #20
(`feat/trim-coupling`: the compiled `ace-jax-coupling` library).

## Intent

A user fits an ACE model from labelled data **without knowing that a basis is
a separate artifact, and without ever meeting Julia**. The minimal command is

```bash
aj fit --order 3 --max-degree 10 --train train.xyz --out fit/
```

and a finished run can be reproduced from one file it wrote:

```bash
aj fit --config fit/fit.yaml
```

Success criteria:
- no `aj basis` step and no intermediate `.npz` for the common case;
- no user-facing string, env var, extra or install step that mentions Julia;
- a definition-built fit is bit-identical to a fit from the equivalent
  `aj basis` file;
- the run file (`fit.yaml`) is the single place a run is described, and the
  resolved copy written by a fit reproduces it exactly.

ace-jax is unreleased with a single user, so this design carries **no
backward-compatibility shims**: renamed things are renamed everywhere.

## Decisions (taken with the user)

| Topic | Decision |
|---|---|
| Coupling library | `ace-jax-coupling` becomes a **core dependency**, marker-gated to Linux x86_64/aarch64 and macOS arm64. The `basis` extra is removed. |
| Basis in `aj fit` | **Inline flags** (shared with `aj basis`), `--elements` inferred from the data by default, **plus** `--config fit.yaml`. |
| Run file scope | **Whole fit run** (`fit.yaml`), not just the basis. |
| Run file schema | **Flat top-level keys = CLI flag names** (underscored), with a nested **`basis:`** block (or `model: path.npz`). |
| Precedence | explicit CLI flag > `fit.yaml` > default; every override logged. |
| Hand-off | **In memory** (approach A): the built basis is serialised to a buffer the existing file-shaped pipeline reads. A later refactor to object hand-off (approach C) stays possible; the Python API below already takes a `Basis` object. |
| Naming | `Authoring` → `Basis`; `meta["authoring"]` → `meta["basis"]`; package `ace_jax.construct` → `ace_jax.basis`. No aliases, no shims. |

## 1. Command line

### `aj fit`
Exactly one basis source is required:
- `--model PATH.npz` (unchanged), or
- basis flags inline (below), or
- `--config fit.yaml` supplying `model:` or `basis:`.

Giving `--model` together with any basis flag is an error
(`--model and --order/... are alternatives`). With `--config`, command-line
basis flags override the file's `basis:` block key by key (§2).

Basis flags on `aj fit` (same names, types and defaults as `aj basis`, from one
shared argparse helper `add_basis_args(parser, prefix_collisions=True)`):

| flag | meaning | note |
|---|---|---|
| `--order` | correlation order | presence of `--order`/`--max-degree` selects "build a basis" |
| `--max-degree` | total-degree level bound | |
| `--elements` | comma-separated Z or symbols | **optional**: default = sorted-by-Z union of species in train/test/ood (or `--data`) |
| `--wL`, `--rcut`, `--rin`, `--maxl`, `--d-max`, `--reduction`, `--radial-mode`, `--pair-mode` | as `aj basis` | |
| `--basis-embedding` | frozen element embedding of the basis (JSON table or `identity`) | renamed on `aj fit` only: `--embedding` there is already the GP species table |
| `--no-gamma` | skip the smoothness prior | |
| `--no-coupling-cache`, `--coupling-cache-dir` | as `aj basis` | |

`--seed` is shared: one seed drives the basis radial initialisation and the fit.

`--r0` (GP hyperprior centre) stays required with `--model`; when the basis is
built, it **defaults to the mean of the basis's per-pair radial `r0`**
(`meta["basis"]["r0"]`, the tabulated bond lengths the basis itself uses), and
the default is logged: `r0 2.351 A (mean bond length of Si; pass --r0 to override)`.

Log line for a build (no backend vocabulary):
`basis: Si,Ge order 3 max-degree 10 -> 110 B functions (built, cached for next time)`
or `(... from cache)`.

### `aj basis`
Unchanged in purpose (save/share an unfitted basis), plus `--config fit.yaml`:
it reads the file's `basis:` block (and top-level `seed`), command-line flags
override, and it writes `--out basis.npz`. `aj basis` keeps `--embedding` (no
collision there).

## 2. `fit.yaml`

```yaml
# a run file: every top-level key is an `aj fit` flag name, dashes -> underscores
train: train.xyz
test: test.xyz
energy_key: dft_energy
force_key: dft_force
m_per_species: 0
out: fit_si
basis:                       # or:  model: si.npz
  elements: [Si]             # optional; default from data
  order: 3
  max_degree: 10
  wL: 1.5
  # rcut, rin, maxl, d_max, reduction, radial_mode, pair_mode, embedding, no_gamma,
  # no_coupling_cache, coupling_cache_dir
```

Rules:
- Top-level keys: the `aj fit` argparse `dest` names. Inside `basis:`: the
  `aj basis` `dest` names (so `embedding`, not `basis_embedding`). Values keep
  their CLI types; list-valued flags (`elements`, `rungs`) accept YAML lists or
  the CLI comma string.
- Loading: YAML values become argparse defaults, then the command line is
  parsed on top, so an explicit flag wins. Each key where both were given logs
  `override: <key> <yaml value> -> <cli value> (command line)`.
- Required-ness (`--train`/`--data`, `--out`, `--r0` when `model:` is used) is
  checked **after** merging, with the same messages as today.
- Unknown keys (top level or in `basis:`) are errors with a nearest-match hint
  (`difflib`): `fit.yaml: unknown key 'm_per_specie' (did you mean 'm_per_species'?)`.
  `model:` and `basis:` together is an error.
- Relative paths in the file (`train`, `test`, `ood`, `data`, `model`,
  `weights`, `baseline`, `init`, `embedding`s) resolve against the YAML file's
  directory, not the working directory.
- One reserved read-ignored key: `provenance:` (below).

Every fit writes the **resolved** run to `<out>/fit.yaml`: every effective
value (defaults included), inferred `elements` and `r0` written explicitly,
paths absolute, and a `provenance:` block that is never read back as input:

```yaml
provenance:
  ace_jax: 0.1.0
  coupling: {version: 0.1.0, et_repo: ..., et_rev: 229f99d0...}   # when a basis was built
  command: "aj fit --order 3 ..."                                   # argv as typed
  created: 2026-09-30T14:02:11Z
```

`aj fit --config <out>/fit.yaml` (with a different `--out`) reproduces the run:
the same `model.npz` bit-for-bit on the same platform. `config.json` is still
written (it carries run statistics such as `M`), but `fit.yaml` is the input
format.

## 3. In-memory hand-off (approach A)

- `ace_jax.basis.model` gains `build_basis(spec: BasisSpec, *, seed) -> Basis`,
  the one entry point both `aj basis` and `aj fit` call; `BasisSpec` is a frozen
  dataclass of the `basis:` keys (dispatching to `build_model` or
  `build_embedding_model` on `embedding`).
- `FitConfig.model` accepts `str | os.PathLike | Basis`. The fit pipeline
  normalises it once, at the top of `load_fit_data`, to a *model source*: a path
  stays a path; a `Basis` is `save_npz`'d into a `BytesIO`. Everything
  downstream (`load`, `export._ace_arrays`, `prior_diagonal`'s log label) takes
  the source; buffer reads `seek(0)` first. The `.yace` check uses the path
  only when it is a path.
- Python users therefore get build → inspect/modify → fit without files:
  ```python
  from ace_jax.basis.model import build_basis, BasisSpec
  b = build_basis(BasisSpec(elements=["Si"], order=3, max_degree=10), seed=0)
  b = b._replace(model=...)            # inspect / modify
  res = fit(FitConfig(model=b, ...), data)
  ```
- Species inference reads the configs' atomic numbers before the basis is
  built (a cheap pass over the extxyz files already listed); the configs are
  then loaded as today.

## 4. Hiding the backend

- `pyproject.toml`: `dependencies` gains
  `ace-jax-coupling==0.1.0; (sys_platform == 'linux' and (platform_machine == 'x86_64' or platform_machine == 'aarch64')) or (sys_platform == 'darwin' and platform_machine == 'arm64')`;
  `[project.optional-dependencies] basis` is removed; the uv path source stays
  until `ace-jax-coupling` is published.
- Missing library at build time (unsupported platform, or a lib-less dev build)
  raises one message, independent of the backend:
  `building a new basis is not available on this platform (<platform>); fit from an existing basis with --model <file.npz> built elsewhere`
  (dev build: `... the ace-jax-coupling install has no compiled library: <hint>`).
  It is raised when the basis is built — after the configs are read (the species come from them), before any fitting work.
- `ACEJAX_NO_JULIA` → `ACEJAX_COUPLING_CACHE_ONLY` (no alias): with it set, a
  coupling-cache miss raises instead of computing (tests use it).
- No user-facing string (help, log, error, docs outside `coupling/` and
  `docs/coupling-etshim-spec.md`) contains "Julia". `coupling/` (the library's
  build) and the spec keep their Julia content: that is the backend itself.

## 5. Renames

| from | to |
|---|---|
| package `ace_jax.construct` (`src/ace_jax/construct/`) | `ace_jax.basis` (`src/ace_jax/basis/`) |
| class `Authoring` | `Basis` |
| model meta key `meta["authoring"]` (written by `build_model`, `build_embedding_model`; read by `fit/radial_model.py`) | `meta["basis"]` |
| `tests/test_python_authoring.py` | `tests/test_basis_build.py` |
| `docs/python-authoring.md` | `docs/basis.md` |
| `ACEJAX_NO_JULIA` | `ACEJAX_COUPLING_CACHE_ONLY` |
| extra `basis` | removed (core dependency) |

No committed fixture carries `meta["authoring"]` (checked), so no file needs
regenerating. Historical plans/specs under `docs/plans/`, `docs/specs/` and
`docs/tier2-plan.md` are not edited.

## 6. Error handling

| condition | behaviour |
|---|---|
| no basis source | `aj fit: give one of --model, --order/--max-degree, or --config` |
| `--model` + basis flags, or `model:` + `basis:` | error naming both |
| `--order` without `--max-degree` (or vice versa) | error |
| unknown YAML key | error + did-you-mean |
| YAML not a mapping / bad type | error naming the key and expected type |
| inferred species empty (no configs) | error |
| test/ood species absent from an explicit `--elements` | error listing the missing species (a basis cannot evaluate them) |
| library unavailable | §4 message, before data load |

## 7. Testing

- **Parity:** a definition-built fit (`--order/--max-degree` on the Si fixture
  shape) writes a `model.npz` bit-identical to a fit from the `aj basis` file of
  the same definition (both linear arm, same seed). Runs only with the compiled
  library (`require_coupling_lib`).
- **Buffer path, no library needed:** `FitConfig(model=<Basis from the primed
  coupling cache>)` fits and exports identically to the same basis saved to a
  file (uses `ACEJAX_COUPLING_CACHE_ONLY` + the committed cache, so it runs in
  the core CI suite).
- **YAML:** precedence (file vs CLI, logged override), unknown key with hint,
  `model`+`basis` conflict, relative-path resolution, list/comma forms,
  required-after-merge, and the reproduce round-trip (`--config out/fit.yaml` →
  identical `model.npz`).
- **Inference:** elements from train/test/ood union; explicit `--elements`
  missing a test species errors; `r0` default equals the mean of
  `meta["basis"]["r0"]` and is logged.
- **Backend hiding:** unsupported-platform message (monkeypatched import
  failure) raised before data load; a test that `ace_jax` source (help strings,
  messages) and user docs (README, SKILL.md, CLAUDE.md user sections,
  `docs/basis.md`) contain no "Julia" outside the allowed files.
- **Renames:** the full suite passes after the move; `git grep` finds no
  `ace_jax.construct`, `Authoring`, `"authoring"` meta key or `ACEJAX_NO_JULIA`
  outside historical docs.

## Out of scope

- Approach C (object hand-off through the whole pipeline) — later, if more
  in-memory sources appear.
- Publishing `ace-jax-coupling` / ace-jax to PyPI, and the wheel licence
  decision (tracked on PR #20).
- A YAML form for `aj eval`.
