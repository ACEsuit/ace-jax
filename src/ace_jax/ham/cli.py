"""`aj ham`: fit, predict and bands for Hamiltonian (operator) models, on model files.

    aj ham fit --data DIR --out model.npz [--validation DIR]     blocks of DFT -> a model file
    aj ham predict --model model.npz --data STRUCTURES --out DIR  H and S blocks, DeepH layout
    aj ham bands --model model.npz --structure S --out bands.npz [--plot bands.png]

Structures with blocks are a directory of ace-jax operator caches (*.npz, `ham.cache`) or of
DeepH-layout subdirectories (`ham.deeph`; with `up/` and `dn/` inside, collinear spin); structures
alone may also be an extxyz file.  `aj eval --model model.npz` scores a model (`score` here).

    aj density fit --data DIR --out model.npz [--validation DIR]  density files -> a model file
    aj density predict --model model.npz --data STRUCTURES --out DIR

the electron-density model (`ham.density`; its commands are the `density_*` functions here).

The parser and the commands only: `ace_jax.cli` mounts them (`add_parser`, `run`).  Energies are
in the unit of the data (eV for FHI-aims through Quoll)."""
import glob
import json
import os
import time

import numpy as np


class UsageError(ValueError):
    """A user-facing condition: `aj` prints it and exits 2 instead of a traceback."""


PRIORS = ("none", "evidence", "loo", "press")


def onsite(k):
    return k[:3] == (0, 0, 0) and k[3] == k[4]


def electrons(text):
    """--nelectrons: None, a number per cell, or Z:n,Z:n electrons per atom of each element."""
    if text is None:
        return None
    return {int(z): float(n) for z, n in (x.split(":") for x in text.split(","))} if ":" in text else float(text)


def _read_blocks(name):
    from . import cache, deeph
    if name.endswith(".npz"):
        return cache.load(name)[0]
    return deeph.read_spin(name) if os.path.isdir(os.path.join(name, "up")) else deeph.read(name)


def block_names(path):
    """The structures with blocks under `path`: operator caches, else DeepH subdirectories."""
    if not os.path.isdir(path):
        raise UsageError(f"{path}: not a directory of operator caches (*.npz) or DeepH-layout subdirectories")
    names = sorted(glob.glob(os.path.join(path, "*.npz"))) or sorted(d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d))
    if not names:
        raise UsageError(f"{path}: no operator caches (*.npz) and no DeepH directories")
    return names


def structures(path, index=None):
    """[(name, ase.Atoms, OperatorData or None)] of `path`: a directory of structures with blocks
    (or one of them: a cache file or a DeepH directory), or an extxyz file.  index: one of them."""
    from ase import Atoms
    if os.path.isdir(path) or path.endswith(".npz"):
        one = path.endswith(".npz") or any(os.path.exists(os.path.join(path, f)) for f in ("element.dat", "up"))
        names = [path.rstrip("/")] if one else block_names(path)
        names = names if index is None else [names[index]]
        out = []
        for n in names:
            d = _read_blocks(n)
            out.append((os.path.splitext(os.path.basename(n))[0], Atoms(numbers=d.numbers, positions=d.positions, cell=d.cell, pbc=True), d))
        return out
    from ase_extxyz.io import read_cextxyz
    frames = list(read_cextxyz(path, index=":", create_calc=False))
    return [(f"{i:04d}", f, None) for i, f in enumerate(frames)] if index is None else [(f"{index:04d}", frames[index], None)]


def _calc(a, kpts):
    """The calculator of a.model with the command's options; --nelectrons defaults to the one
    the model file records (`aj ham fit --nelectrons`), else the atomic numbers."""
    from ..operators import io
    from .calculator import HamiltonianCalculator
    nel = a.nelectrons if a.nelectrons is not None else io.load(a.model).meta["info"].get("nelectrons")
    calc = HamiltonianCalculator(a.model, kpts=kpts, nelectrons=electrons(nel), width=a.width, smearing=a.smearing, threads=a.threads)
    ch = calc.predictor.channels
    calc.overlap = "S" if "S" in ch else None
    calc.channel = getattr(a, "channel", None) or ("H" if "H" in ch else next(c for c in ch if c != "S"))
    if calc.channel not in ch:
        raise UsageError(f"--channel {calc.channel}: the model has {', '.join(ch)}")
    calc.degeneracy = 2.0 if calc.channel == "H" else 1.0      # a spin channel holds one electron per state
    calc.overlap_threshold = float(getattr(a, "overlap_threshold", 0.0) or 0.0)   # of the scored and plotted bands (`operator.block_bands`), not of the energy
    return calc


def _filled(blocks, keys, numbers, orb):
    """`blocks` on `keys`, zero blocks where it has none (a model's cutoff, or S's, shorter than H's)."""
    return {k: np.asarray(blocks[k]) if k in blocks else np.zeros((orb.n_orb(int(numbers[k[3]])), orb.n_orb(int(numbers[k[4]])))) for k in keys}


def score(calc, data, window, metrics_out=None, blocks_out=None):
    """Print the block errors of `calc`'s model on the reference structures under `data`, and the
    electronic-structure errors of its eigenvalues against the reference's on the calculator's
    k-points (`ham.metrics`).  Returns (block error table, metrics summary or None).
    calc.overlap_threshold > 0 (--overlap-threshold): both spectra in the subspace of S(k) above it (`operator.block_bands`).
    A model without an overlap is scored with the data's S where it has one (its H and the reference
    H in the same generalised problem; summary["overlap"] "reference"), else with S = 1 ("identity")."""
    from . import metrics as M
    from .operator import BlockSum, block_bands
    ch, orb = calc.predictor.channels, calc.predictor.orbitals
    names = block_names(data)
    err, per, bad, overlap = {}, [], [], None
    cut = getattr(calc, "overlap_threshold", 0.0)
    print(f"{len(names)} reference structures; eigenvalues on {len(calc.kpts)} k-points, {window[0]:g} to {window[1]:g} from the Fermi level"
          + (f"; overlap eigenvalues below {cut:g} left out" if cut > 0 else ""))
    for i, (name, at, d) in enumerate(s for n in names for s in structures(n)):
        if d.basis != orb:
            raise UsageError(f"{name}: its orbital basis is not the model's")
        at.calc = calc
        pred = calc._blocks(at)
        for c in ch:
            if c in d.blocks:
                for part, sel in (("on-site", onsite), ("off-site", lambda k: not onsite(k))):
                    err.setdefault((c, part), []).append(np.concatenate([(pred[c][k] - v).ravel() for k, v in d.blocks[c].items()
                                                                          if sel(k) and k in pred[c]] or [np.zeros(0)]))
        if calc.channel in d.blocks and (calc.overlap is None or calc.overlap in d.blocks):
            Hb = d.blocks[calc.channel]
            if calc.overlap is None and "S" in d.blocks:   # a model of H alone: its H and the reference H, both with the reference S
                given = "reference"; Sb, P = d.blocks["S"], pred[calc.channel]
                keys = sorted(set(Hb) | set(Sb) | set(P))
                bs = BlockSum(keys, d.numbers, orb); full = lambda B, keys=keys, z=d.numbers: _filled(B, keys, z, orb); Sf = full(Sb)
                e1, e0 = block_bands(bs, full(P), Sf, calc.kpts, overlap_threshold=cut), block_bands(bs, full(Hb), Sf, calc.kpts, overlap_threshold=cut)
            else:
                given = "model" if calc.overlap else "identity"
                Sb = d.blocks[calc.overlap] if calc.overlap else {k: np.eye(*v.shape) * float(onsite(k)) for k, v in Hb.items()}
                if cut > 0 and calc.overlap:             # the model's H and S, as the reference's: each in the subspace of its own S(k)
                    P = pred[calc.channel]
                    e1 = block_bands(BlockSum(sorted(P), d.numbers, orb), P, pred[calc.overlap], calc.kpts, overlap_threshold=cut)
                else:
                    e1 = np.asarray(calc.get_property("eigenvalues", at))
                e0 = block_bands(BlockSum(sorted(Hb), d.numbers, orb), Hb, Sb, calc.kpts, overlap_threshold=cut if calc.overlap else 0.0) if not np.isnan(e1).any() else None
            if overlap is None:
                overlap = given
                print({"reference": "the model has no overlap: eigenvalues of its H and of the reference H with the reference S",
                       "identity": "the model and the data have no overlap: eigenvalues with S = 1", "model": f"overlap: the model's {calc.overlap}"}[given], flush=True)
            if np.isnan(e1).any():                       # the eigensolver's answer to an overlap that is not positive definite
                print(f"  {name}: no eigenvalues: the predicted {calc.overlap or 'H'} gives a generalised eigenproblem that is not "
                      f"positive definite; no band metrics for this structure", flush=True)
                bad.append(name)
            else:
                per.append(dict(name=name, **M.spectrum_metrics(
                    e0, e1, calc.electrons(d.numbers), len(d.numbers), calc.weights, calc.width, calc.smearing, calc.degeneracy, tuple(window))))
        if blocks_out:
            calc.write_results(os.path.join(blocks_out, f"{i:04d}"), at)
    cat = lambda v: np.concatenate(v) if sum(len(x) for x in v) else np.full(1, np.nan)
    table = {k: dict(rmse=float(np.sqrt(np.mean(cat(v) ** 2))), mae=float(np.mean(np.abs(cat(v)))), max=float(np.abs(cat(v)).max())) for k, v in err.items()}
    print("blocks (1e-3 of the data's unit: meV for H in eV)   RMSE        MAE        max")
    for (c, part), t in table.items():
        print(f"  {c:6s} {part:9s} {1e3 * t['rmse']:22.4f} {1e3 * t['mae']:10.4f} {1e3 * t['max']:10.3f}")
    summary = M.summary(per) if per else None
    if summary is not None:
        summary["overlap"] = overlap
        if cut > 0:
            summary["overlap_threshold"] = cut
        if bad:
            summary["no_eigenvalues"] = bad
    if per:
        print("\n".join(M.format_summary(summary, unit="(1e-3 of the data's unit)")))
        if metrics_out:
            with open(metrics_out, "w") as f:
                json.dump(dict(summary=summary, structures=per), f, indent=1)
    return table, summary


def _subset(d, n, seed):
    """`d` with its on-site blocks and a random `n` of its bonds, both orientations (0: all)."""
    from ..operators.data import OperatorData
    ref = next(iter(d.blocks.values()))
    keys = sorted(k for k in ref if not onsite(k))
    if not n or n >= len(keys):
        return d
    keep = {keys[i] for i in np.random.default_rng(seed).choice(len(keys), size=n, replace=False)}
    keep |= {(-k[0], -k[1], -k[2], k[4], k[3]) for k in keep} | {k for k in ref if onsite(k)}
    return OperatorData(d.cell, d.positions, d.numbers, d.basis, {c: {k: v for k, v in b.items() if k in keep} for c, b in d.blocks.items()})


def cmd_fit(a):
    """Fit the on-site and off-site block models to the structures under --data and write one
    model file.  The basis: site features of order 1 (--site-radials radial functions within
    --site-rcut, every L the orbital basis needs), bond features of --bond-radials radial
    functions out to --bond-rcut (default: the longest bond with a block), the site-coupled
    group (--coupled N L) and the bond-environment group (--env R N L NB LB; --no-env).  --prior:
    the smoothness prior's scale, a number, or chosen per shell pair and channel by the evidence
    or by leaving structures out ("loo", "press")."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from ..operators import bond as B
    from ..operators import io
    from ..operators.features import build_site_basis
    from ..operators.fit import fit_offsite, fit_onsite
    names = block_names(a.data)[:a.max_structures]
    from ..operators.data import LazyData
    budget = a.data_memory * 2.0 ** 30                   # the blocks kept in memory; a larger set is read again on each pass
    train = LazyData(len(names), lambda i: _read_blocks(names[i]), budget)
    orb = train[0].basis
    ch = tuple(a.channels) if a.channels else tuple(train[0].blocks)
    n_atoms, longest = 0, 0.0
    for d in train:                                      # one pass: what the basis and the messages need of every structure
        if d.basis != orb:
            raise UsageError(f"{a.data}: the structures do not share one orbital basis")
        missing = [c for c in ch if c not in d.blocks]
        if missing:
            raise UsageError(f"--channels: {', '.join(missing)} is not in every structure (the first has {', '.join(train[0].blocks)})")
        n_atoms += len(d.numbers)
        if not a.bond_rcut:
            longest = max(longest, max(np.linalg.norm(d.bond(k)) for k in d.blocks[ch[0]]))
    prior = None if a.prior == "none" else a.prior if a.prior in PRIORS else float(a.prior)
    els = [int(z) for z in orb.species]
    rcut = a.bond_rcut or float(np.ceil(10 * longest) / 10)
    sb = build_site_basis(els, order=1, totaldegree=40, lmax_feat=2 * orb.lmax, rcut=a.site_rcut, wL=0.5, nmax_by_order=(a.site_radials,),
                          lmax_by_order=(2 * orb.lmax,)).with_table()
    basis = B.BondBasis(site=sb, rcut_bond=rcut, n_bond=a.bond_radials, n_cpl=a.coupled[0], l_cpl=a.coupled[1])
    if not a.no_env:
        r, n, l, nb, lb = a.env
        basis = basis._replace(r_env=float(r), n_env=int(n), l_env=int(l), n_ebond=int(nb), l_ebond=int(lb))
    print(f"{len(train)} structures, {n_atoms} atoms, elements {els}, channels {', '.join(ch)}; "
          f"site cutoff {a.site_rcut:g}, bond cutoff {rcut:g}; prior {a.prior}" + (f", noise floor {a.noise_floor:g}" if a.noise_floor else ""), flush=True)
    t = time.perf_counter()
    on, rep_on = fit_onsite(sb, train, channels=ch, ridge=a.ridge, sigma_c=prior, noise=a.noise_floor)
    off, rep_off = fit_offsite(basis, LazyData(len(names), lambda i: _subset(train[i], a.bonds, a.seed + i)) if a.bonds else train, channels=ch, ridge=a.ridge, sigma_c=prior, noise=a.noise_floor)
    fit_s = time.perf_counter() - t
    n_coef = int(sum(np.asarray(c).size for m in (on, off) for c in m.coef.values()))
    resid = {}
    for part, rep in (("on-site", rep_on), ("off-site", rep_off)):       # rms residual of the L parts, per channel
        rows = np.array([r[0] for r in rep.values()], float)
        rms = np.array([np.broadcast_to(np.asarray(r[2], float), (len(ch),)) for r in rep.values()])
        resid[part] = [float(v) for v in np.sqrt((rows[:, None] * rms ** 2).sum(0) / rows.sum())]
        print(f"  {part:8s} training residual of the L parts, rms (1e-3 of the data's unit): " + ", ".join(f"{c} {1e3 * v:.4f}" for c, v in zip(ch, resid[part])), flush=True)
    print(f"fitted {n_coef} coefficients in {fit_s:.1f} s", flush=True)
    info = dict(command="aj ham fit", data=os.path.abspath(a.data), n_train=len(train), structures=[os.path.basename(n) for n in names],
                bonds_per_structure=a.bonds, seed=a.seed, prior=a.prior, noise_floor=a.noise_floor, ridge=a.ridge, fit_s=round(fit_s, 1), n_coef=n_coef, nelectrons=a.nelectrons,
                training_residual=resid)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    io.save(a.out, on, off, info=info)                   # before the validation: a failure there does not lose the fit
    if a.validation:
        from .calculator import HamiltonianCalculator
        calc = HamiltonianCalculator((on, off), kpts=tuple(a.kpts), nelectrons=electrons(a.nelectrons), width=a.width, smearing=a.smearing, threads=a.threads)
        calc.overlap = "S" if "S" in ch else None
        calc.channel = "H" if "H" in ch else next(c for c in ch if c != "S")
        calc.degeneracy = 2.0 if calc.channel == "H" else 1.0
        calc.overlap_threshold = float(a.overlap_threshold or 0.0)
        print(f"validation set {a.validation}:")
        table, summary = score(calc, a.validation, a.window)
        info["validation"] = dict(data=os.path.abspath(a.validation), blocks={f"{c} {p}": v for (c, p), v in table.items()}, metrics=summary)
        io.save(a.out, on, off, info=info)
    print(f"wrote {a.out}")


def cmd_predict(a):
    """The H (and S) blocks of every structure of --data, one DeepH-layout directory each under
    --out: the structure files, hamiltonians_pred.h5 and overlaps_pred.h5."""
    import jax
    jax.config.update("jax_enable_x64", True)
    calc = _calc(a, (1, 1, 1))
    todo = structures(a.data)
    for name, at, _ in todo:
        t = time.perf_counter()
        calc.write_results(os.path.join(a.out, name), at, overlap=not a.no_overlap)
        print(f"{name}: {len(at)} atoms, {time.perf_counter() - t:.2f} s", flush=True)
    print(f"wrote {len(todo)} structures to {a.out}")


def band_structure(calc, atoms, path=None, npoints=200, reference=None):
    """Bands of `calc`'s model for `atoms` on a path (ASE's `cell.bandpath`: `path` a string of
    special points, default the lattice's own) and on the calculator's k-points, with the Fermi
    level of those; `reference`: an `OperatorData` of the same structure, for the same from its
    blocks.  A dictionary of arrays (`np.savez`): x, ticks, labels, kpts, bands, grid_kpts,
    grid_weights, grid_bands, fermi_level and, with a reference, bands_ref, grid_bands_ref,
    fermi_level_ref."""
    import jax.numpy as jnp
    from .operator import BlockSum, block_bands, smeared_occupations
    bp = atoms.cell.bandpath(path, npoints=npoints, pbc=atoms.pbc)
    x, ticks, labels = bp.get_linear_kpoint_axis()
    p = calc.predictor
    prep = p.prepare_auto(atoms.positions, np.asarray(atoms.cell), atoms.numbers, atoms.pbc)
    nel = calc.electrons(atoms.numbers)
    cut = getattr(calc, "overlap_threshold", 0.0)
    mu = lambda e: float(smeared_occupations(jnp.asarray(e), jnp.asarray(calc.weights), nel, calc.width, calc.smearing, calc.degeneracy)[1])
    out = dict(x=x, ticks=ticks, labels=np.asarray(labels), kpts=bp.kpts, grid_kpts=calc.kpts, grid_weights=calc.weights,
               nelectrons=nel, degeneracy=calc.degeneracy, formula=atoms.get_chemical_formula())
    for tag, ks in (("bands", bp.kpts), ("grid_bands", calc.kpts)):
        out[tag] = np.asarray(p.bands(prep, ks, overlap=calc.overlap, channel=calc.channel, overlap_threshold=cut), float)
    out["fermi_level"] = mu(out["grid_bands"])
    if reference is not None:
        Hb = reference.blocks[calc.channel]
        Sb = reference.blocks[calc.overlap] if calc.overlap else {k: np.eye(*v.shape) * float(onsite(k)) for k, v in Hb.items()}
        bs = BlockSum(sorted(Hb), reference.numbers, p.orbitals)
        for tag, ks in (("bands_ref", bp.kpts), ("grid_bands_ref", calc.kpts)):
            out[tag] = np.asarray(block_bands(bs, Hb, Sb, ks, overlap_threshold=cut if calc.overlap else 0.0), float)
        out["fermi_level_ref"] = mu(out["grid_bands_ref"])
    return out


def draw_bands(p, q, b, window=(-13.0, 6.0), smear=0.1, labels=("reference", "model")):
    """A `band_structure` record on two axes: p the bands on the path, the reference (if any) in
    ink and the model in one hue; q the density of states of the k-point grid (Gaussians of
    width `smear`) on the same energy axis.  Energies from the reference's Fermi level, else the
    model's."""
    ref = "bands_ref" in b
    mu = float(b["fermi_level_ref"] if ref else b["fermi_level"])
    INK, MODEL, MUTED = "#0b0b0b", "#2a78d6", "#8a8f98"
    x = np.asarray(b["x"])
    cut = np.r_[0, np.flatnonzero(np.diff(x) == 0) + 1, len(x)]            # the path's disconnected pieces
    style = dict(color=MODEL, lw=1.0, ls=(0, (4, 2))) if ref else dict(color=MODEL, lw=1.2)
    for lo, hi in zip(cut[:-1], cut[1:]):
        if ref:
            p.plot(x[lo:hi], b["bands_ref"][lo:hi] - mu, color=INK, lw=1.6, solid_capstyle="round")
        p.plot(x[lo:hi], b["bands"][lo:hi] - mu, **style)
    for t in b["ticks"]:
        p.axvline(t, color=MUTED, lw=0.5)
    p.axhline(0.0, color=MUTED, lw=0.5, ls=":")
    names = {}                                          # a break of the path: both ends at one tick, "Z|X"
    for t, s in zip(b["ticks"], b["labels"]):
        names.setdefault(float(t), []).append("Γ" if str(s) == "G" else str(s))
    p.set_xticks(list(names)); p.set_xticklabels(["|".join(v) for v in names.values()])
    p.set_xlim(x[0], x[-1]); p.set_ylim(*window); p.set_ylabel("E − E$_F$" + (f" ({labels[0]})" if ref else ""))
    if ref:
        p.plot([], [], color=INK, lw=1.6, label=labels[0]); p.plot([], [], label=labels[1], **style)
        p.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=9, borderaxespad=0.2)
    e = np.linspace(window[0], window[1], 1200)
    wk, g = np.asarray(b["grid_weights"]), float(b["degeneracy"])
    dos = lambda ev: g * (wk[None, :, None] * np.exp(-0.5 * ((e[:, None, None] - (ev - mu)[None]) / smear) ** 2)).sum((1, 2)) / (smear * np.sqrt(2 * np.pi))
    if ref:
        d0 = dos(b["grid_bands_ref"])
        q.fill_betweenx(e, 0.0, d0, color=INK, alpha=0.10, lw=0); q.plot(d0, e, color=INK, lw=1.2)
    q.plot(dos(b["grid_bands"]), e, **style)
    q.axhline(0.0, color=MUTED, lw=0.5, ls=":")
    q.set_xlim(0.0, None); q.set_xlabel("DoS / states per energy per cell"); q.tick_params(left=False)
    for ax in (p, q):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)


def plot_bands(b, fig, window=(-13.0, 6.0), smear=0.1, title=None, labels=("reference", "model")):
    """Figure file of a `band_structure` record (`draw_bands`).  Needs matplotlib."""
    try:
        import matplotlib
    except ImportError:
        raise UsageError("--plot needs matplotlib (pip install matplotlib)") from None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, (p, q) = plt.subplots(1, 2, figsize=(9.0, 5.2), sharey=True, gridspec_kw=dict(width_ratios=(3.4, 1), wspace=0.04))
    draw_bands(p, q, b, window, smear, labels)
    if title:
        f.suptitle(title, fontsize=10, x=0.125, y=1.0 if "bands_ref" in b else 0.94, ha="left")
    f.savefig(fig, dpi=200, bbox_inches="tight")
    plt.close(f)


def cmd_bands(a):
    """Bands of one structure on a high-symmetry path and the density of states of a k-point
    grid, written as arrays (--out, .npz) and as a figure (--plot).  A structure that comes
    with blocks (a cache file or a DeepH directory) also gives the reference's bands, and the
    errors of the model against them are printed (--no-reference: the model alone)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from . import metrics as M
    calc = _calc(a, tuple(a.kpts))
    name, at, d = structures(a.structure, a.index or 0)[0]
    ref = None if a.no_reference or d is None or calc.channel not in d.blocks else d
    if ref is not None and ref.basis != calc.predictor.orbitals:
        raise UsageError(f"{a.structure}: its orbital basis is not the model's (--no-reference for the model's bands alone)")
    b = band_structure(calc, at, a.path, a.npoints, ref)
    title = f"{b['formula']} ({name}); DoS: {len(calc.kpts)} k-points, {a.dos_width:g} Gaussian"
    print(f"{name}: {b['formula']}, {len(at)} atoms, {b['bands'].shape[1]} bands on {len(b['kpts'])} path points "
          f"({''.join(str(s) for s in b['labels'])}); Fermi level {b['fermi_level']:.4f}")
    if ref is not None:
        m = M.spectrum_metrics(b["grid_bands_ref"], b["grid_bands"], b["nelectrons"], len(at), calc.weights, calc.width, calc.smearing,
                               calc.degeneracy, tuple(a.window))
        print(f"against the reference's blocks (1e-3 of the data's unit): band RMSE {1e3 * m['band_rmse']:.2f} at a common zero (largest "
              f"{1e3 * m['band_max']:.1f}), {1e3 * m['aligned_rmse']:.2f} from own Fermi levels; Fermi level {1e3 * m['fermi']:+.2f}; "
              f"band energy per atom {1e3 * m['band_energy']:+.2f}; density of states (Wasserstein) {1e3 * m['dos']:.2f}")
        title += f"\nband RMSE {1e3 * m['aligned_rmse']:.1f} from own Fermi levels, {1e3 * m['band_rmse']:.1f} at a common zero (1e-3 of the unit)"
        b.update({f"metric_{k}": v for k, v in m.items()})
    if a.out:
        np.savez(a.out, **b)
        print(f"wrote {a.out}")
    if a.plot:
        plot_bands(b, a.plot, tuple(a.plot_window), a.dos_width, title)
        print(f"wrote {a.plot}")


# ------------------------------------------------------------------ aj density
def density_structures(path):
    """[(name, ase.Atoms, DensityData or None)] of `path`: a directory of density files (or one
    file, or one DeepH directory with `density.npz`), or an extxyz file of structures alone."""
    from ase import Atoms

    from . import density as D
    if os.path.isdir(path) or path.endswith(".npz"):
        one = path.endswith(".npz") or os.path.exists(os.path.join(path, D.IN_DIRECTORY))
        try:
            files = [path.rstrip("/")] if one else D.names(path)
        except ValueError as e:
            raise UsageError(str(e)) from None
        out = []
        for f in files:
            d = D.read(f, overlap="lazy")
            out.append((os.path.splitext(os.path.basename(f))[0], Atoms(numbers=d.numbers, positions=d.positions, cell=d.cell, pbc=d.pbc), d))
        return out
    from ase_extxyz.io import read_cextxyz
    return [(f"{i:04d}", f, None) for i, f in enumerate(read_cextxyz(path, index=":", create_calc=False))]


def density_score(predictor, data, metrics_out=None):
    """Print the errors of `predictor`'s densities against the reference records under `data`
    (`ham.density.density_metrics`): per structure the density error (with the record's overlap),
    the coefficient error over the labels' spread and the electron count error before the
    correction; then the summary.  Returns (per-structure records, summary)."""
    from . import density as D
    per = []
    todo = density_structures(data)
    if any(d is None for _, _, d in todo):
        raise UsageError(f"{data}: scoring a density model needs reference densities (a directory of density files)")
    print(f"{len(todo)} reference structures")
    print("structure            atoms   density error    relative     coefficients   count error")
    for name, at, d in todo:
        if d.shells != predictor.model.shells:
            raise UsageError(f"{name}: its auxiliary basis is not the model's")
        pred = predictor.predict(at, d.nelectrons)
        m = D.density_metrics(d, pred)
        m["name"] = name
        per.append(m)
        lam = np.sqrt(np.mean([v ** 2 for v in m["rmse_l"].values()]))
        num = lambda k, f, w, m=m: format(m[k], f) if m.get(k) is not None else "-".rjust(w)
        print(f"{name:20s} {len(at):5d} {num('error', '15.5f', 15)} {num('relative', '11.3e', 11)} {lam:15.4f} {num('dN_recorded', '+13.5f', 13)}", flush=True)
    s = D.summary(per)
    print("\n".join(D.format_summary(s)))
    if metrics_out:
        with open(metrics_out, "w") as f:
            json.dump(dict(summary=s, structures=per), f, indent=1)
    return per, s


def _density_charges(train):
    """The charges of the auxiliary functions, from the training records (the first that has
    them; a warning when two records differ), else fitted to the electron counts."""
    import warnings

    from . import density as D
    have = [d.charges for d in train if d.charges is not None]
    if not have:
        if any(d.nelectrons is None for d in train):
            return None
        return D.fit_charges(train)
    if any(set(q) != set(have[0]) or any(not np.allclose(q[z], have[0][z], rtol=1e-8, atol=1e-12) for z in q) for q in have[1:]):
        warnings.warn("aj density fit: the training structures record different charges of the auxiliary functions; the first is used", stacklevel=2)
    return have[0]


def cmd_density_fit(a):
    """Fit an electron-density model to the density files under --data and write one model file.
    The model: the expansion coefficients of every auxiliary shell of angular momentum l as a
    linear function of the site features of (l, (-1)^l) (correlation order --order,
    --site-radials radial functions within --site-rcut).  --metric overlap (the default when
    every structure has the overlap of the auxiliary functions) minimises the L2 norm of the
    density error; plain, the error of each coefficient.  --prior: the smoothness prior's scale,
    a number, or chosen by the evidence or by leaving structures out ("loo", "press");
    --scaled R: a ridge of R times the diagonal of the Gram matrix on the feature columns."""
    import warnings

    import jax
    jax.config.update("jax_enable_x64", True)
    from ..operators import io
    from ..operators.features import build_site_basis
    from ..operators.fit import METRICS, fit_centred
    from . import density as D
    try:
        files = D.names(a.data)[:a.max_structures]
    except ValueError as e:
        raise UsageError(str(e)) from None
    train = [D.read(f, overlap="lazy" if a.metric != "plain" else False) for f in files]
    shells = train[0].shells
    if any(d.shells != shells for d in train):
        raise UsageError(f"{a.data}: the structures do not share one auxiliary basis")
    have = all(d.overlap is not None for d in train)
    metric = a.metric or ("overlap" if have else "plain")
    if metric not in METRICS:
        raise UsageError(f"--metric: one of {', '.join(METRICS)}")
    if metric == "overlap" and not have:
        raise UsageError("--metric overlap: not every structure of --data has the overlap of the auxiliary functions (--metric plain fits without it)")
    for key, what in (("fit_metric", "definitions of the coefficients (fit_metric)"), (("kpts", "kpt_offset", "smearing", "width"), "k-points or smearing")):
        vals = {json.dumps([(d.info or {}).get(k) for k in ([key] if isinstance(key, str) else key)]) for d in train}
        if len(vals) > 1:
            warnings.warn(f"aj density fit: the training structures record different {what}: {sorted(vals)}", stacklevel=2)
    prior = None if a.prior == "none" else a.prior if a.prior in PRIORS else float(a.prior)
    if prior is not None and a.scaled is not None:
        raise UsageError("--prior and --scaled: give one (--prior none with --scaled)")
    els, lam = [int(z) for z in shells.species], shells.lmax
    radials = tuple(a.site_radials) + (5,) * (a.order - len(a.site_radials))            # higher orders: few radial functions,
    lmax = tuple(a.lmax or [lam]) + (min(3, lam),) * (a.order - len(a.lmax or [lam]))   # low l (the order-1 terms carry the density)
    sb = build_site_basis(els, order=a.order, totaldegree=40, lmax_feat=lam, rcut=a.site_rcut, wL=0.5, nmax_by_order=radials[:a.order],
                          lmax_by_order=lmax[:a.order], natural_parity=True).with_table()
    print(f"{len(train)} structures, {sum(len(d.numbers) for d in train)} atoms, elements {els}, auxiliary l up to {lam}; site features of order {a.order} within "
          f"{a.site_rcut:g}; metric {metric}, prior {a.prior}" + (f", scaled ridge {a.scaled:g}" if a.scaled is not None else ""), flush=True)
    t = time.perf_counter()
    model, rep = fit_centred(sb, [d.centred() for d in train], channels=(D.CHANNEL,), metric=metric, overlaps=[d.overlap for d in train] if metric == "overlap" else None,
                             ridge=a.ridge, sigma_c=prior, scaled=a.scaled)
    fit_s = time.perf_counter() - t
    clipped = rep.pop(("clipped",), None)
    n_coef = int(sum(np.asarray(c).size for c in model.coef.values()))
    rows = np.array([r[0] for r in rep.values()], float)
    resid = float(np.sqrt((rows * np.array([np.asarray(r[2], float).ravel()[0] for r in rep.values()]) ** 2).sum() / rows.sum()))
    n_rows = int(rows.sum()) if metric == "overlap" else int(sum(len(d.coefficients) for d in train))
    print(f"{n_rows} rows, {n_coef} parameters; training residual, rms over rows" + (" in the overlap metric" if metric == "overlap" else "") + f": {resid:.6g}", flush=True)
    if clipped and any(clipped):
        print(f"overlaps not positive definite: eigenvalues raised per structure {list(clipped)}")
    print(f"fitted in {fit_s:.1f} s", flush=True)
    charges = _density_charges(train)
    if charges is None:
        warnings.warn("aj density fit: the data has neither charges nor electron counts: the model's predictions will not be corrected to an electron count", stacklevel=2)
    rec = dict(electrons={str(z): n for z, n in D.electrons_per_atom(train).items()})
    if charges is not None:
        rec["charges"] = {str(z): [float(v) for v in q] for z, q in charges.items()}
    info = dict(command="aj density fit", data=os.path.abspath(a.data), n_train=len(train), structures=[os.path.basename(f) for f in files], metric=metric,
                prior=a.prior, scaled=a.scaled, ridge=a.ridge, order=a.order, site_rcut=a.site_rcut, site_radials=list(radials[:a.order]), lmax=list(lmax[:a.order]),
                fit_s=round(fit_s, 1), n_coef=n_coef, n_rows=n_rows, training_residual=resid, clipped=list(clipped or []), density=rec,
                fit_metric=(train[0].info or {}).get("fit_metric"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    io.save(a.out, centred=model, info=info)             # before the validation: a failure there does not lose the fit
    if a.validation:
        print(f"validation set {a.validation}:")
        _, s = density_score(D.DensityPredictor((model, rec)), a.validation)
        info["validation"] = dict(data=os.path.abspath(a.validation), metrics=s)
        io.save(a.out, centred=model, info=info)
    print(f"wrote {a.out}")


def cmd_density_predict(a):
    """The predicted density of every structure of --data, one density file each under --out:
    the coefficients corrected to the electron count, and in the file's info `dN`, the count
    error before the correction, which is also printed: a large one marks a structure unlike
    the training set."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from . import density as D
    p = D.DensityPredictor(a.model)
    todo = density_structures(a.data)
    os.makedirs(a.out, exist_ok=True)
    for name, at, d in todo:
        t = time.perf_counter()
        pred = p.predict(at, a.nelectrons if a.nelectrons is not None else None if d is None else d.nelectrons)
        D.write(os.path.join(a.out, name + ".npz"), pred)
        dN = pred.info["dN"]
        print(f"{name}: {len(at)} atoms, electron count error before the correction " + ("-" if dN is None else f"{dN:+.5f}") + f", {time.perf_counter() - t:.2f} s", flush=True)
    print(f"wrote {len(todo)} structures to {a.out}")


def cmd_density_eval(a):
    """`aj eval` of a density model: `density_score` on the reference densities under --data."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from . import density as D
    density_score(D.DensityPredictor(a.model), a.data, getattr(a, "metrics_out", None))


def density_add_parser(sub):
    """Mount `density` on the `aj` subparsers."""
    den = sub.add_parser("density", help="electron-density models: fit to the density's expansion coefficients, predict them")
    ds = den.add_subparsers(dest="density_cmd", required=True)
    f = ds.add_parser("fit", help="fit an electron-density model to density files; writes a model file", description=cmd_density_fit.__doc__)
    f.add_argument("--data", required=True, help="training structures: a directory of density files (*.npz), or of DeepH directories with density.npz")
    f.add_argument("--out", required=True, help="the model file (.npz)")
    f.add_argument("--validation", default=None, help="density files to score the fitted model on (as aj eval)")
    f.add_argument("--metric", choices=["overlap", "plain"], default=None, help="overlap: the L2 norm of the density error (default when every structure "
                   "has the overlap); plain: least squares on each coefficient")
    f.add_argument("--prior", default="none", help="smoothness prior: none (default), its scale as a number, or evidence, loo, press")
    f.add_argument("--scaled", type=float, default=1e-6, metavar="R", help="ridge of R times the diagonal of the Gram matrix on every feature column "
                   "(default 1e-6; used with --prior none)")
    f.add_argument("--order", type=int, default=1, help="correlation order of the site features (default 1)")
    f.add_argument("--site-rcut", type=float, default=7.0, help="cutoff of the site features (default 7.0)")
    f.add_argument("--site-radials", type=int, nargs="+", default=[12], help="radial functions of the site features, per correlation order (default 12; higher orders 5)")
    f.add_argument("--lmax", type=int, nargs="+", default=None, metavar="L", help="largest l of the one-particle functions, per correlation order "
                   "(default: the largest l of the auxiliary basis; higher orders 3)")
    f.add_argument("--ridge", type=float, default=1e-10, help="ridge on every coefficient (default 1e-10)")
    f.add_argument("--max-structures", type=int, default=None, help="use the first N structures of --data (name order)")
    p = ds.add_parser("predict", help="write the predicted density of structures (density files)", description=cmd_density_predict.__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, help="structures: an extxyz file, or a directory of density files")
    p.add_argument("--out", required=True, help="output directory: one density file per structure")
    p.add_argument("--nelectrons", type=float, default=None, help="electrons per cell (default: the structure's record, else the model's electrons per atom)")
    return den


def density_run(a):
    if a.density_cmd == "fit" and a.prior != "none":
        a.scaled = None                                  # the default ridge belongs to --prior none
    {"fit": cmd_density_fit, "predict": cmd_density_predict}[a.density_cmd](a)


def _add_spectrum_args(p, kpts):
    p.add_argument("--kpts", type=int, nargs=3, default=kpts, help=f"Gamma-centred k-point grid (default {' '.join(map(str, kpts))})")
    p.add_argument("--nelectrons", default=None,
                   help="electrons the bands hold: a number per cell, or Z:n,Z:n per atom of each element (default: what the model "
                        "file records from its fit, else the atomic numbers)")
    p.add_argument("--width", type=float, default=0.1, help="smearing width in the model's energy unit (default 0.1)")
    p.add_argument("--smearing", choices=["fermi-dirac", "gaussian"], default="fermi-dirac")
    p.add_argument("--window", type=float, nargs=2, default=[-12.0, 5.0],
                   help="eigenvalues scored: from LO to HI of the reference Fermi level (default -12 5)")
    p.add_argument("--threads", type=int, default=1, help="threads of the off-site evaluation on a CPU (default 1)")
    p.add_argument("--overlap-threshold", type=float, default=0.0, metavar="X",
                   help="drop the combinations of orbitals whose overlap eigenvalue at a k-point is below X before diagonalising "
                        "(default 0: none); 1e-3 for a diffuse basis whose predicted bands are far off although the blocks are close")


def add_parser(sub):
    """Mount `ham` on the `aj` subparsers."""
    ham = sub.add_parser("ham", help="Hamiltonian (operator) models: fit to DFT blocks, predict H and S, bands and density of states")
    hs = ham.add_subparsers(dest="ham_cmd", required=True)
    f = hs.add_parser("fit", help="fit a Hamiltonian model to the H and S blocks of DFT structures; writes a model file", description=cmd_fit.__doc__)
    f.add_argument("--data", required=True, help="training structures with blocks: a directory of operator caches (*.npz) or DeepH-layout subdirectories")
    f.add_argument("--out", required=True, help="the model file (.npz)")
    f.add_argument("--validation", default=None, help="structures with blocks to score the fitted model on (as aj eval)")
    f.add_argument("--channels", nargs="+", default=None, help="operator channels to fit (default: every channel of the data, e.g. H S)")
    f.add_argument("--data-memory", type=float, default=8.0, metavar="GB",
                   help="blocks of the training structures kept in memory (default 8); a larger set is read from disk again on each pass of the fit")
    f.add_argument("--max-structures", type=int, default=None, help="use the first N structures of --data (name order)")
    f.add_argument("--bonds", type=int, default=0, help="fit to a random N off-site blocks of each structure (default 0: all)")
    f.add_argument("--seed", type=int, default=0, help="seed of the --bonds choice")
    f.add_argument("--site-rcut", type=float, default=7.0, help="cutoff of the site features (default 7.0)")
    f.add_argument("--site-radials", type=int, default=12, help="radial functions of the site features (default 12)")
    f.add_argument("--bond-rcut", type=float, default=None, help="longest bond with a block (default: the longest in the data)")
    f.add_argument("--bond-radials", type=int, default=12, help="radial functions of the two-centre features (default 12)")
    f.add_argument("--coupled", type=int, nargs=2, default=[2, 2], metavar=("N", "L"),
                   help="site-coupled features: radial functions and largest bond harmonic (default 2 2; 0 0: off)")
    f.add_argument("--env", type=float, nargs=5, default=[5.0, 8, 4, 6, 4], metavar=("R", "N", "L", "NB", "LB"),
                   help="bond environment: cutoff, radial functions and largest harmonic about the midpoint, radial functions and "
                        "largest harmonic of the bond it couples with (default 5 8 4 6 4; 5 6 4 4 4, the default to 0.7.2, is smaller and faster)")
    f.add_argument("--no-env", action="store_true", help="no bond-environment features (faster, less accurate)")
    f.add_argument("--prior", default="none", help="smoothness prior: none (default), its scale as a number, or evidence, loo, press")
    f.add_argument("--noise-floor", type=float, default=None, metavar="SIGMA",
                   help="with --prior evidence: the smallest noise of H the evidence may select, in the data's unit "
                        "(the scatter of H between SCF runs; approximately 1e-7 eV to 3e-6 eV for sc_accuracy_rho 1e-5)")
    f.add_argument("--ridge", type=float, default=1e-10, help="ridge on every coefficient (default 1e-10)")
    _add_spectrum_args(f, [3, 3, 3])
    p = hs.add_parser("predict", help="write the predicted H and S blocks of structures (DeepH layout)", description=cmd_predict.__doc__)
    p.add_argument("--model", required=True); p.add_argument("--data", required=True, help="structures: an extxyz file, or a directory of structures with blocks")
    p.add_argument("--out", required=True, help="output directory: one subdirectory per structure")
    p.add_argument("--no-overlap", action="store_true", help="write H only")
    p.add_argument("--channel", default=None, help="Hamiltonian channel (default H; H_up or H_dn of a spin model)")
    p.add_argument("--threads", type=int, default=1, help="threads of the off-site evaluation on a CPU (default 1)")
    p.set_defaults(nelectrons=None, width=0.1, smearing="fermi-dirac")
    b = hs.add_parser("bands", help="bands on a high-symmetry path and the density of states of one structure", description=cmd_bands.__doc__)
    b.add_argument("--model", required=True)
    b.add_argument("--structure", required=True, help="an extxyz file, an operator cache (.npz) or a DeepH directory; or a directory of those with --index")
    b.add_argument("--index", type=int, default=None, help="which structure of --structure (default the first)")
    b.add_argument("--path", default=None, help="special points of the band path, e.g. GXMG (default: ASE's path for the lattice)")
    b.add_argument("--npoints", type=int, default=200, help="k-points on the path (default 200)")
    b.add_argument("--channel", default=None, help="Hamiltonian channel (default H; H_up or H_dn of a spin model)")
    b.add_argument("--no-reference", action="store_true", help="do not compute the reference's bands from the structure's own blocks")
    b.add_argument("--out", default=None, help="arrays of the bands, the grid eigenvalues and the Fermi levels (.npz)")
    b.add_argument("--plot", default=None, help="figure file (needs matplotlib)")
    b.add_argument("--plot-window", type=float, nargs=2, default=[-13.0, 6.0], help="energy range of the figure from the Fermi level (default -13 6)")
    b.add_argument("--dos-width", type=float, default=0.1, help="Gaussian width of the density of states in the figure (default 0.1)")
    _add_spectrum_args(b, [8, 8, 8])
    return ham


def run(a):
    {"fit": cmd_fit, "predict": cmd_predict, "bands": cmd_bands}[a.ham_cmd](a)
