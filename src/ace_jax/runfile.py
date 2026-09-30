"""fit.yaml: a whole `aj fit` run in one file.

Top-level keys are `aj fit` flag dests (dashes -> underscores); a nested
`basis:` block holds the `aj basis` dests (or `model:` names a .npz instead).
YAML values become argparse defaults, so explicit command-line flags win.
Relative paths resolve against the YAML file's directory, so a run file keeps
working when it is run from elsewhere; `out` stays relative to the cwd.
"""
import argparse
import difflib
import pathlib

import yaml

PATH_KEYS = ("train", "test", "ood", "data", "model", "baseline", "init", "embedding")
BASIS_PATH_KEYS = ("embedding", "coupling_cache_dir")
IGNORED = ("provenance",)
COMMA_KEYS = ("rungs", "elements")


def _resolve(v, base):
    if not isinstance(v, str) or v == "identity":
        return v
    p = pathlib.Path(v).expanduser()
    return str(p if p.is_absolute() else base / p)


def read(path):
    """The run file as a dict, relative paths resolved against its directory."""
    path = pathlib.Path(path)
    d = yaml.safe_load(path.read_text()) or {}
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected a mapping of flag names to values")
    if "basis" in d and not isinstance(d["basis"], dict):
        raise ValueError(f"{path}: 'basis' must be a mapping of basis settings")
    base = path.resolve().parent
    for k in PATH_KEYS:
        if k in d:
            d[k] = _resolve(d[k], base)
    for k in BASIS_PATH_KEYS:
        if k in d.get("basis", {}):
            d["basis"][k] = _resolve(d["basis"][k], base)
    return d


def _unknown(where, key, known):
    hint = difflib.get_close_matches(key, known, n=1)
    return f"{where}: unknown key '{key}'" + (f" (did you mean '{hint[0]}'?)" if hint else "")


def _dests(parser):
    return {a.dest for a in parser._actions if a.dest not in ("help", argparse.SUPPRESS)}


def _value(k, v):
    return ",".join(map(str, v)) if k in COMMA_KEYS and isinstance(v, (list, tuple)) else v


def defaults_for(parser, cfg, *, basis_fields, basis_dest_map):
    """Flat `dest -> value` defaults for `parser` from a run-file dict, or
    ValueError naming the offending key.  `basis_fields`: the keys allowed in
    the `basis:` block (empty: no block, every key is a parser dest);
    `basis_dest_map`: basis key -> parser dest where they differ."""
    dests = _dests(parser)
    basis_dests = {basis_dest_map.get(f, f) for f in basis_fields}
    top_known = sorted(dests - basis_dests - {"config"}) + (["basis"] if basis_fields else [])
    if "model" in cfg and "basis" in cfg:
        raise ValueError("fit.yaml: 'model' and 'basis' are alternatives: give one")
    out = {}
    for k, v in cfg.items():
        if k in IGNORED or (k == "basis" and basis_fields):
            continue
        if k not in top_known:
            raise ValueError(_unknown("fit.yaml", k, top_known))
        out[k] = _value(k, v)
    for k, v in (cfg.get("basis") or {}).items() if basis_fields else ():
        if k not in basis_fields:
            raise ValueError(_unknown("fit.yaml basis", k, list(basis_fields)))
        dest = basis_dest_map.get(k, k)
        if dest in dests:
            out[dest] = _value(k, v)
    return out


def explicit_dests(parser, argv):
    """Dests given explicitly in `argv` (a SUPPRESS-default shadow parse)."""
    shadow = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    for a in parser._actions:
        if not a.option_strings or a.dest == "help":
            continue
        if a.nargs == 0:
            shadow.add_argument(*a.option_strings, dest=a.dest, action="store_const", const=a.const)
        else:
            shadow.add_argument(*a.option_strings, dest=a.dest, nargs=a.nargs)
    ns, _ = shadow.parse_known_args(argv)
    return set(vars(ns))


def resolved(a, data, *, fit_dests, argv):
    """The effective run as a run-file dict: every `aj fit` dest with its value
    (paths absolute), the basis spelled out (elements from the data included) or
    the model path, r0 explicit, and a `provenance` block that is never read
    back.  `aj fit --config <this>` reproduces the run."""
    import datetime
    import importlib.metadata
    import os

    from .basis.coupling import backend_id
    from .basis.model import BasisSpec

    basis_dests = set(BasisSpec.FIELDS) | {"basis_embedding"}
    d = {k: getattr(a, k) for k in sorted(fit_dests)
         if k not in basis_dests and k not in ("config", "cmd", "model")}
    for k in PATH_KEYS:
        if isinstance(d.get(k), str):
            d[k] = os.path.abspath(d[k])
    if isinstance(d.get("rungs"), str):
        d["rungs"] = [r.strip() for r in d["rungs"].split(",")]
    d["r0"] = float(a.r0 if a.r0 is not None else data.r0)
    coupling = None
    if a.model is not None:
        d["model"] = os.path.abspath(a.model)
    else:
        from ase.data import chemical_symbols
        basis = {f: getattr(a, f) for f in BasisSpec.FIELDS if f not in ("embedding", "elements")}
        basis["elements"] = [chemical_symbols[int(z)] for z in data.meta["elements"]]
        emb = a.basis_embedding
        basis["embedding"] = emb if emb in (None, "identity") else os.path.abspath(emb)
        if basis["coupling_cache_dir"] is not None:
            basis["coupling_cache_dir"] = os.path.abspath(basis["coupling_cache_dir"])
        d["basis"] = {f: basis[f] for f in BasisSpec.FIELDS}
        coupling = {"backend": backend_id()}
        try:
            import ace_jax_coupling
            info = ace_jax_coupling.build_info()
            coupling.update({k: info[k] for k in ("et_repo", "et_rev", "platform") if k in info})
        except Exception:                 # the cache served the basis; no library to describe
            pass
    try:
        version = importlib.metadata.version("ace-jax")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    d["provenance"] = {
        "ace_jax": version, "coupling": coupling, "command": " ".join(["aj", *argv]),
        "created": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
    return d


def write(path, d):
    """Write a run file (keys in insertion order, provenance last)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(d, sort_keys=False, default_flow_style=None)
    path.write_text("# resolved run file written by `aj fit`: `aj fit --config <this file>` reproduces it\n"
                    + body)
