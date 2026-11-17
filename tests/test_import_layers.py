"""The layering of `ace_jax` (docs/dev/specs/2026-11-17-package-structure-design.md, section 1),
and a check that every `ace_jax` name a bench script imports exists.

A module may import its own layer and the layers above it in `LAYERS`, never one below; within
the applications layer, potentials and operators do not import each other.  Imports inside
functions count.  Everything is read with `ast`: nothing under `src/` or `bench/` is executed.

`ALLOWED` lists the edges that break the rule today, each with the step of the spec that removes
it.  The test fails on an edge that is not listed AND on a listed edge that no longer exists, so
the list can only shrink."""
import ast
import builtins
import fnmatch
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ace_jax"

# (pattern, layer, side): first match wins; "" is ace_jax/__init__.py.  A package's __init__ takes
# its package's pattern, so `eval` (which re-exports the models) is an application module.
# `eval.io` is one too, not framework as in the spec's table: `load` constructs the models.
FRAMEWORK, LINEAR, UQ, APP, FRONT = range(5)
LAYERS = [
    *[(f"basis.{m}", FRAMEWORK, None) for m in
      ("spec", "coupling", "prior", "export", "embedding", "radial_init", "radial_ace1")],
    *[(f"eval.{m}", FRAMEWORK, None) for m in
      ("harmonics", "radial", "splinify", "nlist", "aa_dag", "product")],
    ("_moved", FRAMEWORK, None),
    ("fit.linear", LINEAR, None), ("fit.linear.*", LINEAR, None),
    ("uq", UQ, None), ("uq.*", UQ, None),
    ("operators", APP, "operators"), ("operators.*", APP, "operators"),
    ("ham", APP, "operators"), ("ham.*", APP, "operators"),
    ("", FRONT, None), ("cli", FRONT, None), ("runfile", FRONT, None),
    ("tutorials", FRONT, None), ("tutorials.*", FRONT, None),
    ("*", APP, "potentials"),
]

# (importer, imported): the step of the spec's order of work that removes the edge.
ALLOWED = {
}


def _modules(root):
    """{dotted name relative to the package: (path, is_package)}."""
    out = {}
    for p in sorted(root.rglob("*.py")):
        parts = list(p.relative_to(root).with_suffix("").parts)
        pkg = parts[-1] == "__init__"
        out[".".join(parts[:-1] if pkg else parts)] = (p, pkg)
    return out


def _stub_target(path):
    """The new dotted module of a moved-module stub (`__getattr__ = moved(__name__, "ace_jax.x")`), else None."""
    for n in ast.parse(path.read_text()).body:
        if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "moved"
                and [getattr(t, "id", "") for t in n.targets] == ["__getattr__"]):
            return n.value.args[1].value[8:]
    return None


ALL = _modules(SRC)
STUBS = {m: t for m, (p, _) in ALL.items() if (t := _stub_target(p))}
MODS = {m: v for m, v in ALL.items() if m not in STUBS}      # the layering and the bench check see no stub


def _resolve(name):
    """The longest prefix of the dotted `name` that is a module ('' = the package itself)."""
    parts = name.split(".") if name else []
    for k in range(len(parts), 0, -1):
        if ".".join(parts[:k]) in MODS:
            return ".".join(parts[:k])
    return ""


def _targets(node, mod, pkg):
    """The ace_jax modules an import statement in module `mod` names (a `from x import y` names the
    submodule x.y when there is one, else x)."""
    if isinstance(node, ast.Import):
        return [_resolve(a.name[8:]) for a in node.names if a.name == "ace_jax" or a.name.startswith("ace_jax.")]
    if not isinstance(node, ast.ImportFrom):
        return []
    if node.level:
        base = mod.split(".") if mod else []
        base = (base if pkg else base[:-1])
        base = base[:len(base) - (node.level - 1)]
        tgt = ".".join(base + (node.module.split(".") if node.module else []))
    elif node.module and (node.module == "ace_jax" or node.module.startswith("ace_jax.")):
        tgt = node.module[8:]
    else:
        return []
    return [full if (full := f"{tgt}.{a.name}".strip(".")) in MODS else _resolve(tgt) for a in node.names]


def _edges():
    out = set()
    for mod, (p, pkg) in MODS.items():
        for n in ast.walk(ast.parse(p.read_text())):
            out |= {(mod, t) for t in _targets(n, mod, pkg) if t != mod}
    return out


def _layer(mod):
    for pat, layer, side in LAYERS:
        if fnmatch.fnmatchcase(mod, pat):
            return layer, side
    raise AssertionError(mod)


def _violates(a, b):
    (la, sa), (lb, sb) = _layer(a), _layer(b)
    return lb > la or (la == lb == APP and sa != sb)


def test_layers_cover_real_modules():
    """A pattern that matches nothing is a module that moved without its layer."""
    stale = [pat for pat, _, side in LAYERS
             if side != "operators" and not any(fnmatch.fnmatchcase(m, pat) for m in MODS)]
    assert not stale, f"LAYERS patterns that match no module: {stale}"


def test_no_upward_imports():
    bad = {e for e in _edges() if _violates(*e)}
    new = sorted(bad - set(ALLOWED))
    assert not new, "imports of a lower layer (spec section 1):\n" + "\n".join(f"  {a} -> {b}" for a, b in new)
    fixed = sorted(set(ALLOWED) - bad)
    assert not fixed, f"edges that no longer exist: remove them from ALLOWED: {fixed}"


def _from_target(node, mod, pkg):
    """The dotted ace_jax module a `from ... import` names (None: not an ace_jax import)."""
    if node.level:
        base = mod.split(".") if mod else []
        base = (base if pkg else base[:-1])
        base = base[:len(base) - (node.level - 1)]
        return ".".join(base + (node.module.split(".") if node.module else []))
    if node.module and (node.module == "ace_jax" or node.module.startswith("ace_jax.")):
        return node.module[8:]
    return None


def test_internal_imports_resolve():
    """Every `from <ace_jax module> import name` in `src`, inside functions too, names a module that
    exists and a name it defines: a lazy import of a moved name fails here, not at its first call."""
    bad = []
    for mod, (p, pkg) in ALL.items():
        for n in ast.walk(ast.parse(p.read_text())):
            if not isinstance(n, ast.ImportFrom) or (tgt := _from_target(n, mod, pkg)) is None:
                continue
            if tgt not in MODS:
                bad.append(f"{mod}:{n.lineno}: no module ace_jax.{tgt}")
                continue
            bad += [f"{mod}:{n.lineno}: no {a.name!r} in ace_jax.{tgt}" for a in n.names
                    if a.name != "*" and not _has(tgt, a.name)]
    assert not bad, "\n".join(bad)


def test_eval_imports_no_other_package():
    """`eval` is the bottom of the potential stack (spec section 3): it maps edge vectors to site
    energies and knows no calculator, exporter, basis builder or fit, not even inside a function."""
    back = sorted((a, b) for a, b in _edges() if a.split(".")[0] == "eval" and b.split(".")[0] != "eval")
    assert not back, back


def test_stubs_point_at_modules_and_nothing_here_uses_them():
    """A stub names a module that exists; `src`, `tests` and `bench` import the new path (so the stubs
    can be deleted without a search)."""
    assert STUBS and all(t in MODS for t in STUBS.values()), {m: t for m, t in STUBS.items() if t not in MODS}
    users = []
    for d in ("src", "tests", "bench", "docs/user"):
        for p in sorted((ROOT / d).rglob("*.py")):
            if p == pathlib.Path(__file__) or ".venv" in p.parts:
                continue
            users += [(p.relative_to(ROOT).as_posix(), n.lineno, s) for n in ast.walk(ast.parse(p.read_text()))
                      if isinstance(n, ast.Import | ast.ImportFrom)
                      for s in STUBS if f"ace_jax.{s}" in {getattr(n, "module", None), *(a.name for a in n.names)}
                      or (isinstance(n, ast.ImportFrom) and n.module and any(f"{n.module}.{a.name}" == f"ace_jax.{s}" for a in n.names))]
    assert not users, users


def test_stub_warns_and_serves_public_names_only():
    import importlib

    import pytest
    for old, new in sorted(STUBS.items()):
        mod = importlib.import_module(f"ace_jax.{new}")
        name = next(n for n in sorted(vars(mod)) if not n.startswith("_"))
        with pytest.warns(DeprecationWarning, match=f"ace_jax.{old} moved to ace_jax.{new}"):
            assert getattr(importlib.import_module(f"ace_jax.{old}"), name) is getattr(mod, name)
    old = next(iter(STUBS))
    with pytest.raises(AttributeError):
        _ = importlib.import_module(f"ace_jax.{old}")._private


# ---- bench: every ace_jax import names something that exists ------------------------------------

def _defined(mod, seen=()):
    """Top-level names of an ace_jax module (assignments, defs, classes, imports, and the names a
    module `__getattr__` serves), with `import *` followed; None = anything (a dynamic module)."""
    p, pkg = MODS[mod]
    tree = ast.parse(p.read_text())
    names, stack = set(), list(tree.body)
    while stack:
        n = stack.pop()
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(n.name)
            if n.name == "__getattr__":
                names |= {c.value for c in ast.walk(n) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
            continue
        if isinstance(n, ast.Import):
            names |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name != "*":
                    names.add(a.asname or a.name)
                elif (t := _targets(n, mod, pkg)) and t[0] not in seen:
                    sub = _defined(t[0], (*seen, mod))
                    if sub is None:
                        return None
                    names |= {s for s in sub if not s.startswith("_")}
        else:
            names |= {t.id for c in ast.walk(n) if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Store)
                      for t in [c]}
            stack += [c for c in ast.iter_child_nodes(n) if isinstance(c, ast.stmt | ast.ExceptHandler)]
    return names


_DEFINED = {}


def _has(mod, attr):
    """Does ace_jax module `mod` have `attr` (a submodule or a top-level name)?"""
    if f"{mod}.{attr}".strip(".") in MODS:
        return True
    if mod not in _DEFINED:
        _DEFINED[mod] = _defined(mod)
    d = _DEFINED[mod]
    return d is None or attr in d or hasattr(builtins.object, attr) or attr in ("__file__", "__name__", "__doc__")


def _module_of(dotted):
    """'ace_jax.fit.rows' -> 'fit.rows' when that is a module, else None."""
    if dotted != "ace_jax" and not dotted.startswith("ace_jax."):
        return None
    m = dotted[8:]
    return m if m in MODS else None


def _scopes(tree):
    """(scope node, its own nodes): the module and each function, a nested function's nodes its own."""
    todo = [tree]
    while todo:
        scope, own, stack = todo.pop(), [], []
        stack += ast.iter_child_nodes(scope)
        while stack:
            n = stack.pop()
            own.append(n)
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                todo.append(n)
            else:
                stack += ast.iter_child_nodes(n)
        yield scope, own


def _script_problems(path):
    """Imports and module attribute reads in one script that name nothing in the source.  A name is
    followed as a module in the scope that imports it (and, for a module-level import, in the
    functions that do not rebind it)."""
    tree = ast.parse(path.read_text())
    bad, top = [], {}
    for scope, own in _scopes(tree):                  # the module scope comes first
        bound = {}                                    # local name -> the ace_jax module it is bound to
        for n in own:
            if isinstance(n, ast.Import):
                for a in n.names:
                    if a.name == "ace_jax" or a.name.startswith("ace_jax."):
                        if _module_of(a.name) is None:
                            bad.append((n.lineno, f"import {a.name}"))
                        elif a.asname:
                            bound[a.asname] = _module_of(a.name)
                        else:
                            bound["ace_jax"] = ""
            elif isinstance(n, ast.ImportFrom) and not n.level and n.module and n.module.split(".")[0] == "ace_jax":
                m = _module_of(n.module)
                if m is None:
                    bad.append((n.lineno, f"from {n.module} import ..."))
                    continue
                for a in n.names:
                    if a.name == "*":
                        continue
                    if not _has(m, a.name):
                        bad.append((n.lineno, f"from {n.module} import {a.name}"))
                    elif (sub := f"{m}.{a.name}".strip(".")) in MODS:
                        bound[a.asname or a.name] = sub
        if scope is tree:
            top = bound
        else:
            rebound = {c.id for c in own if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Store)}
            rebound |= {a.arg for a in ast.walk(scope.args) if isinstance(a, ast.arg)}
            bound = {**{k: v for k, v in top.items() if k not in rebound}, **bound}
        for n in own:                                 # aj.fit.stats.merge, stats.merge, ...
            if not isinstance(n, ast.Attribute):
                continue
            chain, v = [n.attr], n.value
            while isinstance(v, ast.Attribute):
                chain.append(v.attr)
                v = v.value
            if not (isinstance(v, ast.Name) and v.id in bound):
                continue
            mod = bound[v.id]
            for attr in reversed(chain):
                if not _has(mod, attr):
                    bad.append((n.lineno, f"{v.id}.{'.'.join(reversed(chain))}: no {attr!r} in ace_jax.{mod}".rstrip(".")))
                    break
                if (sub := f"{mod}.{attr}".strip(".")) not in MODS:
                    break
                mod = sub
    return sorted(set(bad))


def _bench_scripts():
    return sorted(p for p in (ROOT / "bench").rglob("*.py") if ".venv" not in p.parts)


def test_bench_imports_exist():
    bad = {p.relative_to(ROOT).as_posix(): b for p in _bench_scripts() if (b := _script_problems(p))}
    assert not bad, "bench scripts import ace_jax names that do not exist:\n" + "\n".join(
        f"  {f}:{line}: {what}" for f, b in bad.items() for line, what in b)


def test_bench_check_sees_a_missing_name(tmp_path):
    """The check itself: a name that exists passes, a missing attribute, module and method chain fail."""
    s = tmp_path / "s.py"
    s.write_text("import ace_jax as aj\nfrom ace_jax.eval import nlist, load\nfrom ace_jax.eval.nlist import sparse_graph\n"
                 "def f():\n    from ace_jax import ACECalculator\n    return aj.load, nlist.sparse_graph, aj.eval.nlist.dense_graph\n")
    assert _script_problems(s) == []
    s.write_text("from ace_jax.eval.nlist import no_such_name\nfrom ace_jax.nowhere import x\n"
                 "from ace_jax.eval import nlist\nimport ace_jax as aj\nnlist.missing()\naj.eval.nlist.gone\n"
                 "def f(nlist):\n    return nlist.anything\ndef g():\n    return nlist.gone_too\n")
    assert [line for line, _ in _script_problems(s)] == [1, 2, 5, 6, 10]
