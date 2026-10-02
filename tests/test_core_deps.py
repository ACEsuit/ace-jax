"""`pip install ace-jax` must be able to import every ace_jax module: a module-level
third-party import is a core dependency (an optional one is imported lazily, inside
the function that needs it, as blackjax is for the pathfinder rung)."""
import ast
import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIST = {"yaml": "pyyaml", "ace_jax_coupling": "ace-jax-coupling"}     # import name -> distribution


def _module_level_imports():
    out = {}
    for p in (ROOT / "src" / "ace_jax").rglob("*.py"):
        for n in ast.parse(p.read_text()).body:
            names = ([a.name for a in n.names] if isinstance(n, ast.Import) else
                     [n.module] if isinstance(n, ast.ImportFrom) and n.level == 0 else [])
            for m in names:
                top = m.split(".")[0]
                if top != "ace_jax" and top not in sys.stdlib_module_names:
                    out.setdefault(top, p.relative_to(ROOT).as_posix())
    return out


def test_module_level_imports_are_core_dependencies():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    core = {re.split(r"[<>=!~;\[ ]", d, maxsplit=1)[0].strip().lower() for d in deps}
    missing = {m: f for m, f in _module_level_imports().items() if DIST.get(m, m).lower() not in core}
    assert not missing, f"module-level imports outside the core dependencies: {missing}"


def test_pathfinder_without_blackjax_says_what_to_install(monkeypatch):
    import pytest
    from ace_jax.fit.ladder import run_pathfinder
    monkeypatch.setitem(sys.modules, "blackjax", None)            # import blackjax -> ImportError
    with pytest.raises(ImportError, match=r"ace-jax\[gp\]"):
        run_pathfinder(lambda a: 0.0, None, None)
