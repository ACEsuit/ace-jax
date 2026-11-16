"""ace_jax.ham may import only the framework parts of ace-jax, and nothing outside
ham may import ham (design: docs/dev/specs/2026-11-16-hamiltonians-design.md,
Constraints).  Imports are read from the source, so a lazy import counts too."""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "ace_jax"
# framework modules ham may use; the potential applications (calc, export, cli,
# runfile, tutorials, batch, jaxmd, fit.pipeline, the GP arm) are not in the list
ALLOWED = ("ace_jax.ham", "ace_jax.eval", "ace_jax.basis", "ace_jax.fit.stats", "ace_jax.fit.objective",
           "ace_jax.fit.solve", "ace_jax.fit.hostcache", "ace_jax.fit.pops", "ace_jax.fit.conformal",
           "ace_jax.fit.jackknife", "ace_jax.fit.lml_host", "ace_jax.fit.xyz")


def _imports(path):
    pkg = ".".join(("ace_jax",) + path.relative_to(SRC).parts[:-1])
    out = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            base = pkg.rsplit(".", n.level - 1)[0] if n.level > 1 else pkg
            mod = (base + "." + n.module if n.module else base) if n.level else n.module
            out |= {mod} | {f"{mod}.{a.name}" for a in n.names}
    return {m for m in out if m.split(".")[0] == "ace_jax"}


def _under(mod, roots):
    return any(mod == r or mod.startswith(r + ".") for r in roots)


def test_ham_imports_only_the_framework():
    bad = {str(p.relative_to(SRC)): sorted(m for m in _imports(p) if not _under(m, ALLOWED) and m != "ace_jax")
           for p in (SRC / "ham").rglob("*.py")}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"ace_jax.ham imports outside the framework: {bad}"


def test_nothing_outside_ham_imports_ham():
    bad = {str(p.relative_to(SRC)): sorted(m for m in _imports(p) if _under(m, ("ace_jax.ham",)))
           for p in SRC.rglob("*.py") if "ham" not in p.relative_to(SRC).parts[:1]}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"modules outside ace_jax.ham import it: {bad}"
