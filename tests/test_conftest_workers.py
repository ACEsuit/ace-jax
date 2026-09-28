"""The conftest's default-parallel hook must never fire inside an xdist worker:
workers re-run pytest_cmdline_main with numprocesses=None, and setting it there
made every worker spawn workers recursively (a process storm, no tests run)."""
import types

import conftest


def _config(**opt):
    option = types.SimpleNamespace(numprocesses=None, **opt)
    return types.SimpleNamespace(
        option=option, args=["tests"], getini=lambda k: ["tests"], getoption=lambda k, d=None: False,
        pluginmanager=types.SimpleNamespace(hasplugin=lambda n: True))


def test_bare_run_on_the_controller_gets_workers(monkeypatch):
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setenv("ACEJAX_TEST_WORKERS", "3")
    c = _config(); conftest.pytest_cmdline_main(c)
    assert c.option.numprocesses == 3


def test_worker_is_left_serial(monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    c = _config(); conftest.pytest_cmdline_main(c)
    assert c.option.numprocesses is None
    monkeypatch.delenv("PYTEST_XDIST_WORKER")
    c = _config(); c.workerinput = {}; conftest.pytest_cmdline_main(c)
    assert c.option.numprocesses is None


def test_targeted_run_is_left_serial(monkeypatch):
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    c = _config(); c.args = ["tests/test_gp_hypers.py"]; conftest.pytest_cmdline_main(c)
    assert c.option.numprocesses is None
