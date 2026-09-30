"""ACEJAX_NLIST pins the neighbour-list backend.  Neighbour order changes
summation order, and order-sensitive checks (the pipeline goldens, recorded
with ASE's list) need the backend they were recorded with."""
import pytest


@pytest.mark.parametrize("pin,expect", [("ase", "ase"), ("matscipy", {"matscipy", "ase"})])
def test_env_pins_the_backend(monkeypatch, pin, expect):
    from ace_jax.eval import nlist
    monkeypatch.setenv("ACEJAX_NLIST", pin)
    got = nlist.backend()
    assert got in expect if isinstance(expect, set) else got == expect
    if pin == "ase":
        assert not nlist.have_matscipy_neighbours() and not nlist.have_matscipy()


def test_unknown_pin_raises(monkeypatch):
    from ace_jax.eval import nlist
    monkeypatch.setenv("ACEJAX_NLIST", "bogus")
    with pytest.raises(ValueError, match="ACEJAX_NLIST"):
        nlist.backend()
