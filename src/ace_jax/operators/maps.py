"""Room for compiled programs on a CPU.

XLA's CPU runtime (jaxlib 0.11) maps a few small memory regions for every kernel of a compiled
program and keeps them while JAX caches the executable.  A model of several elements compiles
one program per species pair and per design shape, and the band-energy program of a four-element
model alone takes about 50,000 regions, so a long fit or evaluation reaches the kernel's limit
(`vm.max_map_count`, 65,530 by default) and dies with "LLVM compilation error: Cannot allocate
memory" although memory is free (docs/dev/ham-results.md).  `make_room(need)` is called before
the stages that compile much: when fewer than `need` regions are left it drops JAX's compiled
programs (`jax.clear_caches()`), which are then compiled again as they are used.  Linux and the
CPU backend only; a no-op elsewhere."""
import gc
import sys

import jax

LIMIT_FILE, MAPS_FILE = "/proc/sys/vm/max_map_count", "/proc/self/maps"


def counts():
    """(regions mapped by this process, the kernel's limit), or None where they cannot be read."""
    try:
        with open(LIMIT_FILE) as f:
            limit = int(f.read())
        with open(MAPS_FILE) as f:
            return sum(1 for _ in f), limit
    except (OSError, ValueError):
        return None


def make_room(need, verbose=False):
    """Drop the compiled programs if fewer than `need` memory regions are left to this process.
    Returns True if it did."""
    if jax.default_backend() != "cpu":
        return False
    c = counts()
    if c is None or c[1] - c[0] >= need:
        return False
    jax.clear_caches(); gc.collect()
    if verbose:
        print(f"ace-jax: {c[0]} of {c[1]} memory regions in use (vm.max_map_count): compiled programs dropped", file=sys.stderr, flush=True)
    return True
