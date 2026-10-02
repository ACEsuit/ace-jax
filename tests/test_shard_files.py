"""ACEJAX_SHARD=i/n splits the suite by FILE (conftest.shard_files): module fixtures and a
module's first compile are paid once, in one shard, so recorded per-test durations add up
(per-test splitting scattered modules and each shard repaid their setup)."""
from conftest import shard_files


def test_shard_files_partitions_balances_and_is_deterministic():
    durations = {f"tests/test_{c}.py::t{i}": float(w) for c, n, w in
                 (("a", 3, 40), ("b", 10, 9), ("c", 2, 30), ("d", 1, 50), ("e", 4, 5), ("f", 6, 1))
                 for i in range(n)}
    files = sorted({k.split("::")[0] for k in durations} | {"tests/test_new.py"})   # one unrecorded file
    shards = [shard_files(files, durations, i, 3) for i in range(1, 4)]
    assert sorted(f for s in shards for f in s) == files                         # each file exactly once
    per_file = {}
    for k, v in durations.items():
        per_file[k.split("::")[0]] = per_file.get(k.split("::")[0], 0.0) + v
    load = [sum(per_file.get(f, 0.0) for f in s) for s in shards]
    lower = max(sum(per_file.values()) / 3, max(per_file.values()))              # no split beats this
    assert max(load) <= 4 / 3 * lower                                             # Graham's LPT bound
    assert shards == [shard_files(files, durations, i, 3) for i in range(1, 4)]   # same answer every time
