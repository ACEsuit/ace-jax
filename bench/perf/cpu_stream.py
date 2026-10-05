"""Achievable memory bandwidth on this host, as a roof for docs/dev/cpu-gap-profile.md:
a numpy copy (b[:] = a, 2 x 256 MB float64: read + write counted) on P processes,
process p pinned to CPU p, all at once.

    python bench/perf/cpu_stream.py [P ...]      # default 1 16
"""
import json
import subprocess
import sys
import time


def one(reps=10):
    import numpy as np
    a = np.ones(32 * 2**20)                       # 256 MB
    b = np.empty_like(a)
    b[:] = a
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); b[:] = a; ts.append(time.perf_counter() - t0)
    return 2 * a.nbytes / min(ts) / 1e9


if __name__ == "__main__":
    if sys.argv[1:2] == ["_one"]:
        print(one())
        sys.exit()
    for P in [int(x) for x in sys.argv[1:]] or [1, 16]:
        ps = [subprocess.Popen(["taskset", "-c", str(c), sys.executable, __file__, "_one"],
                               stdout=subprocess.PIPE, text=True) for c in range(P)]
        bw = [float(p.communicate()[0]) for p in ps]
        print(json.dumps({"P": P, "GBps_per_proc": [round(x, 1) for x in bw],
                          "GBps_total": round(sum(bw), 1),
                          "loadavg": open("/proc/loadavg").read().split()[:3]}))
