"""Core-orbital projection of a folder of npz caches (`ham.project.core_projection`): valence-only
H and S caches in another folder, several structures at a time.
    JAX_PLATFORMS=cpu uv run python bench/ham/project_dataset.py <npz dir> <out dir> <Z>:<core shells>[;<Z>:...] [workers=8]
e.g. 13:0,1,3 for the 1s, 2s and 2p shells of an aluminium basis with shells s s s p p d."""
import os
import sys
from concurrent.futures import ProcessPoolExecutor


def one(job):
    src, dst, core = job
    import jax
    jax.config.update("jax_enable_x64", True)
    from ace_jax.ham import cache
    from ace_jax.ham.project import core_projection
    d, meta = cache.load(src)
    p, info = core_projection(d, core)
    cache.save(dst, p, meta={**(meta.get("source") or {}), "core_projection": {"core": {str(z): list(s) for z, s in core.items()}, **info}})
    return os.path.basename(src), info["core_valence_coupling"]


if __name__ == "__main__":
    src, dst, spec = sys.argv[1:4]
    workers = int(dict(a.split("=") for a in sys.argv[4:]).get("workers", 8))
    core = {int(z): tuple(int(s) for s in sh.split(",")) for z, sh in (x.split(":") for x in spec.split(";"))}
    os.makedirs(dst, exist_ok=True)
    jobs = [(os.path.join(src, n), os.path.join(dst, n), core) for n in sorted(os.listdir(src))
            if n.endswith(".npz") and not os.path.exists(os.path.join(dst, n))]
    with ProcessPoolExecutor(workers) as ex:
        for name, leak in ex.map(one, jobs):
            print(f"{name}: largest core-valence element dropped {leak:.3f}", flush=True)
