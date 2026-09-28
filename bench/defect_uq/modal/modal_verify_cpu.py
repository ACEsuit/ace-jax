"""CPU cross-check of GPU runs (B200 cuBLAS < 13.2 warns of possible silent corruption):
reload the saved model file on a CPU container and recompute a sample of test
predictions, compared with the GPU run's pred_test_map.npz.

  modal deploy modal_verify_cpu.py && modal run modal_verify_cpu.py::main --name cantor_gp_pca128_ms4
"""
import os
import pathlib

import modal

DATA = pathlib.Path(os.environ.get("ACEGP_DATA", pathlib.Path.home() / "acegp-data")) / "cantor"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands("git clone --depth 1 --branch main https://github.com/ACEsuit/ace-jax /root/ace-jax",
                  "pip install -e '/root/ace-jax[gp]'")
    .add_local_file(str(DATA / "cantor4k_b_mh1.xyz"), "/data/cantor4k_b_mh1.xyz")
)
app = modal.App("acegp-verify-cpu")
vol = modal.Volume.from_name("acegp-prod-out")


@app.function(image=image, volumes={"/out": vol}, cpu=16, memory=64 * 1024, timeout=6 * 3600)
def verify(name: str, n: int = 6) -> dict:
    import os
    os.environ["JAX_ENABLE_X64"] = "1"
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    from ase.io import read
    run = f"/out/{name}"
    perm = np.load(f"{run}/split_perm.npy")
    pred = np.load(f"{run}/pred_test_map.npz")
    cfgs = read("/data/cantor4k_b_mh1.xyz", ":")
    test = [cfgs[i] for i in perm[3200:4000]]
    gp = os.path.exists(f"{run}/gp_model.npz")
    if gp:
        from ace_jax import GPCalculator
        calc = GPCalculator.from_file(f"{run}/gp_model.npz")
    else:
        from ace_jax import ACECalculator
        calc = ACECalculator(f"{run}/model.npz")
    nat = pred["nat"].astype(int)
    off = np.concatenate([[0], np.cumsum(nat)])
    idx = np.linspace(0, len(test) - 1, n).astype(int)
    dE, dF, dS = [], [], []
    for i in idx:
        at = test[i].copy(); at.calc = calc
        assert len(at) == nat[i]
        E, F = at.get_potential_energy(), at.get_forces()
        dE.append(abs(E - pred["E_mean"][i]) / nat[i])
        dF.append(float(np.abs(F - pred["F_mean"][off[i]:off[i + 1]]).max()))
        if gp:
            dS.append(abs(calc.results["energy_std"] ** 2 - pred["E_var"][i]) / pred["E_var"][i])
    out = {"name": name, "configs": [int(i) for i in idx], "max_dE_per_atom_eV": float(max(dE)),
           "max_dF_eV_per_A": float(max(dF)), **({"max_rel_dEvar": float(max(dS))} if gp else {})}
    import json
    json.dump(out, open(f"{run}/cpu_verify.json", "w"), indent=1); vol.commit()
    print(out)
    return out


@app.local_entrypoint()
def main(name: str, n: int = 6):
    print(verify.remote(name, n))
