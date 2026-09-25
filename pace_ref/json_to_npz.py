"""unit_ref JSON (stdin) -> fixtures/pace/unit_ref.npz.  Records the PACE commit."""
import json, pathlib, subprocess, sys
import numpy as np

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "~/gits/lammps-user-pace").expanduser()
d = json.load(sys.stdin)
out = {k: np.asarray(v["data"], float).reshape(-1, v["ncol"]).squeeze() if v["ncol"] == 1
       else np.asarray(v["data"], float).reshape(-1, v["ncol"]) for k, v in d.items()}
out["pace_commit"] = np.asarray(subprocess.check_output(
    ["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip())
dst = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "pace" / "unit_ref.npz"
dst.parent.mkdir(parents=True, exist_ok=True)
np.savez(dst, **out)
print(f"wrote {dst} ({len(out)} arrays)")
