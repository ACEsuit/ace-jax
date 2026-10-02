"""`python -m ace_jax.fit.zbl` writes a ZBL prior-mean pair potential that the baseline reader takes."""
import numpy as np

from ace_jax.fit import zbl
from ace_jax.fit.baseline import load_mean


def test_zbl_mean_file_round_trips_and_is_core_only(tmp_path):
    out = tmp_path / "zbl.npz"
    zbl.main([str(out), "--elements", "14,26"])
    m = load_mean(out)
    assert m["elements"] == [14, 26] and set(m["idx"]) == {(14, 14), (14, 26), (26, 26)}
    V, r = np.asarray(m["V"]), np.asarray(m["r"])
    assert np.all(V[:, r >= 2.0] == 0) and np.all(np.diff(V[:, r < 1.0], axis=1) < 0)   # repulsive, switched off
    assert np.isclose(V[m["idx"][(14, 14)], 0], zbl.zbl_pair(r[0], 14, 14))           # unswitched inside r_i
