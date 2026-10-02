"""solver="lstsq": plain weighted least squares, for teaching (tutorial 4).

The readout minimises sum_rows (w (y - Phi c))^2 with the configured per-config-type
weights and unit noise, and no prior: the smoothness prior and sigma_c rows of
fit/solve.py::stacked_design are dropped, and numpy's lstsq returns the minimum-norm
solution when the basis outgrows the data. That is what ACEfit's QR solver does with
ACEpotentials' weights, so a growing basis overfits; the evidence fit
(solver="evidence") is the regularised counterpart.

The weighted design is materialised in host memory (n_obs x len_basis): for
tutorial-scale problems only."""
import time

import numpy as np

from ..hypers import to_array
from ..solve import stacked_design
from .mapfit import MapFit
from .rungs import Rungs


def lstsq_theta(prob):
    """The hyperparameters the rows are weighted at: unit noise for E, F and V, so the
    configured weights alone set the balance (the prior entries are unused)."""
    return prob.prior.mu._replace(log_sigma_E=0.0, log_sigma_F=0.0, log_sigma_V=0.0)


def fit_lstsq(cfg, d, b, log=print):
    """(MapFit, Rungs, readout): the readout, and the theta it is weighted at as the
    single 'lstsq' draw, so prediction and export take the usual path."""
    t = time.time()
    theta = lstsq_theta(b.prob)
    Phi, y = stacked_design(b.prob, d.ds_train, theta)
    Dt = Phi.shape[1]
    A, r = np.asarray(Phi)[:-Dt], np.asarray(y)[:-Dt]          # data rows only: no prior block
    c, _, rank, _ = np.linalg.lstsq(A, r, rcond=None)
    log(f"lstsq: {A.shape[0]} weighted rows x {Dt} basis functions, rank {rank}")
    tm = {"lstsq": time.time() - t}
    return (MapFit(theta, None, None, tm, None),
            Rungs({"lstsq": np.asarray(to_array(theta))[None]}, {}, {}), c)
