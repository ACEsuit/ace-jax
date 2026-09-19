"""Hyperparameters theta in log space, and the log-normal hyperprior.

A Normal prior on log(x) is a log-normal prior on x, so the ladder samples the
log values directly and no change-of-variables Jacobian appears anywhere."""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class Hypers(NamedTuple):
    log_ell: float        # kappa length-scale (cosine-SE or Matern-3/2), scaled-descriptor units
    log_A: float          # delta amplitude, eV
    log_alpha: float      # delta right-hand decay rate, 1/Angstrom
    log_r0: float         # delta transition location, Angstrom
    log_eps: float        # delta excess left-hand sharpness, 1/Angstrom
    log_rho: float        # psi magnitude-factor length, RMS scaled-descriptor units
    log_sigma_c: float    # prior scale of the linear coefficients (ACEfit var_0^0.5)
    log_sigma_E: float    # noise on weighted energy rows, eV
    log_sigma_F: float    # noise on force rows, eV/Angstrom
    log_sigma_V: float    # noise on weighted virial rows, eV


def to_array(h):
    return jnp.stack([jnp.asarray(v, jnp.float64) for v in h])


def from_array(a):
    return Hypers(*[a[i] for i in range(len(Hypers._fields))])


class Prior(NamedTuple):
    mu: Hypers
    sigma: Hypers


def log_prior(theta, prior):
    lp = 0.0
    for x, m, s in zip(theta, prior.mu, prior.sigma):
        lp = lp - 0.5 * ((x - m) / s) ** 2 - 0.5 * jnp.log(2 * jnp.pi * s * s)
    return lp


def default_prior(r0):
    """Settings.  Widths of 1-2 in log space are deliberately weak; the argon
    study found tighter hyperpriors did not help (manuscript, Appendix)."""
    ln = np.log
    mu = Hypers(log_ell=ln(1.0), log_A=ln(0.1), log_alpha=ln(2.0 / r0), log_r0=ln(r0),
                log_eps=ln(0.5 / r0), log_rho=ln(2.0), log_sigma_c=ln(1.0),
                log_sigma_E=ln(1e-3), log_sigma_F=ln(5e-2), log_sigma_V=ln(5e-2))
    sigma = Hypers(log_ell=1.0, log_A=1.5, log_alpha=1.0, log_r0=0.3, log_eps=1.0,
                   log_rho=1.0, log_sigma_c=2.0, log_sigma_E=2.0, log_sigma_F=2.0,
                   log_sigma_V=2.0)
    return Prior(mu, sigma)
