"""UQ metrics of the argon study: accuracy, proper score, calibration and
error discrimination are distinct properties and are all reported."""
import numpy as np
from scipy.stats import norm, spearmanr


def rmse(y, mu):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(mu)) ** 2)))


def mae(y, mu):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(mu))))


def crps_gaussian(y, mu, sigma):
    """Closed form for a Gaussian predictive (Gneiting & Raftery 2007)."""
    z = (np.asarray(y) - np.asarray(mu)) / np.asarray(sigma)
    return np.asarray(sigma) * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def coverage(y, mu, sigma):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(mu)) <= np.asarray(sigma)))


def spearman(a, b):
    return float(spearmanr(np.asarray(a), np.asarray(b)).correlation)


def sigma_ratio(sigma, lo=5, hi=95):
    s = np.asarray(sigma)
    return float(np.percentile(s, hi) / np.percentile(s, lo))


def rms_z(y, mu, sigma):
    return float(np.sqrt(np.mean(((np.asarray(y) - np.asarray(mu)) / np.asarray(sigma)) ** 2)))


def summarise(y, mu, sigma):
    """All metrics over the observations with sigma > 0: an exact prediction
    with zero predictive variance (e.g. an isolated-atom reference, whose
    feature rows are all zero) carries no UQ information and would give
    z = 0/0.  `n_dropped` reports how many were excluded."""
    y, mu, sigma = (np.asarray(a, float) for a in (y, mu, sigma))
    keep = sigma > 0
    y, mu, sigma = y[keep], mu[keep], sigma[keep]
    err = np.abs(y - mu)
    return {"rmse": rmse(y, mu), "mae": mae(y, mu), "crps": float(np.mean(crps_gaussian(y, mu, sigma))),
            "coverage": coverage(y, mu, sigma), "rho": spearman(sigma, err),
            "sigma_ratio": sigma_ratio(sigma), "rms_z": rms_z(y, mu, sigma),
            "median_sigma": float(np.median(sigma)), "n_dropped": int((~keep).sum())}
