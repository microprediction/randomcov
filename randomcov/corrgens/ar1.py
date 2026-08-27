import numpy as np


def ar1_corr(n, rho=None, rng=None):
    """Kac-Murdock-Szego / AR(1) Toeplitz correlation rho^|i-j| --
    the classic banded-decay ensemble (time-series, spatial lines)."""
    rng = np.random.default_rng(rng)
    if rho is None:
        rho = 0.2 + 0.75 * rng.random()
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])
