import numpy as np


def sparse_precision_corr(n, density=0.05, strength=0.4, rng=None):
    """Gaussian graphical model: a random sparse precision matrix
    (Erdos-Renyi conditional-independence graph, made diagonally
    dominant), inverted and normalized. Dense correlation with sparse
    CONDITIONAL structure -- the opposite texture to factor models."""
    rng = np.random.default_rng(rng)
    P = np.zeros((n, n))
    mask = np.triu(rng.random((n, n)) < density, 1)
    vals = strength * (2 * rng.random((n, n)) - 1)
    P[mask] = vals[mask]
    P = P + P.T
    np.fill_diagonal(P, np.abs(P).sum(axis=1) + 0.5 + rng.random(n))
    C = np.linalg.inv(P)
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C
