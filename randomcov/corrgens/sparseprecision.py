import numpy as np


def sparse_precision_corr(n, density=0.05, strength=0.4, rng=None):
    """Gaussian graphical model: a random sparse precision matrix
    (Erdos-Renyi conditional-independence graph, made diagonally
    dominant), inverted and normalized. Dense correlation with sparse
    CONDITIONAL structure -- the opposite texture to factor models."""
    # density is the Erdos-Renyi edge probability. It was compared
    # against uniform draws unchecked, so NaN and anything negative gave
    # the empty graph (the identity) and anything above one, Inf
    # included, gave the complete graph: a different ensemble returned
    # as a well-formed correlation matrix (#18).
    try:
        density = float(density)
    except (TypeError, ValueError):
        raise ValueError(f"density must be a number in [0, 1]; got {density!r}")
    if not np.isfinite(density) or not 0.0 <= density <= 1.0:
        raise ValueError(f"density must be a finite edge probability in [0, 1]; got {density}")
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
