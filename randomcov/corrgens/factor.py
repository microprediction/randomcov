import numpy as np


def factor_corr(n, k=3, strength=None, sparse_links=0, link_size=0.3,
                rng=None):
    """k-factor model plus idiosyncratic diagonal, optionally with a few
    sparse off-grammar links (the 'approximate factor' world of the
    financial econometrics literature): C = B B' + sparse + D,
    renormalized to unit diagonal and PSD-repaired if links break it."""
    rng = np.random.default_rng(rng)
    if strength is None:
        strength = 0.9 * np.exp(-0.7 * np.arange(k))
    B = rng.normal(size=(n, k)) * np.asarray(strength)
    S = B @ B.T
    for _ in range(sparse_links):
        i, j = rng.choice(n, 2, replace=False)
        S[i, j] += link_size
        S[j, i] = S[i, j]
    S = S + np.diag(np.maximum(1.0 - np.diag(S), 0.05))
    w_, U_ = np.linalg.eigh((S + S.T) / 2.0)
    S = (U_ * np.maximum(w_, 1e-6)) @ U_.T
    d = np.sqrt(np.diag(S))
    C = S / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C
