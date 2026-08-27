import numpy as np


def onion_corr(n, eta=1.0, rng=None):
    """Extended onion method (Ghosh & Henderson 2003; Lewandowski,
    Kurowicka & Joe 2009): exact sampling from the LKJ(eta) density,
    eta = 1 uniform over the elliptope. Grows the matrix one variable at
    a time; the new column's squared radius is Beta distributed."""
    rng = np.random.default_rng(rng)
    C = np.eye(1)
    beta_par = eta + (n - 2) / 2.0
    for k in range(1, n):
        r2 = rng.beta(k / 2.0, beta_par)
        beta_par -= 0.5
        u = rng.normal(size=k)
        u = u / np.linalg.norm(u)
        w_, U_ = np.linalg.eigh(C)
        root = U_ @ (np.sqrt(np.maximum(w_, 0))[:, None] * U_.T)
        q = root @ (np.sqrt(r2) * u)
        C = np.block([[C, q[:, None]], [q[None, :], np.ones((1, 1))]])
    return C
