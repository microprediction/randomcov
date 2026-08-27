import numpy as np


def vine_corr(n, eta=1.0, rng=None):
    """C-vine partial correlation method (Joe 2006; LKJ 2009): partial
    correlations sampled as shifted Beta variables, level-dependent
    parameters giving exactly the LKJ(eta) density; recursion converts
    partials to correlations. Any other partial-correlation law can be
    substituted for the Beta to target other elliptope measures."""
    rng = np.random.default_rng(rng)
    P = np.zeros((n, n))              # partial correlations
    C = np.eye(n)
    for k in range(n - 1):
        a = eta + (n - 1 - (k + 1)) / 2.0
        for i in range(k + 1, n):
            p = 2.0 * rng.beta(a, a) - 1.0
            P[k, i] = p
            # walk back the vine recursion to a plain correlation
            for l in range(k - 1, -1, -1):
                p = p * np.sqrt((1 - P[l, i] ** 2) * (1 - P[l, k] ** 2)) \
                    + P[l, i] * P[l, k]
            C[k, i] = C[i, k] = p
    return C
