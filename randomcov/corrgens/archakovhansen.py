import numpy as np
from scipy.linalg import expm, logm


def archakov_hansen_corr(n, scale=0.6, rng=None, max_iter=200, tol=1e-12):
    """Archakov & Hansen (Econometrica 2021) parameterization: the
    off-diagonal of the matrix LOG of a correlation matrix is an
    unconstrained vector. Sample it Gaussian(0, scale^2), then run their
    fixed-point iteration on the diagonal so exp(A) has unit diagonal.
    A modern route to 'Fisher-z-like' random correlation matrices."""
    rng = np.random.default_rng(rng)
    A = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    z = rng.normal(scale=scale, size=len(iu[0]))
    A[iu] = z
    A = A + A.T
    x = np.zeros(n)
    for _ in range(max_iter):
        np.fill_diagonal(A, x)
        d = np.diag(expm(A))
        step = np.log(d)
        x = x - step
        if np.abs(step).max() < tol:
            break
    np.fill_diagonal(A, x)
    C = expm(A)
    # symmetrize and pin the diagonal against roundoff
    C = (C + C.T) / 2.0
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C
