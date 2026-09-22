import numpy as np


def lkj_corr(n, eta=1.0, rng=None):
    """
    Generates a random correlation matrix using the LKJ prior.
    https://en.wikipedia.org/wiki/Lewandowski-Kurowicka-Joe_distribution

    Args:
        n (int): Dimension of the correlation matrix.
        eta (float): Shape parameter controlling the concentration around the identity matrix.

    Returns:
        np.ndarray: A random n x n correlation matrix.
    """
    if n < 2:
        raise ValueError("Dimension must be at least 2.")
    if eta <= 0:
        raise ValueError("Eta must be greater than 0.")

    rng = np.random.default_rng(rng)
    L = lkj_cholesky(n, eta, rng)
    corr_matrix = np.dot(L, L.T)
    return corr_matrix


def lkj_cholesky(n, eta, rng=None):
    """
    Generates the Cholesky factor of a correlation matrix using the LKJ prior.

    Args:
        n (int): Dimension of the correlation matrix.
        eta (float): Shape parameter controlling the concentration around the identity matrix.

    Returns:
        np.ndarray: The Cholesky factor (lower-triangular matrix) of the correlation matrix.
    """
    rng = np.random.default_rng(rng)

    # Initialize the Cholesky factor
    L = np.zeros((n, n))

    # The first diagonal element is always 1
    L[0, 0] = 1.0

    for i in range(1, n):
        # Squared norm of row i's off-diagonal part. LKJ (2009, Sec. 3.2):
        # y_i ~ Beta(i/2, eta + (n - 1 - i)/2). The first shape depends on the
        # row index; a symmetric Beta here makes eta inert and the law
        # non-exchangeable.
        y = rng.beta(0.5 * i, eta + 0.5 * (n - 1 - i))
        L[i, i] = np.sqrt(1.0 - y)

        # Sample from the unit sphere for the off-diagonal elements
        vi = rng.standard_normal(i)
        vi_norm = np.linalg.norm(vi)
        L[i, :i] = vi / vi_norm * np.sqrt(y)

    return L

