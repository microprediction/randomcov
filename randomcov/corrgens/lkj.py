import numpy as np
from scipy.stats import beta, norm


def lkj_corr(n, eta=0.5):
    """
    Generates a random correlation matrix using the LKJ prior.

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

    L = lkj_cholesky(n, eta)
    corr_matrix = np.dot(L, L.T)
    return corr_matrix


def lkj_cholesky(n, eta):
    """
    Generates the Cholesky factor of a correlation matrix using the LKJ prior.

    Args:
        n (int): Dimension of the correlation matrix.
        eta (float): Shape parameter controlling the concentration around the identity matrix.

    Returns:
        np.ndarray: The Cholesky factor (lower-triangular matrix) of the correlation matrix.
    """
    # Initialize the Cholesky factor
    L = np.zeros((n, n))

    # The first diagonal element is always 1
    L[0, 0] = 1.0

    for i in range(1, n):
        # Sample Beta distributed variable for the diagonal element
        beta_param = eta + 0.5 * (n - i - 1)
        ui = beta.rvs(a=beta_param, b=beta_param)
        L[i, i] = np.sqrt(ui)

        # Sample from the unit sphere for the off-diagonal elements
        vi = norm.rvs(size=i)
        vi_norm = np.linalg.norm(vi)
        L[i, :i] = vi / vi_norm * np.sqrt(1 - ui)

    return L

