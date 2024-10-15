
\
import numpy as np
from numpy.linalg import eigh
import statsmodels
from statsmodels.stats.correlation_tools import cov_nearest

import numpy as np

def remove_negative_eigenvalues(cov_matrix, eps=1e-6):
    """
    Adjusts a covariance matrix to make it positive definite by removing negative eigenvalues.

    Parameters:
    - cov_matrix: The original covariance matrix (must be square and symmetric).
    - eps: Small positive value to replace negative eigenvalues.

    Returns:
    - cov_positive_def: The adjusted positive definite covariance matrix.
    """
    # Ensure the matrix is symmetric
    cov_sym = (cov_matrix + cov_matrix.T) / 2

    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_sym)

    # Adjust eigenvalues: set negative eigenvalues to eps
    eigvals_adjusted = np.clip(eigvals, a_min=eps, a_max=None)

    # Reconstruct the covariance matrix
    cov_positive_def = eigvecs @ np.diag(eigvals_adjusted) @ eigvecs.T

    # Ensure the matrix is symmetric
    cov_positive_def = (cov_positive_def + cov_positive_def.T) / 2

    return cov_positive_def



def nearest_positive_def(cov):
    """
    Adjusts a symmetric matrix to the nearest positive semi-definite matrix.

    Args:
        matrix (np.ndarray): The matrix to adjust.

    Returns:
        np.ndarray: A positive semi-definite matrix.
    """
    cov_near = cov_nearest(cov, method='clipped', threshold=0, n_fact=100, return_all=False)
    cov_nearer = remove_negative_eigenvalues(cov_matrix=cov_near)
    return cov_nearer




if __name__ == "__main__":
    # Original covariance matrix (may not be positive definite)
    cov_matrix = np.array([[4.0, 1.2, 0.8],
                           [1.2, 9.0, 2.1],
                           [0.8, 2.1, 16.0]])

    # Create a covariance matrix with perfect correlation
    std_devs = np.sqrt(np.diag(cov_matrix))
    perfect_corr_cov = np.outer(std_devs, std_devs)
    np.fill_diagonal(perfect_corr_cov, np.diag(cov_matrix))

    print("Covariance matrix with perfect correlation (before adjustment):\n", perfect_corr_cov)

    # Adjust to nearest positive definite matrix
    nearest_cov = nearest_positive_def(perfect_corr_cov)

    # Verify positive definiteness
    eigenvalues = np.linalg.eigvalsh(nearest_cov)
    print("\nEigenvalues of the adjusted covariance matrix:\n", eigenvalues)

    print("\nNearest positive definite covariance matrix:\n", nearest_cov)