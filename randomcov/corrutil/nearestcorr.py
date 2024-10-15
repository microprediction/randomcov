
import numpy as np
from scipy.linalg import eigh

def nearest_corr(corr_matrix):
    """
    Project a symmetric matrix to the nearest correlation matrix using spectral decomposition.

    Parameters:
    - corr_matrix: A symmetric matrix that may not be a valid correlation matrix.

    Returns:
    - corr_matrix_projected: A valid correlation matrix.
    """
    # Perform eigenvalue decomposition of the symmetric matrix
    eigenvalues, eigenvectors = eigh(np.array(corr_matrix))

    # Set any negative eigenvalues to a small positive value to maintain positive semidefiniteness
    eigenvalues[eigenvalues < 0] = 1e-10

    # Reconstruct the matrix
    corr_matrix_projected = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), eigenvectors.T))

    # Ensure diagonal elements are exactly 1 (correlations should be 1 with themselves)
    D = np.diag(1.0 / np.sqrt(np.diag(corr_matrix_projected)))
    corr_matrix_projected = np.dot(D, np.dot(corr_matrix_projected, D))

    return corr_matrix_projected
