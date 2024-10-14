import numpy as np

def nearest_positive_def(matrix):
    """
    Adjusts a symmetric matrix to the nearest positive semi-definite matrix.

    Args:
        matrix (np.ndarray): The matrix to adjust.

    Returns:
        np.ndarray: A positive semi-definite matrix.
    """
    # Ensure the matrix is symmetric
    sym_matrix = (matrix + matrix.T) / 2

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(sym_matrix)

    # Clip negative eigenvalues to zero
    eigenvalues_clipped = np.clip(eigenvalues, a_min=0, a_max=None)

    # Reconstruct the matrix
    adjusted_matrix = eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.T

    # Ensure the diagonal is 1
    np.fill_diagonal(adjusted_matrix, 1.0)

    return adjusted_matrix
