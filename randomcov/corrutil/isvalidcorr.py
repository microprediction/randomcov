import numpy as np

def is_valid_corr(corr_matrix):
    """
    Validates whether a matrix is a valid correlation matrix.

    Args:
        corr_matrix (np.ndarray): The matrix to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        corr_matrix = np.asarray(corr_matrix, dtype=float)
    except (TypeError, ValueError):
        return False

    # Check if the matrix is 2D and square
    if corr_matrix.ndim != 2 or corr_matrix.shape[0] != corr_matrix.shape[1]:
        return False

    # Reject NaN/Inf before allclose, which treats matching infinities as equal
    if not np.isfinite(corr_matrix).all():
        return False

    # Check if the diagonal elements are 1
    if not np.allclose(np.diag(corr_matrix), 1):
        return False

    # Check if the matrix is symmetric
    if not np.allclose(corr_matrix, corr_matrix.T):
        return False

    # Check if the matrix is positive semi-definite
    try:
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
    except np.linalg.LinAlgError:
        return False
    if not np.isfinite(eigenvalues).all():
        return False
    if np.any(eigenvalues < -1e-8):  # Allowing a small numerical tolerance
        return False

    return True

# Example usage
if __name__ == "__main__":
    from randomcov.corrgens.normalcopula import copula_corr
    corr = copula_corr(n=5)
    print("Generated Correlation Matrix:\n", corr)
    print("Is valid correlation matrix:", is_valid_corr(corr))
