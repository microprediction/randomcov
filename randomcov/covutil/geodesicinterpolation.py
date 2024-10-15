import numpy as np
from scipy.linalg import sqrtm, inv, fractional_matrix_power


def geodesic_interpolation(start_cov, end_cov, gamma):
    """
    Interpolate between two SPD matrices along the geodesic under the affine-invariant Riemannian metric.

    Parameters:
    - start_cov: The starting SPD matrix.
    - end_cov: The ending SPD matrix.
    - gamma: A scalar between 0 and 1, where gamma=0 returns start_cov, and gamma=1 returns end_cov.

    Returns:
    - interpolated_cov: The interpolated SPD matrix at parameter gamma.
    """
    # Ensure the matrices are symmetric positive-definite
    assert np.allclose(start_cov, start_cov.T), "start_cov must be symmetric."
    assert np.allclose(end_cov, end_cov.T), "end_cov must be symmetric."
    eigvals_start = np.linalg.eigvalsh(start_cov)
    eigvals_end = np.linalg.eigvalsh(end_cov)
    assert np.all(eigvals_start > 0), "start_cov must be positive definite."
    assert np.all(eigvals_end > 0), "end_cov must be positive definite."

    # Compute the matrix square root and its inverse of the starting covariance matrix
    start_cov_sqrt = sqrtm(start_cov)
    start_cov_inv_sqrt = inv(start_cov_sqrt)

    # Compute the middle term
    middle_term = start_cov_inv_sqrt @ end_cov @ start_cov_inv_sqrt

    # Compute the fractional power of the middle term
    middle_term_power = fractional_matrix_power(middle_term, gamma)

    # Compute the interpolated covariance matrix
    interpolated_cov = start_cov_sqrt @ middle_term_power @ start_cov_sqrt

    # Ensure the result is symmetric
    interpolated_cov = (interpolated_cov + interpolated_cov.T) / 2

    return interpolated_cov


def geodesic_interpolation_towards_perfect(cov, gamma:float):
    perfect_cov = covariance_with_perfect_correlation(cov_matrix=cov)
    return geodesic_interpolation(start_cov=cov, end_cov=perfect_cov, gamma=gamma)


def covariance_with_perfect_correlation(cov_matrix, correlation=1.0):
    """
    Given a covariance matrix, construct a new covariance matrix
    with perfect correlation between all variables.

    Parameters:
    - cov_matrix: The original covariance matrix (must be square and symmetric).
    - correlation: The desired correlation coefficient (default is 1.0 for perfect positive correlation).

    Returns:
    - cov_perfect: The new covariance matrix with perfect correlation.
    """
    # Ensure the input matrix is square
    n = cov_matrix.shape[0]
    assert cov_matrix.shape == (n, n), "Input must be a square matrix."
    # Ensure the matrix is symmetric
    assert np.allclose(cov_matrix, cov_matrix.T), "Input matrix must be symmetric."

    # Compute the standard deviations (square roots of the variances)
    std_devs = np.sqrt(np.diag(cov_matrix))

    # Initialize the new covariance matrix
    cov_perfect = np.zeros((n, n))

    # Fill in the covariance matrix
    for i in range(n):
        for j in range(n):
            cov_perfect[i, j] = std_devs[i] * std_devs[j] * correlation

    # Set the diagonal elements to the original variances
    np.fill_diagonal(cov_perfect, np.diag(cov_matrix))

    return cov_perfect


if __name__ == "__main__":
    # Original covariance matrix
    cov_matrix = np.array([[4.0, 1.2, 0.8],
                           [1.2, 9.0, 2.1],
                           [0.8, 2.1, 16.0]])

    # Construct the covariance matrix with perfect correlation
    cov_perfect = covariance_with_perfect_correlation(cov_matrix)

    print("Original covariance matrix:\n", cov_matrix)
    print("\nCovariance matrix with perfect correlation:\n", cov_perfect)


    # Define two SPD matrices
    start_cov = np.array([[2.0, 0.6],
                          [0.6, 1.0]])

    end_cov = np.array([[1.0, 0.2],
                        [0.2, 1.5]])

    gamma = 0.5  # Halfway interpolation

    interpolated_cov = geodesic_interpolation(start_cov, end_cov, gamma)

    print("Start covariance matrix:\n", start_cov)
    print("\nEnd covariance matrix:\n", end_cov)
    print(f"\nInterpolated covariance matrix at gamma={gamma}:\n", interpolated_cov)
