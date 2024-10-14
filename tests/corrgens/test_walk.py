import numpy as np
from numpy.linalg import eigvals
from randomcov.corrgens.walk import walk_corr


def test_walk_corr_shape():
    """Test if the output matrix has the correct shape."""
    n = 5
    corr_matrix = walk_corr(n)
    assert corr_matrix.shape == (n, n), f"Expected shape ({n}, {n}), got {corr_matrix.shape}"


def test_walk_corr_symmetry_many_steps():
    """Test if the output matrix is symmetric."""
    n = 5
    corr_matrix = walk_corr(n,steps=50)
    assert np.allclose(corr_matrix, corr_matrix.T), "Matrix is not symmetric"


def test_walk_corr_positive_definite_large_epsilon():
    """Test if the output matrix is positive definite."""
    n = 5
    corr_matrix = walk_corr(n, steps=5, epsilon=0.5)
    # Check if all eigenvalues are positive (i.e., the matrix is positive definite)
    eigenvalues = eigvals(corr_matrix)
    assert np.all(eigenvalues > 0), "Matrix is not positive definite"


def test_walk_corr_diagonal_elements():
    """Test if the diagonal elements of the correlation matrix are all 1."""
    n = 5
    corr_matrix = walk_corr(n)
    diag_elements = np.diag(corr_matrix)
    assert np.allclose(diag_elements, np.ones(n)), "Diagonal elements are not all 1"

