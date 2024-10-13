import numpy as np
from numpy.linalg import eigvals
from randomcov.corrgens.lkj import lkj_corr


def test_lkj_corr_shape():
    """Test if the output matrix has the correct shape."""
    n = 5
    eta = 2.0
    corr_matrix = lkj_corr(n, eta)
    assert corr_matrix.shape == (n, n), f"Expected shape ({n}, {n}), got {corr_matrix.shape}"


def test_lkj_corr_symmetry():
    """Test if the output matrix is symmetric."""
    n = 5
    eta = 2.0
    corr_matrix = lkj_corr(n, eta)
    assert np.allclose(corr_matrix, corr_matrix.T), "Matrix is not symmetric"


def test_lkj_corr_positive_definite():
    """Test if the output matrix is positive definite."""
    n = 5
    eta = 2.0
    corr_matrix = lkj_corr(n, eta)
    # Check if all eigenvalues are positive (i.e., the matrix is positive definite)
    eigenvalues = eigvals(corr_matrix)
    assert np.all(eigenvalues > 0), "Matrix is not positive definite"


def test_lkj_corr_diagonal_elements():
    """Test if the diagonal elements of the correlation matrix are all 1."""
    n = 5
    eta = 2.0
    corr_matrix = lkj_corr(n, eta)
    diag_elements = np.diag(corr_matrix)
    assert np.allclose(diag_elements, np.ones(n)), "Diagonal elements are not all 1"


def test_lkj_corr_eta_greater_than_one():
    """Test if higher eta values result in correlation matrices closer to identity matrix."""
    n = 5
    eta = 10.0  # High value for eta should produce correlations close to zero
    corr_matrix = lkj_corr(n, eta)
    off_diag_elements = corr_matrix - np.eye(n)  # Subtract identity to focus on off-diagonal
    max_corr = np.max(np.abs(off_diag_elements))
    assert max_corr > 0.5, f"Off-diagonal correlations are too high with eta={eta}"


def test_lkj_corr_eta_equals_one():
    """Test if eta=1 gives uniform distribution, resulting in moderate correlations."""
    n = 5
    eta = 1.0  # Should result in more random off-diagonal values
    corr_matrix = lkj_corr(n, eta)
    # Check if off-diagonal values are not all close to zero, allowing for a wide spread
    off_diag_elements = np.abs(corr_matrix - np.eye(n))
    assert np.any(off_diag_elements > 0.1), "Off-diagonal correlations are too close to zero for eta=1"


def test_lkj_corr_eta_less_than_one():
    """Test if eta < 1 results in stronger correlations (closer to -1 or 1)."""
    n = 50
    eta = 0.2  # Low eta should result in stronger correlations
    corr_matrix = lkj_corr(n, eta)
    off_diag_elements = corr_matrix - np.eye(n)  # Subtract identity matrix to isolate off-diagonals
    max_corr = np.max(np.abs(off_diag_elements))
    assert max_corr > 0.5, f"Off-diagonal correlations are too weak with eta={eta}"


def test_lkj_corr_different_dimensions():
    """Test lkj_corr function for different matrix sizes."""
    for n in [2, 3, 10, 50]:
        eta = 2.0
        corr_matrix = lkj_corr(n, eta)
        assert corr_matrix.shape == (n, n), f"Expected shape ({n}, {n}), got {corr_matrix.shape}"


def test_lkj_corr_invalid_eta():
    """Test that the function raises an error for invalid eta values (eta <= 0)."""
    n = 5
    invalid_eta = 0.0
    try:
        lkj_corr(n, invalid_eta)
        assert False, "The function should raise an error for eta <= 0"
    except ValueError:
        pass  # Test passes if ValueError is raised
    except Exception as e:
        assert False, f"Expected ValueError but got {type(e).__name__}"


def test_lkj_corr_invalid_n():
    """Test that the function raises an error for invalid n values (n < 2)."""
    invalid_n = 1  # Correlation matrix must be at least 2x2
    eta = 2.0
    try:
        lkj_corr(invalid_n, eta)
        assert False, "The function should raise an error for n < 2"
    except ValueError:
        pass  # Test passes if ValueError is raised
    except Exception as e:
        assert False, f"Expected ValueError but got {type(e).__name__}"
