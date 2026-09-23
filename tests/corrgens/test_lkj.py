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
    """Higher eta concentrates LKJ toward the identity, so correlations shrink.

    Under LKJ(eta) at n=5 each correlation is 2*Beta(a, a) - 1 with
    a = eta + 3/2, so at eta=10 the marginal sd is about 0.2. The old version
    of this test asserted max |r| > 0.5, which only the broken generator
    (a symmetric Beta that pinned row 1 at +/-0.71 whatever eta was) could
    satisfy. Seeded, and compared against eta=1 on the same seeds.
    """
    n = 5
    high = [lkj_corr(n, eta=10.0, rng=np.random.default_rng(k)) for k in range(20)]
    low = [lkj_corr(n, eta=1.0, rng=np.random.default_rng(k)) for k in range(20)]
    off = lambda C: np.abs(C - np.eye(n))
    mean_high = np.mean([off(C).mean() for C in high])
    mean_low = np.mean([off(C).mean() for C in low])
    assert mean_high < 0.3, f"eta=10 correlations too large: mean |r| = {mean_high:.3f}"
    assert mean_high < mean_low, "eta=10 should give smaller correlations than eta=1"


def test_lkj_corr_eta_equals_one():
    """Test if eta=1 gives uniform distribution, resulting in moderate correlations."""
    n = 5
    eta = 1.0  # Should result in more random off-diagonal values
    corr_matrix = lkj_corr(n, eta)
    # Check if off-diagonal values are not all close to zero, allowing for a wide spread
    off_diag_elements = np.abs(corr_matrix - np.eye(n))
    assert np.any(off_diag_elements > 0.1), "Off-diagonal correlations are too close to zero for eta=1"


def test_lkj_corr_eta_less_than_one():
    """Lower eta spreads LKJ toward the boundary, so correlations grow.

    The comparison has to be at fixed n: under LKJ the marginal sd of a
    correlation is 1/sqrt(2*eta + n - 1), so dimension shrinks correlations
    as strongly as eta does, and "eta=0.2 gives max |r| > 0.5 at n=50" is
    not a property of LKJ (it is about 0.5 on a lucky seed). The broken
    generator passed it only because row 1 carried +/-0.71 at every eta.
    """
    n = 5
    low = [lkj_corr(n, eta=0.2, rng=np.random.default_rng(k)) for k in range(20)]
    high = [lkj_corr(n, eta=2.0, rng=np.random.default_rng(k)) for k in range(20)]
    off = lambda C: np.abs(C - np.eye(n))
    mean_low = np.mean([off(C).mean() for C in low])
    mean_high = np.mean([off(C).mean() for C in high])
    assert mean_low > 0.25, f"eta=0.2 correlations too weak: mean |r| = {mean_low:.3f}"
    assert mean_low > mean_high, "eta=0.2 should give larger correlations than eta=2"


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
