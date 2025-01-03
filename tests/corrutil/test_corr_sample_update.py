#tests.corrutil.test_corr_sample_update
import pytest
import numpy as np
from numpy.testing import assert_allclose
from randomcov.corrutil.corrsampleupdate import corr_sample_update


# Assume the function is defined in a file named correlation_utils.py:
# from correlation_utils import corr_sample_update

def test_corr_sample_update_basic():
    """
    Test with synthetic data in 3 dimensions. We'll create samples
    that have a known correlation, then request a new target correlation
    and verify the result is close.
    """

    # Define a small 3D target correlation
    target_corr = np.array([
        [1.0,  0.5,  0.2],
        [0.5,  1.0,  0.3],
        [0.2,  0.3,  1.0]
    ])

    # Generate synthetic samples (N=10,000) from a known correlation:
    # Let's create an original correlation distinct from target_corr
    old_corr = np.array([
        [1.0,  0.55,  0.2],
        [0.55,  1.0,  0.25],
        [0.2,  0.25,  1.0]
    ])
    rng = np.random.default_rng(42)
    # Cholesky factor of old_corr
    L_old = np.linalg.cholesky(old_corr)
    N = 100_000
    z = rng.normal(size=(N, 3))
    samples_old = z @ L_old.T  # shape (N, 3)

    # Possibly add small mean offset or scale so it's not purely perfect
    samples_old += rng.normal(scale=0.01, size=samples_old.shape)

    # Import the function under test
    updated_samples = corr_sample_update(samples_old, target_corr, remove_mean=True)

    # Check shape
    assert updated_samples.shape == samples_old.shape

    # Compute the new correlation
    updated_cov = np.cov(updated_samples, rowvar=False)
    diag_std = np.sqrt(np.diag(updated_cov))
    updated_corr = updated_cov / np.outer(diag_std, diag_std)

    # Check it's close to the target correlation
    # Tolerances depend on sample size & dimension:
    assert_allclose(updated_corr, target_corr, atol=0.02, rtol=0.05)


def test_corr_sample_update_no_mean_removal():
    """
    Test with remove_mean=False, ensuring it doesn't crash
    and results still approximate the target correlation.
    """

    target_corr = np.array([
        [1.0,  0.4],
        [0.4,  1.0]
    ])

    rng = np.random.default_rng(123)
    # Create some 2D samples that have a certain correlation + a non-zero mean
    N = 5000
    old_corr = np.array([
        [1.0, 0.3],
        [0.3, 1.0]
    ])
    L_old = np.linalg.cholesky(old_corr)
    samples_old = rng.normal(size=(N, 2)) @ L_old.T
    samples_old += 5.0  # shift mean to 5

    updated_samples = corr_sample_update(samples_old, target_corr, remove_mean=False)

    # Even if we don't remove mean, correlation should be close
    updated_cov = np.cov(updated_samples, rowvar=False)
    diag_std = np.sqrt(np.diag(updated_cov))
    updated_corr = updated_cov / np.outer(diag_std, diag_std)

    assert_allclose(updated_corr, target_corr, atol=0.03, rtol=0.05)


def test_corr_sample_update_target_not_posdef():
    """
    Provide a target correlation that is not positive-definite
    (e.g., negative or zero diagonal entry).
    Expect a ValueError.
    """

    # Non-PD target (diagonal must be 1 for correlation, but let's break it)
    bad_target_corr = np.array([
        [0.0, 0.5],
        [0.5, 1.0]
    ])
    samples = np.random.normal(size=(1000, 2))

    with pytest.raises(ValueError, match="Target correlation not positive-definite"):
        _ = corr_sample_update(samples, bad_target_corr)




def test_corr_sample_update_low_rank():
    """
    Test near-singular (low-rank) sample correlation,
    ensuring the regularization helps avoid a crash.
    We'll create correlated data in 5D but with effectively rank ~2 or 3.
    """

    rng = np.random.default_rng(999)
    N = 50000
    d = 5

    # Create a matrix with rank=2 or 3
    # E.g., random rank-2 factor => data
    rank = 2
    factor = rng.normal(size=(d, rank))  # shape (5,2)
    data_low_rank = rng.normal(size=(N, rank)) @ factor.T  # shape (N,5)

    # Slight random noise
    data_low_rank += 0.00001 * rng.normal(size=data_low_rank.shape)

    # target correlation - just pick something
    target_corr = np.eye(d)
    target_corr[0, 1] = 0.5
    target_corr[1, 0] = 0.5

    updated_samples = corr_sample_update(data_low_rank, target_corr, regularization=1e-8)

    # Check shape
    assert updated_samples.shape == (N, d)

    # Empirical corr
    cov_new = np.cov(updated_samples, rowvar=False)
    diag_std = np.sqrt(np.diag(cov_new))
    new_corr = cov_new / np.outer(diag_std, diag_std)

    # Should not be perfect, but hopefully not crash
    assert new_corr.shape == (d, d)
    # We can at least check that the correlation is finite
    assert np.all(np.isfinite(new_corr))


def test_corr_sample_update_stability():
    """
    Simple stability test: If target_corr is close to old sample corr,
    the transform shouldn't change the data too drastically.
    """

    rng = np.random.default_rng(1234)
    d = 3
    N = 4000
    # Make some random correlation
    corr_random = np.array([
        [1.0,  0.7, -0.1],
        [0.7,  1.0,  0.0],
        [-0.1, 0.0,  1.0]
    ])
    L_rand = np.linalg.cholesky(corr_random)
    data = rng.normal(size=(N, d)) @ L_rand.T

    # Suppose new_corr is almost the same:
    new_corr = np.copy(corr_random)
    new_corr[0, 2] = new_corr[0, 2] -0.01  # small tweak
    new_corr[2, 0] = new_corr[2, 0] -0.01  # small tweak

    updated_data = corr_sample_update(data, new_corr, remove_mean=True, regularization=1e-12)

    # Check empirical correlation is close
    cov_updated = np.cov(updated_data, rowvar=False)
    diag_std = np.sqrt(np.diag(cov_updated))
    corr_updated = cov_updated / np.outer(diag_std, diag_std)

    # Should be near new_corr
    assert_allclose(corr_updated, new_corr, atol=0.03, rtol=0.05)

    # Check that the average Euclidean movement isn't huge
    movement = np.mean(np.linalg.norm(updated_data - data, axis=1))
    # It's arbitrary, but let's ensure it isn't too large:
    assert movement < 1.0, f"Movement was too large: {movement}"
