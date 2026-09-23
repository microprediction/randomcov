"""block_equicorr must never return a matrix that is not PSD, and its
exact check must agree with a dense eigendecomposition."""
import numpy as np
import pytest

from randomcov.corrgens.blockequi import block_equicorr


def test_defaults_are_psd_and_unchanged_by_the_guard():
    for k in range(60):
        C = block_equicorr(30, rng=np.random.default_rng(k))
        assert np.linalg.eigvalsh(C).min() > -1e-10


def test_negative_between_that_breaks_psd_raises():
    # the old guard (rho_within >= rho_between) passed this; min eig is about -3
    with pytest.raises(ValueError, match="positive semidefinite"):
        block_equicorr(30, blocks=4, rho_within=0.5, rho_between=-0.2,
                       rng=np.random.default_rng(0))


def test_mildly_negative_between_that_keeps_psd_is_allowed():
    # two blocks, rho_between = -0.05: R is PSD, so this must not raise
    C = block_equicorr(20, blocks=2, rho_within=0.6, rho_between=-0.05,
                       rng=np.random.default_rng(1))
    assert np.linalg.eigvalsh(C).min() > -1e-10


def test_reduced_matrix_check_matches_dense_eigenvalues():
    """Random parameters straddling the boundary: the guard must raise
    exactly when the dense matrix has a negative eigenvalue."""
    rng = np.random.default_rng(7)
    agree = 0
    for _ in range(200):
        blocks = int(rng.integers(2, 6))
        rho_b = float(rng.uniform(-0.6, 0.3))
        rho_w = rng.uniform(max(rho_b, -0.3), 0.9, size=blocks)
        seed = int(rng.integers(1 << 30))
        try:
            C = block_equicorr(25, blocks=blocks, rho_within=rho_w,
                               rho_between=rho_b, rng=np.random.default_rng(seed))
            dense_psd = np.linalg.eigvalsh(C).min() > -1e-8
            assert dense_psd, "guard passed a non-PSD matrix"
        except ValueError:
            # rebuild without the guard to confirm it really was non-PSD
            labels = np.random.default_rng(seed).integers(0, blocks, size=25)
            C = np.full((25, 25), rho_b)
            for b in range(blocks):
                idx = np.where(labels == b)[0]
                C[np.ix_(idx, idx)] = rho_w[b]
            np.fill_diagonal(C, 1.0)
            assert np.linalg.eigvalsh(C).min() < 1e-8, "guard rejected a PSD matrix"
        agree += 1
    assert agree == 200
