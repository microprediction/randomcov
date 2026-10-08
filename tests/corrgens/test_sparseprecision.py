import numpy as np
import pytest
from randomcov.corrgens.sparseprecision import sparse_precision_corr
from randomcov.randomcorrelationmatrix import random_correlation_matrix


def _is_correlation(C, n):
    return (C.shape == (n, n) and np.allclose(C, C.T)
            and np.allclose(np.diag(C), 1.0)
            and np.all(np.linalg.eigvalsh(C) > 0))


@pytest.mark.parametrize("density", [0.0, 0.5, 1.0])
def test_in_range_density_prices_a_correlation(density):
    C = sparse_precision_corr(6, density=density, rng=7)
    assert _is_correlation(C, 6)


def test_density_zero_is_the_identity():
    assert np.array_equal(sparse_precision_corr(6, density=0.0, rng=7), np.eye(6))


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_out_of_range_or_non_finite_density_is_refused(bad):
    # an Erdos-Renyi edge probability; a bad value used to be compared
    # against uniform draws and silently became the empty or the
    # complete graph (#18)
    with pytest.raises(ValueError, match="density"):
        sparse_precision_corr(6, density=bad, rng=7)


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan, np.inf])
def test_dispatcher_refuses_the_same_values(bad):
    with pytest.raises(ValueError, match="density"):
        random_correlation_matrix(6, corr_method="sparse_precision",
                                  corr_kwargs={"density": bad})


def test_dispatcher_accepts_in_range_density():
    C = random_correlation_matrix(6, corr_method="sparse_precision",
                                  corr_kwargs={"density": 0.5})
    assert _is_correlation(C, 6)
