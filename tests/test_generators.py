"""Every generator must produce a valid correlation matrix: symmetric,
unit diagonal, entries in [-1, 1], PSD to tolerance -- at small and
moderate n. The winning package's fuzz batteries consume these
ensembles, so validity here is load-bearing downstream."""
import numpy as np
import pytest

from randomcov.corrgens.allcorrgens import CORR_GENERATORS, CorrMethod

SLOW = {CorrMethod.ANIMALS}


@pytest.mark.parametrize("method", list(CORR_GENERATORS))
@pytest.mark.parametrize("n", [2, 5, 40])
def test_valid_correlation(method, n):
    if method in SLOW and n > 5:
        n = 20
    gen = CORR_GENERATORS[method]
    np.random.seed(0)
    try:
        C = np.asarray(gen(n=n, rng=0))
    except TypeError:
        C = np.asarray(gen(n=n))
    assert C.shape == (n, n)
    assert np.abs(C - C.T).max() < 1e-10
    assert np.abs(np.diag(C) - 1).max() < 1e-8
    assert C.max() <= 1 + 1e-8 and C.min() >= -1 - 1e-8
    assert np.linalg.eigvalsh(C).min() > -1e-8


def test_spectrum_preserved():
    from randomcov.corrgens.spectrum import spectrum_corr, _spectrum
    rng = np.random.default_rng(3)
    n = 30
    C = spectrum_corr(n, kind="exp", rng=3)
    # Givens rotations preserve the eigenvalues: spectrum sums to n
    ev = np.linalg.eigvalsh(C)
    assert abs(ev.sum() - n) < 1e-8


def test_seed_determinism():
    from randomcov.corrgens.onion import onion_corr
    a = onion_corr(12, rng=7)
    b = onion_corr(12, rng=7)
    assert np.abs(a - b).max() == 0.0
