"""The three elliptope samplers claim the same measure, so they must agree.

Shape, symmetry and positive-definiteness tests all pass for a generator that
samples the wrong distribution entirely -- which is how `lkj` shipped for a
while drawing a symmetric Beta, making eta inert and the law non-exchangeable.
These tests check the measure itself.
"""
import numpy as np
import pytest

from randomcov.corrgens.lkj import lkj_corr
from randomcov.corrgens.onion import onion_corr
from randomcov.corrgens.vine import vine_corr

GENERATORS = [("lkj", lkj_corr), ("onion", onion_corr), ("vine", vine_corr)]


def _mean_det(f, n, eta, reps=600):
    return np.mean([np.linalg.det(f(n, eta=eta, rng=np.random.default_rng(k)))
                    for k in range(reps)])


@pytest.mark.parametrize("name,f", GENERATORS)
@pytest.mark.parametrize("eta", [0.5, 1.0, 2.0])
def test_determinant_matches_lkj_theory(name, f, eta):
    """E[det C] under LKJ(eta) is a known product of Beta means."""
    n = 5
    expected = np.prod([
        (eta + 0.5 * (n - 1 - i)) / (eta + 0.5 * (n - 1 - i) + 0.5 * i)
        for i in range(1, n)
    ])
    assert _mean_det(f, n, eta) == pytest.approx(expected, rel=0.08)


@pytest.mark.parametrize("name,f", GENERATORS)
def test_eta_is_not_inert(name, f):
    """Larger eta must concentrate toward the identity."""
    dets = [_mean_det(f, 5, eta) for eta in (0.5, 1.0, 2.0, 10.0)]
    assert all(b > a for a, b in zip(dets, dets[1:])), f"{name}: {dets}"


@pytest.mark.parametrize("name,f", GENERATORS)
def test_exchangeable(name, f):
    """LKJ is exchangeable: every off-diagonal entry has the same marginal."""
    n = 40
    C = [f(n, eta=1.0, rng=np.random.default_rng(k)) for k in range(300)]
    first = np.std([c[0, 1] for c in C])
    last = np.std([c[n - 2, n - 1] for c in C])
    theory = 1.0 / np.sqrt(n + 1)
    assert first == pytest.approx(theory, rel=0.25), f"{name}: {first} vs {theory}"
    assert last == pytest.approx(theory, rel=0.25), f"{name}: {last} vs {theory}"
