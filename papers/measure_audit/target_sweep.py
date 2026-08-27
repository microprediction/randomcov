"""Seeded sweep behind the shrinkage-target table: oracle linear shrinkage
toward the scaled-identity target versus the constant-correlation target
(Ledoit-Wolf 2004 vs 2003), by named ensemble at fixed (n, T).

For each draw and each target F the oracle intensity
delta* = <Sigma - S, F - S> / ||F - S||^2 (clipped to [0, 1]) is applied and
the Frobenius loss of the shrunk estimator recorded; the table reports the
loss ratio constant-correlation / identity, so values below one favor the
constant-correlation target.

Run: python papers/measure_audit/target_sweep.py
"""
import warnings

import numpy as np

from randomcov import CORR_GENERATORS, random_covariance_matrix

warnings.filterwarnings("ignore")
rng = np.random.default_rng(3)
n, T, reps = 30, 60, 20


def identity_target(S):
    return (np.trace(S) / len(S)) * np.eye(len(S))


def const_corr_target(S):
    s = np.sqrt(np.diag(S))
    corr = S / np.outer(s, s)
    rbar = (corr.sum() - len(S)) / (len(S) * (len(S) - 1))
    F = rbar * np.outer(s, s)
    np.fill_diagonal(F, np.diag(S))
    return F


def oracle_loss(Sigma, S, F):
    D = F - S
    delta = float(np.sum((Sigma - S) * D) / np.sum(D * D))
    delta = min(max(delta, 0.0), 1.0)
    est = S + delta * D
    return float(np.linalg.norm(est - Sigma, "fro"))


rows = []
for mi, m in enumerate(CORR_GENERATORS):
    ratios = []
    for rep in range(reps):
        Sigma = np.asarray(random_covariance_matrix(n=n, corr_method=m,
                                                 rng=200000 + 1000 * mi + rep))
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(n))
        X = (L @ rng.standard_normal((n, T))).T
        S = np.cov(X, rowvar=False)
        ratios.append(oracle_loss(Sigma, S, const_corr_target(S))
                      / oracle_loss(Sigma, S, identity_target(S)))
    ratios = np.array(ratios)
    rows.append((m.value, float(np.median(ratios)),
                 float(np.mean(ratios < 1.0))))
for name, med, frac in sorted(rows, key=lambda r: r[1]):
    print(f"{name:16s} median CC/ID {med:5.2f}   CC wins {frac:.0%}")
