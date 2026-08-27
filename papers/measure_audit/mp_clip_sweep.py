"""Seeded sweep behind the eigenvalue-clipping table: Marchenko-Pastur
cleaning of the sample correlation matrix (Laloux et al.) versus the raw
sample correlation, by named ensemble at fixed (n, T).

Eigenvalues below the MP edge (1 + sqrt(n/T))^2 are flattened to their
average (trace preserved); the table reports the Frobenius-loss ratio
clipped / raw against the true correlation, so values below one mean
cleaning helped.

Run: python papers/measure_audit/mp_clip_sweep.py
"""
import warnings

import numpy as np

from randomcov import CORR_GENERATORS, random_correlation_matrix

warnings.filterwarnings("ignore")
rng = np.random.default_rng(5)
n, T, reps = 30, 60, 20
edge = (1.0 + np.sqrt(n / T)) ** 2


def sample_corr(X):
    S = np.cov(X, rowvar=False)
    s = np.sqrt(np.diag(S))
    return S / np.outer(s, s)


def mp_clip(R):
    lam, V = np.linalg.eigh(R)
    bulk = lam < edge
    if bulk.any():
        lam = lam.copy()
        lam[bulk] = lam[bulk].mean()
    out = (V * lam) @ V.T
    d = np.sqrt(np.diag(out))
    return out / np.outer(d, d)


rows = []
for mi, m in enumerate(CORR_GENERATORS):
    ratios = []
    for rep in range(reps):
        C = np.asarray(random_correlation_matrix(n=n, corr_method=m,
                                                  rng=300000 + 1000 * mi + rep))
        L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
        X = (L @ rng.standard_normal((n, T))).T
        R = sample_corr(X)
        ratios.append(np.linalg.norm(mp_clip(R) - C, "fro")
                      / np.linalg.norm(R - C, "fro"))
    ratios = np.array(ratios)
    rows.append((m.value, float(np.median(ratios)),
                 float(np.mean(ratios < 1.0))))
for name, med, frac in sorted(rows, key=lambda r: r[1]):
    print(f"{name:16s} median clip/raw {med:5.2f}   clip wins {frac:5.0%}")
