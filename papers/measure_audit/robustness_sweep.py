"""Seeded robustness check: the shrinkage-intensity spread and the
Marchenko-Pastur clipping verdicts at T = 120 instead of T = 60, to show
the audits are not artifacts of the T = 2n regime.

Run: python papers/measure_audit/robustness_sweep.py
"""
import warnings

import numpy as np
from sklearn.covariance import LedoitWolf

from randomcov import CORR_GENERATORS, random_correlation_matrix

warnings.filterwarnings("ignore")
rng = np.random.default_rng(28)
n, T, reps = 30, 120, 20
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
    lw, clip = [], []
    for rep in range(reps):
        C = np.asarray(random_correlation_matrix(
            n=n, corr_method=m, rng=1500000 + 1000 * mi + rep))
        L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
        X = (L @ rng.standard_normal((n, T))).T
        lw.append(LedoitWolf().fit(X).shrinkage_)
        R = sample_corr(X)
        clip.append(np.linalg.norm(mp_clip(R) - C, "fro")
                    / np.linalg.norm(R - C, "fro"))
    rows.append((m.value, float(np.mean(lw)), float(np.median(clip))))
print(f"T = {T}")
for name, mu, med in sorted(rows, key=lambda r: r[1]):
    print(f"{name:16s} LW intensity {mu:5.3f}   clip/raw {med:5.2f}")
