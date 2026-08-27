"""Seeded sweep behind the covariance-localization table: Gaspari-Cohn
tapering of a rank-deficient ensemble sample covariance (Houtekamer-Mitchell
2001, Hamill et al. 2001) versus the raw sample covariance, by named
ensemble.

The data-assimilation regime is member-starved: N = 20 members for n = 30
variables. The taper uses index distance |i - j| with half-radius c = 6
(support 12), which presumes the variables are ordered on a line -- exactly
the assumption the audit is probing. The table reports the Frobenius-loss
ratio tapered / raw against the true covariance, so values below one mean
localization helped.

Run: python papers/measure_audit/localization_sweep.py
"""
import warnings

import numpy as np

from randomcov import CORR_GENERATORS, random_covariance_matrix

warnings.filterwarnings("ignore")
rng = np.random.default_rng(10)
n, N, reps, c = 30, 20, 20, 6.0


def gaspari_cohn(d, c):
    r = np.abs(d) / c
    out = np.zeros_like(r)
    a = r <= 1.0
    b = (r > 1.0) & (r <= 2.0)
    ra, rb = r[a], r[b]
    out[a] = (-0.25 * ra**5 + 0.5 * ra**4 + 0.625 * ra**3
              - (5.0 / 3.0) * ra**2 + 1.0)
    out[b] = ((1.0 / 12.0) * rb**5 - 0.5 * rb**4 + 0.625 * rb**3
              + (5.0 / 3.0) * rb**2 - 5.0 * rb + 4.0
              - (2.0 / 3.0) / rb)
    return out


idx = np.arange(n)
taper = gaspari_cohn(idx[:, None] - idx[None, :], c)

rows = []
for mi, m in enumerate(CORR_GENERATORS):
    ratios = []
    for rep in range(reps):
        Sigma = np.asarray(random_covariance_matrix(n=n, corr_method=m,
                                                 rng=600000 + 1000 * mi + rep))
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(n))
        X = (L @ rng.standard_normal((n, N))).T
        S = np.cov(X, rowvar=False)
        ratios.append(np.linalg.norm(taper * S - Sigma, "fro")
                      / np.linalg.norm(S - Sigma, "fro"))
    ratios = np.array(ratios)
    rows.append((m.value, float(np.median(ratios)),
                 float(np.mean(ratios < 1.0))))
for name, med, frac in sorted(rows, key=lambda r: r[1]):
    print(f"{name:16s} median taper/raw {med:5.2f}   taper wins {frac:5.0%}")
