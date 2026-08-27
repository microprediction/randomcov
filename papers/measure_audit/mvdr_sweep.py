"""Seeded sweep behind the diagonal-loading table: loaded versus unloaded
MVDR (Capon) beamformer weights in the snapshot-limited regime (Cox,
Zeskind and Owen 1987; Carlson 1988), by named ensemble.

With a random unit steering vector a and the distortionless constraint
w'a = 1, weights are built from the sample covariance of T = 40 snapshots,
with and without loading delta = 0.1 tr(S)/n, and scored by true output
power w' Sigma w. The table reports the median power ratio loaded /
unloaded, so values below one mean loading helped.

Run: python papers/measure_audit/mvdr_sweep.py
"""
import warnings

import numpy as np

from randomcov import CORR_GENERATORS, random_covariance_matrix

warnings.filterwarnings("ignore")
rng = np.random.default_rng(12)
n, T, reps, load = 30, 40, 20, 0.1


def mvdr(S, a):
    w = np.linalg.solve(S, a)
    return w / (a @ w)


rows = []
for mi, m in enumerate(CORR_GENERATORS):
    ratios = []
    for rep in range(reps):
        Sigma = np.asarray(random_covariance_matrix(n=n, corr_method=m,
                                                 rng=700000 + 1000 * mi + rep))
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(n))
        X = (L @ rng.standard_normal((n, T))).T
        S = np.cov(X, rowvar=False)
        a = rng.standard_normal(n)
        a /= np.linalg.norm(a)
        w0 = mvdr(S, a)
        wl = mvdr(S + load * (np.trace(S) / n) * np.eye(n), a)
        ratios.append((wl @ Sigma @ wl) / (w0 @ Sigma @ w0))
    ratios = np.array(ratios)
    rows.append((m.value, float(np.median(ratios)),
                 float(np.mean(ratios < 1.0))))
for name, med, frac in sorted(rows, key=lambda r: r[1]):
    print(f"{name:16s} median load/raw {med:5.2f}   loading wins {frac:5.0%}")
