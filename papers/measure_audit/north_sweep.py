"""Seeded sweep behind the North-rule table: reliability of the North et
al. (1982) rule of thumb for EOF separation, by named ensemble at fixed
(n, T).

Practitioners declare the leading EOF trustworthy when the eigenvalue
sampling error delta = lambda * sqrt(2/T) is smaller than the gap to the
next eigenvalue, both estimated from the sample. The table reports how
often the leading sample EOF is declared separated, and the
false-reassurance rate: the share of declared-separated draws whose leading
eigenvector is mostly wrong (squared alignment with the true leading
eigenvector below one half).

Run: python papers/measure_audit/north_sweep.py
"""
import warnings

import numpy as np

from randomcov import CORR_GENERATORS, random_correlation_matrix

warnings.filterwarnings("ignore")
rng = np.random.default_rng(18)
n, T, reps = 30, 60, 40

rows = []
for mi, m in enumerate(CORR_GENERATORS):
    sep, bad_given_sep = 0, 0
    for rep in range(reps):
        C = np.asarray(random_correlation_matrix(
            n=n, corr_method=m, rng=1000000 + 1000 * mi + rep))
        lam_t, vec_t = np.linalg.eigh(C + 1e-10 * np.eye(n))
        e1 = vec_t[:, -1]
        L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
        X = (L @ rng.standard_normal((n, T))).T
        S = np.cov(X, rowvar=False)
        s = np.sqrt(np.diag(S))
        lam, vec = np.linalg.eigh(S / np.outer(s, s))
        delta1 = lam[-1] * np.sqrt(2.0 / T)
        if lam[-1] - lam[-2] > delta1:
            sep += 1
            if (vec[:, -1] @ e1) ** 2 < 0.5:
                bad_given_sep += 1
    rate = bad_given_sep / sep if sep else float("nan")
    rows.append((m.value, sep / reps, rate))
for name, fsep, fr in sorted(rows, key=lambda r: (np.nan_to_num(r[2]), r[1])):
    fr_s = f"{fr:4.0%}" if fr == fr else " n/a"
    print(f"{name:16s} declared separated {fsep:4.0%}   "
          f"false reassurance {fr_s}")
