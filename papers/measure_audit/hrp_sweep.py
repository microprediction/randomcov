"""Seeded sweep behind the HRP-versus-minimum-variance table: out-of-sample
true portfolio variance of hierarchical risk parity (Lopez de Prado 2016)
against unconstrained minimum variance, by named ensemble at fixed (n, T).

Both portfolios are built from the same sample covariance; the scoreboard is
the true variance w' Sigma w under the generating covariance.

Run: python papers/measure_audit/hrp_sweep.py
"""
import warnings

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from randomcov import CORR_GENERATORS, random_covariance_matrix

warnings.filterwarnings("ignore")
rng = np.random.default_rng(2)
n, T, reps = 30, 60, 20


def cov_to_corr(cov):
    s = np.sqrt(np.diag(cov))
    return cov / np.outer(s, s)


def cluster_var(cov, idx):
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(cov):
    corr = cov_to_corr(cov)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    link = sch.linkage(squareform(dist, checks=False), "single")
    order = list(sch.leaves_list(link))
    w = np.ones(len(cov))
    stack = [order]
    while stack:
        cl = stack.pop()
        if len(cl) < 2:
            continue
        k = len(cl) // 2
        a, b = cl[:k], cl[k:]
        va, vb = cluster_var(cov, a), cluster_var(cov, b)
        alpha = 1.0 - va / (va + vb)
        w[a] *= alpha
        w[b] *= 1.0 - alpha
        stack += [a, b]
    return w / w.sum()


def minvar_weights(cov):
    ones = np.ones(len(cov))
    w = np.linalg.solve(cov, ones)
    return w / w.sum()


def minvar_long_only(cov):
    from scipy.optimize import minimize

    k = len(cov)
    res = minimize(lambda w: w @ cov @ w, np.ones(k) / k,
                   jac=lambda w: 2.0 * cov @ w,
                   bounds=[(0.0, 1.0)] * k,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                   method="SLSQP", options={"maxiter": 500})
    return res.x / res.x.sum()


rows = []
for mi, m in enumerate(CORR_GENERATORS):
    r_unc, r_lo = [], []
    for rep in range(reps):
        S = np.asarray(random_covariance_matrix(n=n, corr_method=m,
                                                 rng=100000 + 1000 * mi + rep))
        L = np.linalg.cholesky(S + 1e-10 * np.eye(n))
        X = (L @ rng.standard_normal((n, T))).T
        Shat = np.cov(X, rowvar=False)
        v_hrp = hrp_weights(Shat) @ S @ hrp_weights(Shat)
        w_mv, w_lo = minvar_weights(Shat), minvar_long_only(Shat)
        r_unc.append(v_hrp / (w_mv @ S @ w_mv))
        r_lo.append(v_hrp / (w_lo @ S @ w_lo))
    r_unc, r_lo = np.array(r_unc), np.array(r_lo)
    rows.append((m.value,
                 float(np.median(r_lo)), float(np.mean(r_lo < 1.0)),
                 float(np.median(r_unc)), float(np.mean(r_unc < 1.0))))
print(f"{'ensemble':16s} {'HRP/MVlo':>9s} {'wins':>5s} {'HRP/MVunc':>10s} {'wins':>5s}")
for name, mlo, flo, munc, func_ in sorted(rows, key=lambda r: r[1]):
    print(f"{name:16s} {mlo:9.2f} {flo:5.0%} {munc:10.2f} {func_:5.0%}")
