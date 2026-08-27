"""Seeded sweep behind the regime figure: six audits re-run at n = 30 for
T in {15, 30, 60, 120, 240}, i.e. T/n from one half to eight, under every
named ensemble.

Audits and metrics (medians over twelve seeded draws per cell):
  lw     Ledoit-Wolf shrinkage intensity
  clip   Frobenius-loss ratio, MP-clipped over raw sample correlation
  hrp    true-variance ratio, HRP over long-only minimum variance
  load   true output power ratio, loaded over unloaded MVDR (pinv when
         the sample covariance is singular, which is the T < n case the
         loading literature addresses)
  gl     precision Frobenius-loss ratio, graphical lasso over Ledoit-Wolf
         (T > n only; the LW inverse needs an invertible estimate)
  taper  Frobenius-loss ratio, Gaspari-Cohn-tapered over raw sample
         covariance with T ensemble members

Results are written to regime_results.json for regime_matrix.py.

Full run (twelve reps per cell, roughly 20-40 minutes, dominated by the
cross-validated graphical lasso):
    python papers/measure_audit/regime_sweep.py
Smoke test: REPS=1 python papers/measure_audit/regime_sweep.py
"""
import json
import os
import warnings

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import GraphicalLassoCV, LedoitWolf

from randomcov import CORR_GENERATORS, random_covariance_matrix

warnings.filterwarnings("ignore")
rng = np.random.default_rng(30)
n, reps = 30, int(os.environ.get("REPS", "12"))
TS = [15, 30, 60, 120, 240]


def cov_to_corr(cov):
    s = np.sqrt(np.diag(cov))
    return cov / np.outer(s, s)


def mp_clip(R, T):
    edge = (1.0 + np.sqrt(n / T)) ** 2
    lam, V = np.linalg.eigh(R)
    bulk = lam < edge
    if bulk.any():
        lam = lam.copy()
        lam[bulk] = lam[bulk].mean()
    out = (V * lam) @ V.T
    d = np.sqrt(np.diag(out))
    return out / np.outer(d, d)


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


def minvar_long_only(cov):
    k = len(cov)
    res = minimize(lambda w: w @ cov @ w, np.ones(k) / k,
                   jac=lambda w: 2.0 * cov @ w,
                   bounds=[(0.0, 1.0)] * k,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                   method="SLSQP", options={"maxiter": 500})
    return res.x / res.x.sum()


def gc_taper(c=6.0):
    idx = np.arange(n)
    r = np.abs(idx[:, None] - idx[None, :]) / c
    out = np.zeros_like(r)
    a = r <= 1.0
    b = (r > 1.0) & (r <= 2.0)
    ra, rb = r[a], r[b]
    out[a] = (-0.25 * ra**5 + 0.5 * ra**4 + 0.625 * ra**3
              - (5.0 / 3.0) * ra**2 + 1.0)
    out[b] = ((1.0 / 12.0) * rb**5 - 0.5 * rb**4 + 0.625 * rb**3
              + (5.0 / 3.0) * rb**2 - 5.0 * rb + 4.0 - (2.0 / 3.0) / rb)
    return out


TAPER = gc_taper()

results = {a: {m.value: {} for m in CORR_GENERATORS}
           for a in ["lw", "clip", "hrp", "load", "gl", "taper"]}
for mi, m in enumerate(CORR_GENERATORS):
    for ti, T in enumerate(TS):
        vals = {a: [] for a in results}
        for rep in range(reps):
            Sigma = np.asarray(random_covariance_matrix(
                n=n, corr_method=m, rng=1600000 + 10000 * mi + 100 * ti + rep))
            Sigma = Sigma + 1e-10 * np.eye(n)
            L = np.linalg.cholesky(Sigma)
            X = (L @ rng.standard_normal((n, T))).T
            S = np.cov(X, rowvar=False)
            R = cov_to_corr(S)
            C = cov_to_corr(Sigma)

            vals["lw"].append(LedoitWolf().fit(X).shrinkage_)
            vals["clip"].append(np.linalg.norm(mp_clip(R, T) - C, "fro")
                                / np.linalg.norm(R - C, "fro"))
            w_h, w_m = hrp_weights(S), minvar_long_only(S)
            vals["hrp"].append((w_h @ Sigma @ w_h) / (w_m @ Sigma @ w_m))
            sv = rng.standard_normal(n)
            sv /= np.linalg.norm(sv)
            w0 = np.linalg.pinv(S) @ sv
            w0 /= sv @ w0
            Sl = S + 0.1 * (np.trace(S) / n) * np.eye(n)
            wl = np.linalg.solve(Sl, sv)
            wl /= sv @ wl
            vals["load"].append((wl @ Sigma @ wl) / (w0 @ Sigma @ w0))
            vals["taper"].append(np.linalg.norm(TAPER * S - Sigma, "fro")
                                 / np.linalg.norm(S - Sigma, "fro"))
            if T > n:
                Theta = np.linalg.inv(Sigma)
                try:
                    gl = GraphicalLassoCV(max_iter=200).fit(X).precision_
                    lw = np.linalg.inv(LedoitWolf().fit(X).covariance_)
                    vals["gl"].append(np.linalg.norm(gl - Theta, "fro")
                                      / np.linalg.norm(lw - Theta, "fro"))
                except Exception:
                    pass
        for a in results:
            if vals[a]:
                results[a][m.value][str(T)] = float(np.median(vals[a]))
    print(f"done {m.value}")

with open("papers/measure_audit/regime_results.json", "w") as f:
    json.dump(results, f, indent=1)
print("wrote regime_results.json")
