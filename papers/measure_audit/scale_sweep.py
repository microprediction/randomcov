"""The scaled regime sweep: the six regime audits re-run on a grid of
dimension and sample size, n in {30, 300, 3000} crossed with T/n in
{0.5, 1, 2, 4, 8}, under every named ensemble that is practical at that n.

Audits and metrics (medians over seeded draws per cell, as in
regime_sweep.py):
  lw     Ledoit-Wolf shrinkage intensity
  clip   Frobenius-loss ratio, MP-clipped over raw sample correlation
  hrp    true-variance ratio, HRP over long-only minimum variance
  load   true output power ratio, loaded over unloaded MVDR (pseudo-inverse
         when T <= n, the snapshot-starved case the loading literature
         addresses)
  gl     precision Frobenius-loss ratio, cross-validated graphical lasso
         over Ledoit-Wolf (T > n only; the LW inverse needs an invertible
         estimate)
  glf    the same ratio for a graphical lasso at the theory-driven penalty
         alpha = sqrt(log(n)/T) instead of a cross-validated one. One fit
         rather than a CV path, so it survives to n = 3000, where the CV
         variant does not; run at every n as the comparable series.
  taper  Frobenius-loss ratio, Gaspari-Cohn-tapered over raw sample
         covariance

Design differences from the n = 30 pilot (regime_sweep.py), all forced by
scale and applied at every n so the three panels are comparable:

  paired truths   one Sigma per (n, ensemble, rep), reused across all five
                  T; the regime axis is then a within-truth comparison.
  min-var solver  long-only minimum variance by FISTA with exact simplex
                  projection instead of SLSQP. SLSQP is O(n^3) per iterate
                  and does not reach n = 3000; the two agree to well under
                  a percent of true variance at n = 300 (see DOCKET.md).
  taper width     Gaspari-Cohn c = n/5, i.e. constant bandwidth relative
                  to the matrix, matching the pilot's c = 6 at n = 30.
  MVDR inverse    np.linalg.solve when T > n, an eigh pseudo-inverse
                  otherwise; identical in exact arithmetic to the pilot's
                  pinv, far cheaper at n = 3000, and free of the gesdd
                  non-convergence that np.linalg.pinv hits there.
  PSD floor       eigenvalues floored at 1e-10 of the largest before the
                  Cholesky. Several generators return truths with tiny
                  negative eigenvalues at large n; the floor is recorded
                  per ensemble in the meta block.

Three generators are expensive at n = 3000 because their construction is a
Python loop: animals (O(n^2) interactions x 500 steps, ~65 min a draw),
onion (an eigendecomposition per column, ~90 min) and vine (O(n^3)
recursion, ~42 min). They are skipped by default above n = 1000 and can be
run anyway with --include-slow. The cross-validated graphical lasso is off
by default above n = 300 for the same reason; the fixed-penalty variant
(glf) runs everywhere and is the series to compare across n. Whatever is
skipped is recorded in the meta block rather than silently dropped.

Run (writes regime_results_n<N>.json and regime_raw_n<N>.json):
    python papers/measure_audit/scale_sweep.py --n 30   --reps 24
    python papers/measure_audit/scale_sweep.py --n 300  --reps 24
    python papers/measure_audit/scale_sweep.py --n 3000 --reps 12 --workers 8
    # extend an existing n with the slow generators, keeping prior cells:
    python papers/measure_audit/scale_sweep.py --n 3000 --reps 12 \
        --only animals,onion,vine --include-slow --merge
"""
import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

AUDITS = ["lw", "clip", "hrp", "load", "gl", "glf", "taper"]

# generators whose cost is superquadratic in n inside the Python loop
SLOW_GENERATORS = {"animals", "onion", "vine"}
SLOW_LIMIT = 1000          # excluded above this n
GLASSO_LIMIT = 300         # graphical-lasso audit skipped above this n


def build_cells(raw, prior_cells, audits, verbose=False):
    """Fresh cells merged over prior ones. Fresh audits win; audits this run
    did not compute are carried forward from the prior cell."""
    cells = [{"ensemble": r["ensemble"], "rep": r["rep"],
              "floored": r["floored"], "num_rank": r["num_rank"],
              "notes": r.get("notes", {}),
              "values": {a: {str(k): v for k, v in r["values"][a].items()}
                         for a in AUDITS}} for r in raw]
    if not prior_cells:
        return cells
    by_key = {(c["ensemble"], c["rep"]): c for c in cells}
    kept, extended = [], 0
    for c in prior_cells:
        fresh = by_key.get((c["ensemble"], c["rep"]))
        if fresh is None:
            kept.append(c)
            continue
        for a in AUDITS:
            if a not in audits and c["values"].get(a):
                fresh["values"][a] = c["values"][a]
        extended += 1
    if verbose:
        print(f"merge: {len(kept)} prior cells kept, {len(cells)} recomputed"
              + (f" ({extended} of them extending prior cells with "
                 f"{','.join(audits)})" if extended else ""), flush=True)
    return kept + cells


def build_tasks(n, reps, ensembles, index, opts, rep_start=0):
    return [(n, index[e], e, rep, opts)
            for e in ensembles for rep in range(rep_start, rep_start + reps)]


def run_cell(task):
    """Wrapper: a numerical failure loses one cell, not the whole sweep."""
    try:
        return run_cell_inner(task)
    except Exception as exc:                      # noqa: BLE001
        n, mi, ensemble, rep, opts = task
        return {"ensemble": ensemble, "rep": rep, "floored": False,
                "num_rank": 0, "seconds": 0.0, "values": {a: {} for a in AUDITS},
                "notes": {}, "error": f"{type(exc).__name__}: {exc}"}


def run_cell_inner(task):
    """One (n, ensemble, rep): draw a truth, then every T against it."""
    import numpy as np
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform
    from sklearn.covariance import (GraphicalLassoCV, LedoitWolf,
                                    graphical_lasso)

    from randomcov import random_covariance_matrix

    n, mi, ensemble, rep, opts = task
    TS = [int(round(f * n)) for f in (0.5, 1, 2, 4, 8)]
    want = set(opts["audits"])
    t_cell = time.time()

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

    def proj_simplex(v):
        u = np.sort(v)[::-1]
        css = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1.0))[0][-1]
        return np.maximum(v - (css[rho] - 1.0) / (rho + 1.0), 0.0)

    def minvar_long_only(cov, iters=5000, tol=1e-13):
        """min w'Sw over the simplex, FISTA with exact simplex projection."""
        step = 1.0 / (2.0 * np.linalg.eigvalsh(cov)[-1])
        k = len(cov)
        w = np.ones(k) / k
        y, t = w.copy(), 1.0
        for _ in range(iters):
            wn = proj_simplex(y - step * (2.0 * (cov @ y)))
            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = wn + ((t - 1.0) / tn) * (wn - w)
            if np.max(np.abs(wn - w)) < tol:
                return wn
            w, t = wn, tn
        return w

    def sym_pinv_apply(cov, v, rcond=1e-15):
        """Moore-Penrose solve for a symmetric PSD matrix, via eigh.
        LAPACK's gesdd (behind np.linalg.pinv) fails to converge on some
        n = 3000 sample covariances; eigh is both robust and cheaper."""
        lam, U = np.linalg.eigh(cov)
        keep = lam > rcond * lam[-1]
        Uk = U[:, keep]
        return Uk @ ((Uk.T @ v) / lam[keep])

    def gc_taper(c):
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

    # --- the truth, drawn once and shared by every T -------------------
    Sigma = np.asarray(random_covariance_matrix(
        n=n, corr_method=ensemble,
        rng=np.random.default_rng([n, mi, rep, 0])))
    Sigma = 0.5 * (Sigma + Sigma.T)
    lam, V = np.linalg.eigh(Sigma)
    floor = 1e-10 * lam[-1]
    floored = bool((lam < floor).any())
    # numerical rank of the truth: several generators are built from a
    # finite history or a low-dimensional latent, and degenerate as n grows
    num_rank = int((lam > floor).sum())
    if floored:
        Sigma = (V * np.maximum(lam, floor)) @ V.T
        Sigma = 0.5 * (Sigma + Sigma.T)
    L = np.linalg.cholesky(Sigma)
    C = cov_to_corr(Sigma)
    TAPER = gc_taper(n / 5.0)
    Theta = None

    out = {a: {} for a in AUDITS}
    notes = {}          # audit@T -> why the metric is missing, never silent
    for ti, T in enumerate(TS):
        noise = np.random.default_rng([n, mi, rep, 1, ti])
        X = (L @ noise.standard_normal((n, T))).T
        S = np.cov(X, rowvar=False)
        R = cov_to_corr(S)

        if "lw" in want:
            out["lw"][T] = float(LedoitWolf().fit(X).shrinkage_)
        if "clip" in want:
            out["clip"][T] = float(np.linalg.norm(mp_clip(R, T) - C, "fro")
                                   / np.linalg.norm(R - C, "fro"))
        if "hrp" in want:
            w_h, w_m = hrp_weights(S), minvar_long_only(S)
            out["hrp"][T] = float((w_h @ Sigma @ w_h) / (w_m @ Sigma @ w_m))

        # the steering vector is drawn whether or not the MVDR audit runs,
        # so that omitting an audit never shifts another audit's stream
        sv = noise.standard_normal(n)
        if "load" in want:
            sv = sv / np.linalg.norm(sv)
            w0 = np.linalg.solve(S, sv) if T > n else sym_pinv_apply(S, sv)
            w0 /= sv @ w0
            Sl = S + 0.1 * (np.trace(S) / n) * np.eye(n)
            wl = np.linalg.solve(Sl, sv)
            wl /= sv @ wl
            out["load"][T] = float((wl @ Sigma @ wl) / (w0 @ Sigma @ w0))

        if "taper" in want:
            out["taper"][T] = float(np.linalg.norm(TAPER * S - Sigma, "fro")
                                    / np.linalg.norm(S - Sigma, "fro"))

        run_cv = opts["glasso_cv"] and "gl" in want
        run_fixed = opts["glasso_fixed"] and "glf" in want
        if T > n and (run_cv or run_fixed):
            if Theta is None:
                Theta = np.linalg.inv(Sigma)
            lw_prec, lw_loss = None, None
            try:
                lw_prec = np.linalg.inv(LedoitWolf().fit(X).covariance_)
                lw_loss = np.linalg.norm(lw_prec - Theta, "fro")
            except Exception as exc:              # noqa: BLE001
                notes[f"lw_inverse@{T}"] = f"{type(exc).__name__}"
            if lw_loss and run_cv:
                try:
                    gl = GraphicalLassoCV(max_iter=200).fit(X).precision_
                    out["gl"][T] = float(np.linalg.norm(gl - Theta, "fro")
                                         / lw_loss)
                except Exception as exc:          # noqa: BLE001
                    notes[f"gl@{T}"] = f"{type(exc).__name__}"
            if lw_loss and run_fixed:
                # theory-driven penalty rather than a CV path: one fit, so
                # it survives to n = 3000. Applied to the sample correlation
                # and mapped back, the usual scale-free convention.
                alpha = np.sqrt(np.log(n) / T)
                try:
                    _, P = graphical_lasso(R, alpha=alpha,
                                           max_iter=opts["glasso_max_iter"])
                except Exception as exc:          # noqa: BLE001
                    # The solver refuses the theory penalty on densely
                    # correlated truths: the solution there is near-singular
                    # and the dual update leaves the SPD cone. This is not
                    # ill-conditioning of the input (block_equicorr fails at
                    # cond 2.6e3) and neither the lars mode nor a ridge on R
                    # rescues it -- only a larger penalty does, which is what
                    # a practitioner would be forced to do. Climb the ladder,
                    # record the multiplier that worked, and fall back to a
                    # ridge only if the whole ladder fails. A cell rescued at
                    # 16x or beyond is effectively a diagonal estimate; the
                    # multiplier is the finding, so it is always recorded.
                    P = None
                    for mult in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
                        try:
                            _, P = graphical_lasso(
                                R, alpha=mult * alpha,
                                max_iter=opts["glasso_max_iter"])
                            notes[f"glf_alpha@{T}"] = f"{mult:g}x"
                            break
                        except Exception:
                            continue
                    if P is None:
                        for eps in (1e-2, 1e-1):
                            Re = (R + eps * np.eye(n)) / (1.0 + eps)
                            try:
                                _, P = graphical_lasso(
                                    Re, alpha=alpha,
                                    max_iter=opts["glasso_max_iter"])
                                notes[f"glf_ridge@{T}"] = f"{eps:g}"
                                break
                            except Exception:
                                continue
                    if P is None:
                        notes[f"glf@{T}"] = f"{type(exc).__name__}"
                if P is not None:
                    d = 1.0 / np.sqrt(np.diag(S))
                    P = P * np.outer(d, d)
                    out["glf"][T] = float(np.linalg.norm(P - Theta, "fro")
                                          / lw_loss)

    return {"ensemble": ensemble, "rep": rep, "floored": floored,
            "num_rank": num_rank, "seconds": time.time() - t_cell,
            "values": out, "notes": notes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--blas-threads", type=int, default=1)
    ap.add_argument("--outdir", default="papers/measure_audit")
    ap.add_argument("--only", default=None,
                    help="comma-separated ensembles to run (default: all "
                         "practical at this n)")
    ap.add_argument("--include-slow", action="store_true",
                    help="run animals/onion/vine even above the slow limit")
    ap.add_argument("--merge", action="store_true",
                    help="merge into any existing results at this n, "
                         "replacing recomputed (ensemble, rep) cells")
    ap.add_argument("--glasso-cv", choices=["auto", "on", "off"],
                    default="auto",
                    help="cross-validated graphical lasso (auto: on up to "
                         f"n = {GLASSO_LIMIT})")
    ap.add_argument("--glasso-fixed", choices=["on", "off"], default="on",
                    help="fixed-penalty graphical lasso, alpha = "
                         "sqrt(log n / T)")
    ap.add_argument("--glasso-max-iter", type=int, default=100)
    ap.add_argument("--audits", default=None,
                    help="comma-separated audits to compute (default: all). "
                         "With --merge, audits absent here are kept from the "
                         "prior run, so a single audit can be added to a "
                         "finished grid without recomputing the rest.")
    ap.add_argument("--recover", action="store_true",
                    help="rebuild the grid from the append-only cell log "
                         "regime_cells_n<N>.jsonl without recomputing "
                         "anything")
    ap.add_argument("--rep-start", type=int, default=0,
                    help="first rep index; with --merge, adds reps to an "
                         "existing run instead of recomputing them")
    args = ap.parse_args()

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(args.blas_threads)

    import numpy as np
    from multiprocessing import get_context

    from randomcov import CORR_GENERATORS

    all_ens = [m.value for m in CORR_GENERATORS]
    index = {e: i for i, e in enumerate(all_ens)}
    if args.n > SLOW_LIMIT and not args.include_slow:
        ensembles = [e for e in all_ens if e not in SLOW_GENERATORS]
        excluded = sorted(SLOW_GENERATORS)
    else:
        ensembles = all_ens
        excluded = []
    if args.only:
        want = [e.strip() for e in args.only.split(",")]
        unknown = [e for e in want if e not in index]
        if unknown:
            raise SystemExit(f"unknown ensembles: {unknown}")
        ensembles = want
        excluded = [e for e in excluded if e not in want]

    glasso_cv = (args.n <= GLASSO_LIMIT if args.glasso_cv == "auto"
                 else args.glasso_cv == "on")
    audits = AUDITS if args.audits is None else [a.strip() for a
                                                 in args.audits.split(",")]
    unknown = [a for a in audits if a not in AUDITS]
    if unknown:
        raise SystemExit(f"unknown audits: {unknown}")
    opts = {"glasso_cv": glasso_cv,
            "glasso_fixed": args.glasso_fixed == "on",
            "glasso_max_iter": args.glasso_max_iter,
            "audits": audits}

    n = args.n
    TS = [int(round(f * n)) for f in (0.5, 1, 2, 4, 8)]
    tasks = build_tasks(n, args.reps, ensembles, index, opts, args.rep_start)
    print(f"n={n} T={TS} ensembles={len(ensembles)} reps={args.reps} "
          f"cells={len(tasks)} workers={args.workers} "
          f"blas={args.blas_threads} glasso_cv={glasso_cv} "
          f"glasso_fixed={opts['glasso_fixed']} merge={args.merge}",
          flush=True)

    # read any prior grid once, up front: emit() merges against this copy in
    # memory, so checkpoints are idempotent
    raw_path = os.path.join(args.outdir, f"regime_raw_n{n}.json")
    prior_meta, prior_cells = {}, []
    if args.merge and os.path.exists(raw_path):
        with open(raw_path) as f:
            prior = json.load(f)
        prior_meta = prior.get("meta", {})
        prior_cells = prior.get("cells", [])
        for e in prior_meta.get("ensembles", []):
            if e not in ensembles:
                ensembles.append(e)
        ensembles.sort(key=lambda e: index[e])
        excluded = [e for e in excluded if e not in ensembles]

    t0 = time.time()
    raw, floored, seconds, ranks, errors = [], {}, {}, {}, []

    # Append-only sink, the last line of defence. Every finished cell is
    # flushed here the moment it arrives, before any derived bookkeeping can
    # raise. A 20-hour draw must survive a bug anywhere downstream of it;
    # rebuild the grid from this file with --recover.
    jsonl_path = os.path.join(args.outdir, f"regime_cells_n{n}.jsonl")
    jsonl = open(jsonl_path, "a", buffering=1)

    def sink(res):
        try:
            jsonl.write(json.dumps(res) + "\n")
            jsonl.flush()
            os.fsync(jsonl.fileno())
        except Exception as exc:                  # noqa: BLE001
            print(f"WARNING: could not append to {jsonl_path}: {exc}",
                  flush=True)

    def emit(final=False):
        """Write results from whatever has completed so far.

        Called on every finished cell as well as at the end: an n = 3000
        animals draw costs the better part of a day, and an exception or a
        kill after the pool drains must never discard that. Writes are
        atomic (temp file + replace) so a checkpoint can never leave a
        half-written grid on disk.
        """
        cells = build_cells(raw, prior_cells, audits, verbose=final)

        medians = {a: {e: {} for e in ensembles} for a in AUDITS}
        counts = {a: {e: {} for e in ensembles} for a in AUDITS}
        for a in AUDITS:
            for e in ensembles:
                for T in TS:
                    vals = [c["values"][a][str(T)] for c in cells
                            if c["ensemble"] == e
                            and str(T) in c["values"].get(a, {})]
                    if vals:
                        medians[a][e][str(T)] = float(np.median(vals))
                        counts[a][e][str(T)] = len(vals)

        def merged(key, fresh_value):
            """Carry a prior meta dict forward under a merge, fresh wins."""
            out = dict(prior_meta.get(key) or {})
            out.update(fresh_value)
            return out

        reps_by_ensemble, note_tally = {}, {}
        for c in cells:
            reps_by_ensemble[c["ensemble"]] = reps_by_ensemble.get(
                c["ensemble"], 0) + 1
            for k, v in (c.get("notes") or {}).items():
                key = f"{c['ensemble']} {k.split('@')[0]} {v}"
                note_tally[key] = note_tally.get(key, 0) + 1

        # coverage: cells where an audit produced no number. An audit this
        # run was not asked to compute is only reported if prior merged
        # cells hold a partial series; "not computed" is not "missing".
        missing = {}
        for a in AUDITS:
            for e in ensembles:
                want_T = ([T for T in TS if T > n] if a in ("gl", "glf")
                          else TS)
                expect = len(want_T) * reps_by_ensemble.get(e, 0)
                got = sum(len([T for T in want_T
                               if str(T) in c["values"].get(a, {})])
                          for c in cells if c["ensemble"] == e)
                # an audit deliberately run at fewer reps than the grid (the
                # cross-validated glasso costs ~10 min a fit, so it runs at
                # half the reps) is complete at its own rep count: judge it
                # by whole cells covered, not by the grid's cell count
                cells_e = [c for c in cells if c["ensemble"] == e]
                with_any = sum(1 for c in cells_e if c["values"].get(a))
                if with_any and got == len(want_T) * with_any:
                    continue                  # every cell it ran is full
                if not expect or got == expect:
                    continue
                disabled = ((a == "gl" and not glasso_cv)
                            or (a == "glf" and not opts["glasso_fixed"]))
                if got == 0 and (a not in audits or disabled):
                    continue
                missing[f"{a}/{e}"] = f"{got}/{expect}"

        meta = {
            "n": n, "reps": args.reps, "TS": TS,
            "audits_computed": audits,
            "complete": final,
            "cells_done_this_pass": len(raw),
            "cells_planned_this_pass": len(tasks),
            "reps_by_ensemble": reps_by_ensemble,
            "ensembles": ensembles,
            "excluded_generators": excluded,
            "excluded_reason": ("superquadratic Python generator loops"
                                if excluded else None),
            "glasso_cv_run": glasso_cv,
            "glasso_cv_skip_reason": (None if glasso_cv else
                                      "GraphicalLassoCV is O(n^3) per sweep"),
            "glasso_fixed_run": opts["glasso_fixed"],
            "glasso_fixed_alpha": "sqrt(log(n)/T)",
            "minvar_solver": "FISTA + exact simplex projection",
            "workers": args.workers, "blas_threads": args.blas_threads,
            "taper_c": n / 5.0,
            "psd_floored_ensembles": sorted(
                set(prior_meta.get("psd_floored_ensembles", []))
                | {e for e, f in floored.items() if any(f)}),
            "median_numerical_rank": merged(
                "median_numerical_rank",
                {e: float(np.median(r)) for e, r in ranks.items()}),
            "median_cell_seconds": merged(
                "median_cell_seconds",
                {e: float(np.median(s)) for e, s in seconds.items()}),
            "wall_seconds": time.time() - t0,
            "failed_cells": errors,
            "solver_notes": note_tally,
            "incomplete_coverage": missing,
        }

        res_path = os.path.join(args.outdir, f"regime_results_n{n}.json")
        for path, payload in ((res_path, medians),
                              (raw_path, {"meta": meta, "counts": counts,
                                          "cells": cells})):
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=1)
            os.replace(tmp, path)
        if final:
            print(f"wrote regime_results_n{n}.json / regime_raw_n{n}.json "
                  f"in {time.time()-t0:.0f}s", flush=True)

    if args.recover:
        # rebuild the grid from the append-only sink; no compute at all
        seen = set()
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    res = json.loads(line)
                except json.JSONDecodeError:
                    continue              # torn final line, ignore
                key = (res["ensemble"], res["rep"])
                if key in seen:
                    raw[:] = [r for r in raw
                              if (r["ensemble"], r["rep"]) != key]
                seen.add(key)
                raw.append(res)
        print(f"recover: {len(raw)} cells read from {jsonl_path}", flush=True)
        for res in raw:
            floored.setdefault(res["ensemble"], []).append(res["floored"])
            seconds.setdefault(res["ensemble"], []).append(res["seconds"])
            ranks.setdefault(res["ensemble"], []).append(res["num_rank"])
        emit(final=True)
        jsonl.close()
        return

    ctx = get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_cell, tasks), 1):
            sink(res)                 # on disk before anything can fail
            raw.append(res)
            floored.setdefault(res["ensemble"], []).append(res["floored"])
            seconds.setdefault(res["ensemble"], []).append(res["seconds"])
            ranks.setdefault(res["ensemble"], []).append(res["num_rank"])
            if res.get("error"):
                errors.append({"ensemble": res["ensemble"],
                               "rep": res["rep"], "error": res["error"]})
            print(f"[{i}/{len(tasks)}] {res['ensemble']} rep{res['rep']} "
                  f"{res['seconds']:.1f}s rank {res['num_rank']}/{n}  "
                  f"elapsed {time.time()-t0:.0f}s"
                  + (f"  ERROR {res['error']}" if res.get("error") else ""),
                  flush=True)
            # a failure in derived bookkeeping must never stop the sweep or
            # discard cells: the sink already holds them, and the next
            # checkpoint will try again
            try:
                emit()
            except Exception as exc:              # noqa: BLE001
                print(f"WARNING: checkpoint failed ({type(exc).__name__}: "
                      f"{exc}); cells are safe in {jsonl_path}", flush=True)

    try:
        emit(final=True)
    except Exception as exc:                      # noqa: BLE001
        print(f"ERROR: final write failed ({type(exc).__name__}: {exc}). "
              f"Rebuild with --recover from {jsonl_path}", flush=True)
        raise
    finally:
        jsonl.close()


if __name__ == "__main__":
    main()
