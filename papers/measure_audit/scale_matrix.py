"""Render the scaled regime figure from regime_results_n{30,300,3000}.json:
six audits (rows) across three dimensions (columns), each panel a metric
against T/n with one line per ensemble, colored by structural family.
Rows share a y scale so the drift with n is visible.

Run scale_sweep.py at each n first.

Run: python papers/measure_audit/scale_matrix.py -> scale.pdf / scale.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import (FixedFormatter, FixedLocator, NullFormatter,
                               NullLocator)

HERE = os.path.dirname(os.path.abspath(__file__))
NS = [30, 100, 300, 1000, 3000]

# Generators native to this package rather than to a published method.
# They stay in the data but are drawn dashed and excluded from the headline
# verdict counts: two of the three are built from a fixed construction
# budget (residuals from a 1000-point regression sample, animals from 500
# simulation steps) and so hand back rank-deficient truths as n grows.
PACKAGE_NATIVE = {"residuals", "walk", "animals"}

FAMILIES = {
    "precision-structured": (["sparse_precision", "ar1", "hierarchical",
                              "block_equicorr"], "#2a7e43"),
    "factor-heavy": (["factor", "residuals", "animals", "walk",
                      "archakov_hansen"], "#d97817"),
    "dense elliptope / spectral": (["lkj", "onion", "vine", "wishart",
                                    "spectrum"], "#6a51a3"),
    "kernel": (["kernel"], "#c22f2f"),
}
COLOR = {e: c for _, (es, c) in FAMILIES.items() for e in es}

PANELS = [
    ("lw", "Ledoit--Wolf intensity", False, None),
    ("clip", "MP clip / raw (Frobenius)", True, 1.0),
    ("hrp", "HRP / long-only min-var (true variance)", True, 1.0),
    ("load", "loaded / unloaded MVDR (true power)", True, 1.0),
    ("gl", "CV glasso / Ledoit--Wolf (precision loss)", True, 1.0),
    ("glf", "fixed-$\\alpha$ glasso / Ledoit--Wolf", True, 1.0),
    ("taper", "GC-tapered / raw (Frobenius)", True, 1.0),
]

results, metas = {}, {}
for n in NS:
    path = os.path.join(HERE, f"regime_results_n{n}.json")
    if os.path.exists(path):
        with open(path) as f:
            results[n] = json.load(f)
        raw = os.path.join(HERE, f"regime_raw_n{n}.json")
        if os.path.exists(raw):
            with open(raw) as f:
                metas[n] = json.load(f)["meta"]
present = [n for n in NS if n in results]

fig, axes = plt.subplots(len(PANELS), len(present),
                         figsize=(3.9 * len(present), 2.35 * len(PANELS)),
                         squeeze=False)
for ri, (key, title, logy, hline) in enumerate(PANELS):
    lo, hi = np.inf, -np.inf
    for ci, n in enumerate(present):
        ax = axes[ri][ci]
        for e, series in results[n].get(key, {}).items():
            ts = sorted(int(t) for t in series)
            if not ts:
                continue
            ys = [series[str(t)] for t in ts]
            lo, hi = min(lo, min(ys)), max(hi, max(ys))
            native = e in PACKAGE_NATIVE
            ax.plot([t / n for t in ts], ys, marker="o", markersize=3,
                    linewidth=0.9 if native else 1.4, color=COLOR[e],
                    alpha=0.45 if native else 0.8,
                    linestyle="--" if native else "-")
        if hline is not None:
            ax.axhline(hline, color="black", linewidth=0.8, linestyle=":")
        # a log x-axis re-labels its own minor ticks unless both the locator
        # and the minor formatter are pinned
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2", "4",
                                                     "8"]))
        ax.xaxis.set_minor_locator(NullLocator())
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=8)
        if ri == 0:
            ax.set_title(f"$n = {n}$", fontsize=11)
        if ri == len(PANELS) - 1:
            ax.set_xlabel("$T/n$", fontsize=9)
        if ci == 0:
            ax.set_ylabel(title, fontsize=7.5)
        else:
            ax.tick_params(labelleft=False)
        if not any(results[n].get(key, {}).values()):
            ax.text(0.5, 0.5, "not run at this $n$", fontsize=8,
                    ha="center", va="center", color="0.45",
                    transform=ax.transAxes)
            ax.tick_params(labelleft=False, labelbottom=False,
                           left=False, bottom=False)
    if np.isfinite(lo) and np.isfinite(hi):
        for ci in range(len(present)):
            axes[ri][ci].set_ylim(lo * 0.85 if logy else lo - 0.05,
                                  hi * 1.18 if logy else hi + 0.05)

handles = [Line2D([0], [0], color=c, linewidth=2.2)
           for _, (_, c) in FAMILIES.items()]
labels = list(FAMILIES)
handles.append(Line2D([0], [0], color="0.35", linewidth=1.2, linestyle="--"))
labels.append("package-native curiosity (not in the counts)")
fig.legend(handles, labels, loc="lower center", ncol=5,
           fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.002))
fig.suptitle(f"{len(PANELS)} audits across dimension and regime: "
             "one line per ensemble", fontsize=12)
fig.tight_layout(rect=[0, 0.028, 1, 0.985])
fig.savefig(os.path.join(HERE, "scale.pdf"))
fig.savefig(os.path.join(HERE, "scale.png"), dpi=170)
print("wrote scale.pdf / scale.png")

# --- verdict flips: does the sign of each claim survive the growth in n? --
print("\nverdict table over the literature ensembles (fraction of cells with "
      "metric > 1, by n; lw reports the median intensity).")
print("package-native curiosities are excluded; the bracketed figure "
      "includes them.")
for key, title, _, _ in PANELS:
    row = []
    for n in present:
        lit = [v for e, s in results[n].get(key, {}).items()
               if e not in PACKAGE_NATIVE for v in s.values()]
        allv = [v for e, s in results[n].get(key, {}).items()
                for v in s.values()]
        if not lit:
            row.append("      --      ")
        elif key == "lw":
            row.append(f" {np.median(lit):6.3f} [{np.median(allv):5.3f}]")
        else:
            row.append(f" {np.mean(np.array(lit) > 1.0):6.2f} "
                       f"[{np.mean(np.array(allv) > 1.0):5.2f}]")
    print(f"{key:6s} " + "".join(row))
