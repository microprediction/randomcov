"""Render the regime figure from regime_results.json: six audits, each a
panel of metric against T/n (log axis), one line per ensemble colored by
structural family. Run regime_sweep.py first.

Run: python papers/measure_audit/regime_matrix.py -> regime.pdf/.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

with open("papers/measure_audit/regime_results.json") as f:
    results = json.load(f)

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
    ("gl", "glasso / Ledoit--Wolf (precision loss)", True, 1.0),
    ("taper", "GC-tapered / raw (Frobenius)", True, 1.0),
]

n = 30
fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), sharex=True)
for ax, (key, title, logy, hline) in zip(axes.flat, PANELS):
    for e, series in results[key].items():
        ts = sorted(int(t) for t in series)
        ax.plot([t / n for t in ts], [series[str(t)] for t in ts],
                marker="o", markersize=3, linewidth=1.4,
                color=COLOR[e], alpha=0.75)
    if hline is not None:
        ax.axhline(hline, color="black", linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
for ax in axes[1]:
    ax.set_xlabel("$T/n$", fontsize=10)

handles = [Line2D([0], [0], color=c, linewidth=2.2)
           for _, (_, c) in FAMILIES.items()]
fig.legend(handles, list(FAMILIES), loc="lower center", ncol=4,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Six audits across the regime axis: one line per ensemble, "
             "$n=30$", fontsize=12)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("papers/measure_audit/regime.pdf")
fig.savefig("papers/measure_audit/regime.png", dpi=170)
print("wrote regime.pdf / regime.png")
