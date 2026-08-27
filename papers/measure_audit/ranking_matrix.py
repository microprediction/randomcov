"""Render the leaderboard matrix from ranking_results.json: ensembles
(rows) x covariance estimators from scikit-learn and precise (columns),
each cell the estimator's within-ensemble rank by median relative
Frobenius loss (1 = best of seventeen), with rank-1 cells outlined.

Run: python papers/measure_audit/ranking_matrix.py -> ranking.pdf/.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("papers/measure_audit/ranking_results.json") as f:
    results = json.load(f)

ENSEMBLES = ["sparse_precision", "ar1", "hierarchical", "block_equicorr",
             "factor", "kernel", "walk", "animals", "residuals",
             "archakov_hansen", "spectrum", "wishart", "lkj", "onion",
             "vine"]
names = list(next(iter(results.values()))["rank"].keys())
# order columns by mean rank, best overall on the left
names.sort(key=lambda s: np.mean([results[e]["rank"][s] for e in ENSEMBLES]))
R = np.array([[results[e]["rank"][s] for s in names] for e in ENSEMBLES],
             dtype=float)
N = len(names)

fig, ax = plt.subplots(figsize=(12.5, 7.0))
im = ax.imshow(R, cmap="RdYlGn_r", vmin=1, vmax=N, aspect="auto")

for i in range(len(ENSEMBLES)):
    for j in range(N):
        r = int(R[i, j])
        color = "white" if r <= 2 or r >= N - 2 else "black"
        ax.text(j, i, str(r), ha="center", va="center", fontsize=8.5,
                color=color)
        if r == 1:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="black", linewidth=2.2))

ax.set_xticks(range(N))
ax.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
ax.set_yticks(range(len(ENSEMBLES)))
ax.set_yticklabels(ENSEMBLES, fontsize=9, family="monospace")
ax.set_xticks(np.arange(-0.5, N), minor=True)
ax.set_yticks(np.arange(-0.5, len(ENSEMBLES)), minor=True)
ax.grid(which="minor", color="white", linewidth=1.0)
ax.tick_params(which="both", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[1, N])
cbar.ax.set_yticklabels(["best", "worst"], fontsize=9)
cbar.outline.set_visible(False)

ax.set_title("Rank of seventeen covariance estimators by generating "
             "ensemble\n(median relative Frobenius loss, $n=30$, $T=60$; "
             "outlined cells: rank 1)", fontsize=11, pad=12)
fig.tight_layout()
fig.savefig("papers/measure_audit/ranking.pdf")
fig.savefig("papers/measure_audit/ranking.png", dpi=170)
print("wrote ranking.pdf / ranking.png")
