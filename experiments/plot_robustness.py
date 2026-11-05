"""Plot robustness results: evaluation reward vs blindness probability per training condition.

Usage: python experiments/plot_robustness.py [results.json]  (default: robustness_results.json)
Writes a PNG with the same name next to the results file.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
results = os.path.join(here, sys.argv[1] if len(sys.argv) > 1 else "robustness_results.json")
data = json.load(open(results))
seeds = list(data["seeds"].values())
trains = list(seeds[0])
etypes = list(seeds[0][trains[0]]["eval"])
probs = list(seeds[0][trains[0]]["eval"][etypes[0]])

fig, axes = plt.subplots(1, len(etypes), figsize=(12, 4.8), sharey=True)
for ax, etype in zip(axes, etypes):
    for train in trains:
        scores = np.array([[s[train]["eval"][etype][p] for p in probs] for s in seeds])  # [seeds, probs]
        mean, std = scores.mean(0), scores.std(0, ddof=1)
        x = [float(p) for p in probs]
        ax.errorbar(x, mean, yerr=std, marker="o", capsize=3, label=f"Trained: {train.lower()}")
        print(f"{train:24s} | eval {etype:24s} | " + "  ".join(f"{m:5.1f}±{s:4.1f}" for m, s in zip(mean, std)))
    ax.set_title(f"Evaluated with: {etype.lower()}")
    ax.set_xlabel("Blindness probability at evaluation")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Episode reward (deterministic policy)")
axes[0].legend(fontsize=8)
iters = "100 training iterations" if "100 iterations" in data["description"] else "25 training iterations"
fig.suptitle(f"Robustness after {iters}: training condition vs blindness at evaluation (mean ± std over {len(seeds)} seeds)")
fig.tight_layout()
png = results.replace("_results.json", ".png")
fig.savefig(png, dpi=120)
print("wrote", png)
