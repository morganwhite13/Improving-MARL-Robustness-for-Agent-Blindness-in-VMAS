"""Plot experiments/robustness_results.json: evaluation reward vs blindness probability per training condition."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(here, "robustness_results.json")))
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
fig.suptitle(f"Robustness: training condition vs blindness at evaluation (mean ± std over {len(seeds)} seeds)")
fig.tight_layout()
fig.savefig(os.path.join(here, "robustness.png"), dpi=120)
print("wrote", os.path.join(here, "robustness.png"))
