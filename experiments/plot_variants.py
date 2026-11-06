"""Plot experiments/variants_results.json against the 100-iteration robustness test.

Evaluation reward against blindness probability for policies trained with
plain blindness, a blind flag, a flag plus memory, and a curriculum.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
variants = [r for r in json.load(open(os.path.join(here, "variants_results.json")))["runs"]
            if r.get("scenario", "balance") == "balance"]  # other scenarios are reported in the README
robust = json.load(open(os.path.join(here, "robustness_100iters_results.json")))["seeds"]
etypes = ["One agent, 1-10 steps", "Each agent, 1-10 steps"]
probs = ["0.0", "0.05", "0.1", "0.2", "0.3", "0.5"]

series = {
    "trained: no blindness": [robust[s]["No blindness"]["eval"] for s in robust],
    "trained: one-agent blindness": [robust[s]["One agent, 1-10 steps"]["eval"] for s in robust],
    "trained: each-agent blindness": [robust[s]["Each agent, 1-10 steps"]["eval"] for s in robust],
}
for name, label in [("aware-flag", "one-agent blindness + blind flag"),
                    ("aware-memory", "one-agent blindness + flag and memory"),
                    ("curriculum", "one-agent blindness, curriculum"),
                    ("aware-memory-each", "each-agent blindness + flag and memory")]:
    series["trained: " + label] = [r["eval"] for r in variants if r["variant"] == name]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, etype in zip(axes, etypes, strict=True):
    for name, evals in series.items():
        scores = np.array([[e[etype][p] for p in probs] for e in evals])
        mean, std = scores.mean(0), scores.std(0, ddof=1)
        ax.errorbar([float(p) for p in probs], mean, yerr=std, marker="o", capsize=3, label=name)
        print(f"{name:48s} | {etype:22s} | " + "  ".join(f"{m:5.1f}±{s:4.1f}" for m, s in zip(mean, std, strict=True)))
    ax.set_title(f"Evaluated with: {etype.lower()}")
    ax.set_xlabel("Blindness probability at evaluation")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Episode reward (deterministic policy)")
axes[0].legend(fontsize=7)
fig.suptitle("Blindness-aware agents and curriculum vs plain blindness training (100 iterations, 3 seeds)")
fig.tight_layout()
fig.savefig(os.path.join(here, "variants.png"), dpi=120)
print("wrote", os.path.join(here, "variants.png"))
