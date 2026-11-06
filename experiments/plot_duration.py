"""Merge and plot the Experiment 7 re-run (maximum blind duration at evaluation).

Usage:
  python experiments/plot_duration.py <dir of run_duration.py outputs>   # merge into duration_results.json, then plot
  python experiments/plot_duration.py                                    # plot duration_results.json
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "duration_results.json")
if len(sys.argv) > 1:
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(sys.argv[1], "dur_*.json")))]
    json.dump({
        "description": "Experiment 7 re-run with the fixed code: policies trained for 100 iterations (defaults otherwise) with no blindness or with one random agent blinded for 1-10 steps (probability 0.1), then evaluated with deterministic actions on fresh, separately seeded environments at blindness probability 0.1 and a range of maximum blind durations ('0' = no blindness). Mean episode reward over 3 rollouts x 30 environments x 200 steps.",
        "versions": json.load(open(os.path.join(here, "robustness_results.json")))["versions"],
        "runs": runs,
    }, open(path, "w"), indent=1)

runs = json.load(open(path))["runs"]
trains = sorted({r["train"] for r in runs}, reverse=True)
etypes = list(runs[0]["eval"])
durations = list(runs[0]["eval"][etypes[0]])
fig, axes = plt.subplots(1, len(etypes), figsize=(12, 4.6), sharey=True)
for ax, etype in zip(axes, etypes):
    for train in trains:
        scores = np.array([[r["eval"][etype][d] for d in durations] for r in runs if r["train"] == train])
        mean, std = scores.mean(0), scores.std(0, ddof=1)
        x = [int(d) for d in durations]
        ax.errorbar(x, mean, yerr=std, marker="o", capsize=3, label=f"Trained: {train.lower()}")
        print(f"{train:24s} | eval {etype:10s} | " + "  ".join(f"{d}:{m:5.1f}±{s:4.1f}" for d, m, s in zip(durations, mean, std)))
    ax.set_title(f"Evaluated with: {etype.lower()} blind (p = 0.1)")
    ax.set_xlabel("Maximum blind spell at evaluation (steps; 0 = no blindness)")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([int(d) for d in durations], durations)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Episode reward (deterministic policy)")
axes[0].legend(fontsize=8)
fig.suptitle(f"Experiment 7: blind spell length at evaluation (100 training iterations, mean ± std over {len(runs) // len(trains)} seeds)")
fig.tight_layout()
fig.savefig(os.path.join(here, "duration.png"), dpi=120)
print("wrote", os.path.join(here, "duration.png"))
