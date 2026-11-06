"""Merge and plot the Experiment 3-5 re-runs (clip value, minibatch size, epochs).

Usage:
  python experiments/plot_sweep.py <dir of run_sweep.py outputs>   # merge into sweep_results.json, then plot
  python experiments/plot_sweep.py                                 # plot sweep_results.json

The default setting (clip 0.2, batch 200, 20 epochs) is shared by all three
sweeps. It is the same training run as the robustness test's 25-iteration
"One agent, 1-10 steps" policy (runs are deterministic), so it is taken from
robustness_results.json.
"""
import glob
import json
import os
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "sweep_results.json")
DEFAULT = {"clip": 0.2, "batch": 200, "epochs": 20}

if len(sys.argv) > 1:
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(sys.argv[1], "sweep_*.json")))]
    robust = json.load(open(os.path.join(here, "robustness_results.json")))
    for seed, res in robust["seeds"].items():
        rewards = res["One agent, 1-10 steps"]["train_rewards"]
        runs.append({"seed": int(seed), "setup": DEFAULT, "rewards": rewards, "final": rewards[-1], "source": "robustness_results.json"})
    json.dump({
        "description": "Experiments 3-5 regenerated with the fixed code: final episode_reward_mean on env6 (one random agent blinded for 1-10 steps, probability 0.1) with normalization on, varying one PPO setting at a time from clip 0.2, batch 200, 20 epochs. 25 iterations x 6,000 frames.",
        "versions": robust["versions"],
        "runs": runs,
    }, open(path, "w"), indent=1)

runs = json.load(open(path))["runs"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, (key, label) in zip(axes, [("clip", "PPO clip value"), ("batch", "Minibatch size"), ("epochs", "Epochs per iteration")]):
    others = [k for k in DEFAULT if k != key]
    values = sorted({r["setup"][key] for r in runs if all(r["setup"][o] == DEFAULT[o] for o in others)})
    means, stds = [], []
    for v in values:
        f = np.array([r["final"] for r in runs if r["setup"][key] == v and all(r["setup"][o] == DEFAULT[o] for o in others)])
        means.append(f.mean())
        stds.append(f.std(ddof=1))
        ax.scatter([v] * len(f), f, color="black", s=10, zorder=3)
        print(f"{key:6s} {v:>6}: {f.mean():6.1f} ± {f.std(ddof=1):4.1f}  (n={len(f)})")
    ax.errorbar(values, means, yerr=stds, marker="o", capsize=3, color="#7aa6d6")
    ax.axvline(DEFAULT[key], color="grey", lw=0.8, ls="--")
    ax.set_xscale("log" if key == "batch" else "linear")
    ax.set_xticks(values, [str(v) for v in values])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(label + " (dashed: default)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Final episode reward (mean ± std, dots: seeds)")
fig.suptitle("Experiments 3-5: one PPO setting at a time (one agent blinded 1-10 steps, 3 seeds, 25 iterations)")
fig.tight_layout()
fig.savefig(os.path.join(here, "sweep.png"), dpi=120)
print("wrote", os.path.join(here, "sweep.png"))
