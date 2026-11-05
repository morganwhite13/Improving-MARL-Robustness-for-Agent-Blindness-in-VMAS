"""Merge and plot the VMAS scenarios re-run.

Usage:
  python experiments/plot_scenarios.py [--100] <dir of run_scenarios.py outputs>   # merge, then plot
  python experiments/plot_scenarios.py [--100]                                     # plot the merged results

Without --100 this uses the 25-iteration runs (scenarios_results.json); with
--100, the 100-iteration runs (scenarios_100iters_results.json). Balance
comes from the robustness test (robustness_results.json or
robustness_100iters_results.json): the same training runs, since runs are
deterministic.
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
args = sys.argv[1:]
long_run = "--100" in args
args = [a for a in args if a != "--100"]
suffix = "_100iters" if long_run else ""
iters = 100 if long_run else 25
path = os.path.join(here, f"scenarios{suffix}_results.json")
if args:
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args[0], "*_s[0-9].json"))) if "check_" not in f]
    robust = json.load(open(os.path.join(here, f"robustness{suffix}_results.json")))
    for seed, res in robust["seeds"].items():
        for key, blind in [("No blindness", False), ("One agent, 1-10 steps", True)]:
            rewards = res[key]["train_rewards"]
            runs.append({"seed": int(seed), "scenario": "balance", "n_agents": 3, "blind": blind, "rewards": rewards,
                         "final": rewards[-1], "source": "robustness_results.json"})
    json.dump({
        "description": f"VMAS scenarios re-run with the fixed code: each scenario trained with the training function's defaults (norm=True, clip 0.2, batch 200, 20 epochs, {iters} iterations x 6,000 frames), with no blindness and with one random agent blinded for 1-10 steps (probability 0.1). Final episode_reward_mean. give_way and ball_passage always have 2 agents and multi_give_way 4; they ignore n_agents.",
        "versions": robust["versions"],
        "runs": runs,
    }, open(path, "w"), indent=1)

runs = json.load(open(path))["runs"]
scenarios = ["balance", "wheel", "give_way", "ball_passage", "multi_give_way"]
fig, axes = plt.subplots(1, len(scenarios), figsize=(15, 4))
for ax, sc in zip(axes, scenarios):
    for i, (blind, label, color) in enumerate([(False, "No blindness", "#7aa6d6"), (True, "One agent blinded", "#e8a33d")]):
        f = np.array([r["final"] for r in runs if r["scenario"] == sc and r["blind"] == blind])
        ax.bar(i, f.mean(), yerr=f.std(ddof=1), capsize=4, color=color, label=label)
        ax.scatter([i] * len(f), f, color="black", s=10, zorder=3)
        n = next(r["n_agents"] for r in runs if r["scenario"] == sc)
        print(f"{sc:15s} ({n} agents) {label:18s} {f.mean():8.2f} ± {f.std(ddof=1):6.2f}  seeds {[round(x, 2) for x in f]}")
    ax.set_title(f"{sc} ({n} agents)")
    ax.set_xticks([0, 1], ["none", "blind"])
    ax.axhline(0, color="grey", lw=0.8)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("Final episode reward (mean ± std, dots: seeds)")
axes[0].legend(fontsize=8)
fig.suptitle(f"VMAS scenarios: no blindness vs one agent blinded 1-10 steps (3 seeds, {iters} iterations; reward scales differ by scenario)")
fig.tight_layout()
fig.savefig(os.path.join(here, f"scenarios{suffix}.png"), dpi=120)
print("wrote", os.path.join(here, f"scenarios{suffix}.png"))
