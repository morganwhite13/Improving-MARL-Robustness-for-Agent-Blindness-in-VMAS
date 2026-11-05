"""Plot Experiment 8: final reward by team size, with and without blindness."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
seeds = list(json.load(open(os.path.join(here, "exp8_agents_results.json")))["seeds"].values())
sizes = sorted({r["n_agents"] for r in seeds[0].values()})


def finals(blind):
    return np.array([[s[f"{n} agents, {'blind' if blind else 'no blindness'}"]["final"] for s in seeds] for n in sizes])


fig, ax = plt.subplots(figsize=(8, 5))
for blind, label, color in [(False, "No blindness", "#7aa6d6"), (True, "One agent blinded (p = 0.1, 1-10 steps)", "#e8a33d")]:
    f = finals(blind)
    ax.errorbar(sizes, f.mean(1), yerr=f.std(1, ddof=1), marker="o", capsize=3, label=label, color=color)
    for n, row in zip(sizes, f):
        print(f"{label:42s} n={n:2d}: {row.mean():6.1f} ± {row.std(ddof=1):4.1f}")
ax.set_xticks(sizes)
ax.set_xlabel("Number of agents")
ax.set_ylabel("Final episode reward (mean ± std)")
ax.set_title(f"Experiment 8: team size, with and without blindness ({len(seeds)} seeds, 25 iterations)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(here, "exp8_agents.png"), dpi=120)
print("wrote", os.path.join(here, "exp8_agents.png"))
