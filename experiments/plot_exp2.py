"""Plot Experiment 2: final reward per scenario with advantage normalization off vs on.

Normalization off comes from exp1_results.json, on from exp2_norm_results.json
(same seeds and settings otherwise).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
off = json.load(open(os.path.join(here, "exp1_results.json")))["seeds"]
on = json.load(open(os.path.join(here, "exp2_norm_results.json")))["seeds"]
seeds = sorted(off)
scenarios = list(off[seeds[0]])

fig, ax = plt.subplots(figsize=(10, 5.5))
y = np.arange(len(scenarios))[::-1]
for i, (label, data, color) in enumerate([("Normalization off", off, "#7aa6d6"), ("Normalization on", on, "#e8a33d")]):
    finals = np.array([[data[s][sc]["final"] for s in seeds] for sc in scenarios])
    offset = 0.2 if i == 0 else -0.2
    ax.barh(y + offset, finals.mean(1), height=0.38, xerr=finals.std(1, ddof=1), capsize=3, color=color, label=label)
    for row, yy in zip(finals, y + offset):
        ax.scatter(row, np.full_like(row, yy), color="black", s=10, zorder=3)
    for sc, row in zip(scenarios, finals):
        print(f"{label:18s} {sc:62s} {row.mean():6.1f} ± {row.std(ddof=1):4.1f}")
ax.set_yticks(y, scenarios, fontsize=8)
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel("Final episode reward (bar: mean ± std, dots: seeds)")
ax.set_title(f"Experiment 2: advantage normalization off vs on ({len(seeds)} seeds, 25 iterations)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(here, "exp2_norm.png"), dpi=120)
print("wrote", os.path.join(here, "exp2_norm.png"))
