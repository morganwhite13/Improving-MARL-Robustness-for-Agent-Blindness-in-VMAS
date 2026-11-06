"""Plot experiments/efficiency_results.json: reward against wall-clock minutes per configuration."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
runs = json.load(open(os.path.join(here, "efficiency_results.json")))["runs"]
configs = list(dict.fromkeys(r["config"] for r in sorted(runs, key=lambda r: r["config"])))
grid = np.linspace(0.5, 7, 27)  # minutes, where every configuration has data

fig, ax = plt.subplots(figsize=(8, 5))
for name in configs:
    curves = []
    for r in (r for r in runs if r["config"] == name):
        minutes = np.array(r["seconds"]) / 60
        idx = np.searchsorted(minutes, grid, side="right") - 1  # latest finished iteration at each time
        curves.append(np.where(idx >= 0, np.array(r["rewards"])[np.clip(idx, 0, None)], np.nan))
    curves = np.array(curves)
    mean, std = np.nanmean(curves, 0), np.nanstd(curves, 0, ddof=1)
    ax.plot(grid, mean, label=name)
    ax.fill_between(grid, mean - std, mean + std, alpha=0.15)
    print(f"{name:22s} reward at 3 / 5 / 7 min: " + " / ".join(f"{np.nanmean(curves[:, np.argmin(abs(grid - t))]):.1f}" for t in (3, 5, 7)))
ax.set_xlabel("Wall-clock minutes on one CPU core")
ax.set_ylabel("Training reward (mean ± std over 3 seeds)")
ax.set_title("Reward per minute of training, one agent blinded 1-10 steps")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(here, "efficiency.png"), dpi=120)
print("wrote", os.path.join(here, "efficiency.png"))
