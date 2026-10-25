"""Plot experiments/tuning_results.json: final reward per hyperparameter setup across seeds."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(here, "tuning_results.json")))
seeds = list(data["seeds"].values())
names = list(seeds[0])

finals = np.array([[s[n]["final"] for s in seeds] for n in names])  # [setups, seeds]
fig, ax = plt.subplots(figsize=(9, 5))
y = np.arange(len(names))[::-1]
ax.barh(y, finals.mean(1), xerr=finals.std(1, ddof=1), color="#7aa6d6", capsize=4)
for i, row in zip(y, finals):
    ax.scatter(row, np.full_like(row, i), color="black", s=14, zorder=3)
    print(f"{names[len(names) - 1 - i]:22s} {row.mean():6.1f} ± {row.std(ddof=1):4.1f}  seeds: {', '.join(f'{v:.1f}' for v in row)}")
ax.set_yticks(y, names)
ax.set_xlabel("Final episode reward (bar: mean ± std, dots: seeds)")
ax.set_title(f"Hyperparameter setups on one agent blinded 1-10 steps ({len(seeds)} seeds)")
fig.tight_layout()
fig.savefig(os.path.join(here, "tuning_rewards.png"), dpi=120)
print("wrote", os.path.join(here, "tuning_rewards.png"))
