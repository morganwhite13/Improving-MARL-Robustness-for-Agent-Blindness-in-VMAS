"""Plot experiments/exp1_results.json: mean episode reward per scenario across seeds."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(here, "exp1_results.json")))
seeds = list(data["seeds"].values())

fig, ax = plt.subplots(figsize=(10, 6))
for desc in seeds[0]:
    curves = np.array([s[desc]["rewards"] for s in seeds])
    mean, std = curves.mean(0), curves.std(0)
    x = np.arange(1, len(mean) + 1)
    ax.plot(x, mean, label=desc)
    ax.fill_between(x, mean - std, mean + std, alpha=0.15)
ax.set_xlabel("Training iteration")
ax.set_ylabel("Episode reward (mean over 30 envs)")
ax.set_title(f"Reward by blindness scenario (MAPPO, mean ± std over {len(seeds)} seeds)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(here, "exp1_rewards.png"), dpi=120)
print("wrote", os.path.join(here, "exp1_rewards.png"))
