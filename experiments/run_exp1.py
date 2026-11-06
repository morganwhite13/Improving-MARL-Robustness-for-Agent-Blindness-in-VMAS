"""Regenerate Experiment 1 (reward per blindness scenario) with the fixed code.

Executes the training script up to "#Experiment 1" (config, environments and
train_environment_variables, unchanged), then trains every scenario with the
experiment's own settings (norm=False, defaults otherwise) for one seed.
Usage: python experiments/run_exp1.py . <seed> experiments/seed<seed>.json [norm]
Pass "norm" to train with advantage normalization instead, the other half of
Experiment 2 (normalization on vs off, across the same scenarios).
"""
import json
import os
import sys
import time

from common import load_training_script

repo, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
norm = len(sys.argv) > 4 and sys.argv[4] == "norm"
ns = load_training_script(repo, "exp1")

import torch

torch.set_num_threads(1)
results = json.load(open(out)) if os.path.exists(out) else {}
for env, desc in ns["environments"]:
    if desc in results:
        continue
    torch.manual_seed(seed)
    env.set_seed(seed)
    t = time.time()
    rewards, _ = ns["train_environment_variables"](env, desc, norm=norm)
    rewards = [float(r) for r in rewards]
    results[desc] = {"rewards": rewards, "final": rewards[-1], "minutes": (time.time() - t) / 60}
    json.dump(results, open(out, "w"), indent=1)
    print(f"DONE seed={seed} {desc}: final={rewards[-1]:.1f} ({results[desc]['minutes']:.1f} min)", flush=True)
print("ALL DONE", flush=True)
