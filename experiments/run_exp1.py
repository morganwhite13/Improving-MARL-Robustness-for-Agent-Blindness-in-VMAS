"""Regenerate Experiment 1 (reward per blindness scenario) with the fixed code.

Executes the training script up to "#Experiment 1" (config, environments and
train_environment_variables, unchanged), then trains every scenario with the
experiment's own settings (norm=False, defaults otherwise) for one seed.
Usage: python experiments/run_exp1.py . <seed> experiments/seed<seed>.json
"""
import json, os, sys, time
repo, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, repo)
src = open(os.path.join(repo, "MARL Robustness for Agent Blindness in VMAS.py"), encoding="utf-8").read().replace("\r\n", "\n")
src = src[: src.index("#Experiment 1")]
ns = {"__name__": "exp1"}
exec(compile(src, "vmas_script", "exec"), ns)

import torch
torch.set_num_threads(1)
results = json.load(open(out)) if os.path.exists(out) else {}
for env, desc in ns["environments"]:
    if desc in results:
        continue
    torch.manual_seed(seed)
    env.set_seed(seed)
    t = time.time()
    rewards, _ = ns["train_environment_variables"](env, desc, norm=False)
    rewards = [float(r) for r in rewards]
    results[desc] = {"rewards": rewards, "final": rewards[-1], "minutes": (time.time() - t) / 60}
    json.dump(results, open(out, "w"), indent=1)
    print(f"DONE seed={seed} {desc}: final={rewards[-1]:.1f} ({results[desc]['minutes']:.1f} min)", flush=True)
print("ALL DONE", flush=True)
