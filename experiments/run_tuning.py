"""Re-run the hyperparameter comparison (the script's last experiment) with the fixed code.

Executes the training script up to "#Experiment 1" (config, environments and
train_environment_variables, unchanged), then trains each setup below on env6
(one random agent blinded for 1-10 steps with probability 0.1) for one seed.
Usage: python experiments/run_tuning.py . <seed> experiments/tuning_seed<seed>.json
"""
import json, os, sys, time
repo, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, repo)
src = open(os.path.join(repo, "MARL Robustness for Agent Blindness in VMAS.py"), encoding="utf-8").read().replace("\r\n", "\n")
src = src[: src.index("#Experiment 1")]
ns = {"__name__": "tuning"}
exec(compile(src, "vmas_script", "exec"), ns)

SETUPS = {
    "Original": dict(norm=False, clipVal=0.2, batchSize=200, numEpochs=20),
    "Best": dict(norm=True, clipVal=0.3, batchSize=100, numEpochs=30),
    "Best, but clip 0.2": dict(norm=True, clipVal=0.2, batchSize=100, numEpochs=30),
    "Best, but batch 200": dict(norm=True, clipVal=0.3, batchSize=200, numEpochs=30),
    "Best, but 20 epochs": dict(norm=True, clipVal=0.3, batchSize=100, numEpochs=20),
}

import torch
torch.set_num_threads(1)
env = ns["env6"]
results = json.load(open(out)) if os.path.exists(out) else {}
for name, setup in SETUPS.items():
    if name in results:
        continue
    torch.manual_seed(seed)
    env.set_seed(seed)
    t = time.time()
    rewards, _ = ns["train_environment_variables"](env, name, **setup)
    rewards = [float(r) for r in rewards]
    results[name] = {"setup": setup, "rewards": rewards, "final": rewards[-1], "minutes": (time.time() - t) / 60}
    json.dump(results, open(out, "w"), indent=1)
    print(f"DONE seed={seed} {name}: final={rewards[-1]:.1f} ({results[name]['minutes']:.1f} min)", flush=True)
print("ALL DONE", flush=True)
