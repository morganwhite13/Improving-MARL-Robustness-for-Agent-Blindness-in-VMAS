"""One training run for the Experiment 3-5 re-runs (clip value, minibatch size, epochs).

Executes the training script up to "#Experiment 1" (config, environments and
train_environment_variables, unchanged), then trains env6 (one random agent
blinded for 1-10 steps with probability 0.1, the scenario the original sweeps
reported) with normalization on and the given PPO settings, for one seed.
Writes a small JSON file, so many runs can go in parallel.

Usage: python experiments/run_sweep.py . <seed> <out.json> <clip> <batch> <epochs>
"""
import json, os, sys, time
repo, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
clip, batch, epochs = float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
sys.path.insert(0, repo)
src = open(os.path.join(repo, "MARL Robustness for Agent Blindness in VMAS.py"), encoding="utf-8").read().replace("\r\n", "\n")
src = src[: src.index("#Experiment 1")]
ns = {"__name__": "sweep"}
exec(compile(src, "vmas_script", "exec"), ns)

import torch
torch.set_num_threads(1)
env = ns["env6"]
torch.manual_seed(seed)
env.set_seed(seed)
t = time.time()
rewards, _ = ns["train_environment_variables"](env, f"clip {clip}, batch {batch}, epochs {epochs}",
                                               norm=True, clipVal=clip, batchSize=batch, numEpochs=epochs)
rewards = [float(r) for r in rewards]
setup = {"clip": clip, "batch": batch, "epochs": epochs}
json.dump({"seed": seed, "setup": setup, "rewards": rewards, "final": rewards[-1], "minutes": (time.time() - t) / 60}, open(out, "w"), indent=1)
print(f"DONE seed={seed} {setup}: final={rewards[-1]:.1f} ({(time.time() - t) / 60:.1f} min)", flush=True)
