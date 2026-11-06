"""Does training with blindness make the team more robust to blindness?

Executes the training script up to "#Experiment 1" (config, environments and
train_environment_variables, unchanged), then for one seed:

1. Trains three policies with the script's default PPO settings:
   with no blindness, with one random agent blinded for 1-10 steps
   (probability 0.1), and with each agent independently blinded for 1-10 steps
   (probability 0.1).
2. Evaluates every policy with deterministic actions on fresh environments
   (separately seeded, not the training rollouts) at a range of blindness
   probabilities, for both blindness types.

Usage: python experiments/run_robustness.py . <seed> experiments/robustness_seed<seed>.json [n_iters]
n_iters overrides the script's 25 training iterations (e.g. 100 for a longer run).
"""
import json
import os
import sys
import time

from common import evaluate, load_training_script, make_env

repo, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
n_iters = int(sys.argv[4]) if len(sys.argv) > 4 else None
ns = load_training_script(repo, "robustness", n_iters)

import torch

from blindness import BlindOneRandomAgentIfProbabilityForJSteps, BlindRandomAgentsIfProbabilityForJSteps

torch.set_num_threads(1)
TRAIN = {
    "No blindness": ns["env"],
    "One agent, 1-10 steps": ns["env6"],
    "Each agent, 1-10 steps": ns["env7"],
}
EVAL_TYPES = {
    "One agent, 1-10 steps": BlindOneRandomAgentIfProbabilityForJSteps,
    "Each agent, 1-10 steps": BlindRandomAgentsIfProbabilityForJSteps,
}
EVAL_PROBS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
EVAL_ROLLOUTS = 3  # x 30 parallel environments x 200 steps each


def eval_env(transform_cls, prob):
    """Builder for a fresh evaluation environment with blind spells of up to 10 steps at probability prob."""
    return lambda: make_env(ns, blindness=(lambda n: transform_cls(n, prob, 10)) if prob > 0 else None)


results = json.load(open(out)) if os.path.exists(out) else {}
for name, env in TRAIN.items():
    if name in results:
        continue
    torch.manual_seed(seed)
    env.set_seed(seed)
    t = time.time()
    rewards, policy = ns["train_environment_variables"](env, name)
    evals = {
        etype: {str(p): evaluate(ns, policy, eval_env(cls, p), 1000 + seed, EVAL_ROLLOUTS) for p in EVAL_PROBS}
        for etype, cls in EVAL_TYPES.items()
    }
    results[name] = {"train_rewards": [float(r) for r in rewards], "eval": evals, "minutes": (time.time() - t) / 60}
    json.dump(results, open(out, "w"), indent=1)
    print(f"DONE seed={seed} {name}: {evals} ({results[name]['minutes']:.1f} min)", flush=True)
print("ALL DONE", flush=True)
