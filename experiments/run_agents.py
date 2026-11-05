"""Re-run Experiment 8 (number of agents) with a no-blindness control.

Executes the training script up to "#Experiment 1" (config and
train_environment_variables, unchanged), then for one seed trains each team
size twice with the script's default settings: with no blindness, and with
one random agent blinded for 1-10 steps (probability 0.1), the setting the
original experiment used. More agents also make the balance task itself
easier, so raw reward can't show whether a bigger team copes better with
blindness; the share of the no-blindness reward kept can.

Usage: python experiments/run_agents.py . <seed> experiments/agents_seed<seed>.json [n_iters]
"""
import json, os, sys, time
from common import load_training_script, make_env
repo, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
n_iters = int(sys.argv[4]) if len(sys.argv) > 4 else None
ns = load_training_script(repo, "agents", n_iters)

import torch
from blindness import BlindOneRandomAgentIfProbabilityForJSteps

torch.set_num_threads(1)
TEAM_SIZES = [2, 3, 5, 7, 10]


def one_agent_blindness(n_agents):
    return BlindOneRandomAgentIfProbabilityForJSteps(n_agents, 0.1, 10)


results = json.load(open(out)) if os.path.exists(out) else {}
for n in TEAM_SIZES:
    for blind in (False, True):
        key = f"{n} agents, {'blind' if blind else 'no blindness'}"
        if key in results:
            continue
        torch.manual_seed(seed)
        env = make_env(ns, n_agents=n, blindness=one_agent_blindness if blind else None)
        env.set_seed(seed)
        t = time.time()
        rewards, _ = ns["train_environment_variables"](env, key)
        rewards = [float(r) for r in rewards]
        results[key] = {"n_agents": n, "blind": blind, "rewards": rewards, "final": rewards[-1], "minutes": (time.time() - t) / 60}
        json.dump(results, open(out, "w"), indent=1)
        print(f"DONE seed={seed} {key}: final={rewards[-1]:.1f} ({results[key]['minutes']:.1f} min)", flush=True)
print("ALL DONE", flush=True)
