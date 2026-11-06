"""One training run for the VMAS scenarios re-run.

Executes the training script up to "#Experiment 1" (config and
train_environment_variables, unchanged), then trains one VMAS scenario with
the script's default settings, with no blindness or with one random agent
blinded for 1-10 steps (probability 0.1, the setting the original experiment
used), for one seed. Reward scales differ between scenarios, so the
no-blindness run is the control each blindness run is compared with.
Some scenarios fix their own number of agents and ignore n_agents.

Usage: python experiments/run_scenarios.py . <seed> <out.json> <scenario> <none|one> [n_iters]
"""
import json
import sys
import time
import warnings

from common import load_training_script, make_env

repo, seed, out, scenario, cond = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
n_iters = int(sys.argv[6]) if len(sys.argv) > 6 else None
ns = load_training_script(repo, "scenarios", n_iters)

import torch

from blindness import BlindOneRandomAgentIfProbabilityForJSteps

torch.set_num_threads(1)
warnings.filterwarnings("ignore", message="Scenario kwargs")
blindness = (lambda n: BlindOneRandomAgentIfProbabilityForJSteps(n, 0.1, 10)) if cond == "one" else None
env = make_env(ns, scenario=scenario, blindness=blindness)
torch.manual_seed(seed)
env.set_seed(seed)
t = time.time()
rewards, _ = ns["train_environment_variables"](env, f"{scenario}, {cond}")
rewards = [float(r) for r in rewards]
json.dump({"seed": seed, "n_iters": ns["n_iters"], "scenario": scenario, "n_agents": env.n_agents, "blind": cond == "one",
           "rewards": rewards, "final": rewards[-1], "minutes": (time.time() - t) / 60}, open(out, "w"), indent=1)
print(f"DONE seed={seed} {scenario} {cond}: final={rewards[-1]:.2f} ({(time.time() - t) / 60:.1f} min)", flush=True)
