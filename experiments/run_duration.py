"""Experiment 7 re-run: how does the length of blind spells affect a trained team?

The original experiment retrained a policy for every maximum blind duration
and scored it on its own training rollouts. This trains one policy (with no
blindness, or with one random agent blinded for 1-10 steps at probability
0.1), saves it, then evaluates it with deterministic actions on fresh,
separately seeded environments at a range of maximum blind durations, with
blindness probability 0.1, for both kinds of blindness.

Usage: python experiments/run_duration.py . <seed> <out.json> <none|one> [n_iters] [policy.pt]
"""
import json
import sys
import time

from common import evaluate, load_training_script, make_env

repo, seed, out, cond = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
n_iters = int(sys.argv[5]) if len(sys.argv) > 5 else None
policy_path = sys.argv[6] if len(sys.argv) > 6 else None
ns = load_training_script(repo, "duration", n_iters)

import torch

from blindness import BlindOneRandomAgentIfProbabilityForJSteps, BlindRandomAgentsIfProbabilityForJSteps

torch.set_num_threads(1)
TRAIN = {"none": ("No blindness", ns["env"]), "one": ("One agent, 1-10 steps", ns["env6"])}
EVAL_TYPES = {
    "One agent": BlindOneRandomAgentIfProbabilityForJSteps,
    "Each agent": BlindRandomAgentsIfProbabilityForJSteps,
}
DURATIONS = [1, 2, 3, 5, 10, 20]
EVAL_PROB = 0.1
EVAL_ROLLOUTS = 3


def eval_env(transform_cls, max_steps_blind):
    """Builder for a fresh evaluation environment with blind spells of up to max_steps_blind steps (0 = none)."""
    blindness = (lambda n: transform_cls(n, EVAL_PROB, max_steps_blind)) if max_steps_blind > 0 else None
    return lambda: make_env(ns, blindness=blindness)


name, env = TRAIN[cond]
torch.manual_seed(seed)
env.set_seed(seed)
t = time.time()
rewards, policy = ns["train_environment_variables"](env, name)
if policy_path:
    torch.save(policy.state_dict(), policy_path)
evals = {
    etype: {str(d): evaluate(ns, policy, eval_env(cls, d), 1000 + seed, EVAL_ROLLOUTS) for d in [0] + DURATIONS}
    for etype, cls in EVAL_TYPES.items()
}
json.dump({"seed": seed, "train": name, "train_rewards": [float(r) for r in rewards], "eval_prob": EVAL_PROB,
           "eval": evals, "minutes": (time.time() - t) / 60}, open(out, "w"), indent=1)
print(f"DONE seed={seed} {name}: {evals} ({(time.time() - t) / 60:.1f} min)", flush=True)
