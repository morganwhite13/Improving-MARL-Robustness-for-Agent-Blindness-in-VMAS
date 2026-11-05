"""Training variants beyond the original experiments, one run per call.

Variants (all 100 iterations, the original PPO settings unless stated):

  aware-flag     one random agent blinded for 1-10 steps (probability 0.1),
                 and each agent's observation carries a "blind" flag
  aware-memory   as aware-flag, and a blind agent also keeps its last seen observation
  aware-memory-each
                 as aware-memory, but each agent independently blinded for 1-10
                 steps (probability 0.1), like env7
  curriculum     the same blindness, but its probability rises from 0 to 0.1
                 over the first 50 iterations, then stays at 0.1
  stuck:<scenario>:<setting>
                 no blindness, on a scenario that didn't learn with the
                 original settings; <setting> is "long-horizon" (gamma 0.99,
                 GAE lambda 0.95) or "long-horizon-explore" (also entropy 0.01)

Blindness variants are then evaluated exactly as in the robustness test
(run_robustness.py): deterministic actions on fresh environments seeded
1000 + seed, 3 rollouts, at blindness probabilities 0 to 0.5 for both kinds
of blindness, so they compare directly with robustness_100iters_results.json.

Usage: python experiments/run_variant.py <variant> <seed> <out.json> [n_iters]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402

from marl_blindness import ORIGINAL, evaluate, make_env, train  # noqa: E402

EVAL_TYPES = {"One agent, 1-10 steps": "one-steps", "Each agent, 1-10 steps": "each-steps"}
EVAL_PROBS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
STUCK_SETTINGS = {
    "long-horizon": dict(gamma=0.99, lmbda=0.95),
    "long-horizon-explore": dict(gamma=0.99, lmbda=0.95, entropy_eps=0.01),
}

variant, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
torch.set_num_threads(1)
cfg = ORIGINAL.with_(n_iters=int(sys.argv[4]) if len(sys.argv) > 4 else 100)
aware, progress, scenario = "none", None, "balance"

if variant.startswith("stuck:"):
    _, scenario, setting = variant.split(":")
    cfg = cfg.with_(**STUCK_SETTINGS[setting])
    env = make_env(scenario=scenario)
elif variant in ("aware-flag", "aware-memory", "aware-memory-each"):
    aware = variant.split("-")[1]
    env = make_env(blindness="each-steps" if variant.endswith("-each") else "one-steps", aware=aware)
elif variant == "curriculum":
    env = make_env(blindness="one-steps", blind_prob=0.0)
    blindness = env.transform  # the outermost transform is the blindness

    def progress(i, reward):
        blindness._blind_prob = 0.1 * min(1.0, (i + 1) / 50)  # for the next iteration's data
else:
    raise SystemExit(f"unknown variant {variant}")

torch.manual_seed(seed)
env.set_seed(seed)
start = time.time()
result = train(env, cfg, progress=progress)
record = {"variant": variant, "seed": seed, "settings": cfg.to_dict(), "train_rewards": result.rewards,
          "train_minutes": (time.time() - start) / 60}
if not variant.startswith("stuck:"):
    record["eval"] = {
        label: {
            str(p): evaluate(result.policy, lambda kind=kind, p=p: make_env(
                blindness=kind if p > 0 else "none", blind_prob=p, aware=aware), seed=1000 + seed)
            for p in EVAL_PROBS
        }
        for label, kind in EVAL_TYPES.items()
    }
json.dump(record, open(out, "w"), indent=1)
print(f"DONE {variant} seed={seed}: final train reward {result.rewards[-1]:.2f} "
      f"({record['train_minutes']:.1f} min)", flush=True)
