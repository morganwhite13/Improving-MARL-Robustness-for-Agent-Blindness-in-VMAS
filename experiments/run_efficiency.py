"""Which settings reach a good reward soonest, in wall-clock time on one CPU core?

Trains one configuration for one seed on env6's setting (one random agent
blinded for 1-10 steps, probability 0.1) with the marl_blindness package, and
records the time after every iteration, so configurations that do different
amounts of work per iteration can be compared fairly.

Usage: python experiments/run_efficiency.py <config> <seed> <out.json>
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402

from marl_blindness import ORIGINAL, TUNED, make_env, train  # noqa: E402

CONFIGS = {
    "original": ORIGINAL.with_(n_iters=60),
    "tuned": TUNED.with_(n_iters=40),
    "tuned, lr 1e-3": TUNED.with_(n_iters=40, lr=1e-3),
    "tuned, half batches": TUNED.with_(n_iters=80, frames_per_batch=3_000),
}

name, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = CONFIGS[name]
torch.set_num_threads(1)
env = make_env(blindness="one-steps", num_envs=cfg.num_envs, max_steps=cfg.max_steps)
torch.manual_seed(seed)
env.set_seed(seed)
start = time.time()
seconds = []
result = train(env, cfg, progress=lambda i, r: seconds.append(time.time() - start))
json.dump({"config": name, "settings": cfg.to_dict(), "seed": seed, "rewards": result.rewards, "seconds": seconds},
          open(out, "w"), indent=1)
print(f"DONE {name} seed={seed}: final {result.rewards[-1]:.1f} after {seconds[-1] / 60:.1f} min", flush=True)
