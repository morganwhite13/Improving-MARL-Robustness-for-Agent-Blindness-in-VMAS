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
repo, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
n_iters = int(sys.argv[4]) if len(sys.argv) > 4 else None
sys.path.insert(0, repo)
src = open(os.path.join(repo, "MARL Robustness for Agent Blindness in VMAS.py"), encoding="utf-8").read().replace("\r\n", "\n")
src = src[: src.index("#Experiment 1")]
if n_iters is not None:
    assert src.count("\nn_iters = 25 ") == 1
    src = src.replace("\nn_iters = 25 ", f"\nn_iters = {n_iters} ")
ns = {"__name__": "agents"}
exec(compile(src, "vmas_script", "exec"), ns)

import torch
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from blindness import BlindOneRandomAgentIfProbabilityForJSteps

torch.set_num_threads(1)
TEAM_SIZES = [2, 3, 5, 7, 10]


def make_env(n_agents, blind):
    base = VmasEnv(
        scenario=ns["scenario_name"], num_envs=ns["num_vmas_envs"], continuous_actions=True,
        max_steps=ns["max_steps"], device=ns["vmas_device"], n_agents=n_agents,
    )
    env = TransformedEnv(base, RewardSum(in_keys=[base.reward_key], out_keys=[("agents", "episode_reward")]))
    if blind:
        env = TransformedEnv(env, BlindOneRandomAgentIfProbabilityForJSteps(base.n_agents, 0.1, 10))
    return env


results = json.load(open(out)) if os.path.exists(out) else {}
for n in TEAM_SIZES:
    for blind in (False, True):
        key = f"{n} agents, {'blind' if blind else 'no blindness'}"
        if key in results:
            continue
        torch.manual_seed(seed)
        env = make_env(n, blind)
        env.set_seed(seed)
        t = time.time()
        rewards, _ = ns["train_environment_variables"](env, key)
        rewards = [float(r) for r in rewards]
        results[key] = {"n_agents": n, "blind": blind, "rewards": rewards, "final": rewards[-1], "minutes": (time.time() - t) / 60}
        json.dump(results, open(out, "w"), indent=1)
        print(f"DONE seed={seed} {key}: final={rewards[-1]:.1f} ({results[key]['minutes']:.1f} min)", flush=True)
print("ALL DONE", flush=True)
