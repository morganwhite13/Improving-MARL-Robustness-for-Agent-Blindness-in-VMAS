"""One training run for the VMAS scenarios re-run.

Executes the training script up to "#Experiment 1" (config and
train_environment_variables, unchanged), then trains one VMAS scenario with
the script's default settings, with no blindness or with one random agent
blinded for 1-10 steps (probability 0.1, the setting the original experiment
used), for one seed. Reward scales differ between scenarios, so the
no-blindness run is the control each blindness run is compared with.
Some scenarios fix their own number of agents and ignore n_agents.

Usage: python experiments/run_scenarios.py . <seed> <out.json> <scenario> <none|one>
"""
import json, os, sys, time, warnings
repo, seed, out, scenario, cond = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
sys.path.insert(0, repo)
src = open(os.path.join(repo, "MARL Robustness for Agent Blindness in VMAS.py"), encoding="utf-8").read().replace("\r\n", "\n")
src = src[: src.index("#Experiment 1")]
ns = {"__name__": "scenarios"}
exec(compile(src, "vmas_script", "exec"), ns)

import torch
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from blindness import BlindOneRandomAgentIfProbabilityForJSteps

torch.set_num_threads(1)
warnings.filterwarnings("ignore", message="Scenario kwargs")
base = VmasEnv(scenario=scenario, num_envs=ns["num_vmas_envs"], continuous_actions=True,
               max_steps=ns["max_steps"], device=ns["vmas_device"], n_agents=ns["n_agents"])
env = TransformedEnv(base, RewardSum(in_keys=[base.reward_key], out_keys=[("agents", "episode_reward")]))
if cond == "one":
    env = TransformedEnv(env, BlindOneRandomAgentIfProbabilityForJSteps(base.n_agents, 0.1, 10))
torch.manual_seed(seed)
env.set_seed(seed)
t = time.time()
rewards, _ = ns["train_environment_variables"](env, f"{scenario}, {cond}")
rewards = [float(r) for r in rewards]
json.dump({"seed": seed, "scenario": scenario, "n_agents": base.n_agents, "blind": cond == "one",
           "rewards": rewards, "final": rewards[-1], "minutes": (time.time() - t) / 60}, open(out, "w"), indent=1)
print(f"DONE seed={seed} {scenario} {cond}: final={rewards[-1]:.2f} ({(time.time() - t) / 60:.1f} min)", flush=True)
