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

Usage: python experiments/run_robustness.py . <seed> experiments/robustness_seed<seed>.json
"""
import json, os, sys, time
repo, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, repo)
src = open(os.path.join(repo, "MARL Robustness for Agent Blindness in VMAS.py"), encoding="utf-8").read().replace("\r\n", "\n")
src = src[: src.index("#Experiment 1")]
ns = {"__name__": "robustness"}
exec(compile(src, "vmas_script", "exec"), ns)

import torch
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
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
    base = VmasEnv(
        scenario=ns["scenario_name"], num_envs=ns["num_vmas_envs"], continuous_actions=True,
        max_steps=ns["max_steps"], device=ns["vmas_device"], n_agents=ns["n_agents"],
    )
    env = TransformedEnv(base, RewardSum(in_keys=[base.reward_key], out_keys=[("agents", "episode_reward")]))
    if prob > 0:
        env = TransformedEnv(env, transform_cls(base.n_agents, prob, 10))
    return env


def evaluate(policy, transform_cls, prob, eval_seed):
    """Mean episode reward over EVAL_ROLLOUTS full episodes in every parallel environment."""
    torch.manual_seed(eval_seed)
    env = eval_env(transform_cls, prob)
    env.set_seed(eval_seed)
    scores = []
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(EVAL_ROLLOUTS):
            td = env.rollout(ns["max_steps"], policy=policy, break_when_any_done=False)
            # Average over every episode that ended (some end early), as the training metric does
            ep_reward = td.get(("next", "agents", "episode_reward"))
            done = td.get(("next", "done")).unsqueeze(-1).expand(ep_reward.shape)
            scores.append(ep_reward[done].mean().item())
    return sum(scores) / len(scores)


results = json.load(open(out)) if os.path.exists(out) else {}
for name, env in TRAIN.items():
    if name in results:
        continue
    torch.manual_seed(seed)
    env.set_seed(seed)
    t = time.time()
    rewards, policy = ns["train_environment_variables"](env, name)
    evals = {
        etype: {str(p): evaluate(policy, cls, p, eval_seed=1000 + seed) for p in EVAL_PROBS}
        for etype, cls in EVAL_TYPES.items()
    }
    results[name] = {"train_rewards": [float(r) for r in rewards], "eval": evals, "minutes": (time.time() - t) / 60}
    json.dump(results, open(out, "w"), indent=1)
    print(f"DONE seed={seed} {name}: {evals} ({results[name]['minutes']:.1f} min)", flush=True)
print("ALL DONE", flush=True)
