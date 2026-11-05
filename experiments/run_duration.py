"""Experiment 7 re-run: how does the length of blind spells affect a trained team?

The original experiment retrained a policy for every maximum blind duration
and scored it on its own training rollouts. This trains one policy (with no
blindness, or with one random agent blinded for 1-10 steps at probability
0.1), saves it, then evaluates it with deterministic actions on fresh,
separately seeded environments at a range of maximum blind durations, with
blindness probability 0.1, for both kinds of blindness.

Usage: python experiments/run_duration.py . <seed> <out.json> <none|one> [n_iters] [policy.pt]
"""
import json, os, sys, time
repo, seed, out, cond = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
n_iters = int(sys.argv[5]) if len(sys.argv) > 5 else None
policy_path = sys.argv[6] if len(sys.argv) > 6 else None
sys.path.insert(0, repo)
src = open(os.path.join(repo, "MARL Robustness for Agent Blindness in VMAS.py"), encoding="utf-8").read().replace("\r\n", "\n")
src = src[: src.index("#Experiment 1")]
if n_iters is not None:
    assert src.count("\nn_iters = 25 ") == 1
    src = src.replace("\nn_iters = 25 ", f"\nn_iters = {n_iters} ")
ns = {"__name__": "duration"}
exec(compile(src, "vmas_script", "exec"), ns)

import torch
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
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


def evaluate(policy, transform_cls, max_steps_blind, eval_seed):
    torch.manual_seed(eval_seed)
    base = VmasEnv(
        scenario=ns["scenario_name"], num_envs=ns["num_vmas_envs"], continuous_actions=True,
        max_steps=ns["max_steps"], device=ns["vmas_device"], n_agents=ns["n_agents"],
    )
    env = TransformedEnv(base, RewardSum(in_keys=[base.reward_key], out_keys=[("agents", "episode_reward")]))
    if max_steps_blind > 0:
        env = TransformedEnv(env, transform_cls(base.n_agents, EVAL_PROB, max_steps_blind))
    env.set_seed(eval_seed)
    scores = []
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(EVAL_ROLLOUTS):
            td = env.rollout(ns["max_steps"], policy=policy, break_when_any_done=False)
            ep_reward = td.get(("next", "agents", "episode_reward"))
            done = td.get(("next", "done")).unsqueeze(-1).expand(ep_reward.shape)
            scores.append(ep_reward[done].mean().item())
    return sum(scores) / len(scores)


name, env = TRAIN[cond]
torch.manual_seed(seed)
env.set_seed(seed)
t = time.time()
rewards, policy = ns["train_environment_variables"](env, name)
if policy_path:
    torch.save(policy.state_dict(), policy_path)
evals = {
    etype: {str(d): evaluate(policy, cls, d, eval_seed=1000 + seed) for d in [0] + DURATIONS}
    for etype, cls in EVAL_TYPES.items()
}
json.dump({"seed": seed, "train": name, "train_rewards": [float(r) for r in rewards], "eval_prob": EVAL_PROB,
           "eval": evals, "minutes": (time.time() - t) / 60}, open(out, "w"), indent=1)
print(f"DONE seed={seed} {name}: {evals} ({(time.time() - t) / 60:.1f} min)", flush=True)
