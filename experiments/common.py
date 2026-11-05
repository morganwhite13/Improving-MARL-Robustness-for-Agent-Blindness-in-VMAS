"""Shared setup for the experiment runners in this folder.

The runners reuse the training script as it is: they execute it up to
"#Experiment 1" (config, environments and train_environment_variables) and
work with the resulting namespace.
"""
import os
import sys

SCRIPT = "MARL Robustness for Agent Blindness in VMAS.py"


def load_training_script(repo, name, n_iters=None):
    """Execute the training script up to "#Experiment 1" and return its namespace.

    n_iters overrides the script's 25 training iterations.
    """
    sys.path.insert(0, repo)  # for "from blindness import ..."
    with open(os.path.join(repo, SCRIPT), encoding="utf-8") as f:
        src = f.read().replace("\r\n", "\n")
    src = src[: src.index("#Experiment 1")]
    if n_iters is not None:
        assert src.count("\nn_iters = 25 ") == 1
        src = src.replace("\nn_iters = 25 ", f"\nn_iters = {n_iters} ")
    ns = {"__name__": name}
    exec(compile(src, "vmas_script", "exec"), ns)
    return ns


def make_env(ns, scenario=None, n_agents=None, blindness=None):
    """A fresh VMAS environment built like the script's, with episode rewards summed.

    scenario and n_agents default to the script's. blindness, if given, is
    called with the number of agents and returns the transform to add.
    """
    from torchrl.envs import RewardSum, TransformedEnv
    from torchrl.envs.libs.vmas import VmasEnv

    base = VmasEnv(
        scenario=scenario or ns["scenario_name"], num_envs=ns["num_vmas_envs"], continuous_actions=True,
        max_steps=ns["max_steps"], device=ns["vmas_device"], n_agents=n_agents or ns["n_agents"],
    )
    env = TransformedEnv(base, RewardSum(in_keys=[base.reward_key], out_keys=[("agents", "episode_reward")]))
    if blindness is not None:
        env = TransformedEnv(env, blindness(base.n_agents))
    return env


def evaluate(ns, policy, build_env, eval_seed, rollouts=3):
    """Mean episode reward of a deterministic policy over full episodes.

    Seeds torch, builds the environment with build_env() (in that order, so
    runs are reproducible), then averages the reward of every episode that
    ended, as the training metric does, over `rollouts` rollouts of every
    parallel environment.
    """
    import torch
    from torchrl.envs.utils import ExplorationType, set_exploration_type

    torch.manual_seed(eval_seed)
    env = build_env()
    env.set_seed(eval_seed)
    scores = []
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(rollouts):
            td = env.rollout(ns["max_steps"], policy=policy, break_when_any_done=False)
            ep_reward = td.get(("next", "agents", "episode_reward"))
            done = td.get(("next", "done")).unsqueeze(-1).expand(ep_reward.shape)
            scores.append(ep_reward[done].mean().item())
    return sum(scores) / len(scores)
