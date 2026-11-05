"""Evaluating trained policies on fresh episodes, and saving and loading them."""
from collections.abc import Callable

import torch
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor

from .config import TrainConfig
from .training import build_policy


def evaluate(policy: ProbabilisticActor, build_env: Callable[[], EnvBase], seed: int, rollouts: int = 3) -> float:
    """Mean episode reward of the deterministic policy over full episodes.

    Seeds torch, builds the environment with build_env() (in that order, so runs
    are reproducible), then averages the reward of every episode that ended, as
    the training metric does, over `rollouts` rollouts of every parallel
    environment.
    """
    torch.manual_seed(seed)
    env = build_env()
    env.set_seed(seed)
    scores = []
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(rollouts):
            td = env.rollout(env.max_steps, policy=policy, break_when_any_done=False)
            ep_reward = td.get(("next", "agents", "episode_reward"))
            done = td.get(("next", "done")).unsqueeze(-1).expand(ep_reward.shape)
            scores.append(ep_reward[done].mean().item())
    return sum(scores) / len(scores)


def save_policy(path, policy: ProbabilisticActor, cfg: TrainConfig, env_kwargs: dict) -> None:
    """Save the policy's weights with what's needed to rebuild it."""
    torch.save({"state_dict": policy.state_dict(), "config": cfg.to_dict(), "env": env_kwargs}, path)


def load_policy(path, env: EnvBase) -> tuple[ProbabilisticActor, dict]:
    """Rebuild a saved policy for env (which must have the same observation and action sizes)."""
    saved = torch.load(path, weights_only=False)
    policy = build_policy(env, TrainConfig(**saved["config"]))
    policy.load_state_dict(saved["state_dict"])
    return policy, saved
