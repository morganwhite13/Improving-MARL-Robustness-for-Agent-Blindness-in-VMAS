"""VMAS environments with optional blindness, built the way the training script builds them."""
from collections.abc import Callable

from torchrl.envs import RewardSum, Transform, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv

from .aware import BlindnessAwareObservation
from .blindness import (
    BlindAllAgentsEveryStep,
    BlindOneRandomAgentEveryStep,
    BlindOneRandomAgentIfProbability,
    BlindOneRandomAgentIfProbabilityForJSteps,
    BlindRandomAgentsIfProbability,
    BlindRandomAgentsIfProbabilityForJSteps,
)

# Blindness kinds by name: name -> function(n_agents, prob, max_blind_steps) -> transform
BLINDNESS = {
    "none": None,
    "one-every-step": lambda n, p, j: BlindOneRandomAgentEveryStep(n),
    "all-every-step": lambda n, p, j: BlindAllAgentsEveryStep(n),
    "one": lambda n, p, j: BlindOneRandomAgentIfProbability(n, p),
    "each": lambda n, p, j: BlindRandomAgentsIfProbability(n, p),
    "one-steps": lambda n, p, j: BlindOneRandomAgentIfProbabilityForJSteps(n, p, j),
    "each-steps": lambda n, p, j: BlindRandomAgentsIfProbabilityForJSteps(n, p, j),
}


def make_env(
    scenario: str = "balance",
    n_agents: int = 3,
    num_envs: int = 30,
    max_steps: int = 200,
    device: str = "cpu",
    blindness: str | Callable[[int], Transform] = "none",
    blind_prob: float = 0.1,
    max_blind_steps: int = 10,
    aware: str = "none",
    **scenario_kwargs,
) -> TransformedEnv:
    """A vectorised VMAS environment with episode rewards summed and optional blindness.

    blindness is a name from BLINDNESS, or a function that takes the number of
    agents and returns a transform. aware adds a "blind" flag to each agent's
    observation ("flag"), plus its last seen observation while blind ("memory").

    Some scenarios (give_way, ball_passage, multi_give_way) fix their own number
    of agents and ignore n_agents; the returned env's n_agents is the real one.
    """
    base = VmasEnv(
        scenario=scenario, num_envs=num_envs, continuous_actions=True,
        max_steps=max_steps, device=device, n_agents=n_agents, **scenario_kwargs,
    )
    env = TransformedEnv(base, RewardSum(in_keys=[base.reward_key], out_keys=[("agents", "episode_reward")]))
    if callable(blindness):
        env = TransformedEnv(env, blindness(base.n_agents))
    elif BLINDNESS[blindness] is not None:
        env = TransformedEnv(env, BLINDNESS[blindness](base.n_agents, blind_prob, max_blind_steps))
    if aware != "none":
        env = TransformedEnv(env, BlindnessAwareObservation(memory=aware == "memory"))
    return env
