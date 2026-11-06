"""Observation "blindness" transforms for TorchRL multi-agent environments.

Each transform zeroes some agents' observations at ("agents", "observation"),
simulating sensor failures. Random choices are drawn independently for every
parallel environment, and blindness countdowns are kept per environment and
per agent and cleared when an environment resets.

Randomness comes from torch, so torch.manual_seed makes runs reproducible.
"""

import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from torchrl.envs.transforms import Transform

OBS_KEY = ("agents", "observation")


def _blind(next_tensordict: TensorDictBase, mask: torch.Tensor) -> TensorDictBase:
    """Zero the observations of agents where mask ([*batch, n_agents]) is True."""
    obs = next_tensordict.get(OBS_KEY)
    next_tensordict.set(OBS_KEY, obs.masked_fill(mask.unsqueeze(-1), 0))
    return next_tensordict


def _batch_shape(next_tensordict: TensorDictBase) -> torch.Size:
    """Shape of the environment batch: the observation minus (n_agents, obs_dim)."""
    return next_tensordict.get(OBS_KEY).shape[:-2]


def _random_agent_mask(batch_shape, n_agents, device) -> torch.Tensor:
    """One randomly chosen agent per environment, as a boolean mask."""
    idx = torch.randint(n_agents, batch_shape, device=device)
    return F.one_hot(idx, n_agents).bool()


class BlindOneRandomAgentEveryStep(Transform):
    """Every step, blind one randomly chosen agent in each environment."""

    def __init__(self, n_agents: int):
        super().__init__()
        self._n_agents = n_agents

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        obs = next_tensordict.get(OBS_KEY)
        mask = _random_agent_mask(_batch_shape(next_tensordict), self._n_agents, obs.device)
        return _blind(next_tensordict, mask)


class BlindAllAgentsEveryStep(Transform):
    """Every step, blind every agent. blind_prob is accepted for compatibility and unused."""

    def __init__(self, n_agents: int, blind_prob: float = 0.1):
        super().__init__()
        self._n_agents = n_agents
        self._blind_prob = blind_prob

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        obs = next_tensordict.get(OBS_KEY)
        next_tensordict.set(OBS_KEY, torch.zeros_like(obs))
        return next_tensordict


class BlindOneRandomAgentIfProbability(Transform):
    """Every step, with probability blind_prob per environment, blind one random agent."""

    def __init__(self, n_agents: int, blind_prob: float = 0.1):
        super().__init__()
        self._n_agents = n_agents
        self._blind_prob = blind_prob

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        obs = next_tensordict.get(OBS_KEY)
        batch_shape = _batch_shape(next_tensordict)
        happens = torch.rand(batch_shape, device=obs.device) < self._blind_prob
        mask = _random_agent_mask(batch_shape, self._n_agents, obs.device) & happens.unsqueeze(-1)
        return _blind(next_tensordict, mask)


class BlindRandomAgentsIfProbability(Transform):
    """Every step, blind each agent independently with probability blind_prob."""

    def __init__(self, n_agents: int, blind_prob: float = 0.1):
        super().__init__()
        self._n_agents = n_agents
        self._blind_prob = blind_prob

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        obs = next_tensordict.get(OBS_KEY)
        mask = torch.rand(_batch_shape(next_tensordict) + (self._n_agents,), device=obs.device) < self._blind_prob
        return _blind(next_tensordict, mask)


class _CountdownBlindness(Transform):
    """Base for transforms where a blinded agent stays blind for 1..max_blind_steps steps.

    blind_remaining[env, agent] counts the further steps an agent stays blind
    after the current one. It is created on the first step and cleared for the
    environments being reset.
    """

    def __init__(self, n_agents: int, blind_prob: float = 0.1, max_blind_steps: int = 10):
        super().__init__()
        if max_blind_steps < 1:
            raise ValueError("max_blind_steps must be at least 1")
        self._n_agents = n_agents
        self._blind_prob = blind_prob
        self.max_blind_steps = max_blind_steps
        self.blind_remaining = None

    def _state(self, batch_shape, device) -> torch.Tensor:
        shape = batch_shape + (self._n_agents,)
        if self.blind_remaining is None or self.blind_remaining.shape != shape:
            self.blind_remaining = torch.zeros(shape, dtype=torch.long, device=device)
        return self.blind_remaining

    def _new_blindness(self, currently_blind, batch_shape, device) -> torch.Tensor:
        """Boolean mask of agents that start a new blind spell this step."""
        raise NotImplementedError

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        obs = next_tensordict.get(OBS_KEY)
        batch_shape = _batch_shape(next_tensordict)
        remaining = self._state(batch_shape, obs.device)

        # Agents already blind stay blind this step and count down
        currently_blind = remaining > 0
        remaining = remaining - currently_blind.long()

        # New spells last 1..max_blind_steps steps, including this one
        start = self._new_blindness(currently_blind, batch_shape, obs.device)
        duration = torch.randint(1, self.max_blind_steps + 1, start.shape, device=obs.device)
        remaining = torch.where(start, duration - 1, remaining)

        self.blind_remaining = remaining
        return _blind(next_tensordict, currently_blind | start)

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        if self.blind_remaining is not None:
            reset = tensordict.get("_reset", None) if tensordict is not None else None
            if reset is None:
                self.blind_remaining = torch.zeros_like(self.blind_remaining)
            else:
                # Collapse any trailing dims of "_reset" to one flag per environment
                env_reset = reset.reshape(self.blind_remaining.shape[:-1] + (-1,)).any(-1)
                self.blind_remaining = self.blind_remaining.masked_fill(env_reset.unsqueeze(-1), 0)
        return tensordict_reset


class BlindOneRandomAgentIfProbabilityForJSteps(_CountdownBlindness):
    """At most one blind agent per environment at a time.

    When no agent in an environment is blind, with probability blind_prob one
    random agent goes blind for 1..max_blind_steps steps.
    """

    def _new_blindness(self, currently_blind, batch_shape, device) -> torch.Tensor:
        free = ~currently_blind.any(-1)
        happens = free & (torch.rand(batch_shape, device=device) < self._blind_prob)
        return _random_agent_mask(batch_shape, self._n_agents, device) & happens.unsqueeze(-1)


class BlindRandomAgentsIfProbabilityForJSteps(_CountdownBlindness):
    """Each agent that isn't blind goes blind with probability blind_prob for 1..max_blind_steps steps."""

    def _new_blindness(self, currently_blind, batch_shape, device) -> torch.Tensor:
        draws = torch.rand(batch_shape + (self._n_agents,), device=device) < self._blind_prob
        return draws & ~currently_blind
