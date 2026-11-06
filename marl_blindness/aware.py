"""Blindness-aware observations: let each agent know when its sensors are off.

A blinded agent normally sees all zeros, which it can't tell apart from a real
reading. BlindnessAwareObservation goes after a blindness transform and adds a
flag to each agent's observation (1 while blind). With memory=True it also
replaces a blind agent's zeros with the last observation it actually saw.
"""
import torch
from tensordict import TensorDictBase
from torchrl.data import Unbounded
from torchrl.envs.transforms import Transform

from .blindness import OBS_KEY


class BlindnessAwareObservation(Transform):
    """Append a "blind" flag to each agent's observation; optionally hold the last seen one.

    An agent counts as blind when its whole observation is exactly zero, which
    is what the blindness transforms produce.
    """

    def __init__(self, memory: bool = False):
        super().__init__()
        self.memory = memory
        self.last_seen = None  # [*batch, n_agents, obs_dim]

    def _augment(self, obs: torch.Tensor) -> torch.Tensor:
        blind = (obs == 0).all(-1, keepdim=True)
        if self.memory:
            if self.last_seen is None or self.last_seen.shape != obs.shape:
                self.last_seen = torch.zeros_like(obs)
            obs = torch.where(blind, self.last_seen, obs)
            self.last_seen = obs
        return torch.cat([obs, blind.to(obs.dtype)], dim=-1)

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        next_tensordict.set(OBS_KEY, self._augment(next_tensordict.get(OBS_KEY)))
        return next_tensordict

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        obs = tensordict_reset.get(OBS_KEY)
        if self.memory and self.last_seen is not None and self.last_seen.shape == obs.shape:
            # Forget the memory of environments that are starting a new episode
            reset = tensordict.get("_reset", None) if tensordict is not None else None
            if reset is None:
                self.last_seen = torch.zeros_like(obs)
            else:
                env_reset = reset.reshape(obs.shape[:-2] + (-1,)).any(-1)
                self.last_seen = self.last_seen.masked_fill(env_reset[..., None, None], 0)
        tensordict_reset.set(OBS_KEY, self._augment(obs))
        return tensordict_reset

    def transform_observation_spec(self, observation_spec):
        leaf = observation_spec[OBS_KEY]
        observation_spec[OBS_KEY] = Unbounded(
            shape=leaf.shape[:-1] + (leaf.shape[-1] + 1,), device=leaf.device, dtype=leaf.dtype
        )
        return observation_spec
