"""BlindnessAwareObservation: the flag, the memory, the spec and resets."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("vmas")
import torch
from tensordict import TensorDict
from torchrl.envs.utils import check_env_specs

from marl_blindness import make_env
from marl_blindness.aware import BlindnessAwareObservation
from marl_blindness.blindness import OBS_KEY


def step(transform, obs):
    agents = TensorDict({"observation": obs}, batch_size=obs.shape[:2])
    next_td = TensorDict({"agents": agents}, batch_size=obs.shape[:1])
    return transform._step(TensorDict({}, batch_size=obs.shape[:1]), next_td).get(OBS_KEY)


def test_flag_marks_blind_agents_and_keeps_others_unchanged():
    obs = torch.tensor([[[1.0, 2.0], [0.0, 0.0]]])  # agent 1 is blind
    out = step(BlindnessAwareObservation(), obs)
    assert out.shape == (1, 2, 3)
    assert out[0, 0].tolist() == [1.0, 2.0, 0.0]
    assert out[0, 1].tolist() == [0.0, 0.0, 1.0]


def test_memory_holds_the_last_seen_observation_while_blind():
    t = BlindnessAwareObservation(memory=True)
    step(t, torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))
    out = step(t, torch.tensor([[[5.0, 6.0], [0.0, 0.0]]]))
    assert out[0, 1].tolist() == [3.0, 4.0, 1.0]  # last seen, flagged
    out = step(t, torch.tensor([[[7.0, 8.0], [0.0, 0.0]]]))
    assert out[0, 1].tolist() == [3.0, 4.0, 1.0]  # still held


def test_reset_forgets_memory_only_for_reset_environments():
    t = BlindnessAwareObservation(memory=True)
    step(t, torch.ones(2, 2, 3))
    obs = torch.full((2, 2, 3), 9.0)
    reset = TensorDict({"_reset": torch.tensor([[True], [False]])}, batch_size=[2])
    reset_td = TensorDict({"agents": TensorDict({"observation": obs}, batch_size=[2, 2])}, batch_size=[2])
    t._reset(reset, reset_td)
    assert (t.last_seen[0] == 9).all() and (t.last_seen[1] == 9).all()  # both saw a fresh observation
    blind = torch.zeros(2, 2, 3)
    t.last_seen[1] = 5.0
    t._reset(reset, TensorDict({"agents": TensorDict({"observation": blind}, batch_size=[2, 2])}, batch_size=[2]))
    assert (t.last_seen[0] == 0).all() and (t.last_seen[1] == 5).all()


@pytest.mark.parametrize("aware", ["flag", "memory"])
def test_aware_env_passes_spec_checks_and_flags_blindness(aware):
    env = make_env(blindness="each-steps", blind_prob=0.5, num_envs=8, max_steps=20, aware=aware)
    check_env_specs(env)
    obs = env.rollout(15, break_when_any_done=False).get(("next", "agents", "observation"))
    assert obs.shape[-1] == 17  # balance has 16 observation values, plus the flag
    flags = obs[..., -1]
    assert ((flags == 0) | (flags == 1)).all() and flags.any() and not flags.all()
