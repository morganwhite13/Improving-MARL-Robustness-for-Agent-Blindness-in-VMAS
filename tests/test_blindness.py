import os
import sys

import pytest
import torch
from tensordict import TensorDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blindness import (  # noqa: E402
    OBS_KEY,
    BlindAllAgentsEveryStep,
    BlindOneRandomAgentEveryStep,
    BlindOneRandomAgentIfProbability,
    BlindOneRandomAgentIfProbabilityForJSteps,
    BlindRandomAgentsIfProbability,
    BlindRandomAgentsIfProbabilityForJSteps,
)

N_ENVS, N_AGENTS, OBS_DIM = 2000, 3, 4


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def step(transform, n_envs=N_ENVS):
    """Apply one step of the transform to all-ones observations; return the blind mask."""
    next_td = TensorDict(
        {"agents": TensorDict({"observation": torch.ones(n_envs, N_AGENTS, OBS_DIM)}, batch_size=[n_envs, N_AGENTS])},
        batch_size=[n_envs],
    )
    out = transform._step(TensorDict({}, batch_size=[n_envs]), next_td)
    obs = out.get(OBS_KEY)
    blind = (obs == 0).all(-1)
    seen = (obs == 1).all(-1)
    assert (blind | seen).all(), "observations must be either fully kept or fully zeroed"
    return blind


def test_all_agents_blinds_every_agent():
    assert step(BlindAllAgentsEveryStep(N_AGENTS)).all()


def test_one_random_agent_every_step_blinds_exactly_one_chosen_per_env():
    blind = step(BlindOneRandomAgentEveryStep(N_AGENTS))
    assert (blind.sum(-1) == 1).all()
    # Independent per environment: every agent gets picked somewhere, about equally
    share = blind.float().mean(0)
    assert torch.allclose(share, torch.full((N_AGENTS,), 1 / N_AGENTS), atol=0.05)


def test_one_random_agent_if_probability():
    blind = step(BlindOneRandomAgentIfProbability(N_AGENTS, blind_prob=0.3))
    per_env = blind.sum(-1)
    assert (per_env <= 1).all()
    assert abs(per_env.float().mean().item() - 0.3) < 0.04


def test_random_agents_if_probability_is_independent_per_agent_and_env():
    blind = step(BlindRandomAgentsIfProbability(N_AGENTS, blind_prob=0.25))
    assert torch.allclose(blind.float().mean(0), torch.full((N_AGENTS,), 0.25), atol=0.04)
    # Some environments differ from others in the same step (no shared draw)
    assert not (blind == blind[0]).all()


@pytest.mark.parametrize("cls", [BlindOneRandomAgentIfProbabilityForJSteps, BlindRandomAgentsIfProbabilityForJSteps])
def test_blind_spells_last_between_1_and_max_steps(cls):
    max_steps = 3
    t = cls(N_AGENTS, blind_prob=1.0, max_blind_steps=max_steps)
    first = step(t)
    assert first.any()
    # Stop new spells and count how long each existing one lasts
    t._blind_prob = 0.0
    lengths = first.long()
    still = first.clone()
    for _ in range(max_steps + 2):
        still &= step(t)
        lengths += still.long()
    started = lengths[first]
    assert started.min() >= 1 and started.max() <= max_steps
    # Spell lengths vary across environments
    assert started.unique().numel() > 1
    assert not step(t).any(), "everyone must be able to see again after max_blind_steps"


def test_one_agent_for_j_steps_never_blinds_two_agents_in_one_env():
    t = BlindOneRandomAgentIfProbabilityForJSteps(N_AGENTS, blind_prob=0.5, max_blind_steps=5)
    for _ in range(30):
        assert (step(t).sum(-1) <= 1).all()


def test_countdowns_are_per_environment():
    t = BlindRandomAgentsIfProbabilityForJSteps(N_AGENTS, blind_prob=0.5, max_blind_steps=10)
    step(t)
    assert t.blind_remaining.shape == (N_ENVS, N_AGENTS)
    assert t.blind_remaining.unique().numel() > 1


def test_reset_clears_only_the_environments_being_reset():
    t = BlindRandomAgentsIfProbabilityForJSteps(N_AGENTS, blind_prob=1.0, max_blind_steps=10)
    step(t, n_envs=4)
    t.blind_remaining = torch.full((4, N_AGENTS), 5)
    reset = torch.tensor([[True], [False], [True], [False]])
    t._reset(TensorDict({"_reset": reset}, batch_size=[4]), TensorDict({}, batch_size=[4]))
    assert (t.blind_remaining[[0, 2]] == 0).all()
    assert (t.blind_remaining[[1, 3]] == 5).all()
    # A full reset (no "_reset" key) clears everything
    t._reset(TensorDict({}, batch_size=[4]), TensorDict({}, batch_size=[4]))
    assert (t.blind_remaining == 0).all()


def test_seeded_runs_are_reproducible():
    torch.manual_seed(123)
    a = step(BlindRandomAgentsIfProbability(N_AGENTS, blind_prob=0.3))
    torch.manual_seed(123)
    b = step(BlindRandomAgentsIfProbability(N_AGENTS, blind_prob=0.3))
    assert torch.equal(a, b)


def test_runs_inside_a_real_vmas_environment():
    pytest.importorskip("vmas")
    from torchrl.envs import TransformedEnv
    from torchrl.envs.libs.vmas import VmasEnv
    from torchrl.envs.utils import check_env_specs

    base = VmasEnv(scenario="balance", num_envs=8, continuous_actions=True, max_steps=20, device="cpu", n_agents=N_AGENTS)
    env = TransformedEnv(base, BlindRandomAgentsIfProbabilityForJSteps(N_AGENTS, blind_prob=0.5, max_blind_steps=4))
    check_env_specs(env)
    rollout = env.rollout(30, break_when_any_done=False)
    obs = rollout.get(("next", "agents", "observation"))
    blind = (obs == 0).all(-1)  # [envs, time, agents]
    assert blind.any() and not blind.all()
