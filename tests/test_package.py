"""The marl_blindness package: it reproduces the training script, and its pieces work together."""
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

pytest.importorskip("vmas")
import torch  # noqa: E402

from marl_blindness import BLINDNESS, ORIGINAL, TUNED, evaluate, load_policy, make_env, save_policy, train  # noqa: E402


def test_original_preset_matches_the_script_defaults():
    from common import load_training_script

    ns = load_training_script(REPO, "test_defaults")
    assert ORIGINAL.frames_per_batch == ns["frames_per_batch"]
    assert ORIGINAL.n_iters == ns["n_iters"]
    assert ORIGINAL.max_steps == ns["max_steps"]
    assert ORIGINAL.num_envs == ns["num_vmas_envs"]
    assert (ORIGINAL.lr, ORIGINAL.max_grad_norm) == (ns["lr"], ns["max_grad_norm"])
    assert (ORIGINAL.gamma, ORIGINAL.lmbda, ORIGINAL.entropy_eps) == (ns["gamma"], ns["lmbda"], ns["entropy_eps"])
    assert (TUNED.clip_epsilon, TUNED.num_epochs, TUNED.minibatch_size) == (0.3, 30, 200)


def test_train_reproduces_the_training_script_exactly():
    from common import load_training_script

    torch.set_num_threads(1)
    ns = load_training_script(REPO, "test_equivalence", n_iters=2)
    env = ns["env6"]
    torch.manual_seed(0)
    env.set_seed(0)
    expected, _ = ns["train_environment_variables"](env, "script")

    env = make_env(blindness="one-steps")
    torch.manual_seed(0)
    env.set_seed(0)
    assert train(env, ORIGINAL.with_(n_iters=2)).rewards == expected


@pytest.mark.parametrize("kind", sorted(BLINDNESS))
def test_every_blindness_kind_builds_and_steps(kind):
    env = make_env(blindness=kind, num_envs=4, max_steps=10)
    td = env.rollout(5)
    assert td.get(("next", "agents", "observation")).shape[:2] == (4, 5)


def test_saved_policy_reloads_and_acts_identically(tmp_path):
    torch.set_num_threads(1)
    cfg = ORIGINAL.with_(n_iters=1, frames_per_batch=400, max_steps=20, num_epochs=1)
    env = make_env(num_envs=cfg.num_envs, max_steps=cfg.max_steps)
    torch.manual_seed(0)
    env.set_seed(0)
    policy = train(env, cfg).policy
    save_policy(tmp_path / "p.pt", policy, cfg, {})
    loaded, saved = load_policy(tmp_path / "p.pt", env)
    assert saved["config"]["n_iters"] == 1

    def build():
        return make_env(num_envs=cfg.num_envs, max_steps=cfg.max_steps, blindness="each", blind_prob=0.3)

    assert evaluate(policy, build, seed=5, rollouts=1) == evaluate(loaded, build, seed=5, rollouts=1)


def test_command_line_trains_and_evaluates(tmp_path):
    env = {**os.environ, "PYTHONPATH": REPO}
    run = subprocess.run(
        [sys.executable, "-m", "marl_blindness", "train", "--iters", "1", "--blindness", "one",
         "--out", str(tmp_path / "run.json"), "--save-policy", str(tmp_path / "p.pt")],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert run.returncode == 0, run.stderr
    assert len(json.load(open(tmp_path / "run.json"))["rewards"]) == 1
    run = subprocess.run(
        [sys.executable, "-m", "marl_blindness", "evaluate", str(tmp_path / "p.pt"), "--probs", "0", "0.2",
         "--blindness", "each-steps", "--rollouts", "1"],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.count("mean episode reward") == 2
