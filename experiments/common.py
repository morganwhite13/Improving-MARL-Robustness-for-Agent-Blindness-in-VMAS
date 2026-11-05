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
    """A fresh environment built like the script's (see marl_blindness.make_env).

    scenario and n_agents default to the script's. blindness, if given, is
    called with the number of agents and returns the transform to add.
    """
    from marl_blindness import make_env as package_make_env

    return package_make_env(
        scenario=scenario or ns["scenario_name"], n_agents=n_agents or ns["n_agents"], num_envs=ns["num_vmas_envs"],
        max_steps=ns["max_steps"], device=ns["vmas_device"], blindness=blindness or "none",
    )


def evaluate(ns, policy, build_env, eval_seed, rollouts=3):
    """Mean episode reward of a deterministic policy over full episodes (see marl_blindness.evaluate)."""
    from marl_blindness import evaluate as package_evaluate

    return package_evaluate(policy, build_env, eval_seed, rollouts)
