"""MAPPO training and evaluation under agent blindness in VMAS.

    from marl_blindness import TUNED, make_env, train, evaluate
    env = make_env(blindness="one-steps")
    result = train(env, TUNED)
"""
from .config import ORIGINAL, PRESETS, TUNED, TrainConfig
from .envs import BLINDNESS, make_env
from .evaluation import evaluate, load_policy, save_policy
from .training import TrainResult, train

__all__ = ["ORIGINAL", "PRESETS", "TUNED", "TrainConfig", "BLINDNESS", "make_env",
           "evaluate", "load_policy", "save_policy", "TrainResult", "train"]
