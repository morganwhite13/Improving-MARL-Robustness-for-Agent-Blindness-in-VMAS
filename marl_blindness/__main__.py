"""Command line: train a policy, or evaluate a saved one.

    python -m marl_blindness train --blindness one-steps --preset tuned --iters 100 --save-policy policy.pt
    python -m marl_blindness evaluate policy.pt --blindness each-steps --probs 0 0.1 0.2

Blindness kinds: none, one-every-step, all-every-step, one, each, one-steps,
each-steps ("one" blinds one random agent, "each" each agent independently,
"-steps" for 1..max-blind-steps steps at a time).
"""
import argparse
import json
import time

import torch

from .config import PRESETS
from .envs import BLINDNESS, make_env
from .evaluation import evaluate, load_policy, save_policy
from .training import train


def env_args(parser):
    parser.add_argument("--scenario", default="balance")
    parser.add_argument("--agents", type=int, default=3, help="ignored by scenarios with a fixed team size")
    parser.add_argument("--blindness", choices=BLINDNESS, default="none")
    parser.add_argument("--max-blind-steps", type=int, default=10)


def cmd_train(args):
    cfg = PRESETS[args.preset]
    if args.iters is not None:
        cfg = cfg.with_(n_iters=args.iters)
    env_kwargs = dict(scenario=args.scenario, n_agents=args.agents, num_envs=cfg.num_envs, max_steps=cfg.max_steps,
                      blindness=args.blindness, blind_prob=args.prob, max_blind_steps=args.max_blind_steps)
    torch.set_num_threads(args.threads)
    env = make_env(**env_kwargs)
    torch.manual_seed(args.seed)
    env.set_seed(args.seed)
    start = time.time()
    def report(i, reward):
        print(f"iteration {i + 1}/{cfg.n_iters}: reward {reward:.2f}", flush=True)

    result = train(env, cfg, progress=report)
    minutes = (time.time() - start) / 60
    print(f"final reward {result.rewards[-1]:.2f} after {minutes:.1f} min")
    if args.save_policy:
        save_policy(args.save_policy, result.policy, cfg, env_kwargs)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"seed": args.seed, "config": cfg.to_dict(), "env": env_kwargs, "rewards": result.rewards,
                       "final": result.rewards[-1], "minutes": minutes}, f, indent=1)


def cmd_evaluate(args):
    torch.set_num_threads(args.threads)
    probe = make_env(scenario=args.scenario, n_agents=args.agents, num_envs=1)
    policy, saved = load_policy(args.policy, probe)
    max_steps = saved["config"]["max_steps"]
    num_envs = saved["config"]["frames_per_batch"] // max_steps
    for prob in args.probs:
        kind = args.blindness if prob > 0 else "none"
        env_kwargs = dict(scenario=args.scenario, n_agents=args.agents, num_envs=num_envs, max_steps=max_steps,
                          blindness=kind, blind_prob=prob, max_blind_steps=args.max_blind_steps)
        score = evaluate(policy, lambda kw=env_kwargs: make_env(**kw), seed=args.seed, rollouts=args.rollouts)
        print(f"blindness {kind} p={prob}: mean episode reward {score:.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m marl_blindness", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a MAPPO policy")
    env_args(t)
    t.add_argument("--prob", type=float, default=0.1, help="blindness probability")
    t.add_argument("--preset", choices=PRESETS, default="original")
    t.add_argument("--iters", type=int, help="training iterations (default: the preset's)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--threads", type=int, default=1)
    t.add_argument("--out", help="write rewards and settings to this JSON file")
    t.add_argument("--save-policy", help="save the trained policy to this file")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a saved policy on fresh episodes")
    e.add_argument("policy")
    env_args(e)
    e.add_argument("--probs", type=float, nargs="+", default=[0.0, 0.1], help="blindness probabilities to test")
    e.add_argument("--seed", type=int, default=1000)
    e.add_argument("--rollouts", type=int, default=3)
    e.add_argument("--threads", type=int, default=1)
    e.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
