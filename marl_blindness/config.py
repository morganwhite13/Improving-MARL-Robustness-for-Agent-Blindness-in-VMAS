"""Training settings.

TrainConfig's defaults are the original training script's, so results match
it exactly. TUNED is the best configuration the 3-seed re-runs measured for a
fixed number of iterations (see the README's hyperparameter re-run). For a
fixed amount of compute, ORIGINAL with more iterations does better: TUNED's
iterations take about 1.65x as long (experiments/efficiency_results.json).
"""
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class TrainConfig:
    # Sampling
    frames_per_batch: int = 6_000  # team frames collected per training iteration
    n_iters: int = 25  # sampling and training iterations
    max_steps: int = 200  # episode length
    # Optimisation
    num_epochs: int = 20  # passes over each batch
    minibatch_size: int = 200
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    # PPO
    clip_epsilon: float = 0.2
    gamma: float = 0.9
    lmbda: float = 0.9  # GAE lambda
    entropy_eps: float = 1e-4
    normalize_advantage: bool = True
    # Networks (policy and critic): MLPs shared across agents, centralised critic (MAPPO)
    depth: int = 2
    num_cells: int = 256

    @property
    def num_envs(self) -> int:
        """Parallel environments: one batch holds one full episode from each."""
        return self.frames_per_batch // self.max_steps

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **changes) -> "TrainConfig":
        return replace(self, **changes)


ORIGINAL = TrainConfig()

# 55.6 ± 1.8 vs 21.3 ± 7.6 for ORIGINAL after 25 iterations (one agent blinded
# for 1-10 steps, 3 seeds), but each iteration does 1.5x the gradient steps.
# Given the same wall-clock time, ORIGINAL reached 63.0 vs 49.4 at 7 minutes.
TUNED = TrainConfig(clip_epsilon=0.3, num_epochs=30)

PRESETS = {"original": ORIGINAL, "tuned": TUNED}
