# Improving MARL Robustness for Agent Blindness in VMAS

A Multi-Agent Reinforcement Learning research project investigating how cooperative agents handle partial observability through random "blindness" events. This work demonstrates that MARL systems can learn robust coordination strategies despite unpredictable sensory failures.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
![Balance Scenario](https://github.com/matteobettini/vmas-media/raw/main/media/scenarios/balance.gif?raw=true)

## 🎯 Overview

**The Challenge:** Autonomous systems must cooperate effectively even when sensors fail unexpectedly. This project explores whether multi-agent reinforcement learning can develop strategies that remain robust when agents randomly lose sensory input.

**Key Innovation:** Six custom blindness scenarios implemented as VMAS environment transforms, ranging from simple single-step occlusions to complex multi-agent, variable-duration sensory failures.

**Results:** Re-run over 3 seeds with the fixed code, training without blindness scores highest, short occasional blindness costs relatively little, and blinding every agent prevents learning. Tuned PPO settings (advantage normalization, clip 0.3, more epochs) beat the original settings on average and vary far less between seeds, though the gain is smaller than the original single-seed +86%. A robustness test (train with or without blindness, evaluate on separate episodes under blindness) shows that **training with blindness makes the team markedly more robust, once it has trained long enough**: after 100 iterations, a team trained with each-agent blindness keeps 86% of its reward at 10% blindness, against 43% for a team trained without it. At 25 iterations it hadn't caught up yet.

### Real-World Applications

- **Autonomous vehicles** handling camera/sensor malfunctions
- **Drone swarms** maintaining formation during GPS signal loss
- **Robot teams** completing tasks with failed sensors
- **Satellite networks** coordinating during communication blackouts

## 🔬 Research Highlights

### Custom Blindness Scenarios

Implemented six environment transforms that randomly zero out agent observations:

1. **BlindOneRandomAgentEveryStep** - One agent blinded each step
2. **BlindAllAgentsEveryStep** - All agents simultaneously blinded
3. **BlindOneRandomAgentIfProbability** - Probabilistic single-agent blindness
4. **BlindRandomAgentsIfProbability** - Probabilistic multi-agent blindness
5. **BlindOneRandomAgentIfProbabilityForJSteps** - Extended duration single-agent blindness
6. **BlindRandomAgentsIfProbabilityForJSteps** - Extended duration multi-agent blindness (most realistic)

### Key Findings

The findings below come from the original single-seed runs, before the bug fixes described in "Known issues and status". The re-runs and the robustness test (see "Experimental Results") qualify all five points.

- **Robustness is learnable**: Confirmed by the robustness test, given enough training. After 100 iterations, policies trained with blindness score as well as the no-blindness policy when nobody is blind, and much better when agents are (see "Robustness test"). After only 25 iterations they were still behind
- **Hyperparameters matter**: The original runs reported +86% for normalized advantages, clip=0.3, batch=100, epochs=30. The re-run finds a smaller, noisier gain, and that configuration isn't the best one tested (see "Hyperparameter re-run")
- **Advantage normalization is critical**: Not supported by the re-run. At the experiments' 25 iterations it made no consistent difference to final reward, though it did make seeds agree more (see "Experiment 2 re-run")
- **Redundancy helps**: Confirmed by the re-run with a no-blindness control: a 3-agent team loses about half its reward to one blind agent, a 5-agent team about a fifth, and from 7 agents nothing. The original "5-7 agents is optimal" isn't: 10 agents scored highest, with no sign of coordination overhead (see "Experiment 8 re-run")
- **Performance thresholds exist**: Partly. The threshold depends on how agents go blind: with one agent blind at a time, policies keep most of their reward even at 50% blindness, but when every agent can go blind independently, performance drops sharply from 10-20% (see "Robustness Thresholds")

## Known issues and status

The original results (the tables further down) were produced with code that had several bugs, all now fixed:

- `BlindAllAgentsEveryStep` picked a random agent (with replacement) on each loop pass, so it blinded about two of three agents, not all of them.
- `BlindRandomAgentsIfProbability` blinded a fresh random agent, not the agent that passed the probability check.
- `BlindRandomAgentsIfProbabilityForJSteps` ignored `max_blind_steps` and always drew durations from 1 to 10.
- Every transform made one random choice per step and applied it to all 30 parallel environments at once, kept blind countdowns per agent rather than per environment, and never reset them between episodes. The transforms now live in `blindness.py`, draw independently per environment, and are covered by tests in `tests/`.
- The blindness-probability and blind-duration sweeps wrapped the previous iteration's environment, so transforms stacked up across settings. Each setting now starts from a fresh base environment.
- The first plot read a `"norm"` key that its results did not have, so the script crashed after the first experiment.
- The "VMAS Scenarios" experiment could crash with "Physical actions ... out of its range": a `TanhNormal` sample can overshoot the action bound by floating-point error, which VMAS rejects. The policy now clamps actions into the spec (`safe=True`).

With these fixes, every experiment in the script has been run end to end with a reduced training budget (1 iteration each) to check that it completes.

Experiment 1 has been re-run with the fixed code over 3 seeds (below). The other experiments still need re-running. Two limitations remain: policies are scored on their own training rollouts rather than a separate evaluation, and with the default 25 iterations most reward curves are still rising, so training hasn't converged.

The reward rescaling in the blindness-probability and blind-duration plots is applied for display only, so those curves cannot be compared directly across settings.

## 📊 Experimental Results

### Experiment 1, re-run with the fixed code

Final episode reward (mean over the 30 parallel environments, last training iteration), mean ± standard deviation over 3 seeds. Same settings as the original experiment: `norm=False`, clip 0.2, 25 iterations × 6,000 frames, balance scenario with 3 agents.

| Blindness Scenario | Final Reward |
|-------------------|--------------|
| No blindness (baseline) | 49.2 ± 4.8 |
| Blind 1 random agent with probability 0.1 | 40.5 ± 11.6 |
| Blind each agent with probability 0.1 | 37.8 ± 15.6 |
| Blind 1 random agent for 1–10 steps (probability 0.1) | 34.2 ± 16.1 |
| Blind each agent for 1–10 steps (probability 0.1) | 10.7 ± 9.2 |
| Blind 1 random agent every step | 5.5 ± 12.9 |
| Blind all agents every step | −11.8 ± 1.9 |

![Reward curves per blindness scenario, mean ± std over 3 seeds](experiments/exp1_rewards.png)

What holds up: training without blindness does best, and blinding every agent prevents learning entirely. Occasional, short blindness costs relatively little, while persistent blindness (every step, or several agents for several steps) hurts much more. The spread between seeds is as large as the gaps between the middle scenarios, so their order isn't meaningful from 3 seeds.

These rewards are on a different scale from the original table below (about 50 rather than about 450 for the baseline). The re-run used the pinned versions in `requirements.txt`; the original runs may have used different library versions or reported reward differently.

To reproduce: `python experiments/run_exp1.py . <seed> experiments/seed<seed>.json` for each seed (about 45 minutes per seed on one CPU core), then `python experiments/plot_exp1.py`. The results used here are in `experiments/exp1_results.json`.

### Experiment 2 re-run: advantage normalization

The original runs concluded that advantage normalization is essential under blindness (+40–60 reward). Experiment 1's re-run above trained with normalization off; repeating it with normalization on, with the same seeds and settings otherwise, tests that claim. Final reward, mean ± std over 3 seeds:

| Scenario | Normalization off | Normalization on |
|---|---|---|
| Normal (no blindness) | 49.2 ± 4.8 | 43.8 ± 10.9 |
| Blind 1 random agent every step | 5.5 ± 12.9 | 16.6 ± 6.0 |
| Blind all agents every step | −11.8 ± 1.9 | −10.8 ± 2.4 |
| Blind 1 random agent with probability 0.1 | 40.5 ± 11.6 | 36.8 ± 8.0 |
| Blind each agent with probability 0.1 | 37.8 ± 15.6 | 33.2 ± 6.3 |
| Blind 1 random agent for 1–10 steps | 34.2 ± 16.1 | 21.3 ± 7.6 |
| Blind each agent for 1–10 steps | 10.7 ± 9.2 | −0.2 ± 5.9 |

![Final reward per scenario with normalization off and on, mean ± std with each seed shown](experiments/exp2_norm.png)

Normalization isn't essential here. With it on, the mean is lower in 5 of 7 scenarios, and the per-seed differences mostly change sign. Its one consistent gain is with one agent blinded every step (+11, better in all 3 seeds). What it does do is make seeds agree more: the spread is smaller in 5 of 7 scenarios. Repeated runs reproduce exactly (the no-blindness, normalization-on results match the robustness test's), so these differences are between seeds, not run-to-run noise.

To reproduce: `python experiments/run_exp1.py . <seed> experiments/norm_seed<seed>.json norm` for each seed, merge the seed files into `experiments/exp2_norm_results.json`, then `python experiments/plot_exp2.py`.

### Original results (before the fixes)

| Blindness Scenario | Final Reward | Performance vs. Baseline |
|-------------------|--------------|--------------------------|
| No blindness (baseline) | ~450 | 100% |
| Blind 1 agent randomly (1 step) | ~380 | 84% |
| Blind 1 agent every step | ~320 | 71% |
| Blind 1 agent random duration | ~280 | 62% |
| Blind random agents randomly | ~240 | 53% |
| Blind random agents random duration | ~180 | 40% |

### Hyperparameter Optimization

**Impact of Key Parameters:**
- **Normalization**: Essential (+40-60 reward improvement across scenarios). *Not supported by the 3-seed re-run; see "Experiment 2 re-run".*
- **Clip value**: Optimal at 0.3 (+28% over default 0.2)
- **Batch size**: Smaller batches (50-100) outperform default 200
- **Epochs**: Diminishing returns after 30 epochs
- **Agent count**: 5-7 agents optimal for robustness vs. coordination. *Not supported by the 3-seed re-run; see "Experiment 8 re-run".*

**Configuration reported as best in the original runs:**
```python
normalize_advantage = True
clip_epsilon = 0.3
minibatch_size = 100
num_epochs = 30
# Original single-seed result: ~520 reward (+86% over baseline 280)
```

### Hyperparameter re-run

The comparison above was re-run with the fixed code: 3 seeds per setup, on the scenario where one random agent is blinded for 1–10 steps with probability 0.1. "Best" is the configuration above; the last three rows each revert one of its changes.

| Setup | Normalize | Clip | Batch | Epochs | Final reward (mean ± std) | Seeds |
|-------|-----------|------|-------|--------|---------------------------|-------|
| Original | False | 0.2 | 200 | 20 | 34.2 ± 16.1 | 23.3, 26.6, 52.7 |
| Best | True | 0.3 | 100 | 30 | 45.3 ± 6.8 | 53.2, 41.8, 41.0 |
| Best, but clip 0.2 | True | 0.2 | 100 | 30 | 46.1 ± 10.7 | 36.9, 57.9, 43.5 |
| Best, but batch 200 | True | 0.3 | 200 | 30 | 55.6 ± 1.8 | 55.8, 53.7, 57.2 |
| Best, but 20 epochs | True | 0.3 | 100 | 20 | 52.4 ± 6.8 | 50.9, 59.8, 46.4 |

![Final reward per hyperparameter setup, mean ± std with each seed shown](experiments/tuning_rewards.png)

What holds up: every tuned setup beats the original settings on average (+32% for "Best") and varies much less between seeds. What doesn't: the +86% figure, and the claim that batch 100 and 30 epochs are optimal. Going back to batch 200 or to 20 epochs did as well or better, and with 3 seeds the differences between the tuned setups are within noise.

To reproduce: `python experiments/run_tuning.py . <seed> experiments/tuning_seed<seed>.json` for each seed (about 50 minutes per seed on one CPU core), merge the seed files into `experiments/tuning_results.json`, then `python experiments/plot_tuning.py`.

### Experiment 8 re-run: team size

The original runs found that 5–7 agents are optimal, with 10 agents doing worse from coordination overhead. But in the balance scenario more agents also make the task itself easier, so raw reward can't show whether a bigger team copes better with blindness. The re-run trains each team size both with and without blindness (one random agent blinded for 1–10 steps, probability 0.1, the original setting), 3 seeds each:

| Agents | No blindness | With blindness | Share kept |
|---|---|---|---|
| 2 | 7.4 ± 19.5 | 13.9 ± 12.2 | (didn't learn in 25 iterations) |
| 3 | 43.8 ± 10.9 | 21.3 ± 7.6 | 48% |
| 5 | 76.7 ± 5.0 | 62.4 ± 5.2 | 81% |
| 7 | 81.5 ± 2.9 | 91.9 ± 3.3 | 113% |
| 10 | 94.5 ± 8.9 | 95.5 ± 6.4 | 101% |

![Final reward by team size with and without blindness](experiments/exp8_agents.png)

- **Redundancy helps.** The share of reward kept under blindness rises with team size: about half for 3 agents, about four fifths for 5, and all of it from 7. With 7 agents the blinded runs even scored higher, in all 3 seeds.
- **There's no sweet spot at 5–7.** 10 agents scored highest with and without blindness. Within this range, more agents only helped.
- **Two agents don't learn the task within 25 iterations**, with or without blindness, so their share kept isn't meaningful.

To reproduce: `python experiments/run_agents.py . <seed> experiments/agents_seed<seed>.json` for each seed (about 35 minutes per seed on one CPU core), merge the seed files into `experiments/exp8_agents_results.json`, then `python experiments/plot_exp8.py`.

### Robustness test

The experiments above score each policy on its own training rollouts, in the conditions it was trained in. That can't say whether training *with* blindness makes a team more robust *to* blindness. This test does:

1. Train three policies (3 seeds each, the training function's default settings): with no blindness, with one random agent blinded for 1–10 steps (probability 0.1), and with each agent independently blinded for 1–10 steps (probability 0.1).
2. Evaluate every policy with deterministic actions on fresh, separately seeded environments, at blindness probabilities from 0 to 0.5, for both kinds of blindness (90 episodes per point).

It was run twice: with the script's 25 training iterations, then with 100.

**After 100 training iterations** (mean episode reward ± std over 3 seeds):

| Trained with | Evaluated with | p = 0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|---|
| No blindness | One agent blind | 69.7 ± 2.5 | 64.5 ± 1.8 | 59.7 ± 5.7 | 54.4 ± 8.9 | 48.2 ± 13.6 | 43.2 ± 16.9 |
| One agent blind | One agent blind | 72.3 ± 4.6 | 66.8 ± 3.9 | 65.4 ± 3.8 | 62.7 ± 4.5 | 60.9 ± 4.2 | 58.2 ± 7.1 |
| Each agent blind | One agent blind | 71.8 ± 2.6 | 69.7 ± 1.6 | 69.5 ± 0.9 | 67.7 ± 1.6 | 67.4 ± 2.9 | 65.0 ± 3.7 |
| No blindness | Each agent blind | 69.7 ± 2.5 | 49.0 ± 13.2 | 30.0 ± 21.3 | 15.4 ± 20.2 | 6.7 ± 15.6 | −1.9 ± 9.3 |
| One agent blind | Each agent blind | 72.3 ± 4.6 | 62.3 ± 6.3 | 51.5 ± 7.5 | 32.2 ± 11.6 | 16.0 ± 6.4 | 3.4 ± 2.6 |
| Each agent blind | Each agent blind | 71.8 ± 2.6 | 66.7 ± 4.4 | 61.7 ± 4.5 | 43.1 ± 8.5 | 20.2 ± 10.0 | 5.7 ± 1.4 |

![Evaluation reward against blindness probability for each training condition, after 100 iterations](experiments/robustness_100iters.png)

**After 25 training iterations:**

| Trained with | Evaluated with | p = 0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|---|
| No blindness | One agent blind | 56.3 ± 12.9 | 53.2 ± 11.1 | 51.6 ± 10.3 | 47.5 ± 9.5 | 44.7 ± 10.4 | 41.1 ± 12.0 |
| One agent blind | One agent blind | 38.5 ± 12.3 | 40.4 ± 12.6 | 41.3 ± 13.0 | 41.5 ± 13.6 | 39.6 ± 12.8 | 35.9 ± 13.3 |
| Each agent blind | One agent blind | 15.2 ± 11.3 | 15.5 ± 12.0 | 14.2 ± 11.9 | 12.8 ± 11.1 | 11.5 ± 11.4 | 10.8 ± 10.4 |
| No blindness | Each agent blind | 56.3 ± 12.9 | 44.8 ± 13.6 | 31.6 ± 17.2 | 8.3 ± 13.7 | 0.1 ± 11.1 | −5.4 ± 7.1 |
| One agent blind | Each agent blind | 38.5 ± 12.3 | 38.3 ± 12.4 | 30.9 ± 9.0 | 17.6 ± 6.8 | 7.1 ± 2.3 | −2.5 ± 1.6 |
| Each agent blind | Each agent blind | 15.2 ± 11.3 | 9.6 ± 11.5 | 8.8 ± 11.4 | 4.5 ± 7.7 | 0.2 ± 5.0 | −3.0 ± 5.4 |

![Evaluation reward against blindness probability for each training condition, after 25 iterations](experiments/robustness.png)

What this shows:

- **Training with blindness makes the team more robust, once trained.** After 100 iterations, the more blindness a policy saw in training, the better it holds up: at 10% each-agent blindness, the each-agent-trained policy keeps 86% of its unblinded reward, the one-agent-trained policy 71%, and the no-blindness policy 43%. The blindness-trained policies also vary far less between seeds.
- **It costs nothing when everyone can see.** With no blindness at evaluation, all three policies score about the same after 100 iterations (70–72).
- **But it slows learning.** After 25 iterations the blindness-trained policies were still well behind, which made blindness training look harmful. Judging robustness from a short training run would have given the opposite answer.
- **Many blind agents at once is still hard.** From 30% each-agent blindness, every policy keeps less than a third of its reward.

To reproduce: `python experiments/run_robustness.py . <seed> experiments/robustness_seed<seed>.json [n_iters]` for each seed (about 13 minutes per seed on one CPU core at 25 iterations, 40 at 100), merge the seed files into `experiments/robustness_results.json` (or `robustness_100iters_results.json`), then `python experiments/plot_robustness.py [results file]`.

## 🚀 Getting Started

### Prerequisites

Python 3.11. The script uses the TorchRL API from late 2024, so `requirements.txt` pins the versions it was checked with.

### Installation

```bash
git clone https://github.com/morganwhite13/Improving-MARL-Robustness-for-Agent-Blindness-in-VMAS.git
cd Improving-MARL-Robustness-for-Agent-Blindness-in-VMAS
pip install -r requirements.txt

# Run the tests
pytest -q tests/
```

### Quick Start

```python
# Run the full set of experiments
python "MARL Robustness for Agent Blindness in VMAS.py"

# The script will train agents on multiple blindness scenarios
# and generate performance plots automatically
```

### Training Custom Scenarios

```python
from torchrl.envs import TransformedEnv
from blindness import BlindOneRandomAgentIfProbabilityForJSteps
# train_environment_variables is defined in the main script, which runs every
# experiment when executed; experiments/run_exp1.py shows how to load the
# function and environments without running them

# Create environment with custom blindness
env = VmasEnv(scenario="balance", n_agents=3, ...)
env = TransformedEnv(
    env, 
    BlindOneRandomAgentIfProbabilityForJSteps(
        n_agents=3, 
        blind_prob=0.1,  # 10% chance per step
        max_blind_steps=10
    )
)

# Train with optimal hyperparameters
rewards, policy = train_environment_variables(
    env=env,
    description="Custom Blindness Test",
    norm=True,
    clipVal=0.3,
    batchSize=100,
    numEpochs=30
)

# Visualize trained policy
with torch.no_grad():
    env.rollout(
        max_steps=200,
        policy=policy,
        callback=lambda env, _: env.render()
    )
```

## 🏗️ Architecture

### MAPPO Algorithm

**Multi-Agent Proximal Policy Optimization** with:
- **Centralized critic**: Observes all agent states during training
- **Decentralized actors**: Each agent acts from local observations only
- **Parameter sharing**: Single policy network shared across all agents
- **PPO clipping**: Prevents drastic policy updates for training stability

### Network Architecture

```
Policy Network (Actor):
├── Input: Agent observation (n_obs_per_agent)
├── Hidden: 2 layers × 256 units (Tanh activation)
├── Output: 2 × n_actions (mean & std for continuous actions)
└── Distribution: TanhNormal (bounded continuous actions)

Critic Network:
├── Input: All agent observations (centralized)
├── Hidden: 2 layers × 256 units (Tanh activation)
└── Output: 1 state value per agent
```

### Training Pipeline

1. **Data Collection**: Parallel environments gather experiences
2. **GAE Computation**: Generalized Advantage Estimation for value targets
3. **Replay Buffer**: Store and sample minibatches
4. **PPO Updates**: Multiple optimization epochs per batch
5. **Policy Update**: Sync collector with trained policy weights

## 📁 Repository Structure

```
Improving-MARL-Robustness-for-Agent-Blindness-in-VMAS/
├── MARL Robustness for Agent Blindness in VMAS.py   # Experiments and MAPPO training
├── blindness.py                                     # Blindness transforms
├── tests/test_blindness.py                          # Tests for the transforms
├── experiments/                                     # Re-runs and robustness test: scripts, results, plots
├── .github/workflows/tests.yml                      # Runs the tests in CI
├── README.md                                        # This file
├── LICENSE                                          # MIT License
└── requirements.txt                                 # Python dependencies
```

## 🔬 Experiments Conducted

The project includes comprehensive experiments across 7 dimensions:

1. **Baseline Comparison** - 7 blindness scenarios vs. normal environment
2. **Normalization Impact** - Advantage normalization on/off
3. **PPO Clipping** - 6 clip values (0.01 to 0.75)
4. **Batch Size** - 7 sizes (10 to 1000)
5. **Training Epochs** - 6 values (5 to 50)
6. **Blindness Probability** - 6 probabilities (0.01 to 0.50)
7. **Blindness Duration** - 7 max durations (1 to 20 steps)
8. **Agent Count** - 6 team sizes (2 to 10 agents)
9. **Scenario Generalization** - 5 VMAS scenarios
10. **Combined Optimization** - Best hyperparameter configurations

**Total Training Runs**: 50+ experiments, each 25 iterations × 6,000 frames

## 📈 Performance Metrics

### Robustness Thresholds

From the robustness test after 100 iterations, share of unblinded reward kept:

| Blindness at evaluation | Trained without blindness | Trained with each-agent blindness |
|---|---|---|
| One agent, p = 0.5 | 62% | 91% |
| Each agent, p = 0.1 | 43% | 86% |
| Each agent, p = 0.2 | 22% | 60% |
| Each agent, p = 0.3 | 10% | 28% |

The original single-seed runs had reported >70% up to p = 0.2 and a threshold near p = 0.3.

### Duration Sensitivity

- **1-5 steps**: Minimal impact, agents compensate effectively
- **7-10 steps**: Moderate degradation, coordination suffers
- **20+ steps**: Severe disruption, often unrecoverable

## 🛠️ Technical Stack

- **Framework**: TorchRL, VMAS (Vectorized Multi-Agent Simulator)
- **Deep Learning**: PyTorch 1.13+
- **Algorithm**: MAPPO (Multi-Agent PPO)
- **Environment**: VMAS Balance scenario + custom transforms
- **Optimization**: Adam optimizer with gradient clipping
- **Value Estimation**: Generalized Advantage Estimation (GAE)

## 🔮 Future Work

Potential extensions and improvements:

- **Communication protocols**: Allow agents to signal blindness states
- **Partial blindness**: Noisy observations instead of complete occlusion
- **Multi-algorithm comparison**: Implement MADDPG, QMIX for benchmarking
- **Curriculum learning**: Gradually increase blindness difficulty
- **Meta-learning**: Train agents to adapt quickly to new blindness patterns
- **Real robot deployment**: Test on physical drone/robot swarms
- **Adversarial blindness**: Opponent strategically targets agents
- **Multi-task training**: Train across multiple VMAS scenarios simultaneously

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Matteo Bettini** and **Prorok Lab** (University of Cambridge) for VMAS framework
- **PyTorch and TorchRL teams** for excellent deep RL tools
- **Yu et al. (2022)** for "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"

## 📞 Contact & Support

- [**Portfolio**](https://morganwhite13.github.io/)
- [**Email**](morgan13white@icloud.com)
- [**LinkedIn**](https://www.linkedin.com/in/morgan-white-95b245237/)

---
