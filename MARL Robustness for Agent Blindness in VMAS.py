# Torch
import torch

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

# Tensordict modules
from torch import multiprocessing

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
torch.manual_seed(0)
from matplotlib import pyplot as plt
from tqdm import tqdm

import random
from torchrl.envs.transforms import Transform
from tensordict.tensordict import TensorDictBase


# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
vmas_device = device  # The device where the simulator is run (VMAS can run on GPU)

# Sampling
frames_per_batch = 6_000  # Number of team frames collected per training iteration
n_iters = 25  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 20  # Number of optimization steps per training iteration
minibatch_size = 200  # Size of the mini-batches in each optimization step
lr = 3e-4  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients

# PPO
clip_epsilon = 0.2  # clip value for PPO loss
gamma = 0.9  # discount factor
lmbda = 0.9  # lambda for generalised advantage estimation
entropy_eps = 1e-4  # coefficient of the entropy term in the PPO loss

max_steps = 200  # Episode steps before done
num_vmas_envs = (
    frames_per_batch // max_steps
)  # Number of vectorized envs. frames_per_batch should be divisible by this number
scenario_name = "balance"
n_agents = 3


env = VmasEnv(
    scenario=scenario_name,
    num_envs=num_vmas_envs,
    continuous_actions=True,  # VMAS supports both continuous and discrete actions
    max_steps=max_steps,
    device=vmas_device,
    # Scenario kwargs
    n_agents=n_agents,  # These are custom kwargs that change for each VMAS scenario, see the VMAS repo to know more.
)


print("action_spec:", env.full_action_spec)
print("reward_spec:", env.full_reward_spec)
print("done_spec:", env.full_done_spec)
print("observation_spec:", env.observation_spec)

print("action_keys:", env.action_keys)
print("reward_keys:", env.reward_keys)
print("done_keys:", env.done_keys)




env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)


class BlindOneRandomAgentEveryStep(Transform):
    def __init__(self, n_agents: int):
        super().__init__()
        self._n_agents = n_agents

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
            next_tensordict[("agents", "observation")][..., random.randrange(self._n_agents), :] = 0
            return next_tensordict
        
class BlindAllAgentsEveryStep(Transform):
    def __init__(self, n_agents: int, blind_prob: float = 0.1):
        super().__init__()
        self._n_agents = n_agents
        self._blind_prob = blind_prob  # Probability of blinding an agent

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
            for agent_idx in range(self._n_agents):
                # Blind the agent by setting their observation to zero
                next_tensordict[("agents", "observation")][..., random.randrange(self._n_agents), :] = 0
            return next_tensordict


class BlindOneRandomAgentIfProbability(Transform):
    def __init__(self, n_agents: int, blind_prob: float = 0.1):
        super().__init__()
        self._n_agents = n_agents
        self._blind_prob = blind_prob  # Probability of blinding an agent

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
            if random.random() < self._blind_prob:
                # Blind the agent by setting their observation to zero
                next_tensordict[("agents", "observation")][..., random.randrange(self._n_agents), :] = 0
            return next_tensordict



class BlindRandomAgentsIfProbability(Transform):
    def __init__(self, n_agents: int, blind_prob: float = 0.1):
        super().__init__()
        self._n_agents = n_agents
        self._blind_prob = blind_prob  # Probability of blinding an agent

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
            for agent_idx in range(self._n_agents):
                if random.random() < self._blind_prob:
                    # Blind the agent by setting their observation to zero
                    next_tensordict[("agents", "observation")][..., random.randrange(self._n_agents), :] = 0
            return next_tensordict




class BlindOneRandomAgentIfProbabilityForJSteps(Transform):
    def __init__(self, n_agents: int, blind_prob: float = 0.1, max_blind_steps: int = 10):
        super().__init__()
        self._n_agents = n_agents
        self._blind_prob = blind_prob  # Probability of blinding an agent
        self.max_blind_steps = max_blind_steps  # Max number of steps the agent can be blinded for
        # Dictionary to keep track of blind durations for each agent
        self.blind_remaining = {agent_idx: 0 for agent_idx in range(n_agents)}

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        blindedAgentExists = False
        for agent_idx in range(self._n_agents):
            if self.blind_remaining[agent_idx] > 0:
                # If the agent is already blinded, decrement the counter
                self.blind_remaining[agent_idx] -= 1
                next_tensordict[("agents", "observation")][..., agent_idx, :] = 0
                blindedAgentExists = True
            elif not blindedAgentExists and random.random() < self._blind_prob:
                # Blind the agent randomly and set the blind duration
                blind_duration = random.randint(1, self.max_blind_steps)
                self.blind_remaining[agent_idx] = blind_duration - 1  # Subtract 1 because we're applying the first step of blindness now
                next_tensordict[("agents", "observation")][..., agent_idx, :] = 0
        return next_tensordict


class BlindRandomAgentsIfProbabilityForJSteps(Transform):
    def __init__(self, n_agents: int, blind_prob: float = 0.1, max_blind_steps: int = 10):
        super().__init__()
        self._n_agents = n_agents
        self._blind_prob = blind_prob  # Probability of blinding an agent
        self.max_blind_steps = max_blind_steps  # Max number of steps the agent can be blinded for
        # Dictionary to keep track of blind durations for each agent
        self.blind_remaining = {agent_idx: 0 for agent_idx in range(n_agents)}

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        for agent_idx in range(self._n_agents):
            if self.blind_remaining[agent_idx] > 0:
                # If the agent is already blinded, decrement the counter
                self.blind_remaining[agent_idx] -= 1
                next_tensordict[("agents", "observation")][..., agent_idx, :] = 0
            elif random.random() < self._blind_prob:
                # Blind the agent randomly and set the blind duration
                blind_duration = random.randint(1, 10)
                self.blind_remaining[agent_idx] = blind_duration - 1  
                # Subtract 1 because we're applying the first step of blindness now
                next_tensordict[("agents", "observation")][..., agent_idx, :] = 0
        return next_tensordict


env2 = TransformedEnv(
    env,
    BlindOneRandomAgentEveryStep(env.n_agents)
)

env3 = TransformedEnv(
    env,
    BlindAllAgentsEveryStep(env.n_agents)
)


env4 = TransformedEnv(
    env,
    BlindOneRandomAgentIfProbability(env.n_agents, 0.1)
)


env5 = TransformedEnv(
    env,
    BlindRandomAgentsIfProbability(env.n_agents, 0.1)
)


env6 = TransformedEnv(
    env,
    BlindOneRandomAgentIfProbabilityForJSteps(env.n_agents, 0.1, 10)
)



env7 = TransformedEnv(
    env,
    BlindRandomAgentsIfProbabilityForJSteps(env.n_agents, 0.1, 10)
)





def train_environment_variables(env, description, norm=True, clipVal=0.2, batchSize=200, numEpochs=20, n_rollout_steps=20):
    # Environment setup and checks
    

    check_env_specs(env)


    
    rollout = env.rollout(n_rollout_steps)
    # print("rollout of three steps:", rollout)
    # print("Shape of the rollout TensorDict:", rollout.batch_size)



    share_parameters_policy = True

    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[
                -1
            ],  # n_obs_per_agent
            n_agent_outputs=2 * env.action_spec.shape[-1],  # 2 * n_actions_per_agents
            n_agents=env.n_agents,
            centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
            share_params=share_parameters_policy,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )


    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.unbatched_action_spec[env.action_key].space.low,
            "max": env.unbatched_action_spec[env.action_key].space.high,
        },
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )  # we'll need the log-prob for the PPO loss

    share_parameters_critic = True
    mappo = True  # IPPO if False

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,  # 1 value per agent
        n_agents=env.n_agents,
        centralised=mappo,
        share_params=share_parameters_critic,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )



    # print("Running policy:", policy(env.reset()))
    # print("Running value:", critic(env.reset()))



    collector = SyncDataCollector(
        env,
        policy,
        device=vmas_device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )



    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            frames_per_batch, device=device
        ),  # We store the frames_per_batch collected at each iteration
        sampler=SamplerWithoutReplacement(),
        batch_size=batchSize,  # We will sample minibatches of this size
    )


    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=clipVal,
        entropy_coef=entropy_eps,
        normalize_advantage=norm,  # Important to avoid normalizing across the agent dimension
    )
    loss_module.set_keys(  # We have to tell the loss where to find the keys
        reward=env.reward_key,
        action=env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )


    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
    )  # We build GAE
    GAE = loss_module.value_estimator

    optim = torch.optim.Adam(loss_module.parameters(), lr)


    pbar = tqdm(total=n_iters, desc=f"Training in {description}")

    episode_reward_mean_list = []
    for tensordict_data in collector:
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )  # Compute GAE and add it to the data

        data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
        replay_buffer.extend(data_view)

        for _ in range(numEpochs):
            for _ in range(frames_per_batch // batchSize):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_grad_norm
                )  # Optional

                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()

        # Logging
        done = tensordict_data.get(("next", "agents", "done"))
        episode_reward_mean = (
            tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
        )
        episode_reward_mean_list.append(episode_reward_mean)
        pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
        pbar.update()

    
    
    return episode_reward_mean_list, policy





'''
The following Contains the Code for the experiments I ran:

'''

# Environment configurations
environments = [
    (env, "Normal Balance Scenario"),
    (env2, "Blind One Random Agent Every Step"),
    (env3, "Blind All Agents Every Step"),
    (env4, "Blind One Random Agent Randomly"),
    (env5, "Blind Random Agents Randomly"),
    (env6, "Blind One Random Agent Randomly For a Random Number of Steps"),
    (env7, "Blind Random Agents Randomly For a Random Number of Steps")
]
# environments = [
#     (env6, "Blind One Random Agent Randomly For a Random Number of Steps")
# ]
colors = [
    'blue', 'green', 'red', 'cyan', 'magenta', 
    'black', 'gray', 'orange', 
    'purple', 'brown', 'pink', 'lime', 'olive', 
    'chocolate', 'maroon', 'navy', 'teal', 'silver'
]


#Experiment 1 testing reward across different blindness scenarios
results = []
for env, description in environments:
    rewards, policy = train_environment_variables(env, description, norm=False)
    results.append({"reward": rewards, "policy": policy, "env": env, "desc": description})

colorIndex = 0
# Plotting all environments on a single plot
plt.figure(figsize=(10, 6))
# for desc, rewards in all_rewards.items():
for res in results:
    color = colors[colorIndex]
    plt.plot(res["reward"], label=(res["desc"]+" normalization = " + str(res["norm"])), linestyle='solid', color=color)
    colorIndex += 1
plt.xlabel("Training Iterations")
plt.ylabel("Reward")
plt.title("Episode Reward Mean Across Different Experiments")
plt.legend()
plt.show()

# Display all rollouts
for res in results:
    text = (res["desc"])
    print(f"Rollout for {text}:")
    curEnv = res["env"]
    with torch.no_grad():
        curEnv.rollout(
            max_steps=max_steps,
            policy=res["policy"],
            callback=lambda curEnv, _: curEnv.render(),
            auto_cast_to_device=True,
            break_when_any_done=False,
   )





#Experiment testing reward across different blindness scenarios and normalization
norms = [True, False]
results = []
for env, description in environments:
    for norm in norms:
        # rewards, policy = train_environment(env, description)
        rewards, policy = train_environment_variables(env, description, norm)
        # all_rewards[description] = rewards
        # all_policies[description] = policy
        # all_envs[description] = env
        results.append({"reward": rewards, "policy": policy, "env": env, "desc": description, "norm": norm})


colorIndex = 0
# Plotting all environments on a single plot
plt.figure(figsize=(10, 6))
# for desc, rewards in all_rewards.items():
for res in results:
    color = colors[colorIndex]
    if res["norm"]:
        plt.plot(res["reward"], label=(res["desc"]+" normalization = " + str(res["norm"])), linestyle='solid', color=color)
    else:
        plt.plot(res["reward"], label=(res["desc"]+" normalization = " + str(res["norm"])), linestyle='dashed', color=color)
        colorIndex += 1
plt.xlabel("Training Iterations")
plt.ylabel("Reward")
plt.title("Episode Reward Mean Across Different Experiments")
plt.legend()
plt.show()

# Display all rollouts
for res in results:
    text = (res["desc"]+" normalization = " + str(res["norm"]))
    print(f"Rollout for {text}:")
    curEnv = res["env"]
    with torch.no_grad():
        curEnv.rollout(
            max_steps=max_steps,
            policy=res["policy"],
            callback=lambda curEnv, _: curEnv.render(),
            auto_cast_to_device=True,
            break_when_any_done=False,
   )




#Experiment testing reward across different clip values
norms = [True]
clips = [0.01, 0.1, 0.2, 0.3, 0.5, 0.75]
results = []
for env, description in environments:
    for norm in norms:
        for clip in clips:
            # rewards, policy = train_environment(env, description)
            rewards, policy = train_environment_variables(env, description, norm, clip)
            # all_rewards[description] = rewards
            # all_policies[description] = policy
            # all_envs[description] = env
            results.append({"reward": rewards, "policy": policy, "env": env, "desc": description, "norm": norm, "clip": clip})


# Plotting all environments on a single plot
plt.figure(figsize=(10, 6))
# for desc, rewards in all_rewards.items():
for res in results:
    plt.plot(res["reward"], label=(res["desc"]+" Clipping value = " + str(res["clip"])))
plt.xlabel("Training Iterations")
plt.ylabel("Reward")
plt.title("Episode Reward Mean Across Different Environments")
plt.legend()
plt.show()

# Display all rollouts
for res in results:
    text = (res["desc"]+" Clipping value = " + str(res["clip"]))
    print(f"Rollout for {text}:")
    curEnv = res["env"]
    with torch.no_grad():
        curEnv.rollout(
            max_steps=max_steps,
            policy=res["policy"],
            callback=lambda curEnv, _: curEnv.render(),
            auto_cast_to_device=True,
            break_when_any_done=False,
   )
        
        
        
        
#Experiment testing reward across different batch sizes
batchSizes = [10, 50, 100, 200, 300, 500, 1000]
results = []
for env, description in environments:
    for batchSize in batchSizes:
        # rewards, policy = train_environment(env, description)
        rewards, policy = train_environment_variables(env, description, batchSize=batchSize)
        # all_rewards[description] = rewards
        # all_policies[description] = policy
        # all_envs[description] = env
        results.append({"reward": rewards, "policy": policy, "env": env, "desc": description, "batchSize": batchSize})


# Plotting all environments on a single plot
plt.figure(figsize=(10, 6))
# for desc, rewards in all_rewards.items():
for res in results:
    plt.plot(res["reward"], label=(res["desc"]+" batch Size = " + str(res["batchSize"])))
plt.xlabel("Training Iterations")
plt.ylabel("Reward")
plt.title("Episode Reward Mean Across Different Environments")
plt.legend()
plt.show()

# Display all rollouts
for res in results:
    text = (res["desc"]+" batch Size = " + str(res["batchSize"]))
    print(f"Rollout for {text}:")
    curEnv = res["env"]
    with torch.no_grad():
        curEnv.rollout(
            max_steps=max_steps,
            policy=res["policy"],
            callback=lambda curEnv, _: curEnv.render(),
            auto_cast_to_device=True,
            break_when_any_done=False,
   )


        
#Experiment testing reward across different number of epochs
numEpochsList = [5, 10, 15, 20, 30, 50]
results = []
for env, description in environments:
    for numEpochs in numEpochsList:
        rewards, policy = train_environment_variables(env, description, numEpochs=numEpochs)
        results.append({"reward": rewards, "policy": policy, "env": env, "desc": description, "numEpochs": numEpochs})


# Plotting all environments on a single plot
plt.figure(figsize=(10, 6))
# for desc, rewards in all_rewards.items():
for res in results:
    plt.plot(res["reward"], label=(res["desc"]+" number of Epochs = " + str(res["numEpochs"])))
plt.xlabel("Training Iterations")
plt.ylabel("Reward")
plt.title("Episode Reward Mean Across Different Environments")
plt.legend()
plt.show()

# Display all rollouts
for res in results:
    text = (res["desc"]+" number of Epochs = " + str(res["numEpochs"]))
    print(f"Rollout for {text}:")
    curEnv = res["env"]
    with torch.no_grad():
        curEnv.rollout(
            max_steps=max_steps,
            policy=res["policy"],
            callback=lambda curEnv, _: curEnv.render(),
            auto_cast_to_device=True,
            break_when_any_done=False,
   )



#Experiment testing reward across different rollout values
rolloutStepsList = [10, 15, 20, 25, 35, 50, 100]
rolloutResults = []
for env, description in environments:
    for n_rollout_steps in rolloutStepsList:
        # rewards, policy = train_environment(env, description)
        rewards, policy = train_environment_variables(env, description, n_rollout_steps=n_rollout_steps)
        # all_rewards[description] = rewards
        # all_policies[description] = policy
        # all_envs[description] = env
        rolloutResults.append({"reward": rewards, "policy": policy, "env": env, "desc": description, "n_rollout_steps": n_rollout_steps})


# Plotting all environments on a single plot
plt.figure(figsize=(10, 6))
# for desc, rewards in all_rewards.items():
for res in rolloutResults:
    plt.plot(res["reward"], label=(res["desc"]+" Number of Rollout Steps = " + str(res["n_rollout_steps"])))
plt.xlabel("Training Iterations")
plt.ylabel("Reward")
plt.title("Episode Reward Mean Across Different Environments")
plt.legend()
plt.show()

# Display all rollouts
for res in rolloutResults:
    text = (res["desc"]+" Number of Rollout Steps = " + str(res["n_rollout_steps"]))
    print(f"Rollout for {text}:")
    curEnv = res["env"]
    with torch.no_grad():
        curEnv.rollout(
            max_steps=max_steps,
            policy=res["policy"],
            callback=lambda curEnv, _: curEnv.render(),
            auto_cast_to_device=True,
            break_when_any_done=False,
   )




#Experiment testing reward across different blindness probabilties
blindnessProbabilities = [0.01, 0.05, 0.10, 0.2, 0.3, 0.5]
blindnessResults = []
for blindProb in blindnessProbabilities:
    env = TransformedEnv(env, BlindOneRandomAgentIfProbabilityForJSteps(env.n_agents, blindProb, 10))
    description = "Blind One Random Agent Randomly For a Random Number of Steps"
    rewards, policy = train_environment_variables(env, description)
    # print(f"Rewards for {rewards}:")
    for i in range(len(rewards)):
        rewards[i] = rewards[i]*blindProb*5
        
    blindnessResults.append({"reward": rewards, "policy": policy, "env": env, "desc": description, "blindProb": blindProb})


# Plotting all environments on a single plot
plt.figure(figsize=(10, 6))
colorIndex = 0
# for desc, rewards in all_rewards.items():
for res in blindnessResults:
    plt.plot(res["reward"], label=(res["desc"]+" Blindness Probability = " + str(res["blindProb"])), color=colors[colorIndex])
    colorIndex += 1
plt.xlabel("Training Iterations")
plt.ylabel("Reward * Blindness Probability * Number of Expected Blinded Steps")
plt.title("Episode Reward Mean as a function of the blindness Probability Across Different Blindness Probabilities")
plt.legend()
plt.show()

# Display all rollouts
for res in blindnessResults:
    text = (res["desc"]+" Blindness Probability = " + str(res["blindProb"]))
    print(f"Rollout for {text}:")
    curEnv = res["env"]
    with torch.no_grad():
        curEnv.rollout(
            max_steps=max_steps,
            policy=res["policy"],
            callback=lambda curEnv, _: curEnv.render(),
            auto_cast_to_device=True,
            break_when_any_done=False,
   )


#Experiment testing reward across different maximum blindness steps
blindnessSteps = [1, 2, 3, 5, 7, 10, 20]
blindnessStepResults = []
for blindSteps in blindnessSteps:
    env = TransformedEnv(env, BlindOneRandomAgentIfProbabilityForJSteps(env.n_agents, 0.1, blindSteps))
    description = "Blind One Random Agent Randomly For a Random Number of Steps"
    rewards, policy = train_environment_variables(env, description)
    for i in range(len(rewards)):
        rewards[i] = rewards[i]*0.1*blindSteps/2
    blindnessStepResults.append({"reward": rewards, "policy": policy, "env": env, "desc": description, "blindSteps": blindSteps})


# Plotting all environments on a single plot
plt.figure(figsize=(10, 6))
colorIndex = 0
# for desc, rewards in all_rewards.items():
for res in blindnessStepResults:
    plt.plot(res["reward"], label=(res["desc"]+" Max Blindness Steps = " + str(res["blindSteps"])), color=colors[colorIndex])
    colorIndex += 1
plt.xlabel("Training Iterations")
plt.ylabel("Reward * Blindness Probability * Number of Expected Blinded Steps")
plt.title("Episode Reward Mean as a function of the blindness Probability Across Different Blindness Probabilities")
plt.legend()
plt.show()

# Display all rollouts
for res in blindnessStepResults:
    text = (res["desc"]+" Max Blindness Steps = " + str(res["blindSteps"]))
    print(f"Rollout for {text}:")
    curEnv = res["env"]
    with torch.no_grad():
        curEnv.rollout(
            max_steps=max_steps,
            policy=res["policy"],
            callback=lambda curEnv, _: curEnv.render(),
            auto_cast_to_device=True,
            break_when_any_done=False,
   )




#Experiment testing reward across VMAS Scenarios
scenarioNames = ["balance", "wheel", "give_way", "ball_passage", "multi_give_way"]
scenarioResults = []
for scenarioName in scenarioNames:
    env = VmasEnv(
        scenario=scenarioName,
        num_envs=num_vmas_envs,
        continuous_actions=True,  # VMAS supports both continuous and discrete actions
        max_steps=max_steps,
        device=vmas_device,
        # Scenario kwargs
        n_agents=n_agents,  # These are custom kwargs that change for each VMAS scenario, see the VMAS repo to know more.
    )
    env = TransformedEnv(env, RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]))
    env = TransformedEnv(env, BlindOneRandomAgentIfProbabilityForJSteps(env.n_agents, 0.1, 10))
    description = "Randomly Blind Agent Random Steps"
    rewards, policy = train_environment_variables(env, description)
    scenarioResults.append({"reward": rewards, "policy": policy, "env": env, "desc": description, "scenarioName": scenarioName})


# Plotting all environments on a single plot
plt.figure(figsize=(10, 6))
# for desc, rewards in all_rewards.items():
for res in scenarioResults:
    plt.plot(res["reward"], label=(res["desc"]+" for scenario = " + str(res["scenarioName"])))
plt.xlabel("Training Iterations")
plt.ylabel("Reward")
plt.title("Episode Reward Mean Across Different Scenarios")
plt.legend()
plt.show()

# Display all rollouts
for res in scenarioResults:
    text = (res["desc"]+" for scenario = " + str(res["scenarioName"]))
    print(f"Rollout for {text}:")
    curEnv = res["env"]
    with torch.no_grad():
        curEnv.rollout(
            max_steps=max_steps,
            policy=res["policy"],
            callback=lambda curEnv, _: curEnv.render(),
            auto_cast_to_device=True,
            break_when_any_done=False,
   )



#Experiment testing reward across different numbers of agents
numAgents = [2, 3, 4, 5, 7, 10]
numAgentsResults = []
for n_agents in numAgents:
    env = VmasEnv(
        scenario=scenario_name,
        num_envs=num_vmas_envs,
        continuous_actions=True,  # VMAS supports both continuous and discrete actions
        max_steps=max_steps,
        device=vmas_device,
        # Scenario kwargs
        n_agents=n_agents,  # These are custom kwargs that change for each VMAS scenario, see the VMAS repo to know more.
    )
    env = TransformedEnv(env, RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]))
    env = TransformedEnv(env, BlindOneRandomAgentIfProbabilityForJSteps(env.n_agents, 0.1, 10))
    description = "Randomly Blind Agent Random Steps"
    rewards, policy = train_environment_variables(env, description)
    numAgentsResults.append({"reward": rewards, "policy": policy, "env": env, "desc": description, "n_agents": n_agents})


# Plotting all environments on a single plot
plt.figure(figsize=(10, 6))
# for desc, rewards in all_rewards.items():
for res in numAgentsResults:
    plt.plot(res["reward"], label=(res["desc"]+" for Number of Agents = " + str(res["n_agents"])))
plt.xlabel("Training Iterations")
plt.ylabel("Reward")
plt.title("Episode Reward Mean Across Different Number of Agents")
plt.legend()
plt.show()

# Display all rollouts
for res in numAgentsResults:
    text = (res["desc"]+" for Number of Agents = " + str(res["n_agents"]))
    print(f"Rollout for {text}:")
    curEnv = res["env"]
    with torch.no_grad():
        curEnv.rollout(
            max_steps=max_steps,
            policy=res["policy"],
            callback=lambda curEnv, _: curEnv.render(),
            auto_cast_to_device=True,
            break_when_any_done=False,
   )





#Experiment testing reward across different Algorithm Parameters
env = env6
description = "Randomly Blind Agent for Random Steps"
# algorithmSetUps = [{"norm":False, "clipVal":0.2, "batchSize":200, "numEpochs":20}, {"norm":True, "clipVal":0.3, "batchSize":100, "numEpochs":30}, {"norm":True, "clipVal":0.3, "batchSize":50, "numEpochs":30}, {"norm":True, "clipVal":0.3, "batchSize":100, "numEpochs":20}, {"norm":True, "clipVal":0.2, "batchSize":100, "numEpochs":30}]
algorithmSetUps = [{"norm":True, "clipVal":0.3, "batchSize":100, "numEpochs":30}]
algResults = []

for algs in algorithmSetUps:
    rewards, policy = train_environment_variables(env, description, norm=algs["norm"], clipVal=algs["clipVal"], batchSize=algs["batchSize"], numEpochs=algs["numEpochs"])
    algResults.append({"reward": rewards, "policy": policy, "env": env, "desc": description, "norm":algs["norm"], "clipVal":algs["clipVal"], "batchSize":algs["batchSize"], "numEpochs":algs["numEpochs"]})


# Plotting all environments on a single plot
colorIndex = 0
plt.figure(figsize=(10, 6))
# for desc, rewards in all_rewards.items():
for res in algResults:
    plt.plot(res["reward"], label=(res["desc"]+" for norm = " + str(res["norm"])+" clipVal = " + str(res["clipVal"])+" batchSize = " + str(res["batchSize"])+" numEpochs = " + str(res["numEpochs"])), color=colors[colorIndex])
    colorIndex += 1
plt.xlabel("Training Iterations")
plt.ylabel("Reward")
plt.title("Episode Reward Mean Across Different Algorithm SetUps")
plt.legend()
plt.show()
# Display all rollouts
for res in algResults:
    text = (res["desc"]+" for norm = " + str(res["norm"])+" clipVal = " + str(res["clipVal"])+" batchSize = " + str(res["batchSize"])+" numEpochs = " + str(res["numEpochs"]))
    print(f"Rollout for {text}:")
    curEnv = res["env"]
    with torch.no_grad():
        curEnv.rollout(
            max_steps=max_steps,
            policy=res["policy"],
            callback=lambda curEnv, _: curEnv.render(),
            auto_cast_to_device=True,
            break_when_any_done=False,
   )
        




