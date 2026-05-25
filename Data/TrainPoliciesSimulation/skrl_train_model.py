import gymnasium as gym
from gymnasium.envs.registration import register
import torch
import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import numpy as np

class ActionSmoothingWrapper(gym.Wrapper):
    def __init__(self, env, alpha=0.15):
        super().__init__(env)
        self.alpha = alpha
        self.last_u = np.zeros(env.action_space.shape, dtype=np.float32)

    def __getattr__(self, name):           # forward everything else
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_u.fill(0)
        return obs, info

    def step(self, action):
        a_np = np.asarray(action, dtype=np.float32)
        smoothed = (1 - self.alpha) * self.last_u + self.alpha * a_np
        self.last_u = smoothed
        # keep type consistent
        smoothed_out = torch.as_tensor(smoothed, dtype=action.dtype, device=action.device) \
                       if torch.is_tensor(action) else smoothed
        return self.env.step(smoothed_out)


class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)



# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

class ActorONNX(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.linear_layer_1 = nn.Linear(obs_dim, 32)
        self.linear_layer_2 = nn.Linear(32, 32)
        self.dense1_bn = nn.Identity(32)
        self.dense2_bn = nn.Identity(32)
        self.action_layer = nn.Linear(32, act_dim)
        self.log_std_parameter = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        x = F.relu((self.linear_layer_1(x)))
        x = F.relu((self.linear_layer_2(x)))
        action = torch.tanh(self.action_layer(x))
        return action  # or torch.cat([action, self.log_std_parameter.expand_as(action)], dim=-1)
    
# define models (stochastic and deterministic models) using mixins
class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.linear_layer_1 = nn.Linear(self.num_observations, 32)
        self.linear_layer_2 = nn.Linear(32, 32)
        self.dense1_bn = nn.Identity(32)
        self.dense2_bn = nn.Identity(32)
        self.action_layer = nn.Linear(32, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = F.relu((self.linear_layer_1(inputs["states"])))
        x = F.relu((self.linear_layer_2(x)))
        # Pendulum-v1 action_space is -2 to 2
        return 1* torch.tanh(self.action_layer(x)), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 32)
        self.linear_layer_2 = nn.Linear(32, 32)
        self.linear_layer_3 = nn.Linear(32, 1)
        self.dense1_bn = nn.Identity(32)
        self.dense2_bn = nn.Identity(32)


    def compute(self, inputs, role):
        x = F.relu((self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1))))
        x = F.relu((self.linear_layer_2(x)))
        return self.linear_layer_3(x), {}
        


# load and wrap the gymnasium environment.
# note: the environment version may change depending on the gymnasium version

if __name__ == '__main__':
    register(
        id="ForceControl_BB_Lyap-v0",
        entry_point="src.envs:ForceControlBenchEnvironment_BB_Lyap"
    )

    register(
        id="ForceControl_BB_Lyap_RL_control-v0",
        entry_point="src.envs:ForceControlBenchEnvironment_BB_Lyap_FL_Rl_control"
    )
    
    env = gym.make("ForceControl_BB_Lyap_RL_control-v0", no_random = False, only_control = False, 
                   safe_control_strategy='None'
                   ,Q = 1, R = 5, R2 = 5,smooth_factor = 0.1)
    
    
    #env = ActionSmoothingWrapper(env, alpha=0.15)   # tune α ≈ 0.1–0.2
    env = FrameSkipWrapper(env, skip=6)
    env = wrap_env(env)

    device = env.device


    # instantiate a memory as experience replay
    memory = RandomMemory(memory_size=1000000, num_envs=env.num_envs, device=device, replacement=False)


    # instantiate the agent's models (function approximators).
    # SAC requires 5 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models

    models = {}
    models["policy"] = Actor(env.observation_space, env.action_space, device, clip_actions=True)
    models["critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["critic_2"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

    # initialize models' parameters (weights and biases)
    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg["discount_factor"] = 0.98
    cfg["batch_size"] = 256
    cfg["random_timesteps"] = 50000
    cfg["learning_starts"] = 50000
    cfg["learn_entropy"] = True
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 75
    cfg["experiment"]["checkpoint_interval"] = 2000

    #cfg["initial_entropy_value"] = 0.05   #   ←  low temperature
    #cfg["learn_entropy"]        = False   #   ←  freeze it (no tuning)

    cfg["initial_entropy_value"] = 0.5
    #cfg["learn_entropy"] = False       # ← freeze it at 0.2


    cfg["experiment"]["write_interval"] = 100
    cfg["experiment"]["checkpoint_interval"] = 5000
    cfg["experiment"]["directory"] = "runs/torch/HyD"
    cfg["experiment"]["name"] = "SAC_ForceControlBench"
    cfg["experiment"]["write_tensorboard"] = True


    agent = SAC(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)
    

    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 500000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

    # start training
    trainer.train()
