import torch
import torch.nn as nn
from stable_baselines3 import PPO, DDPG, SAC
from typing import Any, Optional, TypeVar, Union

import numpy as np
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.td3.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, TD3Policy

SelfDDPG = TypeVar("SelfDDPG", bound="DDPG")

from copy import deepcopy

class Reward_decompose:
    """
    1. Reward를 분해. (일단은 manually, 최종적으로는 자동화)
    2. multi-output critic (nn.torch 기반)학습
    3. 이때 actor는 가만히 두고 critic만 TD error로 진행
    """
    def __init__(self, agent, env, env_params, new_reward_f, out_features, deterministic = True):
        self.env = env
        self.env_params = env_params
        self.actor = agent.actor
        self.actor_target = agent.actor_target
        self.critic = agent.critic
        self.gamma = agent.gamma
        self.tau = agent.tau
        self.policy_delay = agent.policy_delay
        self.new_reward_f = new_reward_f
        self.out_features = out_features # new_reward_f에서 get

        self.target_policy_noise = agent.target_policy_noise
        self.target_noise_clip = agent.target_noise_clip

        self.batch_size = 64
        self.deterministic = deterministic

        # Training dcritic
        self.replay_buffer = self._get_rollout()
        self.dcritic = self._decompose_critic(self.critic)
        self.dcritic_target = deepcopy(self.dcritic)

        self._train_dcritic(n_updates = 1000)

    def explain(self, t_query):
        return

    def _get_rollout(self):
        # TODO: Stochastic해서 multiple rollout을 또 얻어낼 수 있는거 아니야?
        replay_buffer = {}
        observations = np.zeros((self.env.N+1, self.env.Nx))
        actions = np.zeros((self.env.N, self.env.Nu))
        rewards = np.zeros((self.env.N, self.out_features))

        o, r = self.env.reset()
        observations[0,:] = (o + 1) * (
                self.env.observation_space_base.high - self.env.observation_space_base.low
        ) / 2 + self.env.observation_space_base.low  # Descaling process

        for i in range(self.env.N - 1):
            a, _s = self.actor.predict(
                o, deterministic=self.deterministic
            )  # TODO: Deterministic vs. Stochastic?

            o, r, term, trunc, info = self.env.step(a)
            rewards[i, :] = self.new_reward_f(self.env, o, a, con=None)

            actions[i, :] = (a + 1) * (
                    self.env.env_params["a_space"]["high"]
                    - self.env.env_params["a_space"]["low"]
            ) / 2 + self.env.env_params["a_space"]["low"]
            observations[i + 1,:] = (o + 1) * (
                    self.env.observation_space_base.high - self.env.observation_space_base.low
            ) / 2 + self.env.observation_space_base.low

        a, _s = self.actor.predict(o, deterministic=self.deterministic)
        actions[self.env.N - 1, :] = (a + 1) * (
                self.env.env_params["a_space"]["high"]
                - self.env.env_params["a_space"]["low"]
        ) / 2 + self.env.env_params["a_space"]["low"]

        dones = np.zeros((self.env.N, 1))
        dones[-1,:] = 1

        replay_buffer['actions'] = actions
        replay_buffer['observations'] = observations[:-1,:]
        replay_buffer['next_observations'] = observations[1:,:]
        replay_buffer['rewards'] = rewards
        replay_buffer['dones'] = dones
        return replay_buffer

    def _decompose_critic(self, critic):
        dcritic = deepcopy(critic)

        old_model = critic.qf0
        layers = list(old_model.children())

        if isinstance(layers[-1], nn.Linear):
            in_features = layers[-1].in_features
            new_output_layer = nn.Linear(in_features, self.out_features)
        else:
            raise ValueError("Last layer is not nn.Linear!")

        # Freeze feature extractor part, for fine-tuning
        for layer in layers[:-1]:
            for param in layer.parameters():
                param.requires_grad = False

        # Constructing decomposed critic
        new_qf0 = nn.Sequential(*layers[:-1], new_output_layer)
        dcritic.qf0 = new_qf0
        dcritic.q_networks[0] = new_qf0

        return dcritic

    def _train_dcritic(self, n_updates):
        """
        Train decomposed critic with rollout data
        Args:
            dcritic:
            env:
        Returns:
        """
        actor_losses, critic_losses = [], []
        for n in range(n_updates):
            # Sample replay buffer
            sample = self._sample(self.replay_buffer, self.batch_size)
            actions = torch.tensor(sample['actions'], dtype=torch.float32)
            observations = torch.tensor(sample['observations'], dtype=torch.float32)
            rewards = torch.tensor(sample['rewards'], dtype=torch.float32)
            next_observations = torch.tensor(sample['next_observations'], dtype=torch.float32)
            dones = torch.tensor(sample['dones'], dtype=torch.float32)

            with torch.no_grad():
                # The actions are not being explored in training dcritics
                next_actions, _ = self.actor.predict(next_observations, deterministic=self.deterministic)
                next_actions = torch.tensor(next_actions)

                # Compute the next Q-values
                next_q_values = torch.cat(self.dcritic_target(next_observations, next_actions), dim=1)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.dcritic(observations, actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, torch.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.dcritic.optimizer.zero_grad()
            critic_loss.backward()
            self.dcritic.optimizer.step()

            # Delayed policy updates
            if n % self.policy_delay == 0:
                # We do not train actor network since it's already been trained.
                polyak_update(self.dcritic.parameters(), self.dcritic_target.parameters(), self.tau)

    def _sample(self, replay_buffer, batch_size):
        N = replay_buffer['actions'].shape[0]
        indices = np.random.choice(N, size=batch_size, replace=False)

        batch = {key: val[indices] for key, val in replay_buffer.items()}
        return batch


def four_tank_reward_decomposed(env, x, u, con):
    Sp_i = 0
    R = 0.1
    costs = []
    if not hasattr(env, 'u_prev'):
        env.u_prev = u

    for k in env.env_params["SP"]:
        i = env.model.info()["states"].index(k)
        SP = env.SP[k]

        o_space_low = env.env_params["o_space"]["low"][i]
        o_space_high = env.env_params["o_space"]["high"][i]

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        r_scale = env.env_params.get("r_scale", {})

        cost_k = (np.sum(x_normalized - setpoint_normalized[env.t]) ** 2) * r_scale.get(k, 1)
        costs.append(cost_k)

        Sp_i += 1

    u_normalized = (u - env.env_params["a_space"]["low"]) / (
            env.env_params["a_space"]["high"] - env.env_params["a_space"]["low"]
    )
    u_prev_norm = (env.u_prev - env.env_params["a_space"]["low"]) / (
            env.env_params["a_space"]["high"] - env.env_params["a_space"]["low"]
    )
    env.u_prev = u

    # Add the control cost
    cost_u = np.sum(R * (u_normalized - u_prev_norm) ** 2)
    costs.append(cost_u)

    rs = [-cost for cost in costs]
    return rs