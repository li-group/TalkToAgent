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
    def __init__(self, agent, env, env_params, new_reward_f, out_features):
        self.env = env
        self.env_params = env_params
        self.actor = agent.actor
        self.critic = agent.critic
        self.new_reward_f = new_reward_f
        self.out_features = out_features # new_reward_f에서 get

        self.replay_buffer = {}
        self.batch_size = 64

    def get_rollout(self):
        observations = np.zeros((self.env.N+1, self.env.Nx))
        actions = np.zeros((self.env.N, self.env.Nu))
        rewards = np.zeros((self.env.N, self.out_features))

        o, r = self.env.reset()
        observations[0,:] = (o + 1) * (
                self.env.observation_space_base.high - self.env.observation_space_base.low
        ) / 2 + self.env.observation_space_base.low  # Descaling process

        for i in range(self.env.N - 1):
            a, _s = self.actor.predict(
                o, deterministic=True
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

        a, _s = self.actor.predict(o, deterministic=True)
        actions[self.env.N - 1, :] = (a + 1) * (
                self.env.env_params["a_space"]["high"]
                - self.env.env_params["a_space"]["low"]
        ) / 2 + self.env.env_params["a_space"]["low"]

        # for i in range(self.env.N):
        #     a, _s = self.actor.predict(
        #         o, deterministic=True
        #     )  # TODO: Deterministic vs. Stochastic?
        #
        #     o, r, term, trunc, info = self.env.step(a)
        #     rewards[i, :] = self.new_reward_f(self.env, o, a, con=None)
        #
        #     actions[i, :] = (a + 1) * (
        #             self.env.env_params["a_space"]["high"]
        #             - self.env.env_params["a_space"]["low"]
        #     ) / 2 + self.env.env_params["a_space"]["low"]
        #     observations[i + 1,:] = (o + 1) * (
        #             self.env.observation_space_base.high - self.env.observation_space_base.low
        #     ) / 2 + self.env.observation_space_base.low

        self.replay_buffer['actions'] = actions
        self.replay_buffer['observations'] = observations[:-1,:]
        self.replay_buffer['next_observations'] = observations[1:,:]
        self.replay_buffer['rewards'] = rewards
        return self.replay_buffer

    # Defining critic function
    def decompose_critic(self, critic):
        self.dcritic = deepcopy(critic)

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
        self.dcritic.qf0 = new_qf0
        self.dcritic.q_networks[0] = new_qf0

        return self.dcritic

    def train_dcritic(self, dcritic, env):
        """
        Train decomposed critic with rollout data
        Args:
            dcritic:
            env:
        Returns:
        """
        # Replay buffer sample
        sample = self._sample(self.replay_buffer, self.batch_size)
        states = sample['states']
        actions = sample['actions']
        rewards = sample['rewards']
        next_states = sample['next_states']
        dones = sample['dones']

        # Compute the next Q values using the target values
        with torch.no_grad():
            next_actions = self.target_actor(next_states, deterministic=True)
            next_q = self.target_critic(torch.cat([next_states, next_actions], dim=-1))
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute critic loss & Optimize the critic networks
        current_q = self.critic(torch.cat([states, actions], dim=-1))
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_mag)
        self.critic_optimizer.step()

        # Compute actor loss & Optimize the actor network
        actor_actions = self.actor(states, deterministic=True)
        actor_loss = self.critic(torch.cat([states, actor_actions], dim=-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_mag)
        self.actor_optimizer.step()

        # Soft update the target networks
        for to_model, from_model in zip(self.target_critic.parameters(), self.critic.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        for to_model, from_model in zip(self.target_actor.parameters(), self.actor.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        loss = np.array([critic_loss.detach().cpu().item(), actor_loss.detach().cpu().item()])

        return loss

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