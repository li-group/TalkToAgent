import torch
import torch.nn as nn
from stable_baselines3 import PPO, DDPG, SAC
from typing import Any, Optional, TypeVar, Union

import numpy as np
from gymnasium import spaces
from torch.nn import functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.td3.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, TD3Policy

SelfDDPG = TypeVar("SelfDDPG", bound="DDPG")

from copy import deepcopy

class D3PG:
    """
    1. Reward를 분해. (일단은 manually, 최종적으로는 자동화)
    2. multi-output critic (nn.torch 기반)학습
    3. 이때 actor는 가만히 두고 critic만 TD error로 진행
    """
    def __init__(self, agent, env, env_params, new_reward_f, out_features, component_names,
                 deterministic = False, n_rollouts = 100, n_updates = 100000):
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
        self.n_updates = n_updates
        self.component_names = component_names

        self.target_policy_noise = agent.target_policy_noise
        self.target_noise_clip = agent.target_noise_clip

        self.n_rollouts = n_rollouts
        self.batch_size = 256
        self.deterministic = deterministic

        # Training dcritic
        self.replay_buffer = self._get_rollout()
        self.dcritic = self._decompose_critic(self.critic)
        self.dcritic_target = deepcopy(self.dcritic)

        self._train_dcritic(n_updates = self.n_updates)

    def explain(self, t_query, data, algo):
        step_index = int(np.round(t_query / self.env_params['delta_t']))

        X = data[algo]['x'][:, step_index] # (Nx, N_instances, 1)
        U = data[algo]['u'][:, step_index] # (Nu, N_instances, 1)
        X, U = self.env._scale_X(X.T), self.env._scale_U(U.T) # (1, Nx), (1, Nu)
        XU = np.concatenate([X, U], axis = 1).squeeze() # ((Nx + Nu))

        import torch
        dec_q = self.dcritic.qf0(torch.tensor(np.array(XU), dtype=torch.float32)).detach().numpy()
        q = data[algo]['q'][:, step_index].squeeze()

        self._plot_result(dec_q, q, self.component_names)

        return dec_q

    def _get_rollout(self):
        # TODO: Deterministic vs. Stochastic?
        #   Stochastic해서 multiple rollout을 또 얻어낼 수 있는거 아니야?
        replay_buffer = {
            'actions': np.zeros((0, self.env.Nu)),
            'next_observations': np.zeros((0, self.env.Nx)),
            'observations': np.zeros((0, self.env.Nx)),
            'rewards': np.zeros((0, self.out_features)),
            'dones': np.zeros((0, 1)),
        }

        for _ in range(self.n_rollouts):
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
                )
                a = self._add_noise(a)

                o, r, term, trunc, info = self.env.step(a)
                rewards[i, :] = self.new_reward_f(self.env, self.env.state, a, con=None) # self.env.state: Unnormalized state

                actions[i, :] = (a + 1) * (
                        self.env.get_env_params["a_space"]["high"]
                        - self.env.get_env_params["a_space"]["low"]
                ) / 2 + self.env.get_env_params["a_space"]["low"]
                observations[i + 1,:] = (o + 1) * (
                        self.env.observation_space_base.high - self.env.observation_space_base.low
                ) / 2 + self.env.observation_space_base.low

            a, _s = self.actor.predict(o, deterministic=self.deterministic)
            a = self._add_noise(a)
            actions[self.env.N - 1, :] = (a + 1) * (
                    self.env.get_env_params["a_space"]["high"]
                    - self.env.get_env_params["a_space"]["low"]
            ) / 2 + self.env.get_env_params["a_space"]["low"]

            dones = np.zeros((self.env.N, 1))
            dones[-1,:] = 1

            replay_buffer['actions'] = np.vstack([replay_buffer['actions'], actions])
            replay_buffer['observations'] = np.vstack([replay_buffer['observations'], observations[:-1,:]])
            replay_buffer['next_observations'] = np.vstack([replay_buffer['next_observations'], observations[1:,:]])
            replay_buffer['rewards'] = np.vstack([replay_buffer['rewards'], rewards])
            replay_buffer['dones'] = np.vstack([replay_buffer['dones'], dones])

        return replay_buffer

    def _decompose_critic(self, critic):
        dcritic = deepcopy(critic)

        layers = list(dcritic.qf0.children())

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
        dcritic.optimizer = optim.Adam(
            # dcritic.parameters(),
            filter(lambda p: p.requires_grad, dcritic.qf0.parameters()),
            lr=1e-4,
            betas = (0.9, 0.999)
        )
        return dcritic

    def _train_dcritic(self, n_updates):
        """
        Train decomposed critic with rollout data
        Args:
            dcritic:
            env:
        Returns:
        """
        self.dcritic.set_training_mode(True)

        actor_losses, critic_losses = [], []
        for n in range(n_updates):
            # Sample replay buffer
            sample = self._sample(self.replay_buffer, self.batch_size)
            actions = torch.tensor(self.env._scale_U(sample['actions']), dtype=torch.float32)
            observations = torch.tensor(self.env._scale_X(sample['observations']), dtype=torch.float32)
            next_observations = torch.tensor(self.env._scale_X(sample['next_observations']), dtype=torch.float32)
            rewards = torch.tensor(sample['rewards'], dtype=torch.float32)
            dones = torch.tensor(sample['dones'], dtype=torch.float32)

            with torch.no_grad():
                # The actions are not being explored in training dcritics
                next_actions, _ = self.actor.predict(next_observations, deterministic=True)
                next_actions = torch.tensor(next_actions) # Normalized actions

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

            if n % 400 == 0:
                print(f'Decomposed critic loss after {n} iterations: {critic_loss.item()}')
                # Check
                # print(f"\n[Iteration {n}] Parameter values and gradients:")
                # for name, param in self.dcritic.named_parameters():
                #     if param.requires_grad:
                #         print(f"Param: {name}")
                #         print(f"  Value: {param.data.norm():.6f}")
                #         print(f"  Grad : {param.grad.norm() if param.grad is not None else 'None'}")

    def _sample(self, replay_buffer, batch_size):
        N = replay_buffer['actions'].shape[0]
        indices = np.random.choice(N, size=batch_size, replace=False)

        batch = {key: val[indices] for key, val in replay_buffer.items()}
        return batch

    def _add_noise(self, a):
        self.target_policy_noise = 0.05
        self.target_noise_clip = 2 * self.target_policy_noise
        noise = np.random.normal(0, self.target_policy_noise, size = a.shape)
        noise = np.clip(noise, -self.target_noise_clip, self.target_noise_clip)
        noised_action = np.clip(a + noise, -1, 1)
        return noised_action

    def _plot_result(self, dec_q, q, component_names):
        colors = ['tab:red', 'tab:blue', 'tab:green']

        fig, ax = plt.subplots(figsize=(6, 5))

        # location
        x_dec_q = 0
        x_q = 1

        # Plotting decomposed q values as stacked bar
        bottom = 0
        for i in range(len(dec_q)):
            ax.bar(x_dec_q, dec_q[i], bottom=bottom, color=colors[i], label=component_names[i])
            bottom += dec_q[i]  # 누적 (모두 음수이므로 밑으로)

        # Plotting q value as bar
        ax.bar(x_q, q, color='gray', label='total q')

        ax.set_xticks([x_dec_q, x_q])
        ax.set_xticklabels(['Decomposed Q', 'Total Q'])
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('Q-value')
        ax.set_title('Decomposed vs Total Q')
        ax.legend()
        ax.grid(True, axis='y')

        plt.tight_layout()
        plt.show()
