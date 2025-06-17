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

class DecomposedDDPG(DDPG):
    def __init__(self):
        super(DecomposedDDPG, self).__init__()


def decompose_critic(critic, out_features):
    dcritic = deepcopy(critic)

    # 기존 Sequential model
    old_model = critic.qf0
    layers = list(old_model.children())

    if isinstance(layers[-1], nn.Linear):
        in_features = layers[-1].in_features
        new_output_layer = nn.Linear(in_features, out_features)
    else:
        raise ValueError("Last layer is not nn.Linear!")

    dcritic.qf0 = nn.Sequential(*layers[:-1], new_output_layer)
    return dcritic

def train_dcritic(dcritic, env):
    """
    Train decomposed critic with rollout data
    Args:
        dcritic:
        env:
    Returns:
    """
    # Replay buffer sample
    sample = self.replay_buffer.sample()
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


class decompose():
    def __init__(self, env, env_params):
        self.env = env
        self.env_params = env_params
        return

    def decompose(self, reward):
        """
        Args:
            reward: [function] reward call

        Returns:

        """
        o_space = self.env.observation_space
        a_space = self.env.action_space
        critic = ContinuousCritic()


def reward_decomposition():
    # TODO:
    #  1. Reward를 분해. (일단은 manually, 최종적으로는 자동화)
    #  2. multi-output critic (nn.torch 기반)학습
    #  3. 이때 actor는 가만히 두고 critic만 TD error로 진행

class DDPG_decomposed(DDPG):
    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            _init_setup_model=False,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size,
                                                    env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations,
                                                     self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

def reward_decomposition(env, critic, critic_target, replay_buffer, gamma=0.99, batch_size=64, device="cpu"):
    """
    Reward decomposition and critic-only training.

    Args:
        env: 환경 (four-tank 등), reward_decomposed() 메서드 포함.
        critic: multi-output critic 네트워크
        critic_target: target critic 네트워크
        replay_buffer: experience replay buffer
        gamma: discount factor
        batch_size: training batch size
        device: cpu 또는 cuda
    """
    # 1. Sample batch from buffer
    batch = replay_buffer.sample(batch_size)
    states = torch.tensor(batch['obs'], dtype=torch.float32).to(device)
    actions = torch.tensor(batch['acts'], dtype=torch.float32).to(device)
    next_states = torch.tensor(batch['next_obs'], dtype=torch.float32).to(device)
    dones = torch.tensor(batch['done'], dtype=torch.float32).unsqueeze(1).to(device)

    # 2. Compute decomposed rewards manually using environment logic
    reward_components = []
    for i in range(batch_size):
        reward_dict = env.four_tank_reward_decomposed(batch['obs'][i], batch['acts'][i], None)
        reward_components.append([reward_dict[k] for k in sorted(reward_dict.keys())])

    rewards = torch.tensor(reward_components, dtype=torch.float32).to(device)  # shape: [B, C]

    # 3. Compute critic targets
    with torch.no_grad():
        target_Q = critic_target(next_states, actions)  # assume deterministic policy or same action
        target = rewards + gamma * (1 - dones) * target_Q  # shape: [B, C]

    # 4. Predict Q-values from critic
    current_Q = critic(states, actions)  # shape: [B, C]

    # 5. Compute TD loss (per component)
    loss = nn.MSELoss()(current_Q, target)

    # 6. Optimize critic
    critic.optimizer.zero_grad()
    loss.backward()
    critic.optimizer.step()

    return loss.item()


def four_tank_reward_decomposed(self, x, u, con):
    Sp_i = 0
    cost = 0
    R = 0.1
    cost_dict = {}
    if not hasattr(self, 'u_prev'):
        self.u_prev = u

    for k in self.env_params["SP"]:
        i = self.model.info()["states"].index(k)
        SP = self.SP[k]

        o_space_low = self.env_params["o_space"]["low"][i]
        o_space_high = self.env_params["o_space"]["high"][i]

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        r_scale = self.env_params.get("r_scale", {})

        cost_k = (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2) * r_scale.get(k, 1)
        cost_dict[k] = cost
        cost += cost_k

        Sp_i += 1
    u_normalized = (u - self.env_params["a_space"]["low"]) / (
            self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )
    u_prev_norm = (self.u_prev - self.env_params["a_space"]["low"]) / (
            self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )
    self.u_prev = u

    # Add the control cost
    cost_u = np.sum(R * (u_normalized - u_prev_norm) ** 2)
    cost += cost_u
    cost_dict["u"] = cost_u

    r = -cost
    reward_dict = {}
    for k, cost in cost_dict.items():
        reward_dict[k] = - cost

    return reward_dict