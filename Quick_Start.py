# %% Import
from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
import torch.nn.functional as F
import copy
from stable_baselines3 import PPO,SAC
from pcgym import make_env
import jax.numpy as jnp
#Global params
T = 26
nsteps = 100

# %% CSTR
# Setpoint
SP = {'Ca': [0.85 for i in range(int(nsteps/2))] + [0.9 for i in range(int(nsteps/2))]}

# Action and Observation Space
action_space = {'low': np.array([295]), 'high': np.array([302])}
observation_space = {'low': np.array([0.7,300,0.8]),'high': np.array([1,350,0.9])}

# Construct the environment parameter dictionary
env_params = {
    'N': nsteps, # Number of time steps
    'tsim':T, # Simulation Time
    'SP' :SP,
    'o_space' : observation_space,
    'a_space' : action_space,
    'x0': np.array([0.8,330,0.8]), # Initial conditions [Ca, T, Ca_SP]
    'model': 'cstr_ode', # Select the model
}

# Create environment
env = make_env(env_params)

# %%
# Load a pre-trained policy
initial_policy = SAC.load('./pc-gym_paper/train_policies/cstr/policies/SAC_CSTR.zip')
pp = initial_policy.policy_aliases

# %%
# Plot a rollout with a oracle and a reward distribution
env.plot_rollout(pp, reps= 1, oracle = True, dist_reward=True, MPC_params={'N': 10, 'R': 0.001})
