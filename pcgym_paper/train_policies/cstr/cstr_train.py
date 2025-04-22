import sys
sys.path.append("..")  # Adds higher directory to python modules path for callback class.
# sys.path.append("..\..\..\src\pcgym") # Add local pc-gym files to path.

from pcgym import make_env
from pcgym_paper.train_policies.callback import LearningCurveCallback
import numpy as np
from stable_baselines3 import DQN, PPO, DDPG, SAC
from custom_reward import sp_track_reward

# %%
TRAIN_AGENT = False
ALGO = 'DDPG'
SYSTEM = 'cstr_ode'

# Define environment
T = 26
nsteps = 60
training_seed = 1
SP = {
    'Ca': [0.85 for i in range(int(nsteps/3))] + [0.9 for i in range(int(nsteps/3))]+ [0.87 for i in range(int(nsteps/3))],
}

action_space = {
    'low': np.array([295]),
    'high':np.array([302]) 
}

observation_space = {
    'low' : np.array([0.7,300,0.8]),
    'high' : np.array([1,350,0.9])  
}

r_scale = {'Ca':1e3}

# Define reward to be equal to the OCP (i.e the same as the oracle)
env_params = {
    'N': nsteps, 
    'tsim':T, 
    'SP':SP, 
    'o_space' : observation_space, 
    'a_space' : action_space,
    'x0': np.array([0.8,330,0.8]),
    'r_scale': r_scale,
    'model': SYSTEM,
    'normalise_a': True, 
    'normalise_o':True, 
    'noise':True, 
    'integration_method': 'casadi', 
    'noise_percentage':0.001, 
    'custom_reward': sp_track_reward
}

env = make_env(env_params)

# Global timesteps
nsteps_train = 5e4
training_reps = 1
for r_i in range(training_reps):
    print(f'Training repition:{r_i+1}')
    # # Train SAC
    # log_file = f"learning_curves\SAC_CSTR_LC_rep_{r_i}.csv"
    # SAC_CSTR =  SAC("MlpPolicy", env, verbose=1, learning_rate=0.01, seed=training_seed)
    # callback = LearningCurveCallback(log_file=log_file)
    # SAC_CSTR.learn(nsteps_train,callback=callback)
    #
    # # Save SAC Policy
    # SAC_CSTR.save(f'policies\SAC_CSTR_rep_{r_i}.zip')
    #
    # # Train PPO
    # log_file = f"learning_curves\PPO_CSTR_LC_rep_{r_i}.csv"
    # PPO_CSTR =  PPO("MlpPolicy", env, verbose=1, learning_rate=0.001, seed = training_seed)
    # callback = LearningCurveCallback(log_file=log_file)
    # PPO_CSTR.learn(nsteps_train,callback=callback)
    #
    # # Save PPO Policy
    # PPO_CSTR.save(f'policies\PPO_CSTR_rep_{r_i}.zip')

    # Train DDPG
    log_file = f'learning_curves\DDPG_CSTR_LC_rep_{r_i}.csv'
    DDPG_CSTR =  DDPG("MlpPolicy", env, verbose=1, learning_rate=0.001, seed=training_seed)
    if TRAIN_AGENT:
        callback = LearningCurveCallback(log_file=log_file)
        DDPG_CSTR.learn(nsteps_train,callback=callback)

        # Save DDPG Policy
        DDPG_CSTR.save(f'policies\DDPG_CSTR_rep_{r_i}.zip')
    else:
        DDPG_CSTR.load(f'policies\DDPG_CSTR_rep_{r_i}.zip',
                       env = env)

# %% Get rollout data
# evaluator, data = env.get_rollouts({'SAC':SAC_CSTR,'PPO':PPO_CSTR,'DDPG':DDPG_CSTR}, reps=50, oracle=True, MPC_params={'N':17, 'R':1e-8})
evaluator, data = env.get_rollouts({'DDPG':DDPG_CSTR}, reps=10, oracle=True, MPC_params={'N':17, 'R':1e-8})

# %%
observation_space = {
    'low' : np.array([0.7,300,0.8]),
    'high' : np.array([1,350,0.9])
}

import torch
actor = DDPG_CSTR.actor.mu
actor(torch.tensor(np.array([0.7,350,0.8]), dtype=torch.float32))

# observation variables: [Ca, T, Setpoint(Ca)]

# %% LIME 분석
from explainer.LIME import LIME
explainer = LIME(model = actor,
                 target = 'Tc',
                 algo = ALGO,
                 system = SYSTEM)
X = data['DDPG']['x']
U = data['DDPG']['u']
X = X.reshape(X.shape[0], -1).T
U = U.reshape(U.shape[0], -1).T
X_scaled, U_scaled = explainer.scale(X, U, observation_space, action_space)
lime_values = explainer.explain(X = X_scaled,
                                feature_names = env.model.info()["states"] + ["Setpoint_Ca"])
explainer.plot(lime_values)

# %% SHAP 분석
# TODO: SHAP으로 분석한 이후 descale 과정 추가
from explainer.SHAP import SHAP
