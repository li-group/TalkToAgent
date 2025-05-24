import sys
sys.path.append("..")  # Adds higher directory to python modules path for callback class.
# sys.path.append("..\..\..\src\pcgym") # Add local pc-gym files to path.

import torch
from pcgym import make_env
from pcgym_paper.train_policies.callback import LearningCurveCallback
import numpy as np
from stable_baselines3 import DQN, PPO, DDPG, SAC
from custom_reward import sp_track_reward

# %%
TRAIN_AGENT = False
ALGO = 'DDPG'
SYSTEM = 'cstr_ode'
REPS = 10

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
nsteps_train = 1e3
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
        DDPG_CSTR.set_parameters(f'policies\DDPG_CSTR_rep_{r_i}.zip')
        DDPG_CSTR.set_parameters(f'policies\DDPG_CSTR_rep_1.zip')

    # Train DQN
    # TODO: DQN의 discretization 구현
    # log_file = f'learning_curves\DQN_CSTR_LC_rep_{r_i}.csv'
    # DQN_CSTR =  DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, seed=training_seed)
    # if TRAIN_AGENT:
    #     callback = LearningCurveCallback(log_file=log_file)
    #     DQN_CSTR.learn(nsteps_train,callback=callback)
    #
    #     # Save DDPG Policy
    #     DQN_CSTR.save(f'policies\DDPG_CSTR_rep_{r_i}.zip')
    # else:
    #     # DDPG_CSTR.load(f'policies\DDPG_CSTR_rep_{r_i}.zip',
    #     #                env = env)
    #     DQN_CSTR.set_parameters(f'policies\DDPG_CSTR_rep_{r_i}.zip')

    trained_dict = {}

# %% Get rollout data
# evaluator, data = env.get_rollouts({'SAC':SAC_CSTR,'PPO':PPO_CSTR,'DDPG':DDPG_CSTR}, reps=50, oracle=True, MPC_params={'N':17, 'R':1e-8})
# evaluator, data = env.get_rollouts({'DDPG':DDPG_CSTR}, reps=60, oracle=True, MPC_params={'N':17, 'R':1e-8})

# %%
evaluator, data = env.plot_rollout({'DDPG':DDPG_CSTR}, reps = REPS,
                                   oracle=True, dist_reward = True, cons_viol = False,
                                   MPC_params={'N':17, 'R':1e-8})

# %%
actor = DDPG_CSTR.actor.mu

X = data['DDPG']['x']
X = X.reshape(X.shape[0], -1).T
# observation variables: [Ca, T, Setpoint(Ca)]

# %% LIME 분석
# from explainer.LIME import LIME
# explainer = LIME(model = actor,
#                  bg = X,
#                  target = 'Tc',
#                  feature_names = env.model.info()["states"] + ["Setpoint_Ca"],
#                  algo = ALGO,
#                  env_params = env_params)
# lime_values = explainer.explain(X = X)
# explainer.plot(lime_values)

# %% SHAP 분석 (global)
# from explainer.SHAP import SHAP
# explainer = SHAP(model = actor,
#                  bg = X,
#                  target = 'Tc',
#                  feature_names = env.model.info()["states"] + ["Setpoint_Ca"],
#                  algo = ALGO,
#                  env_params = env_params)
# shap_values = explainer.explain(X = X)
# explainer.plot(shap_values)

# %% SHAP 분석 (local)
# instance = X[0,:]
# shap_values_local = explainer.explain(X = instance)
# explainer.plot(shap_values_local)

# %%
# TODO: Critic을 추출하여 각 time step당 Q value tracking 및 state, action과 함께 plot
critic = DDPG_CSTR.critic.qf0
X = data['DDPG']['x']
X = X.reshape(X.shape[0], -1, order = 'F').T
U = data['DDPG']['u']
U = U.reshape(U.shape[0], -1, order = 'F').T
X, U = env._scale_X(X), env._scale_U(U)
XU = np.hstack([X,U])
# critic(torch.tensor(np.array([0.7,350,0.8,300]), dtype=torch.float32))
Q = critic(torch.tensor(np.array(XU), dtype=torch.float32)).detach().numpy()

Q = Q.T
Q = Q.reshape(Q.shape[0], -1, REPS, order = 'F')

data[ALGO]['q'] = Q

# %%
evaluator.plot_data(data)

# %%
# TODO: DQN 등의 value network에 대해서도 구현
# TODO: 각 feature의 의미에 대한 dictionary 생성.
# TODO: GPT4로 설명할 수 있나 볼까? 최소한의 prompt도 만들어보고. Prompt는 system에 대한 prompt도 있어야 할 것 같고 XRL에 대한 prompt도 있어야 할 것 같다.
# TODO: Long-term reward가 필요한 system에 대해서 생각해보기.
# TODO: Actor, critic distinguish 반영해야. verbose도 추가하면 좋을 듯
# TODO: 다른 시스템에 대해서도 extend
