import sys
sys.path.append("..")  # Adds higher directory to python modules path for callback class.
# sys.path.append("..\..\..\src\pcgym") # Add local pc-gym files to path.

import torch
from src.pcgym import make_env
from pcgym_paper.train_policies.callback import LearningCurveCallback
import numpy as np
from stable_baselines3 import DQN, PPO, DDPG, SAC
from custom_reward import sp_track_reward

# %%
TRAIN_AGENT = False
ALGO = 'DDPG'
SYSTEM = 'cstr_ode'
REPS = 10
SHAP_INTERPRET = False
LIME_INTERPRET = False

# Define environment
T = 300 # Total simulated time (min)
nsteps = 600 # Total number of steps
delta_t = T/nsteps # Minutes per step
training_seed = 1

sps = []
for i in range(nsteps):
    if i % 20 == 0:
        sp = np.random.uniform(low=0.8, high=0.9)
    sps.append(sp)

# SP = {
#     'Ca': [0.85 for i in range(int(nsteps/3))] + [0.9 for i in range(int(nsteps/3))]+ [0.87 for i in range(int(nsteps/3))],
# }

SP = {
    'Ca': sps,
}
action_space = {
    'low': np.array([295]),
    'high':np.array([302]) 
}

observation_space = {
    'low' : np.array([0.7,300,-0.1]),
    'high' : np.array([1,350,0.1])
}

r_scale = {'Ca':1e3}

# Define reward to be equal to the OCP (i.e the same as the oracle)
env_params = {
    'N': nsteps, 
    'tsim':T, 
    'SP':SP, 
    'o_space' : observation_space, 
    'a_space' : action_space,
    'x0': np.array([0.8,330,0.0]),
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
        DDPG_CSTR.set_parameters(f'policies\DDPG_CSTR_rep_{r_i}.zip')

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
evaluator, data = env.get_rollouts({'DDPG':DDPG_CSTR}, reps = 1, get_Q = True)
# evaluator, data = env.plot_rollout({'DDPG':DDPG_CSTR}, reps = REPS,
#                                    oracle=True, dist_reward = True, cons_viol = False,
#                                    MPC_params={'N':17, 'R':1e-8}, get_Q = True)

# %%
actor = DDPG_CSTR.actor.mu

X = data['DDPG']['x']
X = X.reshape(X.shape[0], -1).T
# observation variables: [Ca, T, Error(Ca)]

# %% LIME 분석
if LIME_INTERPRET:
    from explainer.LIME import LIME
    explainer = LIME(model = actor,
                     bg = X,
                     target = 'Tc',
                     feature_names = env.model.info()["states"] + ["Error_Ca"],
                     algo = ALGO,
                     env_params = env_params)
    lime_values = explainer.explain(X = X)
    explainer.plot(lime_values)

# %% SHAP 분석 (global)
if SHAP_INTERPRET:
    from explainer.SHAP import SHAP
    explainer = SHAP(model = actor,
                     bg = X,
                     target = 'Tc',
                     feature_names = env.model.info()["states"] + ["Error_Ca"],
                     algo = ALGO,
                     env_params = env_params)
    shap_values = explainer.explain(X = X)
    explainer.plot(shap_values)

    # %% SHAP 분석 (local)
    instance = X[0,:]
    shap_values_local = explainer.explain(X = instance)
    explainer.plot(shap_values_local)


# %%
import os
# evaluator.plot_data(data, savedir = f'./[{ALGO}][{SYSTEM}] Rollout.png')

# %% Sensitivity analysis of actions to state values
# TODO: 1. ICE, PDP plots 이용

# %%
# TODO: 2. Future trajectory에 대한 설명 추가: 현재 state에서 어떤 action을 취했을 때 앞으로 현재 policy라고 가정하고 어떻게 시스템이 돌아갈 것이냐
# TODO: In the trajectory, why does the policy made this action at this specific timepoint?
# User는 전체 trajectory를 보고 특정 time step의 decision에 대해 궁금해 함. trajectory object가 주어져야 하고, function에서 특정 state
# 와 action, 또는 특정 time index가 argument로 주어져야 함.
# TODO: 언제쯤 setpoint에 도달할 것으로 예상하는지?
# Rollout data
trajectory = data[ALGO]

state_history = trajectory['x'].squeeze().T
action_history = trajectory['u'].squeeze().T
q_history = trajectory['q'].squeeze().T

time_step_query = 120
step_index = int(time_step_query // delta_t)

current_state = state_history[step_index]
current_action = action_history[step_index]
current_q = q_history[step_index]

# Sensitivity of an action: 현재 state로부터 action을 구현
sim_trajs = []
window_length = action_space['high'] - action_space['low']
actions = [current_action + dev * window_length for dev in [-0.2, -0.1, 0, 0.1, 0.2]]
actions_dict = dict(zip(["-20%", "-10%", "0%", "10%", "20%"], actions)) # Perturbation of actions
actions_dict = {k: v for k, v in actions_dict.items() if (v <= action_space['high']).all() or (v >= action_space['low']).all()}

for pertb, action in actions_dict.items():
    sim_info = {'step_index': step_index,
                'trajectory': trajectory,
                'perturbation': pertb,
                'action': action}
    evaluator, sim_traj = env.get_rollouts({'DDPG':DDPG_CSTR}, reps=1, sim_info=sim_info, get_Q = True)
    sim_trajs.append(sim_traj)

xs = np.array([s[ALGO]['x'] for s in sim_trajs]).squeeze().T
us = np.array([s[ALGO]['u'] for s in sim_trajs]).squeeze().T[:,np.newaxis,:]
qs = np.array([s[ALGO]['q'] for s in sim_trajs]).squeeze().T

# %%
HORIZON = 20
LABELS = ["-20%", "-10%", "0%", "10%", "20%"]
def plot_results(xs, us, qs, horizon, labels = None):
    import matplotlib.pyplot as plt
    import numpy as np

    xs_sliced = xs[:, :-1, :]  # Eliminating error term
    time_range = np.arange(step_index - 10, step_index + horizon)
    labels = labels if labels is not None else ["Label {i}".format(i=i) for i in range(xs.shape[-1])]

    cmap = plt.get_cmap('viridis')
    n_lines = len(labels)
    if n_lines == 1:
        colors = ['black']
    else:
        colors = [cmap(i / (n_lines - 1)) for i in range(n_lines)]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    for i in range(us.shape[-1]):
        axes[0].plot(time_range[:11], us[step_index-10:step_index+1, 0, i], color='black', linewidth=3)
        axes[0].plot(time_range[10:], us[step_index:step_index+horizon, 0, i], color=colors[i], label=labels[i])
    axes[0].axvline(step_index, linestyle='--', color='red')
    axes[0].set_ylabel(env.model.info()['inputs'][0])
    axes[0].set_ylim([action_space['low'][0], action_space['high'][0]])
    axes[0].legend()
    axes[0].grid(True)

    for i in range(xs_sliced.shape[1]):
        for j in range(xs_sliced.shape[2]):
            axes[i+1].plot(time_range[:11], xs_sliced[step_index-10:step_index+1, i, j], color='black', linewidth=3)
            axes[i+1].plot(time_range[10:], xs_sliced[step_index:step_index+horizon, i, j], color=colors[j], label=labels[j])
        axes[i+1].axvline(step_index, linestyle='--', color='red')
        axes[i+1].set_ylabel(env.model.info()['states'][i])
        axes[i+1].legend()
        axes[i+1].grid(True)
        axes[i+1].set_ylim([observation_space['low'][i], observation_space['high'][i]])
        if env.model.info()["states"][i] in env.SP:
            axes[i+1].step(
                time_range,
                env.SP[env.model.info()["states"][i]][time_range[0]:time_range[-1]+1],
                where="post",
                color="black",
                linestyle="--",
                label="Set Point",
            )

    for i in range(qs.shape[1]):
        axes[3].plot(time_range[:11], qs[step_index-10:step_index+1, i], color='black', linewidth=3)
        axes[3].plot(time_range[10:], qs[step_index:step_index+horizon, i], color=colors[i], label=labels[i])
    axes[3].axvline(step_index, linestyle='--', color='red')
    axes[3].set_ylabel('Q value')
    axes[3].legend()
    axes[3].grid(True)

    plt.xlabel('Time (min)')
    plt.tight_layout()
    plt.show()

plot_results(xs, us, qs, HORIZON, LABELS)
# %%
plot_results(data['DDPG']['x'].transpose(1,0,2),
             data['DDPG']['u'].transpose(1,0,2),
             data['DDPG']['q'].transpose(1,0,2),
             HORIZON)

# %% Until 5/2 meeting
# TODO: sim_info 반영하기. step_index 전까지는 trajectory data를 그대로 이용하고, step_index에서 특
#   특정 action을 시행, rollout 추출 후 trajectory visualization.
# TODO: Perturbation approach말고도 alternative_action을 집어주면 그거랑 비교하는 과정도 추가하면 좋을 듯.

# TODO: User query: 이 trajectory에서 이 time step에서 왜 이렇게 행동했을까?
# TODO: 변형: 이 trajectory에서 이 state에서 왜 이렇게 행동했을까?
# TODO: Plot: Optimal action by policy를 특별히 remark. 그 기준으로 +-5%, 10% 등으로 possible 계산을 해보는 건 어떨까.



# %%
# TODO: DQN 등의 value network에 대해서도 구현 - discretization 필요
# TODO: 각 feature의 의미에 대한 dictionary 생성.
# TODO: GPT4로 설명할 수 있나 볼까? 최소한의 prompt도 만들어보고. Prompt는 system에 대한 prompt도 있어야 할 것 같고 XRL에 대한 prompt도 있어야 할 것 같다.
# TODO: Long-term reward가 필요한 system에 대해서 생각해보기.
# TODO: Actor, critic distinguish 반영해야. verbose도 추가하면 좋을 듯
# TODO: 다른 시스템에 대해서도 extend

# TODO: Manuscript의 본 방법론의 contribution 중 하나로 figure와 text가 공존하는 multi-modal explanation으로 설명하는 건 어떨까
# TODO: 일반적인 제어에 관해서도 추가를 하는 게 좋을 것 같다. 예) 지금 이 상태에서 setpoint를 갑자기 올려버리면 어떻게 action을 하게 될지?
# TODO: Future work: User의 query에 따라서 자동으로 reward function을 조절해주는 inverse RL 방법을 연구해보는 것도 가능하지 않을까?
#     예시: "지금 policy를 조금 더 보수적으로 학습해보고 시뮬레이션을 돌려볼 수 있을까?"
