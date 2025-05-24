import sys
sys.path.append("..")  # Adds higher directory to python modules path for callback class.
# sys.path.append("..\..\..\src\pcgym") # Add local pc-gym files to path.

import torch
from src.pcgym import make_env
from callback import LearningCurveCallback
import numpy as np
from stable_baselines3 import DQN, PPO, DDPG, SAC
from custom_reward import sp_track_reward

# %% Define RL training/rollout-specific parameters
TRAIN_AGENT = True
ALGO = 'DDPG'
ROLLOUT_REPS = 1

NSTEPS_TRAIN = 5e4
TRAINING_REPS = 1

# Define explanation-specific parameters
LIME_INTERPRET = True
SHAP_INTERPRET = True
PDP_INTERPRET = True
TRAJECTORY_ANALYSIS = True

# Define environment-specific parameters
SYSTEM = 'cstr_ode'
TARGET = 'Ca'

T = 300 # Total simulated time (min)
nsteps = 600 # Total number of steps
delta_t = T/nsteps # Minutes per step
training_seed = 1

setpoints = []
for i in range(nsteps):
    if i % 20 == 0:
        setpoint = np.random.uniform(low=0.8, high=0.9)
    setpoints.append(setpoint)

SP = {TARGET: setpoints}
action_space = {'low': np.array([295]),
                'high':np.array([302])}

observation_space = {'low' : np.array([0.7,300,-0.1]),
                     'high' : np.array([1,350,0.1])}

initial_point = np.array([0.8,330,0.0])

r_scale = {TARGET:1e3}

# Define reward to be equal to the OCP (i.e the same as the oracle)
env_params = {
    'N': nsteps, 
    'tsim':T, 
    'SP':SP,
    'delta_t': delta_t,
    'o_space' : observation_space, 
    'a_space' : action_space,
    'x0': initial_point,
    'r_scale': r_scale,
    'model': SYSTEM,
    'normalise_a': True, 
    'normalise_o':True, 
    'noise':False,
    'integration_method': 'casadi', 
    'noise_percentage':0.001, 
    'custom_reward': sp_track_reward
}

env = make_env(env_params)
feature_names  = env.model.info()["states"] + [f"Error_{TARGET}"]

# %% Train RL agents
for r_i in range(TRAINING_REPS):
    print(f'Training repition:{r_i+1}')

    # Train RL agents (DDPG, PPO, SAC, and DQN(which needs a priori discretization))
    log_file = f'.\learning_curves\{ALGO}_{SYSTEM}_LC_rep_{r_i}.csv'
    if ALGO == 'DDPG':
        agent =  DDPG("MlpPolicy", env, verbose=1, learning_rate=0.001, seed = training_seed)
    elif ALGO == 'SAC':
        agent = SAC("MlpPolicy", env, verbose=1, learning_rate=0.001, seed = training_seed)
    elif ALGO == 'PPO':
        agent =  PPO("MlpPolicy", env, verbose=1, learning_rate=0.001, seed = training_seed)
    else:
        raise ValueError(f'Algorithm {ALGO} not supported')


    if TRAIN_AGENT:
        callback = LearningCurveCallback(log_file=log_file)
        agent.learn(NSTEPS_TRAIN, callback=callback)

        # Save DDPG Policy
        agent.save(f'./policies/{ALGO}_{SYSTEM}_rep_{r_i}.zip')
    else:
        agent.set_parameters(f'./policies/{ALGO}_{SYSTEM}_rep_{r_i}')

    trained_dict = {}

# %%
# evaluator, data = env.plot_rollout({ALGO : agent}, reps = REPS,
#                                    oracle=True, dist_reward = True, cons_viol = False,
#                                    MPC_params={'N':17, 'R':1e-8}, get_Q = True)
evaluator, data = env.plot_rollout({ALGO : agent}, reps = 10, get_Q = True)


# %%
actor = agent.actor.mu

X = data['DDPG']['x']
X = X.reshape(X.shape[0], -1).T
# observation variables: [Ca, T, Error(Ca)]

# %% Clustering states
# Extract last activation values from the actor
import torch.nn as nn
actor_hidden = nn.Sequential(*list(actor.children())[:-2])
X_scaled = env._scale_X(X)
x_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to('cpu')
activation = actor_hidden(x_tensor).detach().numpy().squeeze()

# %%
from explainer.Cluster_states import get_params, Reducer, Cluster
params = get_params(X.shape[0])
reducer = Reducer(params)
X_reduced = reducer.reduce(X_scaled, algo = 'TSNE')
y = actor(torch.tensor(X_scaled, dtype=torch.float32)).detach().numpy().squeeze()
reducer.plot_scatter(X_reduced, hue = y)

# %%
q = data['DDPG']['q']
q = q.reshape(q.shape[0], -1).T
reducer.plot_scatter_grid(X_reduced, hue =np.hstack([y[:,np.newaxis],q,X[:,[0,2]]]), hue_names = ['y','q',TARGET,'errors_Ca'])

# %%
params = get_params(X.shape[0])
cluster = Cluster(params, feature_names=feature_names)
cluster_labels = cluster.cluster(X_reduced,
                                 # y = y,
                                 algo = 'HDBSCAN')
cluster.plot_scatter(X_reduced, cluster_labels)
# cluster.plot_scatter_with_arrows(X_reduced, cluster_labels)

cluster.plot_violin(X, cluster_labels)

# TODO: Raw state를 그대로 추출하는 것 vs. 마지막 activation을 어떻게 추출하는지.
#     아직까지는 raw state를 그대로 넣어도 괜찮을 듯 하다.

# %% LIME analysis
if LIME_INTERPRET:
    from explainer.LIME import LIME
    explainer = LIME(model = actor,
                     bg = X,
                     target = TARGET,
                     feature_names = feature_names,
                     algo = ALGO,
                     env_params = env_params)
    lime_values = explainer.explain(X = X)
    explainer.plot(lime_values)

# %% SHAP analysis (global)
if SHAP_INTERPRET:
    from explainer.SHAP import SHAP
    explainer = SHAP(model = actor,
                     bg = X,
                     target = TARGET,
                     feature_names = feature_names,
                     algo = ALGO,
                     env_params = env_params)
    shap_values = explainer.explain(X = X)
    # explainer.plot(shap_values)
    explainer.plot(shap_values, cluster_labels = cluster_labels)

    # %% SHAP analysis (local)
    instance = X[0,:]
    shap_values_local = explainer.explain(X = instance)
    explainer.plot(shap_values_local)

# %% Partial dependence analysis of actions to state values (global)
if PDP_INTERPRET:
    from explainer.PDP import PDP
    explainer = PDP(model = actor,
                    bg = X,
                    target = TARGET,
                    feature_names = feature_names,
                    algo = ALGO,
                    env_params = env_params,
                    grid_points = 100)
    ice_curves = explainer.explain(X = X)
    explainer.plot(ice_curves)
    # explainer.plot(ice_curves, cluster_labels)

    # ICE plot (local)
    # ice_curves = explainer.explain(X = X[3]) # Specific data point instance
    # explainer.plot(ice_curves)

# %% Future trajectory analysis of specific action
if TRAJECTORY_ANALYSIS:
    from explainer.Futuretrajectory import sensitivity, counterfactual
    sensitivity(t_query = 180,
                perturbs = [-0.2, -0.1, 0, 0.1, 0.2],
                data = data,
                env_params = env_params,
                policy = agent,
                algo = ALGO,
                horizon=20)

    counterfactual(t_query = 180,
                   a_cf = [300],
                   data = data,
                   env_params = env_params,
                   policy = agent,
                   algo = ALGO,
                   horizon=20)


# %% 5.9. meeting
# TODO: 실제 process operator들이 할 수 있는 counterfactual에 대해 생각해보기
# TODO: LLM을 한번 연결 해봐야 할 것 같은데. user query를 얻어서 문제 (또는 code로) formulate를 하는 게 제일 bottleneck이 될 듯.

# %%
# TODO: Convergence analysis 언제쯤 setpoint에 도달할 것으로 예상하는지?
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
