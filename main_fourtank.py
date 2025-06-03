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
SYSTEM = 'four_tank'
TARGETS = ['h3', 'h4']

T = 2000 # Total simulated time (min)
nsteps = 600 # Total number of steps
delta_t = T/nsteps # Minutes per step
training_seed = 1

SP={}
for target in TARGETS:
    setpoints = []
    for i in range(nsteps):
        if i % 60 == 0:
            setpoint = np.random.uniform(low=0.1, high=0.5)
        setpoints.append(setpoint)
    SP[target]= setpoints

# SP = {
#         'h3': [0.5 for i in range(int(nsteps/2))] + [0.1 for i in range(int(nsteps/2))],
#         'h4': [0.2 for i in range(int(nsteps/2))] + [0.3 for i in range(int(nsteps/2))],
#     }

action_space = {
    'low': np.array([0.1,0.1]),
    'high':np.array([10,10])
}

observation_space = {
    'low' : np.array([0,]*6),
    'high' : np.array([0.6]*6)
}

initial_point = np.array([0.141, 0.112, 0.072, 0.42,SP['h3'][0],SP['h4'][0]])

r_scale = {'h3':1e3,
           'h4':1e3}


def oracle_reward(self, x, u, con):
    Sp_i = 0
    cost = 0
    R = 0.1
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

        cost += (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2) * r_scale.get(k, 1)

        Sp_i += 1
    u_normalized = (u - self.env_params["a_space"]["low"]) / (
            self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )
    u_prev_norm = (self.u_prev - self.env_params["a_space"]["low"]) / (
            self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )
    self.u_prev = u

    # Add the control cost
    cost += np.sum(R * (u_normalized - u_prev_norm) ** 2)
    r = -cost
    try:
        return r[0]
    except Exception:
        return r

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
    'custom_reward': oracle_reward
}

env = make_env(env_params)
feature_names  = env.model.info()["states"] + [f"Error_{target}" for target in TARGETS]

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
from explainer.Cluster_states import cluster_params, Reducer, Cluster
params = cluster_params(X.shape[0])
reducer = Reducer(params)
X_reduced = reducer.reduce(X_scaled, algo = 'TSNE')
y = actor(torch.tensor(X_scaled, dtype=torch.float32)).detach().numpy().squeeze()
# reducer.plot_scatter(X_reduced, hue = y)

# %%
# q = data['DDPG']['q']
# q = q.reshape(q.shape[0], -1).T
# reducer.plot_scatter_grid(X_reduced, hue =np.hstack([y[:,0], y[:,1],q,X[:,[0,2]]]), hue_names = ['y1', 'y2', 'q', TARGET ,'errors_Ca'])

# %%
params = cluster_params(X.shape[0])
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
                     target = 'h3',
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
                     target = 'h3',
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
                    target = 'h3',
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
