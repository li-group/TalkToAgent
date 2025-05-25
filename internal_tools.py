import sys
sys.path.append("..")

import torch
import numpy as np
from callback import LearningCurveCallback
from stable_baselines3 import PPO, DDPG, SAC
from params import running_params, env_params

running_params = running_params()
env, env_params = env_params(running_params['system'])

def train_agent():
    algo = running_params.get("algo")
    system = running_params.get("system")

    training_seed = running_params.get("seed", 1)
    nsteps_train = running_params.get("nsteps", int(5e4))
    train = running_params.get("train", True)

    if algo == 'DDPG':
        agent = DDPG("MlpPolicy", env, verbose=1, learning_rate=0.001, seed=training_seed)
    elif algo == 'SAC':
        agent = SAC("MlpPolicy", env, verbose=1, learning_rate=0.001, seed=training_seed)
    elif algo == 'PPO':
        agent = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001, seed=training_seed)
    else:
        raise ValueError(f'Algorithm {algo} not supported')

    if train:
        callback = LearningCurveCallback(log_file=f'.\learning_curves\{algo}_{system}_LC_rep.csv')
        agent.learn(total_timesteps=int(nsteps_train), callback=callback)
        agent.save(f'./policies/{algo}_{system}.zip')
    else:
        agent.set_parameters(f'./policies/{algo}_{system}')

    return agent

def get_rollout_data(agent):
    algo = running_params.get("algo")
    reps = running_params.get("rollout_reps")
    evaluator, data = env.plot_rollout({algo: agent}, reps=reps, get_Q=True)
    return data

def cluster_states(agent, data):
    feature_names = env_params.get("feature_names")
    target = env_params.get("target")
    actor = agent.actor.mu
    X = data['DDPG']['x'].reshape(data['DDPG']['x'].shape[0], -1).T
    q = data['DDPG']['q'].reshape(data['DDPG']['q'].shape[0], -1).T

    from explainer.Cluster_states import cluster_params, Reducer, Cluster

    params = cluster_params(X.shape[0])
    reducer = Reducer(params)
    X_scaled = X  # Optionally add preprocessing
    X_reduced = reducer.reduce(X_scaled, algo='TSNE')
    y = actor(torch.tensor(X_scaled, dtype=torch.float32)).detach().numpy().squeeze()

    reducer.plot_scatter(X_reduced, hue=y)
    reducer.plot_scatter_grid(X_reduced, hue=np.hstack([y[:, None], q, X[:, [0, 2]]]), hue_names=['y', 'q', target, f'errors_{target}'])

    cluster = Cluster(params, feature_names=feature_names)
    cluster_labels = cluster.cluster(X_reduced, algo='HDBSCAN')
    cluster.plot_scatter(X_reduced, cluster_labels)
    cluster.plot_violin(X, cluster_labels)
    return cluster

def feature_importance_global(agent, data, cluster_labels=None, lime=False, shap=True):
    algo = running_params.get("algo")
    feature_names = env_params.get("feature_names")
    target = env_params.get("target")
    actor = agent.actor.mu
    X = data['DDPG']['x'].reshape(data['DDPG']['x'].shape[0], -1).T

    if lime:
        from explainer.LIME import LIME
        explainer = LIME(model=actor, bg=X, target=target, feature_names=feature_names, algo=algo, env_params=env_params)
        lime_values = explainer.explain(X=X)
        explainer.plot(lime_values)
        return lime_values

    if shap:
        from explainer.SHAP import SHAP
        explainer = SHAP(model=actor, bg=X, target=target, feature_names=feature_names, algo=algo, env_params=env_params)
        shap_values = explainer.explain(X=X)
        explainer.plot(shap_values, cluster_labels=cluster_labels)
        return shap_values

def feature_importance_local(agent, data):
    algo = running_params.get("algo")
    feature_names = env_params.get("feature_names")
    target = env_params.get("target")
    actor = agent.actor.mu
    X = data['DDPG']['x'].reshape(data['DDPG']['x'].shape[0], -1).T

    from explainer.SHAP import SHAP
    explainer = SHAP(model=actor, bg=X, target=target, feature_names=feature_names, algo=algo, env_params=env_params)
    instance = X[0, :]
    shap_values_local = explainer.explain(X=instance)
    explainer.plot(shap_values_local)
    return shap_values_local

def partial_dependence_plot(agent, data):
    algo = running_params.get("algo")
    feature_names = env_params.get("feature_names")
    target = env_params.get("target")
    actor = agent.actor.mu
    X = data['DDPG']['x'].reshape(data['DDPG']['x'].shape[0], -1).T

    from explainer.PDP import PDP
    explainer = PDP(model=actor, bg=X, target=target, feature_names=feature_names, algo=algo, env_params=env_params, grid_points=100)
    ice_curves = explainer.explain(X=X)
    explainer.plot(ice_curves)
    return ice_curves

def trajectory_sensitivity(agent, data, t_query:float):
    algo = running_params.get("algo")
    from explainer.Futuretrajectory import sensitivity
    xs, us, qs = sensitivity(t_query=t_query,
                             perturbs=[-0.2, -0.1, 0, 0.1, 0.2],
                             data=data,
                             env_params=env_params,
                             policy=agent,
                             algo=algo,
                             horizon=20)
    return xs, us, qs

def trajectory_counterfactual(agent, data, t_query:float, cf_actions:list):
    algo = running_params.get("algo")
    from explainer.Futuretrajectory import counterfactual
    xs, us, qs = counterfactual(t_query=t_query,
                                a_cf=cf_actions,
                                data=data,
                                env_params=env_params,
                                policy=agent,
                                algo=algo,
                                horizon=20)
    return xs, us, qs
