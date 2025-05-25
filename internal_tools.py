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
    """
    Use when: You want to train or load a reinforcement learning agent on the specified environment.
    Example: "Train a DDPG agent for the CSTR environment."
    Example: "Load a pretrained PPO model and skip training."
    """
    algo = running_params.get("algo")
    system = running_params.get("system")

    training_seed = running_params.get("seed", 1)
    nsteps_train = running_params.get("nsteps", int(5e4))
    train_agent = running_params.get("train_agent", True)

    if algo == 'DDPG':
        agent = DDPG("MlpPolicy", env, verbose=1, learning_rate=0.001, seed=training_seed)
    elif algo == 'SAC':
        agent = SAC("MlpPolicy", env, verbose=1, learning_rate=0.001, seed=training_seed)
    elif algo == 'PPO':
        agent = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001, seed=training_seed)
    else:
        raise ValueError(f'Algorithm {algo} not supported')

    if train_agent:
        callback = LearningCurveCallback(log_file=f'.\learning_curves\{algo}_{system}_LC_rep.csv')
        agent.learn(total_timesteps=int(nsteps_train), callback=callback)
        agent.save(f'./policies/{algo}_{system}.zip')
    else:
        agent.set_parameters(f'./policies/{algo}_{system}')

    return agent

def get_rollout_data(agent):
    """
    Use when: You want to simulate and extract state-action-reward data after training.
    Example: "Evaluate the agent's policy through rollouts."
    Example: "Get the Q-values and state trajectories from the rollout."
    """
    algo = running_params.get("algo")
    reps = running_params.get("rollout_reps")
    evaluator, data = env.plot_rollout({algo: agent}, reps=reps, get_Q=True)
    return data

def cluster_states(agent, data):
    """
    Use when: You want to reduce the dimensionality of state space and perform unsupervised clustering.
    Example: "Cluster the agent's behavior using HDBSCAN on the state-action space."
    Example: "Visualize states using t-SNE and group into behavioral clusters."
    """
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

def feature_importance_global(agent, data, cluster_labels=None, lime=False, shap=True):
    """
    Use when: You want to understand which features most influence the agent’s policy across all states.
    Example: "Explain the global feature importance using SHAP."
    Example: "Visualize LIME-based feature importance for the trained agent."
    """
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
    """
    Use when: You want to inspect how features affected the agent's decision at a specific point.
    Example: "Provide local SHAP values for a single instance."
    Example: "What influenced the agent most at timestep 0?"
    """
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

def partial_dependence_plot(agent, data):
    """
    Use when: You want to examine how changing one input feature influences the agent's action.
    Example: "Plot ICE and PDP curves to understand sensitivity to temperature."
    Example: "How does action vary with concentration change?"
    """
    algo = running_params.get("algo")
    feature_names = env_params.get("feature_names")
    target = env_params.get("target")
    actor = agent.actor.mu
    X = data['DDPG']['x'].reshape(data['DDPG']['x'].shape[0], -1).T

    from explainer.PDP import PDP
    explainer = PDP(model=actor, bg=X, target=target, feature_names=feature_names, algo=algo, env_params=env_params, grid_points=100)
    ice_curves = explainer.explain(X=X)
    explainer.plot(ice_curves)

def trajectory_sensitivity(agent, data, t_query: float):
    """
    Use when: You want to simulate how small action perturbations influence future trajectory.
    Example: "Evaluate sensitivity of state trajectory to action perturbations at t=180."
    Example: "How robust is the policy to action noise?"
    """
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

def trajectory_counterfactual(agent, data, t_query: float, cf_actions: list):
    """
    Use when: You want to simulate a counterfactual scenario with manually chosen action.
    Example: "What would have happened if we had chosen action = 300 at t=180?"
    Example: "Show the trajectory if a different control input is applied."
    """

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
