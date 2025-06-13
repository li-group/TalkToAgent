import sys
sys.path.append("..")

import torch
import numpy as np
from callback import LearningCurveCallback
from stable_baselines3 import PPO, DDPG, SAC
from params import running_params, env_params

running_params = running_params()
env, env_params = env_params(running_params['system'])
actions = env_params.get("actions")

def train_agent(lr = 0.001, gamma = 0.9):
    """
    Use when: You want to train or load a reinforcement learning agent on the specified environment.
    Example: "Train a DDPG agent for the CSTR environment."
    Example: "Load a pretrained PPO model and skip training."
    """
    algo = running_params.get("algo")
    system = running_params.get("system")

    training_seed = running_params.get("seed", 1)
    nsteps_train = running_params.get("nsteps_train", int(5e4))
    train_agent = running_params.get("train_agent", True)

    if algo == 'DDPG':
        agent = DDPG("MlpPolicy", env, learning_rate=lr, seed=training_seed, gamma=gamma, verbose=1)
    elif algo == 'SAC':
        agent = SAC("MlpPolicy", env, learning_rate=lr, seed=training_seed, gamma=gamma, verbose=1)
    elif algo == 'PPO':
        agent = PPO("MlpPolicy", env, learning_rate=lr, seed=training_seed, gamma=gamma, verbose=1)
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
    Use when: You want to perform unsupervised clustering of states and classify each cluster's characteristic.
    Example: "Cluster the agent's behavior using HDBSCAN on the state-action space."
    Example: "Visualize states using t-SNE and group into behavioral clusters."
    """
    feature_names = env_params.get("feature_names")
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
    fig_sct = reducer.plot_scatter_grid(X_reduced, hue=np.hstack([y[:, None], q, X[:, [0, 2]]]), hue_names=['y', 'q', target, f'errors_{target}'])

    # TODO: Settling time에 대한 분석
    cluster = Cluster(params, feature_names=feature_names)
    cluster_labels = cluster.cluster(X_reduced, algo='HDBSCAN')
    fig_clus = cluster.plot_scatter(X_reduced, cluster_labels)
    fig_vio = cluster.plot_violin(X, cluster_labels)
    return fig_sct, fig_clus, fig_vio

def feature_importance_global(agent, data, action = None, cluster_labels=None, lime=False, shap=True):
    """
    Use when: You want to understand which features most influence the agent’s policy across all states.
    Example: "Explain the global feature importance using SHAP."
    Example: "Visualize LIME-based feature importance for the trained agent."
    """
    algo = running_params.get("algo")
    feature_names = env_params.get("feature_names")
    actor = agent.actor.mu
    X = data['DDPG']['x'].reshape(data['DDPG']['x'].shape[0], -1).T

    if lime:
        from explainer.LIME import LIME
        explainer = LIME(model=actor, bg=X, feature_names=feature_names, algo=algo, env_params=env_params)
        lime_values = explainer.explain(X=X)
        fig = explainer.plot(lime_values)
        return [fig]

    if shap:
        from explainer.SHAP import SHAP
        explainer = SHAP(model=actor, bg=X, feature_names=feature_names, algo=algo, env_params=env_params)
        shap_values = explainer.explain(X=X)
        figures = explainer.plot(local = False,
                                 action = action,
                                 cluster_labels=cluster_labels)
        return figures

def feature_importance_local(agent, data, t_query, action = None):
    """
    Use when: You want to inspect how features affected the agent's decision at a specific time point.
    Example: "Provide local SHAP values for a single instance."
    Example: "What influenced the agent most at timestep 120?"
    """
    step_index = int(t_query // env_params['delta_t'])

    algo = running_params.get("algo")
    feature_names = env_params.get("feature_names")
    actor = agent.actor.mu
    X = data['DDPG']['x'].reshape(data['DDPG']['x'].shape[0], -1).T

    from explainer.SHAP import SHAP
    explainer = SHAP(model=actor, bg=X, feature_names=feature_names, algo=algo, env_params=env_params)
    instance = X[step_index, :]
    shap_values_local = explainer.explain(X=instance)
    figures = explainer.plot(local = True, action = action)
    return figures

def partial_dependence_plot_global(agent, data, action = None, features = None):
    """
    Use when: You want to examine how changing one input feature influences the agent's action.
    Example: "Plot ICE and PDP curves to understand sensitivity to temperature."
    Example: "How does action vary with concentration change generally?"
    """
    algo = running_params.get("algo")
    feature_names = env_params.get("feature_names")
    actor = agent.actor.mu
    X = data['DDPG']['x'].reshape(data['DDPG']['x'].shape[0], -1).T

    from explainer.PDP import PDP
    explainer = PDP(model=actor, bg=X, feature_names=feature_names, algo=algo, env_params=env_params, grid_points=100)
    ice_curves = explainer.explain(X=X, action=action, features=features)
    fig = explainer.plot(ice_curves)
    return [fig]

def partial_dependence_plot_local(agent, data, t_query, action = None, features = None):
    """
    Use when: You want to examine how changing one input feature AT SPECIFIC TIME POINT influences the agent's action.
    Example: "Plot ICE curves to understand sensitivity to temperature at timestep 180."
    Example: "How does action can vary with concentration change now?"
    """
    step_index = int(t_query // env_params['delta_t'])

    algo = running_params.get("algo")
    feature_names = env_params.get("feature_names")
    actor = agent.actor.mu
    X = data['DDPG']['x'].reshape(data['DDPG']['x'].shape[0], -1).T

    from explainer.PDP import PDP
    explainer = PDP(model=actor, bg=X, feature_names=feature_names, algo=algo, env_params=env_params,
                    grid_points=100)
    ice_curves = explainer.explain(X = X[step_index], action = action, features=features) # Specific data point instance
    fig = explainer.plot(ice_curves)
    return fig

def trajectory_sensitivity(agent, data, t_query: float, action = None):
    """
    Use when: You want to simulate how small action perturbations influence future trajectory.
    Example: "Evaluate sensitivity of state trajectory to action perturbations at t=180."
    Example: "How robust is the policy to action noise?"
    """
    algo = running_params.get("algo")
    from explainer.Futuretrajectory import sensitivity
    xs, us, qs, fig = sensitivity(t_query=t_query,
                             perturbs=[-0.2, -0.1, 0, 0.1, 0.2],
                             data=data,
                             env_params=env_params,
                             policy=agent,
                             algo=algo,
                             horizon=20)
    return [fig]

def trajectory_counterfactual(agent, data, t_query: float, cf_actions: list, action = None):
    """
    Use when: You want to simulate a counterfactual scenario with manually chosen action.
    Example: "What would have happened if we had chosen action = 300 at t=180?"
    Example: "Show the trajectory if a different control input is applied."
    """

    algo = running_params.get("algo")
    from explainer.Futuretrajectory import counterfactual
    fig = counterfactual(t_query=t_query,
                         a_cf=cf_actions,
                         action=action,
                         data=data,
                         env_params=env_params,
                         policy=agent,
                         algo=algo,
                         horizon=20)
    return fig


# %% Overall function executions
def function_execute(agent, data):
    function_execution = {
        "cluster_states": lambda args: cluster_states(agent, data),
        "feature_importance_global": lambda args: feature_importance_global(
            agent, data,
            cluster_labels=None,
            action=args.get("action", None),
            lime=args.get("lime", False),
            shap=args.get("shap", True)
        ),
        "feature_importance_local": lambda args: feature_importance_local(
            agent, data,
            action=args.get("action", None),
            t_query=args.get("t_query")
        ),
        "partial_dependence_plot_global": lambda args: partial_dependence_plot_global(
            agent, data,
            action=args.get("action", None),
            features=args.get("features", None),
        ),
        "partial_dependence_plot_local": lambda args: partial_dependence_plot_local(
            agent, data,
            action=args.get("action", None),
            features=args.get("features", None),
            t_query=args.get("t_query")
        ),
        "trajectory_sensitivity": lambda args: trajectory_sensitivity(
            agent, data,
            action=args.get("action", None),
            t_query=args.get("t_query")
        ),
        "trajectory_counterfactual": lambda args: trajectory_counterfactual(
            agent, data,
            t_query=args.get("t_query"),
            action=args.get("action", None),
            cf_actions=args.get("cf_actions")
        ),
        "raise_error": lambda args: raise_error(
            message=args.get("message")
        ),
    }
    return function_execution


def raise_error(message):
    """
    Raises error
    """
    raise Exception(message)