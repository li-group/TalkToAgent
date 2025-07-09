import sys
sys.path.append("..")

import torch
import numpy as np
from typing import Union
from callback import LearningCurveCallback
import traceback
from stable_baselines3 import PPO, DDPG, SAC
from params import running_params, env_params
from stable_baselines3.common.base_class import BaseAlgorithm

running_params = running_params()
env, env_params = env_params(running_params['system'])

actions = env_params.get("actions")
algo = running_params.get("algo")
system = running_params.get("system")
gamma = running_params.get("gamma")

def train_agent(lr = 0.001, gamma = 0.9):
    """
    Use when: You want to train or load a reinforcement learning agent on the specified environment.
    Example:
        1) "Train a DDPG agent for the CSTR environment."
        2) "Load a pretrained PPO model and skip training."
    Args:
        lr (float): Learning rate
        gamma (float): Discount factor when calculating Q values
    Returns:
        agent (BaseAlgorithm): Trained RL agent, using stable-baselines3
    """
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
    Example:
        1) "Evaluate the agent's policy through rollouts."
        2) "Get the Q-values and state trajectories from the rollout."
    Args:
        agent (BaseAlgorithm): Trained RL agent
    Return:
        data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
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
    Args:
        agent (BaseAlgorithm): Trained RL agent
        data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
    Return:
        [fig_sct, fig_clus, fig_vio]: List of figures
    """
    feature_names = env_params.get("feature_names")
    actor = agent.actor.mu
    X = data[algo]['x'].reshape(data[algo]['x'].shape[0], -1).T
    q = data[algo]['q'].reshape(data[algo]['q'].shape[0], -1).T

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
    return [fig_sct, fig_clus, fig_vio]

def feature_importance_global(agent, data, action = None, cluster_labels=None, lime=False, shap=True):
    """
    Use when: You want to understand which features most influence the agent’s policy across all states.
    Example:
        1) "Explain the global feature importance using SHAP."
        2) "Visualize LIME-based feature importance for the trained agent."
    Args:
        agent (BaseAlgorithm): Trained RL agent
        data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
        action (str): Name of the agent action to be explained
        cluster_labels (list) List of cluster labels of data points
        lime (bool): Whether to use LIME to extract feature importance
        shap (bool): Whether to use SHAP to extract feature importance
    Return:
        figures (list): List of resulting figures
    """
    algo = running_params.get("algo")
    feature_names = env_params.get("feature_names")
    # TODO: This code is only valid for DDPG. Make it adjustable to SAC.
    actor = agent.actor.mu
    # import torch.nn as nn
    # class DeterministicActorWrapper(nn.Module):
    #     def __init__(self, actor):
    #         super().__init__()
    #         self.actor = actor
    #
    #     def forward(self, x):
    #         return self.actor(x, deterministic=True)
    #
    # actor = agent.actor
    # actor = DeterministicActorWrapper(actor)
    # actor.predict(torch.tensor(X, dtype=torch.float32))
    X = data[algo]['x'].reshape(data[algo]['x'].shape[0], -1).T

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
    Example:
        1) "Provide local SHAP values for a single instance."
        2) "What influenced the agent most at timestep 120?"
    Args:
        agent (BaseAlgorithm): Trained RL agent
        data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
        t_query (Union[int, float]): Specific time point in simulation to be interpreted
        action (str): Name of the agent action to be explained
    Return:
        figures (list): List of resulting figures
    """
    step_index = int(t_query // env_params['delta_t'])

    algo = running_params.get("algo")
    feature_names = env_params.get("feature_names")
    actor = agent.actor.mu
    X = data[algo]['x'].reshape(data[algo]['x'].shape[0], -1).T

    from explainer.SHAP import SHAP
    explainer = SHAP(model=actor, bg=X, feature_names=feature_names, algo=algo, env_params=env_params)
    instance = X[step_index, :]
    shap_values_local = explainer.explain(X=instance)
    figures = explainer.plot(local = True, action = action)
    return figures

def partial_dependence_plot_global(agent, data, action = None, states = None):
    """
    Use when: You want to examine how changing one input feature influences the agent's action.
    Example:
        1) "Plot ICE and PDP curves to understand sensitivity to temperature."
        2) "How does action vary with concentration change generally?"
    Args:
        agent (BaseAlgorithm): Trained RL agent
        data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
        action (str): Name of the agent action to be explained
        states (list): List of states whose impact to action needs to be explained
    Return:
        figures (list): List of resulting figures
    """
    algo = running_params.get("algo")
    feature_names = env_params.get("feature_names")
    actor = agent.actor.mu
    X = data[algo]['x'].reshape(data[algo]['x'].shape[0], -1).T

    from explainer.PDP import PDP
    explainer = PDP(model=actor, bg=X, feature_names=feature_names, algo=algo, env_params=env_params, grid_points=100)
    ice_curves = explainer.explain(X=X, action=action, features=states)
    figures = explainer.plot(ice_curves)
    return figures

def partial_dependence_plot_local(agent, data, t_query, action = None, states= None):
    """
    Use when: You want to examine how changing one input feature AT SPECIFIC TIME POINT influences the agent's action.
    Example:
        1) "Plot ICE curves to understand sensitivity to temperature at timestep 180."
        2) "How does action can vary with concentration change now?"
    Args:
        agent (BaseAlgorithm): Trained RL agent
        data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
        t_query (Union[int, float]): Specific time point in simulation to be interpreted
        action (str): Name of the agent action to be explained
        states (list): List of states whose impact to action needs to be explained
    Return:
        figures (list): List of resulting figures
    """
    step_index = int(t_query // env_params['delta_t'])

    algo = running_params.get("algo")
    feature_names = env_params.get("feature_names")
    actor = agent.actor.mu
    X = data[algo]['x'].reshape(data[algo]['x'].shape[0], -1).T

    from explainer.PDP import PDP
    explainer = PDP(model=actor, bg=X, feature_names=feature_names, algo=algo, env_params=env_params,
                    grid_points=100)
    ice_curves = explainer.explain(X = X[step_index], action = action, features=states) # Specific data point instance
    figures = explainer.plot(ice_curves)
    return figures

def trajectory_sensitivity(agent, data, t_query, action):
    """
    Use when: You want to simulate how small action perturbations influence future trajectory.
    Example:
        1) "Evaluate sensitivity of state trajectory to action perturbations at t=180."
        2) "How robust is the policy to action noise?"
    Args:
        agent (BaseAlgorithm): Trained RL agent
        data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
        t_query (Union[int, float]): Specific time point in simulation to be interpreted
        action (str): Name of the agent action to be explained
    Return:
        figures (list): List of resulting figures
    """
    from explainer.Futuretrajectory import sensitivity
    figures = sensitivity(t_query=t_query,
                          perturbs=[-0.2, -0.1, 0, 0.1, 0.2],
                          action=action,
                          data=data,
                          env_params=env_params,
                          policy=agent,
                          algo=algo,
                          horizon=20)
    return figures

def trajectory_counterfactual(agent, t_begin, t_end, actions, values):
    """
    Use when: You want to simulate a counterfactual scenario with manually chosen action.
    Example:
        1) "What would have happened if we had chosen action = 300 at t=180?"
        2) "Show the trajectory if a different control input is applied."
    Args:
        agent (BaseAlgorithm): Trained RL agent
        t_begin (Union[float, int]): First time step within the simulation interval to be interpreted
        t_end (Union[float, int]): Last time step within the simulation interval to be interpreted
        actions (list): List of actions to be perturbed
        values (list): List of counterfactual values for each action
    Return:
        figures (list): List of resulting figures
    """
    from explainer.Futuretrajectory import cf_by_action
    figures = cf_by_action(
        t_begin=t_begin,
        t_end=t_end,
        actions = actions,
        values = values,
        policy=agent,
        horizon=20)
    return figures

def q_decompose(agent, data, t_query):
    """
    Use when: You want to know the agent's intention behind certain action, by decomposing q values into both semantic and temporal dimension.
    Example:
        1) "What is the agent trying to achieve in the long run by doing this action at timestep 180?"
        2) "Why is the agent's intention behind the action at timestep 200?"
    Args:
        agent (BaseAlgorithm): Trained RL agent
        data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
        t_query (Union[int, float]): Specific time point in simulation to be interpreted
    Returns:
        figures (list): List of resulting figures
    """
    # Retrieve reward function from file_path-function_name
    from sub_agents.Reward_decomposer import RewardDecomposer
    decomposer = RewardDecomposer()
    file_path = "./custom_reward.py"
    function_name = f"{running_params['system']}_reward"
    new_reward_f, component_names = decomposer.decompose(file_path, function_name)

    from explainer.Q_decompose import decompose_forward
    figures = decompose_forward(
        t_query = t_query,
        data = data,
        env = env,
        policy = agent,
        algo = algo,
        new_reward_f = new_reward_f,
        component_names = component_names,
        gamma = gamma,
    )
    return figures

def policy_counterfactual(agent, data, team_conversation, message, t_query=None):
    """
    Use when: You want to what would the trajectory would be if we chose alternative policy,
            or to compare the optimal policy with other policies.
    Example:
        1) "What would the trajectory change if I use the bang-bang controller instead of the current RL policy?"
        2) "Why don't we just use the PID controller instead of the RL policy?"
        3) "Would you compare the predicted trajectory between our RL policy and bang-bang controller after t-300?"
    Args:
        agent (BaseAlgorithm): Trained RL agent
        data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
        message (str): Brief instruction for constructing the counterfactual policy. It is used as prompts for the Coder agent.
        t_query (Union[int, float]): Specific time point in simulation to be interpreted
    Returns:

    """
    if t_query is None:
        t_query = 0

    from sub_agents.Policy_generator import PolicyGenerator
    from sub_agents.Evaluator import Evaluator
    generator = PolicyGenerator()
    evaluator = Evaluator()

    CF_policy, code = generator.generate(message, agent)
    print(f"[PolicyGenerator] Initial counterfactual policy generated")
    team_conversation.append({"agent": "PolicyGenerator", "summary": f"Initial policy generated", "full_content": generator.prev_codes[-1]})

    success = False
    max_retries = 10
    attempt = 0

    while not success and attempt < max_retries:
        try:
            # Running the simulation with counterfactual policy
            env_params['noise'] = False  # For reproducibility
            from src.pcgym import make_env
            env = make_env(env_params)

            step_index = int(t_query // env_params['delta_t'])
            cf_settings = {
                'CF_mode': 'policy',
                'step_index': step_index,
                'CF_policy': CF_policy
            }
            evaluator, data_cf = env.get_rollouts({'Counterfactual': agent}, reps=1, get_Q=False,
                                                  cf_settings=cf_settings)
            success = True

        except Exception as e:
            attempt += 1
            error_message = traceback.format_exc()

            guidance = evaluator.evaluate(code, error_message)

            print(f"[Evaluator] Error during rollout (attempt {attempt}):\n{error_message}")
            team_conversation.append({"agent": "Evaluator", "full_content": f"Error during rollout (attempt {attempt}):\n{error_message}"
                                      })

            # CF_policy = generator.refine(error_message)
            CF_policy = generator.refine_new(error_message, guidance)
            team_conversation.append({"agent": "PolicyGenerator", "summary": f"CF policy refined",
                                      "full_content": generator.prev_codes[-1]})

    print("[Evaluator] Rollout complete." if success else "[Evaluator] Failed after multiple attempts.")

    if success:
        # Append counterfactual results to evaluator object
        evaluator.n_pi += 1
        evaluator.policies[running_params['algo']] = agent
        evaluator.data = data | data_cf

        figures = [evaluator.plot_data(evaluator.data)]
        return figures



# %% Overall function executions
def function_execute(agent, data, team_conversation):
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
            states=args.get("features", None),
        ),
        "partial_dependence_plot_local": lambda args: partial_dependence_plot_local(
            agent, data,
            action=args.get("action", None),
            states=args.get("features", None),
            t_query=args.get("t_query")
        ),
        "trajectory_sensitivity": lambda args: trajectory_sensitivity(
            agent, data,
            action=args.get("action", None),
            t_query=args.get("t_query")
        ),
        "trajectory_counterfactual": lambda args: trajectory_counterfactual(
            agent,
            t_begin=args.get("t_begin"),
            t_end=args.get("t_end"),
            actions=args.get("actions"),
            values=args.get("values")
        ),
        "q_decompose": lambda args: q_decompose(
            agent, data,
            t_query=args.get("t_query"),
        ),
        "policy_counterfactual": lambda args: policy_counterfactual(
            agent, data, team_conversation,
            t_query=args.get("t_query"),
            message=args.get("message")
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