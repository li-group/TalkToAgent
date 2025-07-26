import sys
sys.path.append("..")

from typing import Union
from callback import LearningCurveCallback
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

def feature_importance_global(agent, data, action = None, cluster_labels=None):
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
    Return:
        figures (list): List of resulting figures
    """
    algo = running_params.get("algo")
    feature_names = env_params.get("feature_names")

    if algo == 'DDPG':
        actor = agent.actor.mu
    elif algo == 'SAC':
        from torch.nn import Sequential
        latent_pi = agent.actor.latent_pi
        mu = agent.actor.mu
        actor = Sequential(*latent_pi, mu) # Sequentially connect two networks

    X = data[algo]['x'].reshape(data[algo]['x'].shape[0], -1).T

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

    if algo == 'DDPG':
        actor = agent.actor.mu
    elif algo == 'SAC':
        from torch.nn import Sequential
        latent_pi = agent.actor.latent_pi
        mu = agent.actor.mu
        actor = Sequential(*latent_pi, mu) # Sequentially connect two networks

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

def counterfactual_action(agent, t_begin, t_end, actions, values):
    """
    Use when: You want to simulate a counterfactual scenario with manually chosen action.
    Example:
        1) "What would happen if we had chosen action = 300 at t=180?"
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
    from explainer.CF_action import cf_by_action
    figures, data = cf_by_action(
        t_begin=t_begin,
        t_end=t_end,
        actions = actions,
        values = values,
        policy=agent,
        horizon=20
    )
    return data

def counterfactual_behavior(agent, t_begin, t_end, actions, alpha=1.0):
    """
    Use when: You want to simulate a counterfactual scenario with different control behaviors
    Example:
        1) "What would the future states would change if we control the system in more conservative way?"
        2) "What would happen if the controller was more aggressive than our current controller?"
        3) "What if we controlled the system in the opposite way from t=4000 to 4200?"
    Args:
        agent (BaseAlgorithm): Trained RL agent
        t_begin (Union[float, int]): First time step within the simulation interval to be interpreted
        t_end (Union[float, int]): Last time step within the simulation interval to be interpreted
        actions (list): List of actions to be perturbed
        alpha (float): Controller behavior. 1.0 means default controller and higher values imply more aggressive control behavior.
    Return:
        figures (list): List of resulting figures
    """
    from explainer.CF_behavior import cf_by_behavior
    figures, data = cf_by_behavior(
        t_begin=t_begin,
        t_end=t_end,
        actions = actions,
        alpha = alpha,
        policy=agent,
        horizon=20,
    )
    rewards = q_decompose(agent, data, t_begin)
    return data, rewards

def counterfactual_policy(agent, t_begin, t_end, team_conversation, message, max_retries=10):
    """
    Use when: You want to know what would the trajectory would be if we chose alternative policy,
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
        figures (list): List of resulting figures
    """
    from explainer.CF_policy import cf_by_policy
    figures, data = cf_by_policy(
        t_begin=t_begin,
        t_end=t_end,
        policy=agent,
        message=message,
        team_conversation=team_conversation,
        max_retries=max_retries,
        horizon=20,
    )
    return data

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

    actions_dict = {}
    for name, traj in data.items():
        actions_dict[name] = traj['u'].squeeze().T

    from explainer.Q_decompose import decompose_forward
    figures, rewards = decompose_forward(
        t_query = t_query,
        actions_dict=actions_dict,
        env = env,
        new_reward_f = new_reward_f,
        component_names = component_names,
    )
    return rewards



# %% Overall function executions
def function_execute(agent, data, team_conversation):
    function_execution = {
        "feature_importance_global": lambda args: feature_importance_global(
            agent, data,
            cluster_labels=None,
            action=args.get("action", None),
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
        "counterfactual_action": lambda args: counterfactual_action(
            agent,
            t_begin=args.get("t_begin"),
            t_end=args.get("t_end"),
            actions=args.get("actions"),
            values=args.get("values")
        ),
        "counterfactual_behavior": lambda args: counterfactual_behavior(
            agent,
            t_begin=args.get("t_begin"),
            t_end=args.get("t_end"),
            actions=args.get("actions"),
            alpha=args.get("alpha")
        ),
        "counterfactual_policy": lambda args: counterfactual_policy(
            agent,
            t_begin=args.get("t_begin"),
            t_end=args.get("t_end"),
            team_conversation=team_conversation,
            message=args.get("message")
        ),
        "q_decompose": lambda args: q_decompose(
            agent, data,
            t_query=args.get("t_query"),
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
