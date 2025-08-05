import sys
sys.path.append("..")

from typing import Union
from callback import LearningCurveCallback
from stable_baselines3 import PPO, DDPG, SAC
from params import get_running_params, get_env_params
from stable_baselines3.common.base_class import BaseAlgorithm

running_params = get_running_params()
env, env_params = get_env_params(running_params['system'])

actions = env_params.get("actions")
algo = running_params.get("algo")
system = running_params.get("system")
gamma = running_params.get("gamma")

def train_agent(lr = 0.001, gamma = 0.9):
    """
    Train or load a reinforcement learning agent.
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
    Simulate and extract state-action-reward data after training.
    Args:
        agent (BaseAlgorithm): Trained RL agent
    Return:
        data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
    """
    algo = running_params.get("algo")
    reps = running_params.get("rollout_reps")
    evaluator, data = env.plot_rollout({algo: agent}, reps=reps, get_Q=True)
    return data

def feature_importance_global(agent, data, action = None):
    """
    Use when: You want to understand which features most influence the agent’s policy across all states.
    Example:
        1) "How do the process states globally influence the agent's decisions?"
        2) "Which feature makes great contribution to the agent's decisions generally?"
    Args:
        agent (BaseAlgorithm): Trained RL agent
        data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
        action (str): Name of the agent action to be explained
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

    from explainer.FI_SHAP import SHAP
    explainer = SHAP(model=actor, bg=X, feature_names=feature_names, algo=algo, env_params=env_params)
    shap_values = explainer.explain(X=X)
    figures = explainer.plot(local = False,
                             action = action)
    return figures

def feature_importance_local(agent, data, t_query, action = None):
    """
    Use when: You want to inspect how features affected the agent's decision at a specific time point.
    Example:
        1) "How do the state variables influence actions at t=400?"
        2) "Which state variable influenced the agent's action most at timestep 120?"
    Args:
        agent (BaseAlgorithm): Trained RL agent
        data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
        t_query (Union[int, float]): Specific time point in simulation to be interpreted
        action (str): Name of the agent action to be explained
    Return:
        figures (list): List of resulting figures
    """
    step_index = round(t_query / env_params['delta_t'])

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

    from explainer.FI_SHAP import SHAP
    explainer = SHAP(model=actor, bg=X, feature_names=feature_names, algo=algo, env_params=env_params)
    instance = X[step_index, :]
    shap_values_local = explainer.explain(X=instance)
    figures = explainer.plot(local = True, action = action)
    return figures

def counterfactual_action(agent, t_begin, t_end, actions, values):
    """
    Use when: You want to simulate a counterfactual scenario with manually chosen action.
    Example:
        1) "Why don't we apply a different action of a=100 at t=400 instead?"
        2) "What would have happened if we had chosen action = 300 from t=200 to t=400?"
    Args:
        agent (BaseAlgorithm): Trained RL agent
        t_begin (Union[float, int]): First time step within the simulation interval to be interpreted
        t_end (Union[float, int]): Last time step within the simulation interval to be interpreted
        actions (list): List of actions to be perturbed
        values (list): List of counterfactual values for each action
    Return:
        figures (list): List of resulting figures
        figures_q (list, optional): List of figures, which compare the decomposed rewards of actual and counterfactual policies
    """
    from explainer.CF_action import cf_by_action
    figures, data = cf_by_action(
        t_begin=t_begin,
        t_end=t_end,
        actions = actions,
        values = values,
        policy=agent,
        horizon=20)
    # figures_q = q_decompose(data, t_begin)
    # return figures + figures_q
    return figures

def counterfactual_behavior(agent, t_begin, t_end, actions, alpha=1.0):
    """
    Use when: You want to simulate a counterfactual scenario with different control behaviors
    Example:
        1) "What would happen if the agent had a more aggressive behavior than our current agent?"
        2) "Why don't we just control the system in an opposite direction from t=4000 to 4200?"
    Args:
        agent (BaseAlgorithm): Trained RL agent
        t_begin (Union[float, int]): First time step within the simulation interval to be interpreted
        t_end (Union[float, int]): Last time step within the simulation interval to be interpreted
        actions (list): List of actions to be perturbed
        alpha (float): Smoothing parameter. 1.0 means default controller and higher values imply more aggressive control behavior.
    Return:
        figures (list): List of resulting figures
        figures_q (list, optional): List of figures, which compare the decomposed rewards of actual and counterfactual policies
    """
    from explainer.CF_behavior import cf_by_behavior
    figures, data = cf_by_behavior(
        t_begin=t_begin,
        t_end=t_end,
        actions = actions,
        alpha = alpha,
        policy=agent,
        horizon=20)
    # figures_q = q_decompose(data, t_begin)
    # return figures + figures_q
    return figures

def counterfactual_policy(agent, t_begin, t_end, team_conversation, message, use_debugger = True, max_retries=10):
    """
    Use when: You want to know what would the trajectory would be if we chose alternative policy,
            or to compare the optimal policy with other policies.
    Example:
        1) "What would the trajectory change if I use the on-off controller instead of the current RL policy?"
        2) "What if a simple threshold rule was applied between timestep 4000 and 4400, setting v1 = 0.1 whenever h3 > 0.9 and v1 = 3.0 whenever h3 < 0.4, instead of using the RL policy?"
    Args:
        agent (BaseAlgorithm): Trained RL agent
        t_begin (Union[float, int]): First time step within the simulation interval to be interpreted
        t_end (Union[float, int]): Last time step within the simulation interval to be interpreted
        team_conversation (list): Conversation history between agents
        message (str): Brief instruction for constructing the counterfactual policy. It is used as prompts for the Coder agent.
        use_debugger (bool): Whether to use the debugger for refining the code
        max_retries (int): Maximum number of iteration allowed for generating the decomposed reward function
    Returns:
        figures (list): List of resulting figures
        figures_q (list, optional): List of figures, which compare the decomposed rewards of actual and counterfactual policies
    """
    from explainer.CF_policy import cf_by_policy
    figures, data = cf_by_policy(
        t_begin=t_begin,
        t_end=t_end,
        policy=agent,
        message=message,
        team_conversation=team_conversation,
        max_retries=max_retries,
        use_debugger=use_debugger,
        horizon=20,
    )
    # figures_q = q_decompose(data, t_begin)
    # return figures + figures_q
    return figures

def q_decompose(data, t_query, team_conversation, max_retries=10):
    """
    Use when: You want to know the agent's intention behind certain action, by decomposing q values into both semantic and temporal dimension.
    Example:
        1) "What is the agent trying to achieve in the long run by doing this action at timestep 180?"
        2) "Why is the agent's intention behind the action at timestep 200?"
    Args:
        data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
        t_query (Union[int, float]): Specific time point in simulation to be interpreted
        team_conversation (list): Conversation history between agents
        max_retries (int): Maximum number of iteration allowed for generating the decomposed reward function
    Returns:
        figures (list): List of resulting figures
    """
    # Retrieve reward function from file_path-function_name

    actions_dict = {}
    for name, traj in data.items():
        actions_dict[name] = traj['u'].squeeze().T

    from explainer.EO_Qdecompose import decompose_forward
    figures, rewards = decompose_forward(
        t_query = t_query,
        a_trajs=actions_dict,
        env = env,
        team_conversation = team_conversation,
        max_retries = max_retries,
    )
    return figures

# %% Overall function executions
def function_execute(agent, data, team_conversation):
    function_execution = {
        "feature_importance_global": lambda args: feature_importance_global(
            agent, data,
            action=args.get("action", None),
        ),
        "feature_importance_local": lambda args: feature_importance_local(
            agent, data,
            action=args.get("action", None),
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
            message=args.get("message"),
            use_debugger=args.get("use_debugger",True),
        ),
        "q_decompose": lambda args: q_decompose(
            data,
            t_query=args.get("t_query"),
            team_conversation=team_conversation,
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
