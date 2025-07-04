import os
import sys
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from typing import Union, Callable, Optional, List
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

current_dir = os.getcwd()

def train_agent(lr:float = 0.001, gamma:float = 0.9) -> BaseAlgorithm:
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
        callback = LearningCurveCallback(log_file=current_dir + f'\learning_curves\{algo}_{system}_LC_rep.csv')
        agent.learn(total_timesteps=int(nsteps_train), callback=callback)
        agent.save(current_dir + f'/policies/{algo}_{system}.zip')
    else:
        agent.set_parameters(current_dir + f'/policies/{algo}_{system}')

    return agent

def get_rollout_data(agent:BaseAlgorithm) -> dict:
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

def cluster_states(agent:BaseAlgorithm, data:dict):
    def _cluster_states() -> list:
        """
        Use when: You want to perform unsupervised clustering of states and classify each cluster's characteristic.
        Example:
            1) "Cluster the agent's behavior using HDBSCAN on the state-action space."
            2) "Visualize states using t-SNE and group into behavioral clusters."
        Return:
            _encode_fig(figures): List of encoded figures
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
        figures = [fig_sct, fig_clus, fig_vio]
        return _encode_fig(figures)
    return _cluster_states

def feature_importance_global(agent:BaseAlgorithm, data:dict):
    def _feature_importance_global(action:str = "", cluster_labels:List[int]= [],
                                   lime:bool=False, shap:bool=True) -> List[str]:
        """
        Use when: You want to understand which features most influence the agent’s policy across all states.
        Example:
            1) "Explain the global feature importance using SHAP."
            2) "Visualize LIME-based feature importance for the trained agent."
        Args:
            action (str): Name of the agent action to be explained
            cluster_labels (list) List of cluster labels of data points
            lime (bool): Whether to use LIME to extract feature importance
            shap (bool): Whether to use SHAP to extract feature importance
        Return:
            _encode_fig(figures): List of encoded figures
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
            figures = explainer.plot(lime_values)
            return _encode_fig(figures)

        if shap:
            from explainer.SHAP import SHAP
            explainer = SHAP(model=actor, bg=X, feature_names=feature_names, algo=algo, env_params=env_params)
            shap_values = explainer.explain(X=X)
            figures = explainer.plot(local = False,
                                     action = action,
                                     cluster_labels=cluster_labels)
            return _encode_fig(figures)
    return _feature_importance_global

def feature_importance_local(agent:BaseAlgorithm, data:dict):
    def _feature_importance_local(t_query:float, action:str = "") -> List[str]:
        """
        Use when: You want to inspect how features affected the agent's decision at a specific time point.
        Example:
            1) "Provide local SHAP values for a single instance."
            2) "What influenced the agent most at timestep 120?"
        Args:
            t_query (float): Specific time point in simulation to be interpreted
            action (str): Name of the agent action to be explained
        Return:
            _encode_fig(figures): List of encoded figures
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
        return _encode_fig(figures)
    return _feature_importance_local

def partial_dependence_plot_global(agent:BaseAlgorithm, data:dict):
    def _partial_dependence_plot_global(action:str = "", states:List[str] = []) -> List[str]:
        """
        Use when: You want to examine how changing one input feature influences the agent's action.
        Example:
            1) "Plot ICE and PDP curves to understand sensitivity to temperature."
            2) "How does action vary with concentration change generally?"
        Args:
            action (str): Name of the agent action to be explained
            states (list): List of states whose impact to action needs to be explained
        Return:
            _encode_fig(figures): List of encoded figures
        """
        algo = running_params.get("algo")
        feature_names = env_params.get("feature_names")
        actor = agent.actor.mu
        X = data[algo]['x'].reshape(data[algo]['x'].shape[0], -1).T

        from explainer.PDP import PDP
        explainer = PDP(model=actor, bg=X, feature_names=feature_names, algo=algo, env_params=env_params, grid_points=100)
        ice_curves = explainer.explain(X=X, action=action, features=states)
        figures = explainer.plot(ice_curves)
        return _encode_fig(figures)
    return _partial_dependence_plot_global

def partial_dependence_plot_local(agent:BaseAlgorithm, data:dict):
    def _partial_dependence_plot_local(t_query:float, action:str = "", states:List[str] = []) -> List[str]:
        """
        Use when: You want to examine how changing one input feature AT SPECIFIC TIME POINT influences the agent's action.
        Example:
            1) "Plot ICE curves to understand sensitivity to temperature at timestep 180."
            2) "How does action can vary with concentration change now?"
        Args:
            t_query (float): Specific time point in simulation to be interpreted
            action (str): Name of the agent action to be explained
            states (list): List of states whose impact to action needs to be explained
        Return:
            _encode_fig(figures): List of encoded figures
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
        return _encode_fig(figures)
    return _partial_dependence_plot_local

def trajectory_sensitivity(agent:BaseAlgorithm, data:dict):
    def _trajectory_sensitivity(t_query:float, action:str = "") -> List[str]:
        """
        Use when: You want to simulate how small action perturbations influence future trajectory.
        Example:
            1) "Evaluate sensitivity of state trajectory to action perturbations at t=180."
            2) "How robust is the policy to action noise?"
        Args:
            agent (BaseAlgorithm): Trained RL agent
            data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
            t_query (float): Specific time point in simulation to be interpreted
            action (str): Name of the agent action to be explained
        Return:
            _encode_fig(figures): List of encoded figures
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
        return _encode_fig(figures)
    return _trajectory_sensitivity

def trajectory_counterfactual(agent:BaseAlgorithm, data:dict):
    def _trajectory_counterfactual(t_query:float, cf_actions:List[int], action:str = "") -> List[str]:
        """
        Use when: You want to simulate a counterfactual scenario with manually chosen action.
        Example:
            1) "What would have happened if we had chosen action = 300 at t=180?"
            2) "Show the trajectory if a different control input is applied."
        Args:
            t_query (float): Specific time point in simulation to be interpreted
            cf_actions (list): List of counterfactual actions
            action (str): Name of the agent action to be explained
        Return:
            _encode_fig(figures): List of encoded figures
        """
        from explainer.Futuretrajectory import counterfactual
        figures = counterfactual(t_query=t_query,
                                 a_cf=cf_actions,
                                 action=action,
                                 data=data,
                                 env_params=env_params,
                                 policy=agent,
                                 algo=algo,
                                 horizon=20)
        return _encode_fig(figures)
    return _trajectory_counterfactual

def q_decompose(agent:BaseAlgorithm, data:dict):
    def _q_decompose(t_query:float, new_reward_f:Callable, component_names:List[int]) -> List[str]:
        """
        Use when: You want to know the agent's intention behind certain action, by decomposing q values into both semantic and temporal dimension.
        Example:
            1) "What is the agent trying to achieve in the long run by doing this action at timestep 180?"
            2) "Why is the agent's intention behind the action at timestep 200?"
        Args:
            agent (BaseAlgorithm): Trained RL agent
            data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
            t_query (float): Specific time point in simulation to be interpreted
            new_reward_f (Callable): New decomposed reward function, written in python
            component_names (list): List of names of components that consist the reward function
        Returns:
            _encode_fig(figures): List of encoded figures
        """
        # TODO: reward function을 file_path과 function_name으로부터 불러오기

        from explainer.decomposed.Decompose_forward import decompose_forward
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
        return _encode_fig(figures)
    return _q_decompose


# %% Overall function executions
def function_execute(agent, data):
    def _execute(fn_name):
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
                agent, data,
                t_query=args.get("t_query"),
                action=args.get("action", None),
                cf_actions=args.get("cf_actions")
            ),
            "q_decompose": lambda args: q_decompose(
                agent, data,
                t_query=args.get("t_query"),
                new_reward_f=args.get("new_reward_f"),
                component_names=args.get("component_names", None)
            ),
            "raise_error": lambda args: raise_error(
                message=args.get("message")
            ),
        }
        return function_execution[fn_name]
    return _execute


def raise_error(message):
    """
    Raises error
    """
    raise Exception(message)

def _encode_fig(figures):
    from io import BytesIO
    import base64
    fig_codes = []
    def fig_to_bytes(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return buf
    for fig in figures:
        buf = fig_to_bytes(fig)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        fig_codes.append(img_base64)
    return fig_codes