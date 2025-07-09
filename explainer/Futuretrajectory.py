# %%
import numpy as np
import matplotlib.pyplot as plt
from src.pcgym import make_env

from params import running_params, env_params

running_params = running_params()
env, env_params = env_params(running_params['system'])

def sensitivity(t_query, perturbs, data, env_params, policy, algo, action=None, horizon=20):
    """
    Sensitivity analysis of action to future trajectories.
    i.e.) "What would the future states would change if we adjust an action at certain time step?"
          "Why does the policy made this action at this specific time?"
    - Get a rollout data of trained policy, except only for 't_begin<=t<=t_end', where we adjust action values by 'perturbs'
      arguments

    Args:
        t_query:
        perturbs:
        data:
        env_params:
        policy:
        algo:
        horizon:

    Returns:

    """
    # Rollout data
    trajectory = data[algo]
    step_index = int(np.round(t_query / env_params['delta_t']))

    def _plot_result(a_traj, action_index, figures):
        action_history = a_traj.squeeze().T
        a_query = action_history[step_index]

        labels = [f"{int(p * 100)}%" for p in perturbs]
        sim_trajs = []
        window_length = env_params['a_space']['high'] - env_params['a_space']['low']
        actions = [a_query + dev * window_length for dev in perturbs]
        actions_dict = dict(zip(labels, actions))  # Perturbation of actions
        actions_dict = {k: v for k, v in actions_dict.items() if
                        (v <= env_params['a_space']['high']).all() and (v >= env_params['a_space']['low']).all()}

        env_params['noise'] = False # For reproducibility
        env = make_env(env_params)

        for label, a in actions_dict.items():
            cf_settings = {
                'CF_mode': 'action',
                'step_index': step_index,
                'action_index': action_index,
                'CF_action': a}
            evaluator, sim_traj = env.get_rollouts({'Counterfactual': policy}, reps=1, cf_settings=cf_settings, get_Q=True)
            sim_trajs.append(sim_traj)

        evaluator.n_pi += 1
        evaluator.policies[running_params['algo']] = policy
        evaluator.data = data | sim_traj

        fig = evaluator.plot_data(evaluator.data)

        # xs = np.array([s[algo]['x'] for s in sim_trajs]).squeeze(-1).T
        # us = np.array([s[algo]['u'] for s in sim_trajs]).squeeze(-1).T
        # qs = np.array([s[algo]['q'] for s in sim_trajs]).squeeze(-1).T
        #
        # fig = plot_results(xs, us, qs, step_index, horizon, env, env_params, labels)
        figures.append(fig)

    figures = []
    # Extract action of query
    if not action:
        # If action not specified by LLM, we extract figures for all agent actions.
        for action in env_params['actions']:
            action_index = env_params['actions'].index(action)
            a_traj = trajectory['u'][action_index:action_index+1, :, :]
            _plot_result(a_traj, action_index, figures)
    else:
        action_index = env_params['actions'].index(action)
        a_traj = trajectory['u'][action_index:action_index + 1, :, :]
        _plot_result(a_traj, action_index, figures)

    return figures

def counterfactual(t_begin, t_end, a_cf, env_params, policy, action = None, horizon=10):
    """
    Counterfactual analysis of action to future trajectories.
    i.e.) "What would the future states would change if we execute this action at specific time step?"
          "Why does the policy made this action instead of this?"
    - Get a rollout data of trained policy, except only for 't_begin<=t<=t_end', where we execute predefined counterfactual action
    """

    assert (a_cf <= env_params['a_space']['high']).all() and (a_cf >= env_params['a_space']['low']).all(),\
        "Counterfactual out of action space"

    begin_index = int(np.round(t_begin / env_params['delta_t']))
    end_index = int(np.round(t_end / env_params['delta_t']))
    horizon += end_index - begin_index # Re-adjusting horizon

    env_params['noise'] = False  # For reproducibility
    env = make_env(env_params)

    def _plot_result(action_index, figures):
        evaluator, data = env.get_rollouts({'Actual': policy}, reps=1, get_Q=True)
        len_indices = end_index - begin_index + 1

        for a in a_cf:
            actions = [a for _ in range(len_indices)]
            cf_settings = {
                'CF_mode': 'action',
                'begin_index': begin_index,
                'end_index': end_index,
                'action_index': action_index,
                'CF_action': actions}
            _, cf_data = env.get_rollouts({f'CF: {action}={a}': policy}, reps=1, cf_settings=cf_settings, get_Q=True)

            evaluator.n_pi += 1
            evaluator.policies[f'CF: {action}={a}'] = policy
            data = data | cf_data
        evaluator.data = data

        for al, traj in evaluator.data.items():
            for k, v in traj.items():
                if k != 'r':
                    evaluator.data[al][k] = v[:,begin_index-1:begin_index + horizon,:]
        interval = [begin_index-1, begin_index + horizon] # Interval to watch the control results

        fig = evaluator.plot_data(evaluator.data, interval=interval)
        # fig = evaluator.plot_data(evaluator.data)
        figures.append(fig)

    figures = []
    # Extract action of query
    if not action:
        # If action not specified by LLM, we extract figures for all agent actions.
        for action in env_params['actions']:
            action_index = env_params['actions'].index(action)
            _plot_result(action_index, figures)
    else:
        action_index = env_params['actions'].index(action)
        _plot_result(action_index, figures)

    return figures
