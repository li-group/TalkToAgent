import numpy as np
from src.pcgym import make_env

def cf_trajectory_by_gain(t_begin, t_end, data, gain, env_params, algo, action=None, horizon=10):
    """
    Create an aggressive counterfactual trajectory from original action sequence u.

    Parameters:
    - u: np.ndarray of shape (action_dim, instance_dim)
    - gain: float, aggression factor (e.g., >1 for aggressive)
    - t_begin: int, start time index (inclusive)
    - t_end: int, end time index (inclusive)
    - action: int or None, index of action to perturb. If None, apply to all actions.

    Returns:
    - u_aggressive: np.ndarray of same shape as u, with perturbed segment
    """
    begin_index = int(np.round(t_begin / env_params['delta_t']))
    end_index = int(np.round(t_end / env_params['delta_t']))
    horizon += end_index - begin_index # Re-adjusting horizon

    orig_traj = data[algo]['u'].squeeze()
    action_dim, instance_dim = orig_traj.shape
    if end_index is None:
        end_index = instance_dim - 1

    cf_traj = orig_traj.copy()

    # For delta_u computation: prepend one step before t_begin
    t0 = max(begin_index - 1, 0)
    u_prev = orig_traj[:, t0]

    # Apply perturbation over time window
    if gain < 0:
        for step in range(begin_index, end_index + 1):
            delta = gain * (orig_traj[:, step] - u_prev)
            if action is None:
                cf_traj[:, step] = u_prev + delta
            else:
                action_index = env_params['actions'].index(action)
                cf_traj[action_index, step] = u_prev[action_index] + delta[action_index]

    else:
        # Autoregressive
        for step in range(begin_index, end_index + 1):
            delta = gain * (orig_traj[:, step] - u_prev)
            if action is None:
                cf_traj[:, step] = u_prev + delta
            else:
                action_index = env_params['actions'].index(action)
                cf_traj[action_index, step] = u_prev[action_index] + delta[action_index]
            u_prev = cf_traj[:, step]

    env_params['noise'] = False  # For reproducibility
    env = make_env(env_params)

    for label, a in actions_dict.items():
        cf_settings = {
            'CF_mode': 'action',
            'begin_index': begin_index,
            'end_index': end_index,
            'action_index': action_index,
            'CF_action': a}
        evaluator, cf_data = env.get_rollouts({'Counterfactual': policy}, reps=1, cf_settings=cf_settings, get_Q=True)

    evaluator.n_pi += 1
    # evaluator.policies['Counterfactual'] = policy
    evaluator.policies[algo] = policy
    evaluator.data = data | cf_data

    for al, traj in evaluator.data.items():
        for k, v in traj.items():
            if k != 'r':
                evaluator.data[al][k] = v[:, begin_index - 1:begin_index + horizon, :]
    interval = [begin_index - 1, begin_index + horizon]  # Interval to watch the control results

    fig = evaluator.plot_data(evaluator.data, interval=interval)
    # fig = evaluator.plot_data(evaluator.data)
    figures.append(fig)
