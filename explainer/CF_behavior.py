import numpy as np
from src.pcgym import make_env

from params import running_params, env_params

running_params = running_params()
env, env_params = env_params(running_params['system'])

# %%
def cf_by_behavior(t_begin, t_end, alpha, actions, policy, horizon=10, return_figure=True):
    """
    Counterfactual analysis to future trajectories, according to its behavior.
    i.e.) "What would the future states would change if we control the system in more conservative way?"
          "What would happen if the controller was more aggressive than our current controller?"
          "What if we controlled the system in the opposite way from t=4000 to 4200?"
    - Get a rollout data of trained policy, except only for 't_begin<=t<=t_end', where we execute predefined counterfactual action
    """
    if actions is None:
        actions = env_params['actions']

    begin_index = int(np.round(t_begin / env_params['delta_t']))
    end_index = int(np.round(t_end / env_params['delta_t']))
    len_indices = end_index - begin_index + 1
    horizon += len_indices # Re-adjusting horizon

    env_params['noise'] = False  # For reproducibility
    env = make_env(env_params)
    figures = []

    # Regenerating trajectory data with noise disabled
    evaluator, data = env.get_rollouts({'Actual': policy}, reps=1, get_Q=True)

    # Action CF generations here
    orig_traj = data['Actual']['u'].squeeze() # (action_dim, instances)
    cf_traj = orig_traj.copy()

    action_dim, instance_dim = orig_traj.shape
    if end_index is None:
        end_index = instance_dim - 1

    # For delta_u computation: prepend one step before t_begin
    t0 = max(begin_index - 1, 0)

    # Apply perturbation over time window
    for a in actions:
        i = env_params['actions'].index(a)
        u_prev = orig_traj[i, t0]
        if alpha < 0:
            for step in range(begin_index, end_index + 1):
                delta = alpha * (orig_traj[i, step] - u_prev)
                cf_traj[i, step] = u_prev + delta

        else:
            # Autoregressive
            for step in range(begin_index, end_index + 1):
                delta = alpha * (orig_traj[i, step] - u_prev)
                cf_traj[i, step] = u_prev + delta
                u_prev = cf_traj[i, step]
                # u_prev = orig_traj[i, step]

    cf_settings = {
        'CF_mode': 'action',
        'begin_index': begin_index,
        'end_index': end_index,
        'cf_traj': cf_traj,
    }

    qual = 'Aggressive' if alpha > 1.0 else ('Opposite' if alpha < 0.0 else 'Conservative')
    _, cf_data = env.get_rollouts({f'{qual}, alpha = {alpha}': policy}, reps=1, cf_settings=cf_settings, get_Q=True)

    evaluator.n_pi += 1
    evaluator.policies[f'{qual}, alpha = {alpha}'] = policy
    evaluator.data = data | cf_data

    for al, traj in evaluator.data.items():
        for k, v in traj.items():
            evaluator.data[al][k] = v[:, begin_index - 1:begin_index + horizon, :]
    interval = [begin_index - 1, begin_index + horizon]  # Interval to watch the control results
    fig = evaluator.plot_data(evaluator.data, interval=interval)

    figures.append(fig)

    if return_figure:
        return figures
    return evaluator.data
