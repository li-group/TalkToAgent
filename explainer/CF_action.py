import numpy as np
from src.pcgym import make_env

from params import running_params, env_params

running_params = running_params()
env, env_params = env_params(running_params['system'])

# %%
def cf_by_action(t_begin, t_end, actions, values, policy, horizon=10):
    """
    Counterfactual analysis of action to future trajectories.
    i.e.) "What would the future states would change if we execute this action at specific time step?"
          "Why does the policy made this action instead of this?"
    - Get a rollout data of trained policy, except only for 't_begin<=t<=t_end', where we execute predefined counterfactual action
    """

    assert (values <= env_params['a_space']['high']).all() and (values >= env_params['a_space']['low']).all(),\
        "Counterfactual out of action space"

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
    orig_traj = data['Actual']['u'] # (action_dim, instances, n_reps=1)
    cf_traj = orig_traj.copy()

    for i, a in enumerate(actions):
        action_index = env_params['actions'].index(a)
        v = values[i]
        cf_traj[action_index, begin_index:end_index + 1, :] = np.array([v for _ in range(len_indices)])[:,np.newaxis]

    cf_settings = {
        'CF_mode': 'action',
        'begin_index': begin_index,
        'end_index': end_index,
        'cf_traj': cf_traj,
    }
    label = [f"{a}={v}" for a, v in zip(actions, values)]
    _, cf_data = env.get_rollouts({f'CF: {label}': policy}, reps=1, cf_settings=cf_settings, get_Q=True)

    evaluator.n_pi += 1
    evaluator.policies[f'CF: {label}'] = policy
    evaluator.data = data | cf_data

    interval = [begin_index-1, begin_index + horizon] # Interval to watch the control results
    fig = evaluator.plot_data(evaluator.data, interval=interval)

    figures.append(fig)

    return figures, evaluator.data
