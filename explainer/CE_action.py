import numpy as np
from src.pcgym import make_env

from params import get_running_params, get_env_params

running_params = get_running_params()
env, env_params = get_env_params(running_params['system'])

# %%
def ce_by_action(t_begin, t_end, actions, values, policy, horizon=10):
    """
    Contrastive analysis of action to future trajectories.
    i.e.) "What would the future states change if we execute this action at specific time step?"
          "Why does the policy made this action instead of this?"
    Get a rollout data of trained policy, except only for 't_begin<=t<=t_end', where we execute predefined contrastive action
    Args:
        t_begin (Optional[int, float]): Start of the time interval to be queried
        t_end (Optional[int, float]): End of the time interval to be queried
        actions (list): Action variables to be intervened
        values (list): Intervened values of each action variable
        policy (BaseAlgorithm): Trained RL actor, using stable-baselines3
        horizon (int): Length of future horizon to be explored
    Returns:
        figures (list): List of decomposed reward figures
        evaluator.data (dict): Forward rollout data of actual and contrastive scenarios
    """
    # Translate queried timesteps to indices
    begin_index = int(np.round(t_begin / env_params['delta_t']))
    end_index = int(np.round(t_end / env_params['delta_t']))
    len_indices = end_index - begin_index + 1
    horizon += len_indices # Re-adjusting horizon

    # Regenerate trajectory data with noise disabled
    env_params['noise'] = False  # For reproducibility
    env = make_env(env_params)
    figures = []

    evaluator, data = env.get_rollouts({'Actual': policy}, reps=1)

    # Obtain contrastive action trajectories
    orig_traj = data['Actual']['u'] # (action_dim, instances, n_reps=1)
    ce_traj = orig_traj.copy()

    for i, a in enumerate(actions):
        action_index = env_params['actions'].index(a)
        v = values[i]
        # Assert that the queried action values are within the action space
        assert (v <= env_params['a_space']['high'][action_index]) and (v >= env_params['a_space']['low'][action_index]),\
            "Contrastive scenario out of action space"
        ce_traj[action_index, begin_index:end_index + 1, :] = np.array([v for _ in range(len_indices)])[:,np.newaxis]

    # Obtain rollout data from contrastive action trajectories
    ce_settings = {
        'CE_mode': 'action',
        'begin_index': begin_index,
        'end_index': end_index,
        'ce_traj': ce_traj,
    }
    label = [f"{a}={v}" for a, v in zip(actions, values)]
    _, ce_data = env.get_rollouts({f'CE: {label}': policy}, reps=1, ce_settings=ce_settings)

    evaluator.n_pi += 1
    evaluator.policies[f'CE: {label}'] = policy
    evaluator.data = data | ce_data

    # Get rollout data results for actual & contrastive trajectories
    interval = [begin_index-1, begin_index + horizon] # Interval to watch the control results
    fig = evaluator.plot_data(evaluator.data, interval=interval)

    figures.append(fig)

    return figures, evaluator.data
