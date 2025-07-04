# %%
import numpy as np
import matplotlib.pyplot as plt
from src.pcgym import make_env

"""User는 전체 trajectory를 보고 특정 time step의 decision에 대해 궁금해 함."""

def sensitivity(t_query, perturbs, data, env_params, policy, algo, action=None, horizon=20):
    """
    Sensitivity analysis of action to future trajectories.
    i.e.) "What would the future states would change if we adjust an action at certain time step?"
          "Why does the policy made this action at this specific time?"
    - Get a rollout data of trained policy, except only for 't_query', where we adjust action values by 'perturbs'
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

        for pertb, a in actions_dict.items():
            cf_settings = {
                'CF_mode': 'action',
                'step_index': step_index,
                'action_index': action_index,
                'CF_action': a}
            evaluator, sim_traj = env.get_rollouts({algo: policy}, reps=1, cf_settings=cf_settings, get_Q=True)
            sim_trajs.append(sim_traj)

        xs = np.array([s[algo]['x'] for s in sim_trajs]).squeeze(-1).T
        us = np.array([s[algo]['u'] for s in sim_trajs]).squeeze(-1).T
        qs = np.array([s[algo]['q'] for s in sim_trajs]).squeeze(-1).T

        fig = plot_results(xs, us, qs, step_index, horizon, env, env_params, labels)
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

def counterfactual(t_query, a_cf, data, env_params, policy, algo, action = None, horizon=20):
    """
    Counterfactual analysis of action to future trajectories.
    i.e.) "What would the future states would change if we execute this action at specific time step?"
          "Why does the policy made this action instead of this?"
    - Get a rollout data of trained policy, except only for 't_query', where we execute predefined counterfactual action

    Args:
        t_query:
        a_cf:
        data:
        env_params:
        policy:
        algo:
        horizon:

    Returns:

    """
    assert (a_cf <= env_params['a_space']['high']).all() and (a_cf >= env_params['a_space']['low']).all(),\
        "Counterfactual out of action space"

    # Rollout data
    trajectory = data[algo]
    step_index = int(np.round(t_query / env_params['delta_t']))

    def _plot_result(a_traj, action_index, figures):
        action_history = a_traj.squeeze().T
        a_query = action_history[step_index]

        labels = ["Actual", "Counterfactual"]
        sim_trajs = []
        actions = [a_query, a_cf]
        actions_dict = dict(zip(labels, actions))  # Actual, and counterfactual actions

        env_params['noise'] = False # For reproducibility
        env = make_env(env_params)

        for pertb, a in actions_dict.items():
            cf_settings = {
                'CF_mode': 'action',
                'step_index': step_index,
                'action_index': action_index,
                'CF_action': a}
            evaluator, sim_traj = env.get_rollouts({algo: policy}, reps=1, cf_settings=cf_settings, get_Q=True)
            sim_trajs.append(sim_traj)

        xs = np.array([s[algo]['x'] for s in sim_trajs]).squeeze(-1).T
        us = np.array([s[algo]['u'] for s in sim_trajs]).squeeze(-1).T
        qs = np.array([s[algo]['q'] for s in sim_trajs]).squeeze(-1).T

        fig = plot_results(xs, us, qs, step_index, horizon, env, env_params, labels)
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

def plot_results(xs, us, qs, step_index, horizon, env, env_params, labels=None):
    t_query_adj = step_index * env_params['delta_t']
    xs_sliced = xs[:, :-len(env_params['targets']), :]  # Eliminating error term
    step_range = np.arange(step_index - 10, step_index + horizon)
    time_range = step_range * env_params['tsim'] / env_params['N']
    labels = labels if labels is not None else ["Label {i}".format(i=i) for i in range(xs.shape[-1])]

    cmap = plt.get_cmap('viridis')
    n_lines = len(labels)
    if n_lines == 1:
        colors = ['black']
    else:
        colors = [cmap(i / (n_lines - 1)) for i in range(n_lines)]

    total_vars = us.shape[1] + xs_sliced.shape[1] + qs.shape[1]
    fig, axes = plt.subplots(total_vars, 1, figsize=(12, 12), sharex=True)

    # Visualize control actions as zero-order input
    for i in range(us.shape[1]):
        for j in range(us.shape[2]):
            t_past = time_range[:11]
            u_past = us[step_index - 10:step_index + 1, i, j]
            t_past_zoh = np.repeat(t_past, 2)[1:]  # time duplicated and shifted
            u_past_zoh = np.repeat(u_past, 2)[:-1]
            axes[i].plot(t_past_zoh, u_past_zoh, color='black', linewidth=3)

            t_future = time_range[10:]
            u_future = us[step_index:step_index + horizon, i, j]
            t_future_zoh = np.repeat(t_future, 2)[1:]
            u_future_zoh = np.repeat(u_future, 2)[:-1]
            axes[i].plot(t_future_zoh, u_future_zoh, color=colors[j], label=labels[j])
        axes[i].axvline(t_query_adj, linestyle='--', color='red')
        axes[i].set_ylabel(env.model.info()['inputs'][i])
        axes[i].set_ylim([env_params['a_space']['low'][i], env_params['a_space']['high'][i]])
        axes[i].legend()
        axes[i].grid(True)

    lu = us.shape[1]
    for i in range(xs_sliced.shape[1]):
        for j in range(xs_sliced.shape[2]):
            axes[i + lu].plot(time_range[:11], xs_sliced[step_index - 10:step_index + 1, i, j], color='black',
                             linewidth=3)
            axes[i + lu].plot(time_range[10:], xs_sliced[step_index:step_index + horizon, i, j], color=colors[j],
                             label=labels[j])
        axes[i + lu].axvline(t_query_adj, linestyle='--', color='red')
        axes[i + lu].set_ylabel(env.model.info()['states'][i])
        axes[i + lu].legend()
        axes[i + lu].grid(True)
        axes[i + lu].set_ylim([env_params['o_space']['low'][i], env_params['o_space']['high'][i]])
        if env.model.info()["states"][i] in env.SP:
            axes[i + lu].step(
                time_range,
                env.SP[env.model.info()["states"][i]][step_range[0]:step_range[-1] + 1],
                where="post",
                color="black",
                linestyle="--",
                label="Set Point",
            )

    luxu = us.shape[1] + xs_sliced.shape[1]
    # TODO: Convert q to reward
    for i in range(qs.shape[1]):
        for j in range(qs.shape[2]):
            axes[i + luxu].plot(time_range[:11], qs[step_index - 10:step_index + 1, i, j], color='black', linewidth=3)
            axes[i + luxu].plot(time_range[10:], qs[step_index:step_index + horizon, i, j], color=colors[j], label=labels[j])
        axes[i + luxu].axvline(t_query_adj, linestyle='--', color='red')
        axes[i + luxu].set_ylabel('Q value')
        axes[i + luxu].legend()
        axes[i + luxu].grid(True)

    plt.xlabel('Time (min)')
    plt.tight_layout()
    plt.show()
    return fig
