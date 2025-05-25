# %%
import numpy as np
import matplotlib.pyplot as plt
from src.pcgym import make_env

"""User는 전체 trajectory를 보고 특정 time step의 decision에 대해 궁금해 함."""

def sensitivity(t_query, perturbs, data, env_params, policy, algo, horizon=20):
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
    step_index = int(t_query // env_params['delta_t'])

    # Extract action of query
    action_history = trajectory['u'].squeeze().T
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

    for pertb, action in actions_dict.items():
        sim_info = {'step_index': step_index,
                    'action': action}
        evaluator, sim_traj = env.get_rollouts({algo: policy}, reps=1, sim_info=sim_info, get_Q=True)
        sim_trajs.append(sim_traj)

    xs = np.array([s[algo]['x'] for s in sim_trajs]).squeeze().T
    us = np.array([s[algo]['u'] for s in sim_trajs]).squeeze().T[:, np.newaxis, :]
    qs = np.array([s[algo]['q'] for s in sim_trajs]).squeeze().T

    plot_results(xs, us, qs, step_index, horizon, env, env_params, labels)

    # plot_results(data['DDPG']['x'].transpose(1, 0, 2),
    #              data['DDPG']['u'].transpose(1, 0, 2),
    #              data['DDPG']['q'].transpose(1, 0, 2),
    #              step_index, horizon, env, env_params, labels)

    return xs, us, qs

def counterfactual(t_query, a_cf, data, env_params, policy, algo, horizon=20):
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
    step_index = int(t_query // env_params['delta_t'])

    # Extract action of query
    action_history = trajectory['u'].squeeze().T
    a_query = action_history[step_index]

    labels = ["Actual", "Counterfactual"]
    sim_trajs = []
    actions = [a_query, a_cf]
    actions_dict = dict(zip(labels, actions))  # Actual, and counterfactual actions

    env_params['noise'] = False # For reproducibility
    env = make_env(env_params)

    for pertb, action in actions_dict.items():
        sim_info = {'step_index': step_index,
                    'action': action}
        evaluator, sim_traj = env.get_rollouts({algo: policy}, reps=1, sim_info=sim_info, get_Q=True)
        sim_trajs.append(sim_traj)

    xs = np.array([s[algo]['x'] for s in sim_trajs]).squeeze().T
    us = np.array([s[algo]['u'] for s in sim_trajs]).squeeze().T[:, np.newaxis, :]
    qs = np.array([s[algo]['q'] for s in sim_trajs]).squeeze().T

    plot_results(xs, us, qs, step_index, horizon, env, env_params, labels)

    # plot_results(data['DDPG']['x'].transpose(1, 0, 2),
    #              data['DDPG']['u'].transpose(1, 0, 2),
    #              data['DDPG']['q'].transpose(1, 0, 2),
    #              step_index, horizon, env, env_params, labels)

    return xs, us, qs

def plot_results(xs, us, qs, step_index, horizon, env, env_params, labels=None):
    xs_sliced = xs[:, :-1, :]  # Eliminating error term
    time_range = np.arange(step_index - 10, step_index + horizon)
    labels = labels if labels is not None else ["Label {i}".format(i=i) for i in range(xs.shape[-1])]

    cmap = plt.get_cmap('viridis')
    n_lines = len(labels)
    if n_lines == 1:
        colors = ['black']
    else:
        colors = [cmap(i / (n_lines - 1)) for i in range(n_lines)]

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    for i in range(us.shape[-1]):
        axes[0].plot(time_range[:11], us[step_index - 10:step_index + 1, 0, i], color='black', linewidth=3)
        axes[0].plot(time_range[10:], us[step_index:step_index + horizon, 0, i], color=colors[i], label=labels[i])
    axes[0].axvline(step_index, linestyle='--', color='red')
    axes[0].set_ylabel(env.model.info()['inputs'][0])
    axes[0].set_ylim([env_params['a_space']['low'][0], env_params['a_space']['high'][0]])
    axes[0].legend()
    axes[0].grid(True)

    for i in range(xs_sliced.shape[1]):
        for j in range(xs_sliced.shape[2]):
            axes[i + 1].plot(time_range[:11], xs_sliced[step_index - 10:step_index + 1, i, j], color='black',
                             linewidth=3)
            axes[i + 1].plot(time_range[10:], xs_sliced[step_index:step_index + horizon, i, j], color=colors[j],
                             label=labels[j])
        axes[i + 1].axvline(step_index, linestyle='--', color='red')
        axes[i + 1].set_ylabel(env.model.info()['states'][i])
        axes[i + 1].legend()
        axes[i + 1].grid(True)
        axes[i + 1].set_ylim([env_params['o_space']['low'][i], env_params['o_space']['high'][i]])
        if env.model.info()["states"][i] in env.SP:
            axes[i + 1].step(
                time_range,
                env.SP[env.model.info()["states"][i]][time_range[0]:time_range[-1] + 1],
                where="post",
                color="black",
                linestyle="--",
                label="Set Point",
            )

    for i in range(qs.shape[1]):
        axes[3].plot(time_range[:11], qs[step_index - 10:step_index + 1, i], color='black', linewidth=3)
        axes[3].plot(time_range[10:], qs[step_index:step_index + horizon, i], color=colors[i], label=labels[i])
    axes[3].axvline(step_index, linestyle='--', color='red')
    axes[3].set_ylabel('Q value')
    axes[3].legend()
    axes[3].grid(True)

    plt.xlabel('Time (min)')
    plt.tight_layout()
    plt.show()
