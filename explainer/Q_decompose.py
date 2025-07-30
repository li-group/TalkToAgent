import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
def decompose_forward(t_query, actions_dict, env, new_reward_f, component_names, horizon=10):
    """
    actions_dict: dict {traj_name -> trajectory (shape=(T,action_dim))}
    """
    out_dim = len(component_names)
    traj_rewards = {}

    step_index = int(np.round(t_query / env.env_params['delta_t']))

    for traj_name, actions in actions_dict.items():
        rewards = np.zeros((env.N, out_dim))
        o, r = env.reset()

        for i in range(env.N - 1):
            a = env._scale_U(actions[i])
            o, r, term, trunc, info = env.step(a)
            rewards[i, :] = new_reward_f(env, env.state, a, con=None)

        rewards = rewards[step_index:, :]
        traj_rewards[traj_name] = rewards

    fig = _plot_results(traj_rewards, env.env_params, t_query, horizon, component_names)

    return fig, traj_rewards


def _plot_results(traj_rewards_dict, env_params, t_query, horizon, component_names):
    n_traj = len(traj_rewards_dict)
    colors = ['green', 'orangered', 'blue']

    fig, axes = plt.subplots(
        n_traj, 1,
        figsize=(8, 4 * n_traj),
        sharex=True
    )

    if n_traj == 1:
        axes = [axes]

    for ax, (traj_name, rewards) in zip(axes, traj_rewards_dict.items()):
        T, C = rewards.shape
        dec_segment = rewards[:horizon]

        x = t_query + np.arange(horizon) * env_params['delta_t']
        bottoms = np.zeros(horizon)

        for i in range(C):
            ax.bar(
                x,
                dec_segment[:, i],
                bottom=bottoms,
                color=colors[i % len(colors)],
                label=component_names[i],
                width=env_params['delta_t'] * 0.8
            )
            bottoms += dec_segment[:, i]

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel("Reward", fontsize=16)
        ax.set_title(f"Expected Rewards (Trajectory: {traj_name}, From {t_query} sec, horizon: {horizon})", fontsize=16)
        ax.grid(True, axis='y')
        ax.legend(fontsize=14)

    axes[-1].set_xlabel("Time (sec)", fontsize=16)
    plt.tight_layout()
    plt.show()

    return [fig]
