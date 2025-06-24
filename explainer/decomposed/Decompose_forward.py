import numpy as np
import matplotlib.pyplot as plt

def decompose_forward(t_query, data, env, policy, algo, new_reward_f, out_features, component_names, gamma, deterministic = False, horizon=10):

    # TODO: Deterministic vs. Stochastic?
    #   Stochastic해서 multiple rollout을 또 얻어낼 수 있는거 아니야?

    trajectory = data[algo]
    actions = trajectory['u'].squeeze().T # (env.N, env.Nu)
    rewards = np.zeros((env.N, out_features))

    actor = policy.actor

    step_index = int(np.round(t_query / env.env_params['delta_t']))
    o, r = env.reset()

    for i in range(env.N - 1):
        if i < step_index:
            a = env._scale_U(actions[step_index])
            o, r, term, trunc, info = env.step(a)  # o_{t+1}
            rewards[i, :] = new_reward_f(env, env.state, a, con=None) # self.env.state: Unnormalized state

        else:
            # After t_query, simulate over agent and environment, and extract decomposed rewards
            a, _s = actor.predict(o, deterministic=deterministic)
            o, r, term, trunc, info = env.step(a) # o_{t+1}
            rewards[i, :] = new_reward_f(env, env.state, a, con=None) # self.env.state: Unnormalized state

    rewards = rewards[step_index:, :]
    discount = gamma ** np.arange(rewards.shape[0])  # shape (T,)
    dec_q = rewards * discount[:, np.newaxis]  # broadcast along axis 1
    figures = _plot_results(dec_q, env.env_params, t_query, horizon, component_names)

    return figures

def _plot_results(dec_q, env_params, t_query, horizon, component_names=None):
    """
    Args:
        dec_q: np.ndarray of shape (T, C) — Q values, decomposed in both component and temporal dimension
        t_query: int — starting timestep
        horizon: int — number of steps to plot
        component_names: optional list of component names
    """
    T, C = dec_q.shape

    # Slice the relevant segment
    dec_segment = dec_q[:horizon]

    x = t_query + np.arange(horizon) * env_params['delta_t']
    bottoms = np.zeros(horizon)

    # Component labels and colors
    if component_names is None:
        component_names = [f"comp{i}" for i in range(C)]

    colors = ['green', 'orangered', 'blue']

    fig, ax = plt.subplots(figsize=(12, 5))

    for i in range(C):
        ax.bar(x, dec_segment[:, i], bottom=bottoms, color=colors[i % len(colors)], label=component_names[i], width=env_params['delta_t'] * 0.8)
        bottoms += dec_segment[:, i]

    # Decoration
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Reward")
    ax.set_title(f"Q Decomposed into Rewards (From {t_query} second, for {horizon} time steps)")
    ax.legend()
    ax.grid(True, axis='y')

    plt.tight_layout()
    plt.show()
    return [fig]
