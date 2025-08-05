import traceback
import numpy as np
import matplotlib.pyplot as plt

from params import get_running_params, get_env_params
from sub_agents.Coder import Coder
from sub_agents.Debugger import Debugger

running_params = get_running_params()
system = running_params['system']
env, env_params = get_env_params(running_params['system'])

plt.rcParams['font.family'] = 'Times New Roman'

# %%
def decompose_forward(t_query, a_trajs, env, team_conversation, max_retries, horizon=10, use_debugger=True):
    """
    Decompose the Q values into both temporal and component-wise dimension
    Args:
        t_query (Optional[int, float]): Timestep to be queried
        a_trajs (dict): Action trajectories of explained policies
        env (make_env object): Environment object
        team_conversation (list): Conversation history between agents
        max_retries (int): Maximum number of iteration allowed for generating the decomposed reward function
        horizon (int): Length of future horizon to be explored
        use_debugger (bool): Whether to use the debugger for refining the code
    Returns:
        fig (list): List of decomposed reward figures
        r_trajs (dict): Trajectories of decomposed rewards obtained for explained policies
    """

    # Initialization
    decomposer = Coder()
    debugger = Debugger()
    file_path = "./custom_reward.py"
    function_name = f"{running_params['system']}_reward"
    new_reward_f, component_names = decomposer.decompose(file_path, function_name)
    success = False
    trial = 0

    # Iterate until no errors are detected
    while not success and trial < max_retries:
        try:
            out_dim = len(component_names)
            r_trajs = {}

            step_index = int(np.round(t_query / env.env_params['delta_t']))

            for traj_name, actions in a_trajs.items():
                rewards = np.zeros((env.N, out_dim))
                o, r = env.reset()

                for i in range(env.N - 1):
                    a = env._scale_U(actions[i])
                    o, r, term, trunc, info = env.step(a)
                    rewards[i, :] = new_reward_f(env, env.state, a, con=None)

                rewards = rewards[step_index:, :]
                r_trajs[traj_name] = rewards

            fig = _plot_results(r_trajs, env.env_params, t_query, horizon, component_names)
            success = True

        # If errors are detected, refine the code.
        except Exception as e:
            trial += 1
            error_message = traceback.format_exc()
            error_type = type(e).__name__
            print(f"[Debugger] Error during rollout (trial {trial}):\n{str(e)}")
            team_conversation.append({"agent": "Debugger",
                                      "content": f"[Trial {trial}] Error during rollout",
                                      "error_message": str(e),
                                      "error_type": error_type
                                      })

            if use_debugger:
                guidance = debugger.debug(code, error_message)
                code = decomposer.refine_with_guidance(error_message, guidance)
            else:
                code = decomposer.refine_with_error(error_message) # Just use the error message

            team_conversation.append({"agent": "Coder",
                                      "content": f"[Trial {trial}] Refined reward code generated.",
                                      "code_length": len(code)
                                      })

    return fig, r_trajs

def _plot_results(r_trajs, env_params, t_query, horizon, component_names):
    """
    Plots the decomposed reward along the both temporal and component-wise dimension
    Args:
        r_trajs (dict): Trajectories of decomposed rewards obtained for explained policies
        env_params (dict): Environmental parameters
        t_query (Optional[int, float]): Timestep to be queried
        horizon (int): Length of future horizon to be explored
        component_names (list): List of reward components
    Returns:
        [fig] (list): List of decomposed reward figures
    """
    n_traj = len(r_trajs)
    colors = ['green', 'orangered', 'blue']

    fig, axes = plt.subplots(
        n_traj, 1,
        figsize=(8, 4 * n_traj),
        sharex=True
    )

    if n_traj == 1:
        axes = [axes]

    for ax, (traj_name, rewards) in zip(axes, r_trajs.items()):
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
