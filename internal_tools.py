import sys
sys.path.append("..")

from stable_baselines3 import PPO, DDPG, SAC

from callback import LearningCurveCallback
from params import get_running_params, get_env_params

# %%
running_params = get_running_params()
env, env_params = get_env_params(running_params['system'])

actions = env_params.get("actions")
algo = running_params.get("algo")
system = running_params.get("system")
gamma = running_params.get("gamma")

def train_agent(lr = 0.001, gamma = 0.9):
    """
    Train or load a reinforcement learning agent.
    Args:
        lr (float): Learning rate
        gamma (float): Discount factor when calculating Q values
    Returns:
        agent (BaseAlgorithm): Trained RL agent, using stable-baselines3
    """
    training_seed = running_params.get("seed", 1)
    nsteps_train = running_params.get("nsteps_train", int(1e4))
    train_agent = running_params.get("train_agent", True)

    if algo == 'DDPG':
        agent = DDPG("MlpPolicy", env, learning_rate=lr, seed=training_seed, gamma=gamma, verbose=1)
    elif algo == 'SAC':
        agent = SAC("MlpPolicy", env, learning_rate=lr, seed=training_seed, gamma=gamma, verbose=1)
    elif algo == 'PPO':
        agent = PPO("MlpPolicy", env, learning_rate=lr, seed=training_seed, gamma=gamma, verbose=1)
    else:
        raise ValueError(f'Algorithm {algo} not supported')

    if train_agent:
        callback = LearningCurveCallback(log_file=f'.\learning_curves\{algo}_{system}_LC_rep.csv')
        agent.learn(total_timesteps=int(nsteps_train), callback=callback)
        agent.save(f'./policies/{algo}_{system}.zip')

        # Plot actor - critic losses
        # plt.figure()
        # plt.plot(callback.actor_losses, label="Actor Loss")
        # plt.plot(callback.critic_losses, label="Critic Loss")
        # plt.legend()
        # plt.xlabel("Data Instances")
        # plt.ylabel("Loss")
        # plt.grid()
        # plt.tight_layout()
        # plt.show()

    else:
        agent.set_parameters(f'./policies/{algo}_{system}')

    return agent

def get_rollout_data(agent):
    """
    Simulate and extract state-action-reward data after training.
    Args:
        agent (BaseAlgorithm): Trained RL agent
    Return:
        data (dict): Trajectory data of r(Cumulated reward), x(observations), u(actions), and q(Q-values)
    """
    algo = running_params.get("algo")
    reps = running_params.get("rollout_reps")
    evaluator, data = env.plot_rollout({algo: agent}, reps=reps)
    return data

def raise_error(message):
    """
    Raises error
    """
    raise Exception(message)
