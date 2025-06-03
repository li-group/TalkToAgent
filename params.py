import numpy as np
from src.pcgym import make_env
from custom_reward import sp_track_reward
def running_params():
    running_params = {
        'system': 'cstr_ode',
        'train_agent': False, # Whether to train agents. If false, Load trained agents.
        'algo': 'DDPG', # RL algorithm
        'nsteps_train': 5e4, # Total time steps during training
        'rollout_reps': 10, # Number of episodes for rollout data
    }
    return running_params

def env_params(system):
    if system == 'cstr_ode':
        # Simulation parameteters
        T = 300  # Total simulated time (min)
        nsteps = 600  # Total number of steps
        delta_t = T / nsteps  # Minutes per step
        training_seed = 1

        # Setting setpoints
        target = 'Ca'
        setpoints = []
        for i in range(nsteps):
            if i % 20 == 0:
                setpoint = np.random.uniform(low=0.8, high=0.9)
            setpoints.append(setpoint)
        SP = {target: setpoints}
        print(setpoints)

        # Action, observation space and initial point
        action_space = {'low': np.array([295]),
                        'high': np.array([302])}
        observation_space = {'low': np.array([0.7, 300, -0.1]),
                             'high': np.array([1, 350, 0.1])}
        initial_point = np.array([0.8, 330, 0.0])

        r_scale = {target: 1e3}
    else:
        raise Exception(f'{system} is not a valid system.')

    # Define reward to be equal to the OCP (i.e the same as the oracle)
    env_params = {
        'target': target,
        'N': nsteps,
        'tsim': T,
        'SP': SP,
        'delta_t': delta_t,
        'o_space': observation_space,
        'a_space': action_space,
        'x0': initial_point,
        'r_scale': r_scale,
        'model': system,
        'normalise_a': True,
        'normalise_o': True,
        'noise': False,
        'integration_method': 'casadi',
        'noise_percentage': 0.001,
        'custom_reward': sp_track_reward
    }

    env = make_env(env_params)
    env_params['feature_names'] = env.model.info()["states"] + [f"Error_{target}"]

    return env, env_params
