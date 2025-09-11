import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from src.pcgym import make_env
from custom_reward import cstr_reward, four_tank_reward, multistage_extraction_reward

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL = 'gpt-4.1' # default

def get_LLM_configs():
    return client, MODEL

def set_LLM_configs(model_name):
    global MODEL
    MODEL = model_name

def get_running_params():
    running_params = {
        'system': 'four_tank',
        'train_agent': True, # Whether to train agents. If false, Load trained agents.
        'algo': 'SAC', # RL algorithm
        'nsteps_train': 1e5, # Total time steps during training
        'rollout_reps': 1, # Number of episodes for rollout data
        'learning_rate': 0.001,
        'gamma': 0.9
    }
    return running_params

def get_env_params(system):
    np.random.seed(21)
    if system == 'cstr':
        # Simulation parameters
        T = 300  # Total simulated time (min)
        nsteps = 600  # Total number of steps
        delta_t = T / nsteps  # Minutes per step
        reward = cstr_reward

        # Setting setpoints
        SP = {}
        targets = ['Ca']
        for target in targets:
            setpoints = []
            for i in range(nsteps):
                if i % 20 == 0:
                    setpoint = np.random.uniform(low=0.8, high=0.9)
                setpoints.append(setpoint)
            SP[target] = setpoints

        # Action, observation space and initial point
        action_space = {'low': np.array([295]),
                        'high': np.array([302])}
        observation_space = {'low': np.array([0.7, 300, -0.1]),
                             'high': np.array([1, 350, 0.1])}
        initial_point = np.array([0.8, 330, 0.0])

        r_scale = dict(zip(targets, [1e3 for _ in targets]))

    elif system == 'four_tank':
        # Simulation parameters
        T = 8000  # Total simulated time (min)
        nsteps = 400  # Total number of steps
        delta_t = T / nsteps  # Minutes per step
        reward = four_tank_reward

        # Setting setpoints
        SP = {}
        targets = ['h1', 'h2']
        for target in targets:
            setpoints = []
            for i in range(nsteps):
                if i % 40 == 0:
                    setpoint = np.random.uniform(low=0.1, high=0.5)
                setpoints.append(setpoint)
            SP[target] = setpoints

        # Action, observation space and initial point
        action_space = {
            'low': np.array([0.1, 0.1]),
            'high': np.array([10, 10])
        }

        observation_space = {
            'low': np.array([0, ] * 4 + [-0.6] * 2),
            'high': np.array([0.6] * 4 + [0.6] * 2)
        }

        initial_point = np.array([0.141, 0.112, 0.072, 0.42, 0.0, 0.0])

        r_scale = dict(zip(targets,[1e2 for _ in targets]))

    elif system == 'multistage_extraction':
        T = 300
        nsteps = 300
        delta_t = T / nsteps  # Minutes per step
        reward = multistage_extraction_reward

        # x(np.ndarray): Current state[X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, error_X5, error_Y1]
        # u(np.ndarray): Input[L, G]

        # Setting setpoints
        SP = {}
        targets = ['X5', 'Y1']
        for target in targets:
            setpoints = []
            for i in range(nsteps):
                if i % 30 == 0:
                    setpoint = np.random.uniform(low=0.1, high=0.9)
                setpoints.append(setpoint)
            SP[target] = setpoints

        action_space = {
            'low': np.array([5, 10]),
            'high': np.array([500, 1000])
        }

        observation_space = {
            'low': np.array([0] * 10 + [-1] * 2),
            'high': np.array([1] * 10 + [1] * 2)
        }

        initial_point = np.array([0.55, 0.3, 0.45, 0.25, 0.4, 0.20, 0.35, 0.15, 0.25, 0.1, 0.0, 0.0])

        r_scale = {
            'X5': 1e2,
            'Y1': 1e2
        }

    else:
        raise Exception(f'{system} is not a valid system.')

    # Define reward to be equal to the OCP (i.e. the same as the oracle)
    env_params = {
        'targets': targets,
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
        'custom_reward': reward
    }

    env = make_env(env_params)
    env_params['feature_names'] = env.model.info()["states"] + [f"Error_{target}" for target in targets]
    env_params['actions'] = env.model.info()["inputs"]

    return env, env_params
