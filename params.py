import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from src.pcgym import make_env
from custom_reward import (cstr_reward,
                           multistage_extraction_reward,
                           crystallization_reward,
                           four_tank_reward,
                           photo_production_reward,
                           biofilm_reward)

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
        # 'system': 'biofilm_reactor',
        'system': 'crystallization', # ['cstr', 'four_tank', 'photo_production', 'multistage_extraction', 'crystallization', 'biofilm_reactor']
        'train_agent': True, # Whether to train agents. If false, Load trained agents.
        'algo': 'SAC', # RL algorithm
        'nsteps_train': 5e4, # Total time steps during training
        'rollout_reps': 1, # Number of episodes for rollout data
        'learning_rate': 0.001,
        'gamma': 0.99
    }
    return running_params

def get_env_params(system):
    np.random.seed(21)
    if system == 'cstr':
        """
        Task: Regulation
        States  (3): [Ca, T, Error_Ca]
        Actions (1): [Tc]
        Target  (1): [Ca]
        """
        # Simulation parameters
        task = 'regulation'
        T = 100  # Total simulated time (min)
        nsteps = 100 # Total number of steps
        delta_t = T / nsteps  # Minutes per step
        reward = cstr_reward
        time_scale = 'min'

        # Action, observation space and initial point
        targets = ['Ca']
        action_space = {'low': np.array([295]),
                        'high': np.array([302])}
        observation_space = {'low': np.array([0.7, 300, -0.1]),
                             'high': np.array([1, 350, 0.1])}
        initial_point = np.array([0.8, 330, 0.0])

        r_scale = dict(zip(targets, [1e2 for _ in targets]))

        # Setting setpoints
        def make_SP(nsteps, targets):
            SP = {}
            for target in targets:
                setpoints = []
                for i in range(nsteps):
                    if i % 10 == 0:
                        setpoint = np.random.uniform(low=0.8, high=0.9)
                    setpoints.append(setpoint)
                SP[target] = setpoints
            return SP

    elif system == 'multistage_extraction':
        """
        Task: Regulation
        States (11): [X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, error_X5]
        Actions (2): [L, G]
        Target  (1): [X5]
        """
        # Simulation parameters
        task = 'regulation'
        T = 180
        nsteps = 180
        delta_t = T / nsteps  # Minutes per step
        reward = multistage_extraction_reward
        time_scale = 'min'

        targets = ['X5']
        action_space = {
            'low': np.array([5, 10]),
            'high': np.array([500, 1000])
        }
        observation_space = {
            'low': np.array([0] * 10 + [-1]),
            'high': np.array([1] * 10 + [1])
        }
        initial_point = np.array([0.55, 0.3, 0.45, 0.25, 0.4, 0.20, 0.35, 0.15, 0.25, 0.1, 0.0])

        r_scale = dict(zip(targets, [1 for _ in targets]))

        # Setting setpoints
        def make_SP(nsteps, targets):
            SP = {}
            SP_bound = {'X5': [0.3 ,0.7]}
            targets = SP_bound
            for target in targets:
                setpoints = []
                for i in range(nsteps):
                    if i % 15 == 0:
                        setpoint = np.random.uniform(low=SP_bound[target][0], high=SP_bound[target][1])
                    setpoints.append(setpoint)
                SP[target] = setpoints
            return SP

    elif system == 'crystallization':
        """
        Task: Regulation
        States  (9): [Mu0, Mu1, Mu2, Mu3, Conc, CV, Ln, error_CV, error_Ln]
        Actions (1): [Tc]
        Target  (2): [CV, Ln]
        """
        # Simulation parameters
        task = 'regulation'
        T = 30  # Total simulated time (min)
        nsteps = 30  # Total number of steps
        delta_t = T / nsteps  # Minutes per step
        reward = crystallization_reward
        time_scale = 'min'

        # Action, observation space and initial point
        targets = ['CV', 'Ln']
        action_space = {
            'low': np.array([-1]),
            'high': np.array([1])
        }
        action_space_act = {
            'low': np.array([10]),
            'high': np.array([40])
        }
        a_0 = 39
        a_delta = True
        # observation_space = {
        #     'low': np.array([0, 0, 0, 0, 0, 0, 0, -2, -20]),
        #     'high': np.array([1e5, 1e6, 1e7, 1e9, 0.5, 2, 20, 2, 20])
        # }
        # initial_point = np.array([1478.01, 22995.82, 1800863.24, 248516167.94, 0.1586, 0.5, 15, 1, 15])

        observation_space = {
            'low': np.array([0, 0, 0, 0, 0, 0, 0, -2, -20]),
            'high': np.array([1e20, 1e20, 1e20, 1e20, 0.5, 2, 20, 2, 20])
        }
        CV_0 = np.sqrt(1800863.24079725 * 1478.00986666666 / (22995.8230590611 ** 2) - 1)
        Ln_0 = 22995.8230590611 / (1478.00986666666 + 1e-6)
        initial_point = np.array([1478.01, 22995.82, 1800863.24, 248516167.94, 0.1586, CV_0, Ln_0, 0, 0])

        r_scale = {
            'CV': 1,
            'Ln': 4
        }

        # Setting setpoints
        def make_SP(nsteps, targets):
            SP = {}
            SP_bound = {'CV': [1, 1],
                        'Ln': [15, 15]}
            for target in targets:
                setpoints = []
                for i in range(nsteps):
                    if i % 60 == 0:
                        setpoint = np.random.uniform(low=SP_bound[target][0], high=SP_bound[target][1])
                    setpoints.append(setpoint)
                SP[target] = setpoints
            return SP

    elif system == 'biofilm_reactor':
        """
        Task: Regulation
        States  (17): [S1_1, S2_1, S3_1, O_1, S1_2, S2_2, S3_2, O_2, S1_3, S2_3, S3_3, O_3, S1_A, S2_A, S3_A, O_A, error_S2_A]
        Actions (5): [F, Fr, S1_F, S2_F, S3_F]
        Target  (1): [S2_A]
        """
        # Simulation parameters
        task = 'regulation'
        T = 100  # Total simulated time (min)
        nsteps = 100  # Total number of steps
        delta_t = T / nsteps  # Minutes per step
        reward = biofilm_reward
        time_scale = 'min'

        # Action, observation space and initial point
        targets = ['S2_A']
        action_space = {
            'low': np.array([0, 1, 0.05, 0.05, 0.05]),
            'high': np.array([10, 30, 1, 1, 1])
        }

        observation_space = {
            'low': np.array([0, 0, 0, 0] * 4 + [-10]),
            'high': np.array([10, 10, 10, 500] * 4 + [10])
        }
        initial_point = np.array([2,0.1,10,0.1] * 4 + [0])

        r_scale = dict(zip(targets,[1 for _ in targets]))

        # Setting setpoints
        def make_SP(nsteps, targets):
            SP = {}
            for target in targets:
                setpoints = []
                for i in range(nsteps):
                    if i % 20 == 0:
                        setpoint = np.random.uniform(low=2, high=8)
                    setpoints.append(setpoint)
                SP[target] = setpoints
            return SP

    elif system == 'four_tank':
        """
        Task: Regulation
        States  (6): [h1, h2, h3, h4, error_h1, error_h2]
        Actions (2): [v1, v2]
        Target  (2): [h1, h2]
        """
        # Simulation parameters
        task = 'regulation'
        T = 8000  # Total simulated time (sec)
        nsteps = 400  # Total number of steps
        delta_t = T / nsteps  # Seconds per step
        reward = four_tank_reward
        time_scale = 'sec'

        # Action, observation space and initial point
        targets = ['h1', 'h2']
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

        # Setting setpoints
        def make_SP(nsteps, targets):
            SP = {}
            for target in targets:
                setpoints = []
                for i in range(nsteps):
                    if i % 40 == 0:
                        setpoint = np.random.uniform(low=0.1, high=0.5)
                    setpoints.append(setpoint)
                SP[target] = setpoints
            return SP

    elif system == 'photo_production':
        """
        States  (3): [c_x, c_N, c_q]
        Actions (2): [I, F_N]
        Target  (1): [c_q]
        """
        task = 'maximization'
        T = 240
        nsteps = 12
        delta_t = T / nsteps  # Hours per step
        reward = photo_production_reward
        time_scale = 'hr'

        # No setpoints specified since it is maximization problem
        make_SP = lambda x: None
        targets = ["c_q"]

        action_space = {
            'low': np.array([120, 0]),
            'high': np.array([400, 40])
        }
        observation_space = {
            'low': np.array([0, 50, 0]),
            'high': np.array([20, 800, 0.18])
        }
        initial_point = np.array([0.1, 20.0, 0.01])

        r_scale = dict(zip(targets, [1e2 for _ in targets]))

    else:
        raise Exception(f'{system} is not a valid system.')

    # Define reward to be equal to the OCP (i.e. the same as the oracle)
    env_params = {
        'targets': targets,
        'N': nsteps,
        'tsim': T,
        'SP': make_SP,
        'delta_t': delta_t,
        'o_space': observation_space,
        'a_space': action_space,
        'x0': initial_point,
        'r_scale': r_scale,
        'model': system,
        'normalise_a': True,
        'normalise_o': True,
        'noise': True,
        'integration_method': 'casadi',
        'noise_percentage': 0.001,
        'custom_reward': reward,
        'task': task,
        'time_scale': time_scale,
    }

    if system == 'crystallization':
        env_params['a_delta'] = True
        env_params['a_0'] = a_0
        env_params['a_space_act'] = action_space_act

    env = make_env(env_params)
    if task == 'regulation':
        env_params['feature_names'] = env.model.info()["states"] + [f"Error_{target}" for target in targets]
    else:
        env_params['feature_names'] = env.model.info()["states"]
    env_params['actions'] = env.model.info()["inputs"]

    return env, env_params
