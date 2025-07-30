import os
from openai import OpenAI
from dotenv import load_dotenv
from internal_tools import (
    train_agent
)
from params import get_running_params, get_env_params

import numpy as np
np.random.seed(21)
# %% OpenAI setting
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL = 'gpt-4.1'
print(f"========= XRL Explainer using {MODEL} model =========")

# 1. Prepare environment and agent
running_params = get_running_params()
running_params['system'] = 'multistage_extraction'
env, env_params = get_env_params(running_params.get("system"))
print(f"System: {running_params.get('system')}")

agent = train_agent(lr = running_params['learning_rate'],
                    gamma = running_params['gamma'])

ALGO = running_params['algo']

evaluator, data = env.plot_rollout({ALGO : agent}, reps = 1, get_Q = False)

raise ValueError


from explainer.CF_behavior import cf_by_behavior
cf_traj = cf_by_behavior(t_begin = 4000,
                         t_end = 4500,
                         alpha= 0.2,
                         policy = agent,
                         actions = None,
                         horizon = 10)


from sub_agents.Trajectory_generator import TrajectoryGenerator
message = 'Generate a counterfactual trajectory that act opposite behavior for both v1 and v2 for time range from 200 to 250'
original_trajectory = data[running_params['algo']]['u'].squeeze()

import numpy as np
tgenerator = TrajectoryGenerator()
cf_traj = tgenerator.generate(message, original_trajectory)

while np.isclose(original_trajectory, cf_traj, 1e-3).all():
    error_message = "The original trajectory and counterfactual trajectory are the same. Please check whether you have perturbed the action trajectory correctly."

    tgenerator.refine(error_message)

from explainer.CF_action import cf_by_action

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(20,8))
plt.subplot(2,1,1)
plt.plot(data[ALGO]['u'][0], label = 'original')
plt.plot(cf_traj[0], label = 'counterfactual')
plt.grid()

plt.subplot(2,1,2)
plt.plot(data[ALGO]['u'][1], label = 'original')
plt.plot(cf_traj[1], label = 'counterfactual')
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()


# counterfactual(4000, 4200, cf_traj, data, env_params, agent, ALGO)

