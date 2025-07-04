import os
from openai import OpenAI
import json
from dotenv import load_dotenv
from internal_tools import (
    train_agent,
    get_rollout_data,
    function_execute
)
from prompts import get_prompts, get_fn_json, get_fn_description, get_system_description, get_figure_description
from params import running_params, env_params

# %% OpenAI setting
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL = 'gpt-4.1'
print(f"========= XRL Explainer using {MODEL} model =========")

# 1. Prepare environment and agent
running_params = running_params()
env, env_params = env_params(running_params.get("system"))
print(f"System: {running_params.get('system')}")

agent = train_agent(lr = running_params['learning_rate'],
                    gamma = running_params['gamma'])

ALGO = running_params['algo']

evaluator, data = env.plot_rollout({ALGO : agent}, reps = 1, get_Q = True)


# %% Counterfactual policy generation
import os
from openai import OpenAI
from dotenv import load_dotenv
from prompts import get_prompts, get_fn_json, get_fn_description, get_system_description, get_figure_description
from utils import str2py, py2func

policy_generator_prompt = """
You are a coding expert that generates rule-based control logic, based on user queries.
Your job is to write a code for the following Python class structure, named 'CF_policy': 

========================
class CF_policy():
    def __init__(self, env):
        self.env = env
        
    def predict(self, state, deterministic=True):
        # INSERT YOUR RULE-BASED LOGIC HERE
        return action    
        
========================

Please consider the following points when writing the 'predict' method:
- The output of the 'predict' method (i.e., the action) should be within the range \[-1,1\], as it will be used by an external function that expects scaled values.
    You can scale the actions values by using the method: 'self.env._scale_U(u)', if needed.
- The input 'state' is also scaled. Ensure that your if-then logic works with scaled variables.
    To scale raw state values, you may use: 'self.env._scale_X(x)'.
- If your code requires any additional Python modules, make sure to import them at the beginning of your code.
- Only return the 'CF_policy' class, without "'''" or "'''python".

For accurate policy generation, here are some descriptions of the control system:
{system_description}
"""

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL = 'gpt-4.1'
print(f"========= XRL Explainer using {MODEL} model =========")

# messages = [
#     {"role": "system", "content": policy_generator_prompt.format(system_description= get_system_description(SYSTEM))},
#     {"role": "user", "content": f"Would you make a Bang-bang controller that satisfies this logic below:"
#                                 f"1. When h1 is below setpoint, maximize the value of v1. Otherwise, minimize v1."
#                                 f"2. When h2 is below setpoint, maximize the value of v2. Otherwise, minimize v2."
#      }
# ]
#
# response = client.chat.completions.create(
#     model=MODEL,
#     messages=messages,
# )
#
# content = response.choices[0].message.content
#
# str2py(content, './example.py')

# %%
CF_policy = py2func('./example.py', 'CF_policy')(env)
cf_settings = {
    't_query': 1200,
    'CF_policy': CF_policy
}
evaluator, data_cf = env.get_rollouts({'Counterfactual' : agent}, reps = 1, get_Q = False)
raise ValueError

sim_trajs = [data, data_cf]
algos = [ALGO, 'Counterfactual']

xs = np.array([s[algos[i]]['x'] for i, s in enumerate(sim_trajs)]).squeeze(-1).T
us = np.array([s[algos[i]]['u'] for i, s in enumerate(sim_trajs)]).squeeze(-1).T

def plot_results(xs, us, step_index, horizon, env, env_params, labels=None):
    import matplotlib.pyplot as plt
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

    total_vars = us.shape[1] + xs_sliced.shape[1]
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

    plt.xlabel('Time (min)')
    plt.tight_layout()
    plt.show()
    return fig

step_index = 80
horizon = 10
fig = plot_results(xs, us, step_index, horizon, env, env_params, labels=None)



# %%
error_h1 = data_cf['Counterfactual']['x'][4]
error_h2 = data_cf['Counterfactual']['x'][5]

import matplotlib.pyplot as plt
plt.figure(figsize = (10,6))
plt.plot(error_h1, label = 'h1error')
plt.plot(error_h2, label = 'h2error')
plt.legend()
plt.tight_layout()
plt.ylabel('Errors')
plt.grid()
plt.show()









# %%
if Q_DECOMPOSE:
    from explainer.decomposed.Decompose_forward import decompose_forward
    from custom_reward import four_tank_reward_decomposed
    dec_rewards = decompose_forward(
        t_query = 200,
        data = data,
        env = env,
        policy = agent,
        algo = ALGO,
        new_reward_f = four_tank_reward_decomposed,
        gamma = 0.9,
        component_names= ['h1 control', 'h2 control', 'input reg'],
        deterministic = True
    )

    # %% Reward decomposition - post-hoc
    from explainer.decomposed.D3PG_offline import D3PG
    from custom_reward import four_tank_reward_decomposed
    rdec = D3PG(
        agent = agent,
        env = env,
        env_params = env_params,
        new_reward_f = four_tank_reward_decomposed,
        component_names= ['h1 control', 'h2 control', 'input reg'],
        out_features = 3,
        deterministic = False,
        n_rollouts = 1,
        n_updates = 10000
    )

    rdec.explain(t_query = 2800, data = data, algo = ALGO)

