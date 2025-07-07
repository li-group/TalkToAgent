import os
from openai import OpenAI
from dotenv import load_dotenv
from internal_tools import (
    train_agent
)
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

evaluator, data = env.plot_rollout({ALGO : agent}, reps = 1, get_Q = False)


# %% Counterfactual policy generation
from sub_agents.Policy_generator import generate

message = """Would you make a Bang-bang controller that satisfies this logic below:
          1. When h1 is below setpoint, maximize the value of v1. Otherwise, minimize v1.
          2. When h2 is below setpoint, maximize the value of v2. Otherwise, minimize v2."""
CF_policy = generate(message)



import os
from openai import OpenAI
from dotenv import load_dotenv
from utils import py2func

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
    'CF_mode': 'policy',
    'step_index': 150,
    'CF_policy': CF_policy
}
_, data_cf = env.get_rollouts({'Counterfactual' : agent}, reps = 1, get_Q = False, cf_settings = cf_settings)

# Append counterfactual results to evaluator object
evaluator.n_pi += 1
evaluator.policies['Counterfactual'] = CF_policy
evaluator.data = data | data_cf

figures = [evaluator.plot_data(evaluator.data)]

raise ValueError



Q_DECOMPOSE = False

# %%
if Q_DECOMPOSE:
    from explainer.Q_decompose import decompose_forward
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
    from explainer.D3PG_offline import D3PG
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

