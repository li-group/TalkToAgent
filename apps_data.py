import os
from openai import OpenAI
import json
from dotenv import load_dotenv
from internal_tools_data import (
    train_agent,
    get_rollout_data,
    function_execute
)
from prompts import get_prompts, get_fn_json, get_fn_description, get_system_description, get_figure_description
from params import running_params, env_params
from utils import encode_fig

# %% OpenAI setting
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL = 'gpt-4.1'
# MODEL = 'gpt-4o'
print(f"========= XRL Explainer using {MODEL} model =========")

# 1. Prepare environment and agent
running_params = running_params()
env, env_params = env_params(running_params.get("system"))
print(f"System: {running_params.get('system')}")

agent = train_agent(lr = running_params['learning_rate'],
                    gamma = running_params['gamma'])
data = get_rollout_data(agent)

# 2. Call OpenAI API with function calling enabled
tools = get_fn_json()
coordinator_prompt = get_prompts('coordinator_prompt').format(
    env_params=env_params,
    system_description=get_system_description(running_params.get("system")),
)

team_conversation = []
messages = [{"role": "system", "content": coordinator_prompt}]

# query = "How do the process states globally influence the agent's decisions of v1?" # Global FI
# query = "Which state variable makes great contribution to the agent's decisions at timestep 4000?" # Local FI
# query = "How would the action variable change if the state variables vary at timestep 4000?" #
# query = "What is the agent trying to achieve in the long run by doing this action at timestep 4000?" # EO
query = "What would happen if I reduce the value of v1 action to 2.5 and v2 action to 7.5 from 4020 to 4220, instead of optimal action?" # CF_action
# query = "What would happen if a more conservative control of 0.3 was taken from 4000 to 4200, instead of optimal policy?" # CF_behavior
# query = "What would happen if an opposite control was taken from 4000 to 4200, instead of optimal policy?" # CF_behavior
# query = "What if we use the bang-bang controller instead of the current RL policy from 4000 to 4200? What hinders the bang-bang controller from using it?" # CF_policy
# query = "What would be the outcome if the agent had set the voltage of valve 1 (v1) to 3.0 at timestep 5200 instead of the action it actually took?"
# query = "How would the system have behaved if we had increased v2 slightly between timestep 4000 and 4200?"
# query = "Why don't we just set v1 a's maximum when the h1 is below 0.2?" # CF_policy
# query = "Why don't we have the MPC controller instead of current RL policy from t=4000 to 4200?" # CF_policy
# query = "At interval 800–900, what would be the result of replacing policy control with passive actions?"

messages.append({"role": "user", "content": query})

# Coordinator agent
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    functions=tools,
    function_call="auto"
)

# 3. Execute returned function call (if any)
functions = function_execute(agent, data, team_conversation)

choice = response.choices[0]
if choice.finish_reason == "function_call":
    fn_name = choice.message.function_call.name
    args = json.loads(choice.message.function_call.arguments)
    print(f"[Coordinator] Calling function: {fn_name} with args: {args}")
    team_conversation.append({"agent": "coordinator", "content": f"[Calling function: {fn_name} with args: {args}]"})
    data = functions[fn_name](args)
else:
    print("No function call was triggered.")

# %% Summarize explanation results in natural language form
explainer_prompt = f"""
You're an expert in both explainable reinforcement learning (XRL).
Your role is to explain the XRL results triggered by XRL functions in natural language form.

- Below are the name of the XRL function triggered and it's description:
    Function name:
        {fn_name}

    Function description:
        {get_fn_description(fn_name)}

- Also, for more clear explanation, the description of the system and its environment parameters are given as below:
    System description:
        {get_figure_description(fn_name)}

    Environment parameters:
        {env_params}

- Make sure to compare between actual and counterfactual trajectories, and what made counterfactual trajectory worse (or better) than the actual one.
- Make sure to emphasize how the XRL results relates to the task of chemical process control, based on the given system description.
- The explanation output must be concise and short enough (below {200} tokens), because users may be distracted by too much information.
- Try to concentrate on providing only the explanation results, not on additional importance of the explanation.

Make sure the explanation must be coherent and easy to understand for the users who are experts in chemical process,
but not quite informed at explainable artificial intelligence tools and their interpretations.  
"""

messages.append(
        {"role": "user", "content": explainer_prompt}
)

import json
import numpy as np

# numpy array를 list로 변환하는 helper
def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_list(v) for v in obj]
    return obj  # 나머지는 그대로

# data dict 안의 numpy array를 전부 list로 변환
data_serializable = numpy_to_list(data)

# JSON 직렬화 (pretty print 가능)
data_json_str = json.dumps(data_serializable)

# messages에 추가
# messages.append(
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "text",
#                 "text": f"Here is the full control data dictionary:\n{data_json_str}"
#             }
#         ]
#     }
# )

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
)
print(response.choices[0].message.content)
team_conversation.append({"agent": "explainer", "content": "Multi-modal explanations are generated."})
