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
from params import get_running_params, get_env_params
from utils import encode_fig

# %% OpenAI setting
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL = 'gpt-4.1'
# MODEL = 'gpt-4o'
print(f"========= XRL Explainer using {MODEL} model =========")

# 1. Prepare environment and agent
running_params = get_running_params()
env, env_params = get_env_params(running_params.get("system"))
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
# query = "Which state variable makes great contribution to the agent's decisions at timestep 4020?" # Local FI
# query = "What is the agent trying to achieve in the long run by doing this action at timestep 4000?" # EO
# query = "What would happen if I reduce the value of v1 action to 2.5 and v2 action to 7.5 from 4020 to 4220, instead of optimal action?" # CF_action
# query = "What would happen if a more conservative control of 0.3 was taken from 4000 to 4200, instead of optimal policy?" # CF_behavior
# query = "How would h1 variable change over time when we execute opposite control from 4000 to 4200?" # CF_behavior
query = "Why don't we execute opposite control from 4000 to 4200, to constrain the instant inverse response shown in h1?" # CF_behavior
# query = "What if we use the bang-bang controller instead of the current RL policy from 4000 to 4200? What hinders the bang-bang controller from using it?" # CF_policy
# query = "What would be the outcome if the agent had set the voltage of valve 1 (v1) to 3.0 at timestep 5200 instead of the action it actually took?"
# query = "How would the system have behaved if we had increased v2 slightly between timestep 4000 and 4200?"
# query = "Why don't we just set v1 a's maximum when the h1 is below 0.2?" # CF_policy
# query = "At interval 800–900, what would be the result of replacing policy control with passive actions?"

messages.append({"role": "user", "content": query})

# Coordinator agent
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    functions=tools,
    seed=21,
    temperature=0,
    top_p=0,
    function_call="auto"
)

# 3. Execute returned function call (if any)
functions = function_execute(agent, data, team_conversation)

choice = response.choices[0]
if choice.finish_reason == "function_call":
    fn_name = choice.message.function_call.name
    args = json.loads(choice.message.function_call.arguments)
    print(f"[Coordinator] Calling function: {fn_name} with args: {args}")
    team_conversation.append({"agent": "Coordinator", "content": f"[Calling function: {fn_name} with args: {args}]"})
    figs = functions[fn_name](args)
else:
    print("No function call was triggered.")

# %% Summarize explanation results in natural language form
explainer_prompt = get_prompts('explainer_prompt').format(
    user_query = query,
    fn_name = fn_name,
    fn_description = get_fn_description(fn_name),
    figure_description = get_figure_description(fn_name),
    env_params=env_params,
    system_description=get_system_description(running_params.get("system")),
    max_tokens = 200
)

messages.append(
        {"role": "user", "content": explainer_prompt}
)

for fig in figs:
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_fig(fig)}",
                        "detail": "auto"
                    }
                }
            ]
        }
    )

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    seed = 21,
    temperature=0,
    top_p=0
)

explanation = response.choices[0].message.content
print(explanation)
print(f"Usage: {response.usage.total_tokens}")
team_conversation.append({"agent": "Explainer", "content": "Multi-modal explanations are generated."})

# %%
# # messages.append({"role": "user", "content":"Given the results, explain about the control trajectory over h1 when executed Actual RL controller. How does it excel from the counterfactual trajectory?"})
# messages.append({"role": "user", "content":"Can you tell us how does the setpoint change of h1 and h2 over time?"})
# # messages.append({"role": "user", "content":"No, you are wrong. The setpoint is the dashed line you can see in the subplot of h1 and h2. In this case, can you tell us how does the setpoint change over time?"})
# # messages.append({"role": "user", "content":"How does the value of h1 change over time when executed opposite counterfactual trajectory"})
# response = client.chat.completions.create(
#     model=MODEL,
#     messages=messages,
#     seed = 21,
#     temperature=0
# )
# explanation = response.choices[0].message.content
# print(explanation)

# %% Advanced LLM related tasks
# TODO: Online explanation에 대해서도 구현 (rollout을 진행하다 멈추고 "지금 왜 이렇게 행동한거야?")
# TODO: Front-end 구체화. 다음 주에 해야할 듯.
# TODO: Follow-up question & reply 구현

# %% Process control or XRL related tasks
# TODO: Convergence analysis 언제쯤 setpoint에 도달할 것으로 예상하는지? Settling time analysis
# TODO: DT로도 local feature importance를 구현해야 할 것 같다.
# TODO: General question -> 여러 explanation을 종합한 통합 설명 제공도 구현해야할 듯.
# TODO: CF의 figure description을 추가하여 몇 가지 category(settling time, overshoot 여부, opposite behavior, comparison)에 집중해서 설명을 진행하도록 해보기.
# TODO: 비교군이 single-timestep counterfactual이니깐 이 비교군은 아무 의미가 없다는 걸 보여줘야 할 것 같은데.

# %% Future work
# TODO: DQN 등의 value network에 대해서도 구현 - discretization 필요
# TODO: 일반적인 제어에 관해서도 추가를 하는 게 좋을 것 같다. 예) 지금 이 상태에서 setpoint를 갑자기 올려버리면 어떻게 action을 하게 될지?
