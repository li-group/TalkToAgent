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
# query = input("Enter your query:)
# 1. Prepare environment and agent
running_params = running_params()
env, env_params = env_params(running_params.get("system"))
print(f"System: {running_params.get('system')}")

agent = train_agent(lr = running_params['learning_rate'],
                    gamma = running_params['gamma'])
data = get_rollout_data(agent)

# 2. Call OpenAI API with function calling enabled
tools = get_fn_json()
system_description_prompt = get_prompts('system_description_prompt').format(
    env_params=env_params,
    system_description=get_system_description(running_params.get("system")),
)
coordinator_prompt = get_prompts('coordinator_prompt')
messages = [
        {"role": "system", "content": system_description_prompt},
        {"role": "system", "content": coordinator_prompt},
        {"role": "user", "content": "How do the process states globally influence the agent's decisions of v1 by SHAP?"}
        # {"role": "user", "content": "Which feature makes great contribution to the agent's decisions at timestep 150?"}
        # {"role": "user", "content": "I want to know at which type of states have the low q values of an actor."}
        # {"role": "user", "content": "What would happen if I execute 300˚C as Tc action value instead of optimal action at timestep 150?"}
        # {"role": "user", "content": "What would happen if I execute 9.5 as v1 action value instead of optimal action at timestep 200?"}
        # {"role": "user", "content": "What would happen if I slight vary v1 action value at timestep 200?"}
        # {"role": "user", "content": "How would the action variable change if the state variables vary at timestep 200?"}
        # {"role": "user", "content": "How does action vary with the state variables change generally?"}
    ]

# TODO: Flexibility - 만약 분류에 실패한다면? User interference를 통해 바로 잡고 memory에 반영해야지.
# TODO: Natural language를 policy로 구현하는 방법이 있지 않을까?

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    functions=tools,
    function_call="auto"
)

# 3. Execute returned function call (if any)
functions = function_execute(agent, data)

choice = response.choices[0]
if choice.finish_reason == "function_call":
    fn_name = choice.message.function_call.name
    args = json.loads(choice.message.function_call.arguments)
    print(f"Calling function: {fn_name} with args: {args}")
    figs = functions[fn_name](args)
else:
    print("No function call was triggered.")

# raise ValueError

# %% 4. Summarize explanation results in natural language form
def encode_fig(fig):
    from io import BytesIO
    import base64
    def fig_to_bytes(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return buf
    buf = fig_to_bytes(fig)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

explainer_prompt = get_prompts('explainer_prompt').format(
    fn_name = fn_name,
    fn_description = get_fn_description(fn_name),
    figure_description = get_figure_description(fn_name),
    max_tokens = 400
)

messages.append(
        {"role": "user", "content": explainer_prompt}
)
# %%
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
    model="gpt-4.1",
    messages=messages,
)
print(response.choices[0].message.content)

# %% 6.13. Meeting
# TODO: range, feature 선택 등의 option도 추가해야 flexibility를 얻을 수 있을 듯.
# TODO: Online explanation에 대해서도 구현 (rollout을 진행하다 멈추고 "지금 왜 이렇게 행동한거야?")

# %% Advanced LLM related tasks
# TODO: Function caller (또는 coordinator)에 대한 prompt 정비 (필요 시)
# TODO: Follow-up question & reply 구현
# TODO: Code writer (Engineer in OptiChat) 구현.
# TODO: 실험 설계에 대해서도 고민해보기. 어떤 실험을 설계해야하는지?
#     2. Figure 기반 vs array 기반 LLM explainer, in terms of accuracy and economic(tokens).

# %% Process control or XRL related tasks
# TODO: 실제 process operator들이 할 수 있는 counterfactual에 대해 생각해보기
# TODO: Convergence analysis 언제쯤 setpoint에 도달할 것으로 예상하는지? Settling time analysis
# TODO: DQN 등의 value network에 대해서도 구현 - discretization 필요
# TODO: Long-term reward가 필요한 system에 대해서 생각해보기.
# TODO: 7월에 걸쳐서 실제 engineer와 feedback 과정을 계속 해야할 것 같은데.

# TODO: 일반적인 제어에 관해서도 추가를 하는 게 좋을 것 같다. 예) 지금 이 상태에서 setpoint를 갑자기 올려버리면 어떻게 action을 하게 될지?
