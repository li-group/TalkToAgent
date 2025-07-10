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

# query = "How do the process states globally influence the agent's decisions of v1 by SHAP?" #
# query = "Which feature makes great contribution to the agent's decisions at timestep 150?" #
# query = "I want to know at which type of states have the low q values of an actor." #
# query = "What would happen if I execute 300˚C as Tc action value instead of optimal action at timestep 150?" #
# query = "What would happen if I reduce the value of v1 action to 2.5 from 4000 to 4200, instead of optimal action?" #
query = "What would happen if a more conservative control of 0.3 was taken from 4000 to 4200, instead of optimal policy?" #
# query = "What would happen if I slight vary v1 action value at timestep 200?" #
# query = "How would the action variable change if the state variables vary at timestep 200?" #
# query = "How does action vary with the state variables change generally?" #
# query = "What is the agent trying to achieve in the long run by doing this action at timestep 4000?" # # future_intention_policy
# query = "What if we use the bang-bang controller instead of the current RL policy? What hinders the bang-bang controller from using it?" # counterfactual_policy
# query = "Why don't we just set v1 as maximum when the h1 is below 0.2?" # counterfactual_policy

messages.append({"role": "user", "content": query})


# TODO: Flexibility - 만약 분류에 실패한다면? User interference를 통해 바로 잡고 memory에 반영해야지.

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
    figs = functions[fn_name](args)
else:
    print("No function call was triggered.")

# %% Summarize explanation results in natural language form
explainer_prompt = get_prompts('explainer_prompt').format(
    fn_name = fn_name,
    fn_description = get_fn_description(fn_name),
    figure_description = get_figure_description(fn_name),
    env_params=env_params,
    system_description=get_system_description(running_params.get("system")),
    max_tokens = 400
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
)
print(response.choices[0].message.content)
team_conversation.append({"agent": "explainer", "content": "Multi-modal explanations are generated."})

# %% 6.13. Meeting
# TODO: Online explanation에 대해서도 구현 (rollout을 진행하다 멈추고 "지금 왜 이렇게 행동한거야?")
# TODO: Coder 검증. policy의 output이 stable-baselines3의 output의 형태와 동일하도록 검증하는 agent 내지 function 구현
# TODO: Optichat이나 Faultexplainer 등을 참고해서 front-end를 구현
# TODO: CF policy를 from scratch가 아니라 기존의 policy로부터 고치고 싶을 수도 있잖아.

# TODO: 7.8. Interval 단위 action counterfactual 구현 완료. 다만, action 하나에 그칠 뿐만 아니라 action value도 하나밖에 지정을 못해 flexibility가 떨어진다. Agent를 이용해야할 듯

# %% Advanced LLM related tasks
# TODO: Follow-up question & reply 구현
# TODO: 실험 설계에 대해서도 고민해보기. 어떤 실험을 설계해야하는지?
#     2. Figure 기반 vs array 기반 LLM explainer, in terms of accuracy and economic(tokens).

# %% Process control or XRL related tasks
# TODO: Convergence analysis 언제쯤 setpoint에 도달할 것으로 예상하는지? Settling time analysis
# TODO: DQN 등의 value network에 대해서도 구현 - discretization 필요
# TODO: 7월에 걸쳐서 실제 engineer와 feedback 과정을 계속 해야할 것 같은데.

# TODO: 일반적인 제어에 관해서도 추가를 하는 게 좋을 것 같다. 예) 지금 이 상태에서 setpoint를 갑자기 올려버리면 어떻게 action을 하게 될지?
# TODO: LIME 대신 DT로 local feature importance를 구현해야 할 것 같다.
# TODO: General question -> 여러 explanation을 종합한 통합 설명 제공도 구현해야할 듯.
# TODO: Counterfactual generation.py라는 곳에 action counterfactual, gain counterfactual, policy counterfactual을 다 모아두는 게 어떨까?