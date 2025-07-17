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

# query = "How do the process states globally influence the agent's decisions of v1?" # Global FI
# query = "Which state variable makes great contribution to the agent's decisions at timestep 4020?" # Local FI
# query = "How would the action variable change if the state variables vary at timestep 4000?" #
query = "What is the agent trying to achieve in the long run by doing this action at timestep 4000?" # EO
# query = "What would happen if I reduce the value of v1 action to 2.5 and v2 action to 7.5 from 4020 to 4220, instead of optimal action?" # CF_action
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
    temperature=0
)

explanation = response.choices[0].message.content
print(explanation)
team_conversation.append({"agent": "explainer", "content": "Multi-modal explanations are generated."})

raise ValueError

# %%
explanation = """**1. States:**  
With the counterfactual action, h1 and h3 (purple) show a large overshoot, rising far above their setpoints, while h2 undershoots and then slowly recovers. h4 drops significantly below its actual trajectory. The actual controller (red) keeps all tank levels much closer to their setpoints, with minimal overshoot or undershoot."""

# explanation = """
# - **h1 and h2 (Target Levels):**
#   Under the counterfactual action, h1 rises sharply and overshoots its setpoint, while h2 initially increases but then undershoots and remains well below its setpoint. In contrast, the actual trajectory keeps both h1 and h2 much closer to their respective setpoints, with less overshoot and faster settling.
# """
# gt = "When using counterfactual policy, the trajectory for h1 state showed more overshooting than the original trajectory."
# gt = """The counterfactual trajectory showed less overshooting in h1 state than the original trajectory."""
true_statements = [
    "The counterfactual trajectory showed overshooting behavior in h1 state.",
    "The counterfactual trajectory did not show overshooting behavior in h2 state.",
    "The counterfactual trajectory showed undershooting behavior in h2 state.",
    "The actual trajectory settled faster than the counterfactual trajectory.",
    "The counterfactual was better in control performance than the original trajectory"
]

false_statements = [
    "The counterfactual trajectory did not show overshooting behavior in h1 state.",
    "The counterfactual trajectory showed overshooting behavior in h2 state.",
    "The counterfactual trajectory did not show undershooting behavior in h2 state.",
    "The counterfactual trajectory settled faster than the actual trajectory.",
    "The counterfactual was worse in control performance than the original trajectory"
]

from transformers import pipeline
print(f"Explanation: {explanation}")

true_results = []
false_results = []
nli = pipeline("text-classification", model="roberta-large-mnli", truncation=True)
for st in true_statements:
    result = nli({"text": explanation, "text_pair": st})
    true_results.append(result)
    print(f'Statement: {st}')
    print(result)

for st in false_statements:
    result = nli({"text": explanation, "text_pair": st})
    false_results.append(result)
    print(f'Statement: {st}')
    print(result)
raise ValueError

# %%
from bert_score import score
# gt = """Cats are so cute, aren't they?"""
# gt = """Michael Jordan is better than Lebron James"""
# gt = """When using counterfactual policy, the trajectory for h1 state showed more overshooting than the original trajectory."""
gt = """This is the controller of four-tank process control"""
from bert_score import BERTScorer
scorer = BERTScorer(model_type='bert-base-uncased')
P, R, F1 = scorer.score([explanation], [gt], verbose=True)
# P, R, F1 = score([explanation], [gt], lang="en", verbose=True)
print(f'Statement: {gt}')
print(f'P: {float(P.detach().numpy().squeeze()):.3f}, R: {float(R.detach().numpy().squeeze()):.3f}, F1: {float(F1.detach().numpy().squeeze()):.3f}')

# %% 6.13. Meeting
# TODO: Online explanation에 대해서도 구현 (rollout을 진행하다 멈추고 "지금 왜 이렇게 행동한거야?")
# TODO: Coder 검증. policy의 output이 stable-baselines3의 output의 형태와 동일하도록 검증하는 agent 내지 function 구현
# TODO: Optichat이나 Faultexplainer 등을 참고해서 front-end를 구현

# %% Advanced LLM related tasks
# TODO: Follow-up question & reply 구현
# TODO: 실험 설계에 대해서도 고민해보기. 어떤 실험을 설계해야하는지?
#     2. Figure 기반 vs array 기반 LLM explainer, in terms of accuracy and economic(tokens).

# %% Process control or XRL related tasks
# TODO: Convergence analysis 언제쯤 setpoint에 도달할 것으로 예상하는지? Settling time analysis
# TODO: 7월에 걸쳐서 실제 engineer와 feedback 과정을 계속 해야할 것 같은데.

# TODO: 일반적인 제어에 관해서도 추가를 하는 게 좋을 것 같다. 예) 지금 이 상태에서 setpoint를 갑자기 올려버리면 어떻게 action을 하게 될지?
# TODO: DT로도 local feature importance를 구현해야 할 것 같다.
# TODO: General question -> 여러 explanation을 종합한 통합 설명 제공도 구현해야할 듯.

# %% Future work
# TODO: DQN 등의 value network에 대해서도 구현 - discretization 필요

# TODO: CF를 이용한 trajectory를 봤을 때 hallucination이 심하다. 없는데 있다고 하는 것.
# TODO: CF의 figure description을 추가하여 몇 가지 category(settling time, overshoot 여부, opposite behavior, comparison)에 집중해서 설명을 진행하도록 해보기.
# TODO: 비교군이 single-timestep counterfactual이니깐 이 비교군은 아무 의미가 없다는 걸 보여줘야 할 것 같은데.