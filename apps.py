import os
from openai import OpenAI
import json
from dotenv import load_dotenv
from internal_tools import (
    train_agent,
    get_rollout_data,
    cluster_states,
    feature_importance_global,
    feature_importance_local,
    partial_dependence_plot_global,
    partial_dependence_plot_local,
    trajectory_sensitivity,
    trajectory_counterfactual
)
from prompts import get_prompts, get_fn_json, get_fn_description, get_system_description, get_figure_description
from params import running_params, env_params

# %% OpenAI setting
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL = 'gpt-4.1'
print(f"========= XRL Explainer using {MODEL} model =========")
# query = input("Enter your query:
# 1. Prepare environment and agent
agent = train_agent()
data = get_rollout_data(agent)

running_params = running_params()
env_params = env_params(running_params.get("system"))

# 2. Call OpenAI API with function calling enabled
tools = get_fn_json()
system_description_prompt = get_prompts('system_description_prompt').format(
    env_params=env_params,
    system_description=get_system_description(running_params.get("system")),
)
messages = [
        {"role": "system", "content": system_description_prompt},
        # {"role": "user", "content": "Can you show how features globally influence the agent's decisions by using SHAP?"}
        # {"role": "user", "content": "Can you show how features globally influence the agent's decisions by using LIME?"}
        # {"role": "user", "content": "Can you show which feature makes great contribution to the agent's decisions at timestep 150?"}
        {"role": "user", "content": "I want to know at which type of states have the low q values of an actor."}
        # {"role": "user", "content": "What would happen if I execute 300˚C as action value instead of optimal action at timestep 150?"}

    ]

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    functions=tools,
    function_call="auto"
)

# 3. Execute returned function call (if any)
functions = {
    "cluster_states": lambda args: cluster_states(agent, data),
    "feature_importance_global": lambda args: feature_importance_global(
        agent, data,
        cluster_labels=None,
        lime=args.get("lime", False),
        shap=args.get("shap", True)
    ),
    "feature_importance_local": lambda args: feature_importance_local(agent, data, t_query=args.get("t_query")),
    "partial_dependence_plot_global": lambda args: partial_dependence_plot_global(agent, data),
    "partial_dependence_plot_local": lambda args: partial_dependence_plot_local(agent, data, t_query=args.get("t_query")),
    "trajectory_sensitivity": lambda args: trajectory_sensitivity(agent, data, t_query=args.get("t_query")),
    "trajectory_counterfactual": lambda args: trajectory_counterfactual(agent, data, t_query=args.get("t_query"), cf_actions = args.get("cf_actions"))
}

choice = response.choices[0]
if choice.finish_reason == "function_call":
    fn_name = choice.message.function_call.name
    args = json.loads(choice.message.function_call.arguments)
    print(f"Calling function: {fn_name} with args: {args}")
    figs = functions[fn_name](args)
else:
    print("No function call was triggered.")

# 4. Summarize explanation results in natural language form
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
    max_tokens=400
)
print(response.choices[0].message.content)

# %% 6.6. Meeting
# TODO: Figure 기반, 또는 numpy(or pd.DataFrame) 기반 LLM explainer 비교. 정확도, taken time, token 사용량.


# %% Advanced LLM related tasks
# TODO: Function caller (또는 coordinator)에 대한 prompt 정비 (필요 시)
# TODO: Follow-up question & reply 구현
# TODO: Code writer (Engineer in OptiChat) 구현.
# TODO: Online explanation에 대해서도 구현 (rollout을 진행하다 멈추고 "지금 왜 이렇게 행동한거야?")
# TODO: 실험 설계에 대해서도 고민해보기. 어떤 실험을 설계해야하는지?
#     2. Figure 기반 vs array 기반 LLM explainer, in terms of accuracy and economic(tokens).

# %% Process control or XRL related tasks
# TODO: 실제 process operator들이 할 수 있는 counterfactual에 대해 생각해보기
# TODO: Convergence analysis 언제쯤 setpoint에 도달할 것으로 예상하는지?
# TODO: DQN 등의 value network에 대해서도 구현 - discretization 필요
# TODO: Long-term reward가 필요한 system에 대해서 생각해보기.
# TODO: 다른 시스템에 대해서도 extend
# TODO: 7월에 걸쳐서 실제 engineer와 feedback 과정을 계속 해야할 것 같은데.

# TODO: 일반적인 제어에 관해서도 추가를 하는 게 좋을 것 같다. 예) 지금 이 상태에서 setpoint를 갑자기 올려버리면 어떻게 action을 하게 될지?
