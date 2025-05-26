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
    partial_dependence_plot,
    trajectory_sensitivity,
    trajectory_counterfactual
)
from prompts import get_prompts, get_fn_json, get_fn_description

# %% OpenAI setting
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
print("========= XRL Explainer using gpt-4-0613 model =========")
# query = input("Enter your query: ")

# 1. Prepare environment and agent
agent = train_agent()
data = get_rollout_data(agent)

# 2. Call OpenAI API with function calling enabled
tools = get_fn_json()
# TODO: Function caller (또는 coordinator)에 대한 prompt 정비
# prompt = get_prompts('explainer_prompt').format()
response = client.chat.completions.create(
    model="gpt-4-0613",
    messages=[
        {"role": "system", "content": "You are a reinforcement learning explanation assistant."},
        {"role": "user", "content": "Can you show how features globally influence the agent's decisions by using SHAP?"}
    ],
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
    "feature_importance_local": lambda args: feature_importance_local(agent, data),
    "partial_dependence_plot": lambda args: partial_dependence_plot(agent, data),
    "trajectory_sensitivity": lambda args: trajectory_sensitivity(agent, data),
    "trajectory_counterfactual": lambda args: trajectory_counterfactual(agent, data)
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

prompt = get_prompts('explainer_prompt').format(
    fn_name = fn_name,
    fn_description = get_fn_description(fn_name),
    # explanation = explanation,
)
messages = [
        {"role": "system", "content": "You are an RL interpretation assistant."},
        {"role": "user", "content": prompt}
    ]
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
    messages=messages
)
print(response.choices[0].message.content)
