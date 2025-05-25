import os
from openai import OpenAI
import json
import pandas as pd
import numpy as np
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
from prompts import get_prompts

# %% OpenAI setting
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
print("========= XRL Explainer using gpt-4-0613 model =========")
query = input("Enter your query: ")

# 1. Prepare environment and agent
agent = train_agent()
data = get_rollout_data(agent)

# 2. Call OpenAI API with function calling enabled
from prompts import get_fn_json
tools = get_fn_json()
response = client.chat.completions.create(
    model="gpt-4-0613",
    messages=[
        {"role": "system", "content": "You are a reinforcement learning explanation assistant."},
        {"role": "user", "content": "Can you show how features globally influence the agent's decisions?"}
    ],
    functions=tools,
    function_call="auto"
)

# 5. Execute returned function call (if any)
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
    explanations = functions[fn_name](args)
else:
    print("No function call was triggered.")
raise ValueError

# 6. Summarize explanation results in natural language form
if isinstance(explanations, pd.DataFrame):
    explanation = data.to_markdown(index=False)
elif isinstance(data, np.ndarray):
    df = pd.DataFrame(data)
    explanation = df.to_markdown(index=False)
else:
    explanation = str(data)

prompt = get_prompts('explainer_prompt').format(
    fn_name = fn_name,
    fn_description = fn_description,
    explanation = explanation,
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an RL interpretation assistant."},
        {"role": "user", "content": prompt}
    ]
)
print(response.choices[0].message.content)
