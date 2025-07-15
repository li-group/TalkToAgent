import os
import sys
from openai import OpenAI
import json
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from internal_tools import (
    train_agent,
    get_rollout_data,
    function_execute
)
from prompts import get_prompts, get_fn_json, get_fn_description, get_system_description, get_figure_description
from params import running_params, env_params
from utils import encode_fig

os.chdir("..")

# %% OpenAI setting
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL = 'gpt-4.1'
# MODEL = 'gpt-4o'
print(f"========= XRL Explainer using {MODEL} model =========")

# Prepare environment and agent
running_params = running_params()
env, env_params = env_params(running_params.get("system"))
print(f"System: {running_params.get('system')}")

agent = train_agent(lr = running_params['learning_rate'],
                    gamma = running_params['gamma'])
data = get_rollout_data(agent)

# %% Constructing dataset
from submission.EX_Queries import get_queries
FI_queries, EO_queries, CF_A_queries, CF_B_queries, CF_P_queries = get_queries()

tools = get_fn_json()
coordinator_prompt = get_prompts('coordinator_prompt').format(
    env_params=env_params,
    system_description=get_system_description(running_params.get("system")),
)

true_labels = ["FI"] * 20 + ["EO"] * 20 + ["CF_A"] * 20 + ["CF_B"] * 20 + ["CF_P"] * 20
total_queries = list(FI_queries.keys()) + list(EO_queries.keys()) + list(CF_A_queries.keys()) + list(CF_B_queries.keys()) + CF_P_queries
predicted_labels = []

# %% Execution
for i, query in enumerate(total_queries):
    team_conversation = []
    messages = [{"role": "system", "content": coordinator_prompt}]
    messages.append({"role": "user", "content": query})

    # Coordinator agent
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        functions=tools,
        function_call="auto"
    )

    mapper = {
        'feature_importance_local': 'FI',
        'q_decompose': 'EO',
        'counterfactual_action': 'CF_A',
        'counterfactual_behavior': 'CF_B',
        'counterfactual_policy': 'CF_P'
    }

    choice = response.choices[0]
    if choice.finish_reason == "function_call":
        fn_name = choice.message.function_call.name
        args = json.loads(choice.message.function_call.arguments)
        predicted_label = mapper[fn_name]
        predicted_labels.append(predicted_label)
        true_label = true_labels[i]
        if predicted_label != true_label:
            print(f"Misclassification in query: {query} \n GroundTruth: {true_label} Prediction: {predicted_label} \n")
    else:
        print("No function call was triggered.")


# %% Results in confusion matrix
labels = ["FI", "EO", "CF_A", "CF_B", "CF_P"]

cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap='Blues', values_format='d')
plt.title("Tool selection Confusion Matrix")
plt.tight_layout()
plt.show()
