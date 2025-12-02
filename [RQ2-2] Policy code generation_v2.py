import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from internal_tools import (
    train_agent,
    get_rollout_data,
    function_execute
)
from prompts import get_prompts, get_fn_json, get_system_description
from params import get_running_params, get_env_params, get_LLM_configs, set_LLM_configs
from example_queries import get_queries

plt.rcParams['font.family'] = 'Times New Roman'

figure_dir = os.getcwd() + '/figures'
os.makedirs(figure_dir, exist_ok=True)
savedir = figure_dir + '/[RQ2-2] Policy generation'
os.makedirs(savedir, exist_ok=True)
result_dir = os.getcwd() + '/results'
os.makedirs(result_dir, exist_ok=True)

np.random.seed(21)

# %% Main parameters for experiments
MODELS = ['gpt-5.1'] # GPT models to be explored
USE_DEBUGGERS = [True, False] # Whether to use debugger or not

LOAD_RESULTS = False # Whether to load the results of the experiments. This expedites visualization without running the experiments again.
NUM_EXPERIMENTS = 1 # Number of independent experiments

# Prepare environment and agent
running_params = get_running_params()
system = running_params.get('system')
env, env_params = get_env_params(system)
print(f"System: {system}")

agent = train_agent(lr = running_params['learning_rate'],
                    gamma = running_params['gamma'])
data = get_rollout_data(agent)

# %% Constructing result
if not LOAD_RESULTS:
    total_iterations = {}
    total_failures = {}
    total_error_messages = {}
    total_error_types = {}

    # Run experiments over independent experiments
    for n in range(NUM_EXPERIMENTS):
        _, _, _, _, CE_P_queries = get_queries(system)

        tools = get_fn_json()
        coordinator_prompt = get_prompts('coordinator').format(
            env_params=env_params,
            system_description=get_system_description(running_params.get("system")),
        )

        iterations = {}
        failures = {}
        error_messages_result = {}
        error_types_result = {}

        # Execute contrastive policy generation for all model and debugger options
        for MODEL in MODELS:
            set_LLM_configs(MODEL)
            client, MODEL = get_LLM_configs()
            print(f"========= XRL Explainer using {MODEL} model =========")
            for USE_DEBUGGER in USE_DEBUGGERS:
                team_conversations = []
                error_messages = []
                error_types = []

                # Generate contrastive policy for all CE_P queries
                for i, query in enumerate(CE_P_queries):
                    query = "Use 'contrastive policy' tool to answer the following question:" + query
                    team_conversation = []
                    messages = [{"role": "system", "content": coordinator_prompt}]
                    messages.append({"role": "user", "content": query})
                    fn_name = ''

                    functions = function_execute(agent, data, team_conversation)

                    # If the coordinator did not use contrastive_policy, reprompt the coordinator to use it.
                    while fn_name != "contrastive_policy":
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=messages,
                            functions=tools,
                            function_call="auto",
                        )
                        choice = response.choices[0]
                        if choice.finish_reason == "function_call":
                            fn_name = choice.message.function_call.name

                    # Execute returned function call (if any)
                    choice = response.choices[0]
                    if choice.finish_reason == "function_call":
                        fn_name = choice.message.function_call.name
                        args = json.loads(choice.message.function_call.arguments)
                        args['use_debugger'] = USE_DEBUGGER
                        print(f"[Coordinator] Calling function: {fn_name} with args: {args}")
                        team_conversation.append(
                            {"agent": "coordinator", "content": f"[Calling function: {fn_name} with args: {args}]"})
                        figs = functions[fn_name](args)
                    else:
                        print("No function call was triggered.")

                    team_conversations.append(team_conversation)
                    error_messages.append([entry['error_message'] for entry in team_conversation if 'error_message' in entry] +
                                          [entry['status_message'] for entry in team_conversation if 'status_message' in entry])
                    error_types.append([entry['error_type'] for entry in team_conversation if 'error_type' in entry] +
                                          [entry['status'] for entry in team_conversation if 'status' in entry])

                # Collect all error and failure counts for each model and debugger configuration
                error_counts = [
                    sum(1 for item in conv if 'error_message' in item)
                    for conv in team_conversations
                ]

                failure_counts = [
                    sum(1 for item in conv if item.get('status', False) == "failure")
                    for conv in team_conversations
                ]

                average_errors = sum(error_counts) / len(error_counts) if error_counts else 0
                average_failures = sum(failure_counts) / len(failure_counts) if failure_counts else 0
                kk = '+debugger' if USE_DEBUGGER else ''
                print(f"Average error counts for model {MODEL}{kk}: {average_errors}")

                iterations[f'{MODEL}{kk}'] = average_errors
                failures[f'{MODEL}{kk}'] = average_failures
                error_messages_result[f'{MODEL}{kk}'] = error_messages
                error_types_result[f'{MODEL}{kk}'] = error_types

        # Aggregate results for all models and debugger options, within a single experiment iteration
        total_iterations[int(n)] = iterations
        total_failures[int(n)] = failures
        total_error_messages[int(n)] = error_messages_result
        total_error_types[int(n)] = error_types_result

    # Save results for all experiment iterations
    with open(result_dir + f"/[RQ2][{system}] total_iterations.pkl", "wb") as f:
        pickle.dump(total_iterations, f)
    with open(result_dir + f"/[RQ2][{system}] total_failures.pkl", "wb") as f:
        pickle.dump(total_failures, f)
    with open(result_dir + f"/[RQ2][{system}] total_error_messages.pkl", "wb") as f:
        pickle.dump(total_error_messages, f)
    with open(result_dir + f"/[RQ2][{system}] total_error_types.pkl", "wb") as f:
        pickle.dump(total_error_types, f)

    all_errors = [
        item
        for iterations in total_error_messages.values()
        for error_messages in iterations.values()
        for sublist in error_messages
        for item in sublist
    ]
    error_names = list(set([
        item
        for iterations in total_error_types.values()
        for error_types in iterations.values()
        for sublist in error_types
        for item in sublist
    ]))
    for key in ["Failure", "Success"]:
        if key in error_names:
            error_names.remove(key)
    error_names.append("Failure")
    error_names.append("Success")

# When LOAD_RESULTS = True, just load the results without running experiments
else:
    with open(result_dir + f"/[RQ2][{system}] total_iterations.pkl", "rb") as f:
        total_iterations = pickle.load(f)
    with open(result_dir + f"/[RQ2][{system}] total_failures.pkl", "rb") as f:
        total_failures = pickle.load(f)
    with open(result_dir + f"/[RQ2][{system}] total_error_messages.pkl", "rb") as f:
        total_error_messages = pickle.load(f)
    with open(result_dir + f"/[RQ2][{system}] total_error_types.pkl", "rb") as f:
        total_error_types = pickle.load(f)
    all_errors = [
        item
        for iterations in total_error_messages.values()
        for error_messages in iterations.values()
        for sublist in error_messages
        for item in sublist
    ]
    error_names = set([
        item
        for iterations in total_error_types.values()
        for error_types in iterations.values()
        for sublist in error_types
        for item in sublist
    ])
    all_embeddings = np.load(result_dir + "/[RQ2] all_embeddings.npy")

print("Done every iteration.")

# %% 1) Plotting average iterations & failures for contrastive policy generation
total_failures = {
    model: {
        error: float(np.mean(values))
        for error, values in subdict.items()
    }
    for model, subdict in total_failures.items()
}
total_iterations_r = pd.DataFrame.from_dict(total_iterations).T
total_failures_r = pd.DataFrame.from_dict(total_failures).T

iter_mean = total_iterations_r.mean(axis=0)
iter_std  = total_iterations_r.std(axis=0)

fail_mean = total_failures_r.mean(axis=0)
fail_std  = total_failures_r.std(axis=0)

models = total_iterations_r.columns
models_wrapped = [m.replace("+", "\n+") if "debugger" in m else m for m in models]
x = np.arange(len(models))

bar_width = 0.4

plt.figure(figsize=(9, 5))
plt.bar(
    x - bar_width/2,
    iter_mean,
    yerr=iter_std,
    capsize=5,
    width=bar_width,
    label="Attempts",
    color="olivedrab",
    alpha=1.0
)
plt.bar(
    x + bar_width/2,
    fail_mean,
    yerr=fail_std,
    capsize=5,
    width=bar_width,
    label="Failures",
    color="orangered",
    alpha=0.9
)
plt.xticks(x, models_wrapped, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Model", fontsize=20)
plt.ylabel("Attempt/Failure counts", fontsize=20)
plt.title("Efficiency in contrastive policy generations", fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=16)

plt.tight_layout()
plt.savefig(savedir + '/CE_generation.png')
plt.show()

unique_labels = list(range(len(error_names)))
error_to_cluster = {error_type: error_names.index(error_type)
                    for error_type in error_names}

# %% 3) Compute and plot error transition matrix
for MODEL in MODELS:
    for USE_DEBUGGER in USE_DEBUGGERS:
        kk = '+debugger' if USE_DEBUGGER else ''

        error_types = []
        for key in total_error_types.keys():
            error_types.extend(total_error_types[key][f'{MODEL}{kk}'])
        error_labels = [[error_to_cluster.get(e) for e in es] for es in error_types]
        def compute_transition_matrix(sequences_list):
            label_to_idx = {label: i for i, label in enumerate(unique_labels)}
            n = len(unique_labels)
            counts = np.zeros((n, n))

            for seq in sequences_list:
                # Build transition matrix
                for a, b in zip(seq[:-1], seq[1:]):
                    i = label_to_idx[a]
                    j = label_to_idx[b]
                    counts[i, j] += 1
            return unique_labels, counts

        unique_labels, W = compute_transition_matrix(error_labels)
        W = W.astype(int)

        W_visual = W[:-2, :]  # remove last two rows
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            W_visual,
            cbar=True,
            annot=True,
            linewidths=0.5,
            annot_kws={"size": 19},
            fmt='d',
            xticklabels=error_names,
            yticklabels=error_names[:-2],  # y-axis has two rows fewer
        )

        plt.xticks(rotation=90, fontsize=13)
        plt.yticks(rotation=0, fontsize=13)
        plt.xlabel("To", fontsize=17)
        plt.ylabel("From", fontsize=17)
        plt.title(f"{MODEL}{kk}", fontsize=18)
        plt.tight_layout()
        savename = savedir + f'/{MODEL}{kk}.png'
        plt.savefig(savename)
        plt.show()
