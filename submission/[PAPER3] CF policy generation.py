import os
import json
import pickle
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from netgraph import Graph
from collections import defaultdict

from internal_tools import (
    train_agent,
    get_rollout_data,
    function_execute
)
from prompts import get_prompts, get_fn_json, get_system_description
from params import running_params, env_params
from submission.EX_Queries import get_queries

font_size = 18
plt.rcParams['axes.titlesize'] = font_size
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size-2
plt.rcParams['ytick.labelsize'] = font_size-2
plt.rcParams['legend.fontsize'] = font_size-6
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

os.chdir("..")

# %% Main parameters for experiments
# MODELS = ['gpt-4.1', 'gpt-4o']
# USE_DEBUGGERS = [True, False]

MODELS = ['gpt-4.1']
USE_DEBUGGERS = [True]

LOAD_MESSAGES = False
NUM_EXPERIMENTS = 1

# %% OpenAI setting
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Prepare environment and agent
running_params = running_params()
env, env_params = env_params(running_params.get("system"))
print(f"System: {running_params.get('system')}")

agent = train_agent(lr = running_params['learning_rate'],
                    gamma = running_params['gamma'])
data = get_rollout_data(agent)

total_iterations = {}
total_failures = {}
total_error_messages = {}

# %% Constructing dataset
if not LOAD_MESSAGES:
    for n in range(NUM_EXPERIMENTS):
        _, _, _, _, CF_P_queries = get_queries()

        tools = get_fn_json()
        coordinator_prompt = get_prompts('coordinator_prompt').format(
            env_params=env_params,
            system_description=get_system_description(running_params.get("system")),
        )

        iterations = {}
        failures = {}
        error_messages_result = {}

        # %% Execution
        for MODEL in MODELS:
            print(f"========= XRL Explainer using {MODEL} model =========")
            for USE_DEBUGGER in USE_DEBUGGERS:
                team_conversations = []
                error_messages = []

                for i, query in enumerate(CF_P_queries):
                    query = "Use 'counterfactual policy' tool to answer the following question:" + query
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

                    # 3. Execute returned function call (if any)
                    functions = function_execute(agent, data, team_conversation)

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

                # %%
                error_counts = [
                    sum(1 for item in conv if 'error_message' in item)
                    for conv in team_conversations
                ]

                failure_counts = [
                    sum(1 for item in conv if item.get('status', False) == "failure")
                    for conv in team_conversations
                ]

                average_errors = sum(error_counts) / len(error_counts) if error_counts else 0
                kk = ' + debugger' if USE_DEBUGGER else ''
                print(f"Average error counts for model {MODEL}{kk}: {average_errors}")

                iterations[f'{MODEL}{kk}'] = average_errors
                failures[f'{MODEL}{kk}'] = failure_counts
                error_messages_result[f'{MODEL}{kk}'] = error_messages

        total_iterations[int(n)] = iterations
        total_failures[int(n)] = failures
        total_error_messages[int(n)] = error_messages_result

    with open("total_iterations.pkl", "wb") as f:
        pickle.dump(total_iterations, f)
    with open("total_failures.pkl", "wb") as f:
        pickle.dump(total_failures, f)
    with open("total_error_messages.pkl", "wb") as f:
        pickle.dump(total_error_messages, f)

else:
    with open("total_iterations.pkl", "rb") as f:
        total_iterations = pickle.load(f)
    with open("total_failures.pkl", "rb") as f:
        total_failures = pickle.load(f)
    with open("total_error_messages.pkl", "rb") as f:
        total_error_messages = pickle.load(f)

# %% Plotting total_results
total_results = pd.DataFrame.from_dict(total_iterations).T
mean, std = total_results.mean(axis=1), total_results.std(axis=1)

plt.figure(figsize=(8, 5))
sns.boxplot(data=total_results, palette='viridis')
sns.stripplot(data=total_results, jitter=True, color='black', size=4, alpha=0.6) # 개별 데이터 포인트 추가

plt.xlabel('Model', fontsize=12)
plt.ylabel('Performance', fontsize=12)
plt.title('Model Performance Distribution (Box Plot)', fontsize=14)
plt.ylim(bottom=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

raise ValueError

# %%
all_errors = [
    item
    for error_messages in error_messages_result.values()
    for sublist in error_messages
    for item in sublist
]

# Get embedding function
def get_embedding(text: str, model="text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

# Vectorize all error messages
all_embeddings = np.array([get_embedding(err) for err in all_errors])

# %% UMAP DR
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reducer = pca.fit(all_embeddings)
embeddings_reduced = reducer.transform(all_embeddings)


# reducer = umap.UMAP(n_neighbors=3,
#                     min_dist=0.0,
#                     n_components=2,
#                     metric='cosine',
#                     random_state=42).fit(all_embeddings)
# embeddings_reduced = reducer.transform(all_embeddings)

plt.figure(figsize=(6, 6))
sns.scatterplot(x=embeddings_reduced[:, 0],
                y=embeddings_reduced[:, 1],
                color = 'darkblue')

plt.title('Clustered embeddings')
plt.xlabel('dim1')
plt.ylabel('dim2')
plt.grid()
plt.tight_layout()
plt.show()

# %%
from sklearn.cluster import KMeans
clusterer = KMeans(n_clusters=8)
labels = clusterer.fit_predict(embeddings_reduced)
unique_labels = sorted(set(labels))
clustered = (labels >= 0)

# clusterer = hdbscan.HDBSCAN(
#     min_samples = 10,
#     min_cluster_size=8,
#     metric='euclidean',
# )
# labels = clusterer.fit_predict(embeddings_reduced)
# unique_labels = sorted(set(labels))
# clustered = (labels >= 0)

plt.figure(figsize=(6, 6))
sns.scatterplot(x=embeddings_reduced[~clustered, 0],
                y=embeddings_reduced[~clustered, 1],
                color=(0.5, 0.5, 0.5), s=6, alpha=0.5)
sns.scatterplot(x=embeddings_reduced[clustered, 0],
                y=embeddings_reduced[clustered, 1],
                hue=labels[clustered],
                palette='Set1')

plt.title('Clustered embeddings')
plt.legend()
plt.xlabel('dim1')
plt.ylabel('dim2')
plt.grid()
plt.tight_layout()
plt.show()

# %%
def group_by_labels(all_errors, labels):
    grouped = defaultdict(list)
    for msg, label in zip(all_errors, labels):
        grouped[label].append(msg)
    return dict(grouped)

grouped_dict = group_by_labels(all_errors, labels)

error_to_cluster = {msg: cluster_id
                    for cluster_id, msgs in grouped_dict.items()
                    for msg in msgs}

# %%
for MODEL in MODELS:
    for USE_DEBUGGER in USE_DEBUGGERS:
        kk = ' + debugger' if USE_DEBUGGER else ' '
        error_messages = error_messages_result[f'{MODEL}{kk}']
        error_labels = [[error_to_cluster.get(e) for e in es] for es in error_messages]
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

            # Normalize transition matrix
            # row_sums = counts.sum(axis=1, keepdims=True)
            # with np.errstate(divide='ignore', invalid='ignore'):
            #     transition_matrix = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)

            # return unique_labels, transition_matrix
            return unique_labels, counts

        unique_labels, W = compute_transition_matrix(error_labels)

        print("Labels:", unique_labels)
        print("Transition Matrix (W):")
        print(W)

        # %%
        sources, targets = np.where(W)
        weights = W[sources, targets]
        edges = list(zip(sources, targets))
        edge_labels = dict(zip(edges, weights))

        # plt.figure(figsize=(6,6))
        # sns.heatmap(W, cbar=True, annot=True, linewidths = .5,)
        # plt.tight_layout()
        # plt.show()

        fig, ax = plt.subplots()
        Graph(edges, edge_labels=edge_labels, edge_label_position=0.5, arrows=True, ax=ax)
        plt.tight_layout()
        plt.show()

