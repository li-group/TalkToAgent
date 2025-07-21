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
from submission.EX_Queries import get_queries

os.chdir("..")

# %% Main parameters for experiments
MODEL = 'gpt-4.1'
# MODEL = 'gpt-4o'

USE_DEBUGGER = True
# USE_DEBUGGER = False

MODELS = ['gpt-4.1', 'gpt-4o']
USE_DEBUGGERS = [True, False]

NUM_EXPERIMENTS = 1

# %% OpenAI setting
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
print(f"========= XRL Explainer using {MODEL} model =========")

# Prepare environment and agent
running_params = running_params()
env, env_params = env_params(running_params.get("system"))
print(f"System: {running_params.get('system')}")

agent = train_agent(lr = running_params['learning_rate'],
                    gamma = running_params['gamma'])
data = get_rollout_data(agent)

# %% Constructing dataset
_, _, _, _, CF_P_queries = get_queries()

tools = get_fn_json()
coordinator_prompt = get_prompts('coordinator_prompt').format(
    env_params=env_params,
    system_description=get_system_description(running_params.get("system")),
)

average_error_result = {}
error_messages_result = {}


# %% Execution
# if True:
#     if True:

for MODEL in MODELS:
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
            error_messages.append([entry['error_message'] for entry in team_conversation if 'error_message' in entry])

        # %%
        error_counts = [
            sum(1 for item in conv if 'error_message' in item)
            for conv in team_conversations
        ]

        average_errors = sum(error_counts) / len(error_counts) if error_counts else 0
        kk = 'with user debugger' if USE_DEBUGGER else ''
        print(f"Average error counts for model {MODEL} {kk}: {average_errors}")
        average_error_result[f'{MODEL} {kk}'] = average_errors
        error_messages_result[f'{MODEL} {kk}'] = error_messages

    all_errors = [item for sublist in error_messages for item in sublist]

    raise ValueError

    # %%
    import numpy as np
    import hdbscan

    # Get embedding function
    def get_embedding(text: str, model="text-embedding-3-small"):
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding

    # Vectorize all error messages
    embeddings = np.array([get_embedding(err) for err in all_errors])

    # HDBSCAN 클러스터링
    clusterer = hdbscan.HDBSCAN(
        min_samples = 6,
        min_cluster_size=2,
        metric='cosine',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(embeddings)
    clustered = (labels >= 0)

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=embeddings[~clustered, 0],
                    y=embeddings[~clustered, 1],
                    color=(0.5, 0.5, 0.5), s=6, alpha=0.5)
    sns.scatterplot(x=embeddings[clustered, 0],
                    y=embeddings[clustered, 1],
                    hue=labels[clustered],
                    palette='Set2')

    plt.legend()
    plt.xlabel('UMAP_1')
    plt.ylabel('UMAP_2')
    plt.savefig(savename)
    plt.show()

    # ✅ 클러스터별 결과 출력
    unique_labels = sorted(set(labels))

    for cluster_id in unique_labels:
        if cluster_id == -1:
            print("\n--- Noise / Unclustered ---")
        else:
            print(f"\n--- Cluster {cluster_id} ---")

        for i, err in enumerate(errors):
            if labels[i] == cluster_id:
                print(f"  • {err}")

    # KMeans clustering
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # ✅ 결과 출력
    for cluster_id in range(n_clusters):
        print(f"\n--- Cluster {cluster_id} ---")
        for i, err in enumerate(errors):
            if labels[i] == cluster_id:
                print(err)
