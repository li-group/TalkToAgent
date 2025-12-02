import os
import time
import pickle
import numpy as np
from openai import OpenAI
import json
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

from internal_tools import (
    train_agent,
    get_rollout_data,
)
from params import get_running_params, get_env_params
from example_queries import get_queries

plt.rcParams['font.family'] = 'Times New Roman'

figure_dir = os.getcwd() + '/figures'
os.makedirs(figure_dir, exist_ok=True)
savedir = figure_dir + '/[RQ1-1] Tool classification'
os.makedirs(savedir, exist_ok=True)
result_dir = os.getcwd() + '/results'
os.makedirs(result_dir, exist_ok=True)

np.random.seed(21)

# %% Experiment settings
MODELS = ['gpt-5.1', 'gpt-5-mini', 'gpt-5-nano']
EXAMPLES = [True, False]

LOAD_RESULTS = False
NUM_EXPERIMENTS = 1

# OpenAI setting
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Prepare environment and agent
running_params = get_running_params()
system = running_params.get('system')
env, env_params = get_env_params(system)
print(f"System: {system}")

agent = train_agent(lr=running_params['learning_rate'],
                    gamma=running_params['gamma'])
data = get_rollout_data(agent)

# %% Constructing result
if not LOAD_RESULTS:
    total_accuracies = {}
    total_allocations = {}
    total_times = {}
    total_error_messages = {}

    # Run experiments over independent experiments
    for n in tqdm(range(NUM_EXPERIMENTS)):
        accuracy_result = {}
        allocation_result = {}
        time_result = {}
        error_result = {}

        # Execute contrastive policy generation for all model and prompting options
        for MODEL in MODELS:
            print(f"========= XRL Explainer using {MODEL} model =========")
            for EXAMPLE in EXAMPLES:
                now = time.time()
                misallocation = 0
                if EXAMPLE:
                    from prompts import get_prompts, get_fn_json, get_system_description
                else:
                    from prompts_wo_examples import get_prompts, get_fn_json, get_system_description

                # Constructing dataset
                tools = get_fn_json()
                coordinator_prompt = get_prompts('coordinator').format(
                    env_params=env_params,
                    system_description=get_system_description(running_params.get("system")),
                )

                FI_queries, EO_queries, CE_A_queries, CE_B_queries, CE_P_queries = get_queries(system)

                true_tools = ["FI"] * 20 + ["EO"] * 20 + ["CE_A"] * 20 + ["CE_B"] * 20 + ["CE_P"] * 10
                true_args = list(FI_queries.values()) + list(EO_queries.values()) + list(CE_A_queries.values()) + list(CE_B_queries.values())
                total_queries = list(FI_queries.keys()) + list(EO_queries.keys()) + list(CE_A_queries.keys()) + list(CE_B_queries.keys()) + CE_P_queries
                predicted_tools = []
                predicted_args = []
                errors = []

                for i, query in enumerate(total_queries):
                    team_conversation = []
                    messages = [{"role": "system", "content": coordinator_prompt}]
                    messages.append({"role": "user", "content": query})

                    # Coordinator agent
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        functions=tools,
                        function_call="auto",
                    )

                    mapper = {
                        'feature_importance_local': 'FI',
                        'q_decompose': 'EO',
                        'contrastive_action': 'CE_A',
                        'contrastive_behavior': 'CE_B',
                        'contrastive_policy': 'CE_P',
                        'raise_error': 'None',
                    }

                    choice = response.choices[0]
                    if choice.finish_reason == "function_call":
                        fn_name = choice.message.function_call.name
                        predicted_tool = mapper[fn_name]
                        predicted_arg = json.loads(choice.message.function_call.arguments)

                        predicted_tools.append(predicted_tool)
                        predicted_args.append(predicted_arg)

                        # Comparing tool selections
                        true_tool = true_tools[i]
                        if predicted_tool != true_tool:
                            print(f"Misclassification in query: {query} \n GroundTruth: {true_tool} Prediction: {predicted_tool} \n")
                            errors.append(f"Misclassification in query: {query} \n GroundTruth: {true_tool} Prediction: {predicted_tool} \n")
                            continue

                        # Comparing arguments for FI, EO, and CE_A queries
                        if true_tool in ['FI', 'EO', 'CE_A']:
                            true_arg = true_args[i]
                            if predicted_arg != true_arg:
                                print(
                                    f"Misallocation in arguments in query: {query} \n GroundTruth: {true_arg} Prediction: {predicted_arg} \n")
                                errors.append(
                                    f"Misallocation in arguments in query: {query} \n GroundTruth: {true_arg} Prediction: {predicted_arg} \n")
                                misallocation += 1

                        # Arguments of CE_B queries are compared differently, since it contains alpha value which is compared with its range, not the exact value
                        elif true_tool in ['CE_B']:
                            true_arg = true_args[i]
                            true_arg_ = {k: v for k, v in true_arg.items() if k != 'alpha'}
                            predicted_arg_ = {k: v for k, v in predicted_arg.items() if k != 'alpha'}


                            # Comparison of alpha parameter of CE_B queries
                            def same_class(alpha1: float, alpha2: float) -> bool:
                                def alpha_map(alpha: float) -> int:
                                    if alpha >= 1.0:
                                        return 1
                                    elif alpha >= 0.0:
                                        return 2
                                    else:
                                        return 3
                                return alpha_map(alpha1) == alpha_map(alpha2)

                            if not same_class(true_arg['alpha'], predicted_arg['alpha']):
                                print(
                                    f"Misallocation in arguments in query: {query} \n GroundTruth: {true_arg} Prediction: {predicted_arg} \n")
                                errors.append(
                                    f"Misallocation in arguments in query: {query} \n GroundTruth: {true_arg} Prediction: {predicted_arg} \n")
                                misallocation += 1

                            # Comparison of parameters of CE_B queries except 'alpha'
                            if predicted_arg_ != true_arg_:
                                print(
                                    f"Misallocation in arguments in query: {query} \n GroundTruth: {true_arg} Prediction: {predicted_arg} \n")
                                errors.append(
                                    f"Misallocation in arguments in query: {query} \n GroundTruth: {true_arg} Prediction: {predicted_arg} \n")
                                misallocation += 1

                    # If no function call was triggered, it is also counted as misclassification
                    else:
                        print("No function call was triggered.")
                        predicted_tools.append('None')
                        predicted_args.append(None)

                kk = ' with few shot' if EXAMPLE else ''
                MODEL = 'gpt-4.1-mini' if MODEL == 'gpt-4.1-mini-2025-04-14' else MODEL
                print(f"[{MODEL}{kk}] {(time.time() - now):.2f}s taken")

                time_result[f"[{MODEL}{kk}]"] = f'{(time.time() - now):.2f}'
                accuracy_result[f"[{MODEL}{kk}]"] = accuracy_score(true_tools, predicted_tools)
                allocation_result[f"[{MODEL}{kk}]"] = misallocation / len(true_tools)
                error_result[f"[{MODEL}{kk}]"] = errors

                # # Results in confusion matrix (optional)
                labels = ["FI", "EO", "CE_A", "CE_B", "CE_P", "None"]
                cm = confusion_matrix(true_tools, predicted_tools, labels=labels, normalize='true')
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

                disp.plot(cmap='Blues', values_format='.2f')
                plt.title(f"[{MODEL}{kk}] Tool selection Confusion Matrix")
                plt.tight_layout()
                plt.savefig(savedir + f'/[{MODEL}{kk}].png')
                plt.show()

        # Aggregate results for all models and prompting options, within a single experiment iteration
        total_accuracies[int(n)] = accuracy_result
        total_allocations[int(n)] = allocation_result
        total_times[int(n)] = time_result
        total_error_messages[int(n)] = error_result

    # Save results for all experiment iterations
    with open(result_dir + f"/[RQ1][{system}] total_accuracy.pkl", "wb") as f:
        pickle.dump(total_accuracies, f)
    with open(result_dir + f"/[RQ1][{system}] total_allocation.pkl", "wb") as f:
        pickle.dump(total_allocations, f)
    with open(result_dir + f"/[RQ1][{system}] total_time.pkl", "wb") as f:
        pickle.dump(total_times, f)
    with open(result_dir + f"/[RQ1][{system}] total_error.pkl", "wb") as f:
        pickle.dump(total_error_messages, f)

# When LOAD_RESULTS = True, just load the results without running experiments
else:
    with open(result_dir + f"/[RQ1][{system}] total_accuracy.pkl", "rb") as f:
        total_accuracies = pickle.load(f)
    with open(result_dir + f"/[RQ1][{system}] total_allocation.pkl", "rb") as f:
        total_allocations = pickle.load(f)
    with open(result_dir + f"/[RQ1][{system}] total_time.pkl", "rb") as f:
        total_times = pickle.load(f)
    with open(result_dir + f"/[RQ1][{system}] total_error.pkl", "rb") as f:
        total_error_messages = pickle.load(f)


# %% Extract mean and variance
import pandas as pd
results = pd.DataFrame(total_accuracies)

mean_result = results.mean(axis=1)
std_result = results.std(axis=1)
