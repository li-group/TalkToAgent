import os
import time
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

from params import get_running_params, get_env_params, get_LLM_configs, set_LLM_configs
from internal_tools import train_agent, get_rollout_data
from example_queries import get_queries
from langgraph_agent.Lang_nodes import coordinator_node

plt.rcParams['font.family'] = 'Times New Roman'

figure_dir = os.getcwd() + '/figures'
os.makedirs(figure_dir, exist_ok=True)
savedir = figure_dir + '/[RQ1-1] Tool classification'
os.makedirs(savedir, exist_ok=True)
result_dir = os.getcwd() + '/results'
os.makedirs(result_dir, exist_ok=True)

np.random.seed(21)

# %% Experiment settings
MODELS = ['gpt-5.1']
EXAMPLES = [True]   # True = few-shot (prompts.py), False = zero-shot (prompts_wo_examples.py)

LOAD_RESULTS = False
NUM_EXPERIMENTS = 1

# Prepare environment and agent
running_params = get_running_params()
system = running_params.get('system')
env, env_params = get_env_params(system)
print(f"System: {system}")

agent = train_agent(lr=running_params['learning_rate'],
                    gamma=running_params['gamma'])
data = get_rollout_data(agent)

# Tool name → short label mapping
MAPPER = {
    'feature_importance_local':  'FI',
    'feature_importance_global': 'FI',
    'q_decompose':               'EO',
    'contrastive_action':        'CE_A',
    'contrastive_behavior':      'CE_B',
    'contrastive_policy':        'CE_P',
    'raise_error':               'None',
}

# %% Constructing results
if not LOAD_RESULTS:
    total_accuracies = {}
    total_allocations = {}
    total_times = {}
    total_error_messages = {}

    for n in tqdm(range(NUM_EXPERIMENTS)):
        accuracy_result = {}
        allocation_result = {}
        time_result = {}
        error_result = {}

        for MODEL in MODELS:
            set_LLM_configs(MODEL)
            client, MODEL = get_LLM_configs()
            print(f"========= XRL Explainer (LangGraph) using {MODEL} model =========")

            for EXAMPLE in EXAMPLES:
                now = time.time()
                misallocation = 0

                # Build coordinator system prompt from the appropriate prompts module
                if EXAMPLE:
                    from prompts import get_prompts, get_fn_json, get_system_description
                else:
                    from prompts_wo_examples import get_prompts, get_fn_json, get_system_description

                # Pre-build the system prompt; coordinator_node uses it via
                # state.get("coordinator_prompt_override")
                prompt_override = get_prompts('coordinator').format(
                    env_params=env_params,
                    system_description=get_system_description(running_params.get("system")),
                )

                # Build dataset
                FI_queries, EO_queries, CE_A_queries, CE_B_queries, CE_P_queries = get_queries(system)

                true_tools = (
                    ["FI"]   * len(FI_queries)   +
                    ["EO"]   * len(EO_queries)   +
                    ["CE_A"] * len(CE_A_queries) +
                    ["CE_B"] * len(CE_B_queries) +
                    ["CE_P"] * len(CE_P_queries)
                )
                true_args = (
                    list(FI_queries.values())   +
                    list(EO_queries.values())   +
                    list(CE_A_queries.values()) +
                    list(CE_B_queries.values())
                )
                total_queries = (
                    list(FI_queries.keys())   +
                    list(EO_queries.keys())   +
                    list(CE_A_queries.keys()) +
                    list(CE_B_queries.keys()) +
                    list(CE_P_queries.keys())
                )

                predicted_tools = []
                predicted_args = []
                errors = []

                for i, query in enumerate(total_queries):
                    # Minimal state for coordinator_node (no full graph execution)
                    state = {
                        "user_query": query,
                        "coordinator_prompt_override": prompt_override,
                        "team_conversation": [],
                    }

                    try:
                        result = coordinator_node(state, verbose=0)
                        predicted_tool = MAPPER.get(result["selected_tool"], "None")
                        predicted_arg = result.get("tool_args") or {}
                    except Exception as e:
                        print(f"No function call was triggered: {e}")
                        predicted_tools.append("None")
                        predicted_args.append(None)
                        continue

                    predicted_tools.append(predicted_tool)
                    predicted_args.append(predicted_arg)

                    true_tool = true_tools[i]
                    if predicted_tool != true_tool:
                        msg = (f"Misclassification: {query}\n"
                               f"  GT={true_tool}  Pred={predicted_tool}")
                        print(msg)
                        errors.append(msg)
                        continue

                    # Argument comparison for FI, EO, CE_A
                    if true_tool in ['FI', 'EO', 'CE_A']:
                        true_arg = true_args[i]
                        if predicted_arg != true_arg:
                            msg = (f"Arg mismatch: {query}\n"
                                   f"  GT={true_arg}  Pred={predicted_arg}")
                            print(msg)
                            errors.append(msg)
                            misallocation += 1

                    # CE_B: alpha compared by class; other args compared exactly
                    elif true_tool in ['CE_B']:
                        true_arg = true_args[i]

                        def same_class(alpha1: float, alpha2: float) -> bool:
                            def alpha_map(a: float) -> int:
                                return 1 if a >= 1.0 else (2 if a >= 0.0 else 3)
                            return alpha_map(alpha1) == alpha_map(alpha2)

                        if not same_class(true_arg['alpha'], predicted_arg.get('alpha', 0)):
                            msg = (f"Alpha class mismatch: {query}\n"
                                   f"  GT={true_arg}  Pred={predicted_arg}")
                            print(msg)
                            errors.append(msg)
                            misallocation += 1

                        true_arg_ = {k: v for k, v in true_arg.items() if k != 'alpha'}
                        pred_arg_ = {k: v for k, v in predicted_arg.items() if k != 'alpha'}
                        if pred_arg_ != true_arg_:
                            msg = (f"Arg mismatch (non-alpha): {query}\n"
                                   f"  GT={true_arg}  Pred={predicted_arg}")
                            print(msg)
                            errors.append(msg)
                            misallocation += 1

                kk = ' with few shot' if EXAMPLE else ''
                MODEL_label = 'gpt-4.1-mini' if MODEL == 'gpt-4.1-mini-2025-04-14' else MODEL
                elapsed = time.time() - now
                print(f"[{MODEL_label}{kk}] {elapsed:.2f}s taken")

                time_result[f"[{MODEL_label}{kk}]"] = f'{elapsed:.2f}'
                accuracy_result[f"[{MODEL_label}{kk}]"] = accuracy_score(true_tools, predicted_tools)
                allocation_result[f"[{MODEL_label}{kk}]"] = misallocation / len(true_tools)
                error_result[f"[{MODEL_label}{kk}]"] = errors

                # Confusion matrix
                labels = ["FI", "EO", "CE_A", "CE_B", "CE_P", "None"]
                cm = confusion_matrix(true_tools, predicted_tools, labels=labels, normalize='true')
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
                disp.plot(cmap='Blues', values_format='.2f')
                plt.title(f"[{MODEL_label}{kk}] Tool selection Confusion Matrix")
                plt.tight_layout()
                plt.savefig(savedir + f'/[{system}][{MODEL_label}{kk}].png')
                plt.show()

        total_accuracies[int(n)] = accuracy_result
        total_allocations[int(n)] = allocation_result
        total_times[int(n)] = time_result
        total_error_messages[int(n)] = error_result

    with open(result_dir + f"/[RQ1][{system}] total_accuracy.pkl", "wb") as f:
        pickle.dump(total_accuracies, f)
    with open(result_dir + f"/[RQ1][{system}] total_allocation.pkl", "wb") as f:
        pickle.dump(total_allocations, f)
    with open(result_dir + f"/[RQ1][{system}] total_time.pkl", "wb") as f:
        pickle.dump(total_times, f)
    with open(result_dir + f"/[RQ1][{system}] total_error.pkl", "wb") as f:
        pickle.dump(total_error_messages, f)

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
results = pd.DataFrame(total_accuracies)
mean_result = results.mean(axis=1)
std_result = results.std(axis=1)