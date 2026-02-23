import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.compat.numpy.function import validate_groupby_func

from internal_tools import (
    train_agent,
    get_rollout_data,
    function_execute
)
from prompts import get_prompts, get_fn_json, get_system_description
from params import get_running_params, get_env_params, get_LLM_configs, set_LLM_configs
from example_queries import get_queries
from utils import py2func
from src.pcgym import make_env

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
USE_DEBUGGERS = [True] # Whether to use debugger or not

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
    total_gt_validations = {}

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
        gt_validations = {}

        # Execute contrastive policy generation for all model and debugger options
        for MODEL in MODELS:
            set_LLM_configs(MODEL)
            client, MODEL = get_LLM_configs()
            print(f"========= XRL Explainer using {MODEL} model =========")
            for USE_DEBUGGER in USE_DEBUGGERS:
                team_conversations = []
                error_messages = []
                error_types = []
                gt_results = []

                # Generate contrastive policy for all CE_P queries
                for query, ground_truth_fn in CE_P_queries.items():
                    query_with_prefix = "Use 'contrastive policy' tool to answer the following question:" + query
                    team_conversation = []
                    messages = [{"role": "system", "content": coordinator_prompt}]
                    messages.append({"role": "user", "content": query_with_prefix})
                    fn_name = ''

                    functions = function_execute(agent, data, query_with_prefix, team_conversation)

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

                        succeeded = any(
                            entry.get("status") == "Success"
                            for entry in team_conversation
                        )

                        # Validation with ground-truth validation
                        def validate_gt(
                                system, agent, env_params, ground_truth_fn, t_begin, t_end, atol_frac=1e-4
                        ):
                            """
                            Validates the generated CE policy against the ground truth lambda.
                            Returns:
                                match_rate : float  – fraction of rule-firing (step × action) pairs matched
                            """
                            policy_path = f"./policies/[{system}] ce_policy.py"
                            if not os.path.isfile(policy_path):
                                print("[Validation] Policy file not found — skipping.")
                                return False, 0.0

                            try:
                                CE_cls = py2func(policy_path, "CE_policy")
                                env_ce = make_env({**env_params, "noise": False})
                                ce_pol = CE_cls(env_ce, agent)
                            except Exception as e:
                                print(f"[Validation] Failed to load CE policy: {e}")
                                return False, 0.0

                            feature_names = env_params["feature_names"]
                            action_names = env_params["actions"]
                            a_low = env_params["a_space"]["low"]
                            a_high = env_params["a_space"]["high"]
                            atol = atol_frac * (a_high - a_low)

                            begin_idx = int(np.round(t_begin / env_params["delta_t"]))
                            end_idx = int(np.round(t_end / env_params["delta_t"]))

                            # Dummy rl_action dicts at action-space extremes for rule-firing detection
                            a_lo_dict = dict(zip(action_names, a_low))
                            a_hi_dict = dict(zip(action_names, a_high))

                            # Reproduce CE trajectory
                            ce_settings = {
                                "CE_mode": "policy",
                                "begin_index": begin_idx,
                                "end_index": end_idx,
                                "CE_policy": ce_pol,
                            }
                            try:
                                _, data_ce = env_ce.get_rollouts(
                                    {"New policy": agent}, reps=1, ce_settings=ce_settings
                                )
                            except Exception as e:
                                print(f"[Validation] CE rollout failed: {e}")
                                return False, 0.0

                            x_ce = data_ce["New policy"]["x"][:, begin_idx:end_idx, :].squeeze(axis=2)  # (nx, T)
                            u_ce = data_ce["New policy"]["u"][:, begin_idx:end_idx, :].squeeze(axis=2)  # (nu, T)

                            matches = []
                            n_rule_fired = 0

                            for t in range(x_ce.shape[1]):
                                state_dict = dict(zip(feature_names, x_ce[:, t]))
                                ce_action_vals = u_ce[:, t]

                                gt_lo = ground_truth_fn(state_dict, a_lo_dict)
                                gt_hi = ground_truth_fn(state_dict, a_hi_dict)

                                for j, key in enumerate(action_names):
                                    if key not in gt_lo or key not in gt_hi:
                                        continue
                                    gt_lo_val = float(gt_lo[key])
                                    gt_hi_val = float(gt_hi[key])
                                    # Output invariant to rl_action → rule fires, expected = fixed value
                                    if abs(gt_lo_val - gt_hi_val) < atol[j]:
                                        n_rule_fired += 1
                                        gt_val = (gt_lo_val + gt_hi_val) / 2
                                        matches.append(bool(abs(ce_action_vals[j] - gt_val) <= atol[j]))
                                    # else: defers to RL → skip

                            if n_rule_fired == 0:
                                print("[Validation] Rule condition never triggered in CE trajectory.")
                                return 0.0
                            match_rate = float(np.mean(matches))
                            print(f"[Validation] match_rate={match_rate:.2%} from {n_rule_fired} rule-firing steps, ")
                            return match_rate

                        if succeeded:
                            match_rate = validate_gt(
                                system=system,
                                agent=agent,
                                env_params=env_params,
                                ground_truth_fn=ground_truth_fn,
                                t_begin=args.get("t_begin", 0),
                                t_end=args.get("t_end", 0),
                            )
                        else:
                            match_rate = 0.0
                        gt_results.append(match_rate)
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
                    sum(1 for item in conv if item.get('status', False) == "Failure")
                    for conv in team_conversations
                ]

                average_errors = np.mean(error_counts) if error_counts else 0
                average_failures = np.mean(failure_counts) if failure_counts else 0
                kk = '+debugger' if USE_DEBUGGER else ''
                print(f"Average error counts for model {MODEL}{kk}: {average_errors} \n")

                iterations[f'{MODEL}{kk}'] = error_counts
                failures[f'{MODEL}{kk}'] = failure_counts
                error_messages_result[f'{MODEL}{kk}'] = error_messages
                error_types_result[f'{MODEL}{kk}'] = error_types
                gt_validations[f'{MODEL}{kk}'] = gt_results

        # Aggregate results for all models and debugger options, within a single experiment iteration
        total_iterations[int(n)] = iterations
        total_failures[int(n)] = failures
        total_error_messages[int(n)] = error_messages_result
        total_error_types[int(n)] = error_types_result
        total_gt_validations[int(n)] = gt_validations

    # Save results for all experiment iterations
    with open(result_dir + f"/[RQ2][{system}] total_iterations.pkl", "wb") as f:
        pickle.dump(total_iterations, f)
    with open(result_dir + f"/[RQ2][{system}] total_failures.pkl", "wb") as f:
        pickle.dump(total_failures, f)
    with open(result_dir + f"/[RQ2][{system}] total_error_messages.pkl", "wb") as f:
        pickle.dump(total_error_messages, f)
    with open(result_dir + f"/[RQ2][{system}] total_error_types.pkl", "wb") as f:
        pickle.dump(total_error_types, f)
    with open(result_dir + f"/[RQ2][{system}] total_gt_validations.pkl", "wb") as f:
        pickle.dump(total_gt_validations, f)

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
    with open(result_dir + f"/[RQ2][{system}] total_gt_validations.pkl", "rb") as f:
        total_gt_validations = pickle.load(f)

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

all_embeddings = np.load(result_dir + "/[RQ2] all_embeddings.npy")

print("========================Done every iteration.========================")

# %% 0) Ground-truth validation match rate summary
models_gt = list(total_gt_validations[0].keys())
total_gt_validations_flatten = {model: [] for model in models_gt}
for num in range(NUM_EXPERIMENTS):
    for model in models_gt:
        total_gt_validations_flatten[model].extend(total_gt_validations[num][model])

gt_match_rate_mean = pd.Series({
    model: np.mean(results) if results else 0.0
    for model, results in total_gt_validations_flatten.items()
})

# %% 1) Plotting average iterations & failures for contrastive policy generation
models = total_iterations[0].keys()
total_iterations_flatten = {model: [] for model in models}
total_failures_flatten = {model: [] for model in models}
for num in range(NUM_EXPERIMENTS):
    for model in models:
        total_iterations_flatten[model] += total_iterations[num][model]
        total_failures_flatten[model] += total_failures[num][model]

iter_mean = pd.Series({model: np.mean(vals) for model, vals in total_iterations_flatten.items()})
iter_std = pd.Series({model: np.std(vals) for model, vals in total_iterations_flatten.items()})

fail_mean = pd.Series({model: np.mean(vals) for model, vals in total_failures_flatten.items()})
fail_std = pd.Series({model: np.std(vals) for model, vals in total_failures_flatten.items()})

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
