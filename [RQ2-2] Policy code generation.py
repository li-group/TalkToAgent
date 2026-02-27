import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from params import get_running_params, get_env_params, get_LLM_configs, set_LLM_configs
from internal_tools import train_agent, get_rollout_data
from example_queries import get_queries
from utils import py2func
from src.pcgym import make_env
from langgraph_agent.Lang_graph import create_xrl_graph

plt.rcParams['font.family'] = 'Times New Roman'

figure_dir = os.getcwd() + '/figures'
os.makedirs(figure_dir, exist_ok=True)
savedir = figure_dir + '/[RQ2-2] Policy generation'
os.makedirs(savedir, exist_ok=True)
result_dir = os.getcwd() + '/results'
os.makedirs(result_dir, exist_ok=True)

np.random.seed(21)

# %% Main parameters for experiments
MODELS = ['gpt-5.1']       # GPT models to be explored
USE_DEBUGGERS = [True]     # Whether to use debugger or not

LOAD_RESULTS = False       # Load cached results without re-running experiments
NUM_EXPERIMENTS = 1        # Number of independent experiment repetitions

# Prepare environment and agent
running_params = get_running_params()
system = running_params.get('system')
env, env_params = get_env_params(system)
print(f"System: {system}")

agent = train_agent(lr=running_params['learning_rate'],
                    gamma=running_params['gamma'])
data = get_rollout_data(agent)


# %% Ground-truth validation function (module-level)
def validate_gt(system, agent, env_params, ground_truth_fn, t_begin, t_end, atol_frac=1e-4):
    """
    Validates the generated CE policy against the ground truth lambda.
    Loads the policy saved by the LangGraph pipeline from ./policies/[{system}] ce_policy.py,
    reproduces the CE trajectory, and checks that rule-firing timesteps match the ground truth.

    Returns:
        match_rate : float — fraction of rule-firing (step × action) pairs matched
    """
    policy_path = f"./policies/[{system}] ce_policy.py"
    if not os.path.isfile(policy_path):
        print("[Validation] Policy file not found — skipping.")
        return 0.0

    try:
        CE_cls = py2func(policy_path, "CE_policy")
        env_ce = make_env({**env_params, "noise": False})
        ce_pol = CE_cls(env_ce, agent)
    except Exception as e:
        print(f"[Validation] Failed to load CE policy: {e}")
        return 0.0

    feature_names = env_params["feature_names"]
    action_names = env_params["actions"]
    a_low = env_params["a_space"]["low"]
    a_high = env_params["a_space"]["high"]
    atol = atol_frac * (a_high - a_low)

    begin_idx = int(np.round(t_begin / env_params["delta_t"]))
    end_idx = int(np.round(t_end / env_params["delta_t"]))

    # Dummy rl_action dicts at action-space extremes — invariance → rule fires
    a_lo_dict = dict(zip(action_names, a_low))
    a_hi_dict = dict(zip(action_names, a_high))

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
        return 0.0

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
            # Output invariant to rl_action → rule fires at this step
            if abs(gt_lo_val - gt_hi_val) < atol[j]:
                n_rule_fired += 1
                gt_val = (gt_lo_val + gt_hi_val) / 2
                matches.append(bool(abs(ce_action_vals[j] - gt_val) <= atol[j]))
            # else: defers to RL action → skip

    if n_rule_fired == 0:
        print("[Validation] Rule condition never triggered in CE trajectory.")
        return 0.0

    match_rate = float(np.mean(matches))
    print(f"[Validation] match_rate={match_rate:.2%} over {n_rule_fired} rule-firing steps")
    return match_rate


# %% Constructing results
if not LOAD_RESULTS:
    total_iterations = {}
    total_failures = {}
    total_error_messages = {}
    total_error_types = {}
    total_gt_validations = {}

    for n in range(NUM_EXPERIMENTS):
        _, _, _, _, CE_P_queries = get_queries(system)

        iterations = {}
        failures = {}
        error_messages_result = {}
        error_types_result = {}
        gt_validations = {}

        for MODEL in MODELS:
            set_LLM_configs(MODEL)
            client, MODEL = get_LLM_configs()
            print(f"========= XRL Explainer (LangGraph) using {MODEL} model =========")

            for USE_DEBUGGER in USE_DEBUGGERS:
                # Compile a fresh graph for each (MODEL, USE_DEBUGGER) combination
                graph = create_xrl_graph()

                team_conversations = []
                error_messages = []
                error_types = []
                gt_results = []
                all_results = []

                for query, ground_truth_fn in CE_P_queries.items():
                    # Prefix ensures the Coordinator routes to contrastive_policy
                    query_with_prefix = (
                        "Use 'contrastive policy' tool to answer the following question: " + query
                    )

                    initial_state = {
                        "messages": [],
                        "user_query": query_with_prefix,
                        "selected_tool": None,
                        "tool_args": None,
                        "figures": None,
                        "explanation": None,
                        "generated_code": None,
                        "code_error": None,
                        "debugger_guidance": None,
                        "evaluation_passed": None,
                        "retry_count": 0,
                        "max_retries": 5,
                        "use_debugger": USE_DEBUGGER,
                        "rl_agent": agent,
                        "data": data,
                        "begin_index": None,
                        "end_index": None,
                        "horizon": None,
                        "env_ce": None,
                        "evaluator_obj": None,
                        "data_actual": None,
                        "data_ce": None,
                        "coder": None,
                        "team_conversation": [],
                        "node_timings": {},
                        "node_token_usage": {},
                    }

                    print(f"\n[Query] {query[:80]}...")
                    result = graph.invoke(initial_state)
                    all_results.append(result)

                    team_conversation = result.get("team_conversation", [])
                    team_conversations.append(team_conversation)

                    # Error messages: runtime errors + terminal status message
                    succeeded = result.get("evaluation_passed") == True
                    terminal_msg = (
                        "[Coder] Code successfully generated. Rollout complete."
                        if succeeded
                        else "[Coder] Failed after multiple attempts."
                    )
                    error_msgs_list = (
                        [entry['error_message'] for entry in team_conversation
                         if 'error_message' in entry]
                        + [terminal_msg]
                    )
                    error_messages.append(error_msgs_list)

                    # Error types: exception class names + terminal status ("Success"/"Failure")
                    error_types_list = (
                        [entry['error_type'] for entry in team_conversation
                         if 'error_type' in entry]
                        + ["Success" if succeeded else "Failure"]
                    )
                    error_types.append(error_types_list)

                    # Ground-truth validation
                    tool_args = result.get("tool_args") or {}
                    if succeeded:
                        match_rate = validate_gt(
                            system=system,
                            agent=agent,
                            env_params=env_params,
                            ground_truth_fn=ground_truth_fn,
                            t_begin=tool_args.get("t_begin", 0),
                            t_end=tool_args.get("t_end", 0),
                        )
                    else:
                        match_rate = 0.0
                    gt_results.append(match_rate)

                # Collect error / failure counts per query
                error_counts = [
                    sum(1 for item in conv if 'error_message' in item)
                    for conv in team_conversations
                ]
                failure_counts = [
                    0 if r.get("evaluation_passed") else 1
                    for r in all_results
                ]

                kk = '+debugger' if USE_DEBUGGER else ''
                average_errors = np.mean(error_counts) if error_counts else 0
                print(f"Average error counts for model {MODEL}{kk}: {average_errors}\n")

                iterations[f'{MODEL}{kk}'] = error_counts
                failures[f'{MODEL}{kk}'] = failure_counts
                error_messages_result[f'{MODEL}{kk}'] = error_messages
                error_types_result[f'{MODEL}{kk}'] = error_types
                gt_validations[f'{MODEL}{kk}'] = gt_results

        # Aggregate results across queries for this experiment run
        total_iterations[int(n)] = iterations
        total_failures[int(n)] = failures
        total_error_messages[int(n)] = error_messages_result
        total_error_types[int(n)] = error_types_result
        total_gt_validations[int(n)] = gt_validations

    # Save results
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

# When LOAD_RESULTS = True, load cached results for visualization only
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

print("======================== Done every iteration. ========================")

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
iter_std  = pd.Series({model: np.std(vals)  for model, vals in total_iterations_flatten.items()})

fail_mean = pd.Series({model: np.mean(vals) for model, vals in total_failures_flatten.items()})
fail_std  = pd.Series({model: np.std(vals)  for model, vals in total_failures_flatten.items()})

models_wrapped = [m.replace("+", "\n+") if "debugger" in m else m for m in models]
x = np.arange(len(models))
bar_width = 0.4

plt.figure(figsize=(9, 5))
plt.bar(x - bar_width / 2, iter_mean, yerr=iter_std, capsize=5,
        width=bar_width, label="Attempts", color="olivedrab", alpha=1.0)
plt.bar(x + bar_width / 2, fail_mean, yerr=fail_std, capsize=5,
        width=bar_width, label="Failures", color="orangered", alpha=0.9)
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
error_to_cluster = {error_type: error_names.index(error_type) for error_type in error_names}

# %% 2) Compute and plot error transition matrix
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
                for a, b in zip(seq[:-1], seq[1:]):
                    i = label_to_idx[a]
                    j = label_to_idx[b]
                    counts[i, j] += 1
            return unique_labels, counts

        unique_labels, W = compute_transition_matrix(error_labels)
        W = W.astype(int)

        W_visual = W[:-2, :]  # remove last two rows (Failure, Success are terminal)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            W_visual,
            cbar=True,
            annot=True,
            linewidths=0.5,
            annot_kws={"size": 19},
            fmt='d',
            xticklabels=error_names,
            yticklabels=error_names[:-2],
        )
        plt.xticks(rotation=90, fontsize=13)
        plt.yticks(rotation=0, fontsize=13)
        plt.xlabel("To", fontsize=17)
        plt.ylabel("From", fontsize=17)
        plt.title(f"{MODEL}{kk}", fontsize=18)
        plt.tight_layout()
        plt.savefig(savedir + f'/{MODEL}{kk}.png')
        plt.show()