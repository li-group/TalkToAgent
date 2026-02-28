from params import get_running_params, get_env_params, get_LLM_configs
from internal_tools import train_agent, get_rollout_data
from langgraph_agent.Lang_graph import create_xrl_graph

# %% Setup
client, MODEL = get_LLM_configs()
print(f"========= XRL Explainer (LangGraph) using {MODEL} model =========")

running_params = get_running_params()
env, env_params = get_env_params(running_params.get("system"))
print(f"System: {running_params.get('system')}")

agent = train_agent(lr=running_params['learning_rate'],
                    gamma=running_params['gamma'])
data = get_rollout_data(agent)

# %% Compile graph
graph = create_xrl_graph()

# %% Define queries
# Queries for Four-tank system (Appendix A)
queries = [
    # "Which state variable makes great contribution to the agent's decisions at t=4820?",   # FI
    # "What is the agent trying to achieve in the long run at t=4800?",                       # EO
    # "Why don't we set the value of v1 action to 2.5 and v2 action to 7.5 from 4800 to 5000?",  # CE_A
    # "Why don't we act a more conservative control from t=4800 to 5000?",                   # CE_B
    # "What would happen if we replaced the current RL policy with an on-off controller "
    # "between 4800 and 5000 seconds, "
    # "such that v1=15.0 whenever the error of h2>0.0, and v1=5.0 otherwise; "
    # "and similarly, v2=15.0 whenever the error of h1>0.0, and v2=5.0 otherwise?",           # CE_P
]

# # Queries for CSTR system (Supplementary materials)
# queries = [
#     "Which state variable makes great contribution to the agent's decisions at t=61?",   # FI
#     "What is the agent trying to achieve in the long run at t=60?",                       # EO
#     "Why don't we set the value of Tc action to 295 from 60 to 70?",  # CE_A
#     "Why don't we act the opposite control from t=60 to 70?",                   # CE_B
#     "What would happen if we replaced the current RL policy with an on-off controller "
#     "between 60 and 80 minutes, "
#     "such that Tc=295 whenever the error of Ca>0.0, and Tc=305 otherwise; "   # CE_P
# ]

# Queries for CSTR system (Supplementary materials)
queries = [
    # "Which state variable makes great contribution to the agent's decisions at t=60?",   # FI
    # "What is the agent trying to achieve in the long run at t=60?",                       # EO
    # "Why don't we set the value of I action to 150 from 60 to 120?",  # CE_A
    # "Why don't we act the conservative control from t=60 to 120?",                   # CE_B
    "What would happen if we replaced the current RL policy with an on-off controller "
    "between 0 and 120 hours, "
    "such that F_N=40 whenever the c_N<750, and F_N=10 otherwise; "   # CE_P
]

# %% Run graph for each query
for query in queries:
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print("=" * 60)

    initial_state = {
        "messages": [],
        "user_query": query,
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
        "use_debugger": True,
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

    result = graph.invoke(initial_state)

    print(f"\n{'─' * 60}")
    print("[Result] Explanation:")
    print(result.get("explanation", "(no explanation produced)"))

    # Token usage summary
    timings     = result.get("node_timings")     or {}
    token_usage = result.get("node_token_usage") or {}

    if timings or token_usage:
        all_nodes = list(dict.fromkeys(list(timings) + list(token_usage)))
        total_prompt     = sum(v["prompt"]     for v in token_usage.values())
        total_completion = sum(v["completion"] for v in token_usage.values())
        total_tokens     = sum(v["total"]      for v in token_usage.values())

        print(f"\n{'─' * 80}")
        print(f"[Stats] {'Node':<30} {'Time':>8}   {'Prompt':>8} {'Compl.':>8} {'Total':>8}")
        print(f"{'─' * 80}")
        for node in all_nodes:
            t_val = timings.get(node, 0.0)
            u = token_usage.get(node, {"prompt": 0, "completion": 0, "total": 0})
            p_str = str(u["prompt"])     if u["total"] else "─"
            c_str = str(u["completion"]) if u["total"] else "─"
            tot   = str(u["total"])      if u["total"] else "─"
            print(f"  {node:<30} {t_val:6.3f} s   {p_str:>8} {c_str:>8} {tot:>8}")
        print(f"{'─' * 80}")
        print(f"  {'TOTAL':<30} {sum(timings.values()):6.3f} s   "
              f"{total_prompt:>8} {total_completion:>8} {total_tokens:>8}")
    print(f"{'─' * 80}")