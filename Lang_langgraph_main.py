"""
Lang_langgraph_main.py  —  LangGraph-based entry point for the XRL explanation pipeline.

Differences from the original main.py:
  - Manual if-else dispatch      →  LangGraph conditional edges
  - while-loop retry logic       →  LangGraph cycle (cp_executor ↔ debugger/coder)
  - Scattered team-conversation  →  unified in AgentState.team_conversation

The existing XRL functions (internal_tools.py), sub-agents (sub_agents/),
and explainer modules (explainer/) are reused without modification.

──────────────────────────────────────────────────────────────────────────────
UI & Observability options
──────────────────────────────────────────────────────────────────────────────
1. LangSmith  (recommended for Windows — web UI with per-node timing)
   ┌─ Setup ─────────────────────────────────────────────────────────────────┐
   │  a) Create a free account at https://smith.langchain.com                │
   │  b) Add the following variables to your .env file:                      │
   │       LANGSMITH_TRACING=true                                            │
   │       LANGSMITH_API_KEY=<your-api-key>                                  │
   │       LANGSMITH_PROJECT=TalktoAgent          # optional project label   │
   │  c) Run this file as usual — all node executions are traced             │
   │     automatically and visible at https://smith.langchain.com/traces     │
   └─────────────────────────────────────────────────────────────────────────┘
   LangSmith captures: execution order, per-node inputs/outputs, wall-clock
   time, token usage, and lets you replay any trace interactively.

2. LangGraph Studio  (Mac Apple Silicon only — NOT available on Windows)
   Full visual IDE; requires Docker + LangSmith account.

3. Local timing  (built-in, no external service needed)
   Every node is wrapped with timed_node() in Lang_graph.py.
   Elapsed time is printed to stdout and stored in result["node_timings"].

4. Stream mode  (real-time console updates between nodes)
   Replace graph.invoke() with graph.stream() using stream_mode="updates".
   Each intermediate state update is printed as it arrives.
   See USE_STREAM_MODE flag below.
──────────────────────────────────────────────────────────────────────────────
"""

import os

# ── LangSmith tracing (uncomment and fill in your API key to enable) ──────────
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = "YOUR_LANGSMITH_API_KEY"
# os.environ["LANGSMITH_PROJECT"] = "TalktoAgent"

from params import get_running_params, get_env_params, get_LLM_configs
from internal_tools import train_agent, get_rollout_data
from langgraph_agent.Lang_graph import create_xrl_graph

# ── Toggle: stream mode prints each node's output in real time ────────────────
USE_STREAM_MODE = False   # set True to see updates as nodes complete

# ── Environment and agent setup ───────────────────────────────────────────────
client, MODEL = get_LLM_configs()
running_params = get_running_params()

print(f"========= XRL Explainer (LangGraph) — {MODEL} =========")
print(f"System : {running_params['system']}")

agent = train_agent(
    lr=running_params["learning_rate"],
    gamma=running_params["gamma"],
)
data = get_rollout_data(agent)

# ── Compile the LangGraph workflow ────────────────────────────────────────────
graph = create_xrl_graph()

# ── Define user queries ───────────────────────────────────────────────────────
queries = [
    # Feature Importance (Global)
    # "Which state variable makes great contribution to the agent's decisions generally?",

    # Feature Importance (Local)
    "Which state variable influenced the agent's action most at timestep 4820?",

    # Contrastive Action
    # "Why don't we set the value of v1 action to 2.5 and v2 action to 7.5 from 4800 to 5000?",

    # Contrastive Behavior
    # "Why don't we act a more conservative control from t=4800 to 5000?",

    # Contrastive Policy  (uses the Coder → Debugger → Evaluator cycle)
    # "What would happen if we replaced the current RL policy with an on-off controller "
    # "between 4800 and 5000 seconds, such that v1=15.0 whenever error of h2>0.0 "
    # "and v1=5.0 otherwise; v2=15.0 whenever error of h1>0.0 and v2=5.0 otherwise?",

    # Q-Decompose
    # "What is the agent trying to achieve in the long run at t=4800?",
]

# ── Run the graph for each query ──────────────────────────────────────────────
for query in queries:
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print("=" * 60)

    # Build a fresh initial state for every query
    initial_state = {
        # Conversation
        "messages": [],
        "user_query": query,

        # Coordinator outputs (initialized to None)
        "selected_tool": None,
        "tool_args": None,

        # XRL results (initialized to None)
        "figures": None,
        "explanation": None,

        # Code generation loop (initialized)
        "generated_code": None,
        "code_error": None,
        "debugger_guidance": None,
        "evaluation_passed": None,
        "retry_count": 0,
        "max_retries": 5,

        # Runtime resources
        "rl_agent": agent,
        "data": data,

        # Contrastive Policy sub-flow (initialized to None)
        "begin_index": None,
        "end_index": None,
        "horizon": None,
        "env_ce": None,
        "evaluator_obj": None,
        "data_actual": None,
        "data_ce": None,
        "coder": None,

        # Logging
        "team_conversation": [],

        # Timing (accumulated by timed_node wrapper)
        "node_timings": {},
    }

    if USE_STREAM_MODE:
        # ── Stream mode: print each node's state update as it arrives ─────────
        result = None
        for step in graph.stream(initial_state, stream_mode="updates"):
            node_name, update = next(iter(step.items()))
            keys_changed = [k for k in update if k != "node_timings"]
            print(f"  → [{node_name}] updated keys: {keys_changed}")
            result = update   # last update contains final state fields
    else:
        # ── Invoke mode: run the full graph and collect the final state ────────
        result = graph.invoke(initial_state)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("[Result] Explanation:")
    print(result.get("explanation", "(no explanation produced)"))

    # ── Print node timing summary ─────────────────────────────────────────────
    timings = result.get("node_timings") or {}
    if timings:
        print(f"\n{'─' * 60}")
        print("[Timing] Node execution times:")
        for node, elapsed in timings.items():
            bar = "█" * int(elapsed * 5)        # simple visual bar
            print(f"  {node:<30} {elapsed:6.3f} s  {bar}")
        print(f"  {'TOTAL':<30} {sum(timings.values()):6.3f} s")
    print(f"{'─' * 60}")
