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

4. Stream mode  (intermediate AgentState access between nodes)
   Set STREAM_MODE below to one of:
     "invoke"  — no streaming, returns only the final state
     "updates" — after each node: yields {node_name: {changed_keys}}  (delta)
     "values"  — after each node: yields the full AgentState snapshot  ← recommended
     "debug"   — after each node: yields verbose metadata (type/step/payload)
──────────────────────────────────────────────────────────────────────────────
"""

from params import get_running_params, get_LLM_configs
from internal_tools import train_agent, get_rollout_data
from langgraph_agent.Lang_graph import create_xrl_graph

# ── Stream mode selector ──────────────────────────────────────────────────────
# "invoke"  : run to completion, return only the final AgentState (no streaming)
# "updates" : after each node, yield {node_name: {keys_that_changed}}   ← delta only
# "values"  : after each node, yield the FULL AgentState snapshot       ← recommended
# "debug"   : after each node, yield verbose execution metadata
STREAM_MODE = "values"

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

# ── Graph structure visualization ─────────────────────────────────────────────
# Option A: Save PNG via Mermaid.ink online API (no extra install needed)
try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("graph_structure.png", "wb") as f:
        f.write(png_bytes)
    print("Graph PNG saved → graph_structure.png")
except Exception as e:
    print(f"PNG export failed ({e}); falling back to Mermaid source.")


# ── Define user queries ───────────────────────────────────────────────────────
queries = [
    # Feature Importance (Global)
    # "Which state variable makes great contribution to the agent's decisions generally?",

    # Feature Importance (Local)
    # "Which state variable influenced the agent's action most at timestep 4820?",

    # Contrastive Action
    # "Why don't we set the value of v1 action to 2.5 and v2 action to 7.5 from 4800 to 5000?",

    # Contrastive Behavior
    # "Why don't we act a more conservative control from t=4800 to 5000?",

    # Contrastive Policy  (uses the Coder → Debugger → Evaluator cycle)
    "What would happen if we replaced the current RL policy with an on-off controller "
    "between 4800 and 5000 seconds, such that v1=15.0 whenever error of h2>0.0 "
    "and v1=5.0 otherwise; v2=15.0 whenever error of h1>0.0 and v2=5.0 otherwise?",

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

    # ── Execute graph with selected stream mode ────────────────────────────────
    result = {}

    if STREAM_MODE == "invoke":
        # No streaming — returns only the final AgentState after all nodes finish.
        result = graph.invoke(initial_state)

    elif STREAM_MODE == "updates":
        # Delta mode — after each node, yields {node_name: {changed_keys_only}}.
        # Useful for seeing exactly what each node wrote, but state is partial.
        for step in graph.stream(initial_state, stream_mode="updates"):
            node_name, delta = next(iter(step.items()))
            keys = [k for k in delta if k != "node_timings"]
            print(f"  [{node_name}] wrote: {keys}")
            # Access individual updated values:
            #   delta["selected_tool"]  — only present if coordinator just ran
            #   delta["generated_code"] — only present if coder just ran
            result.update(delta)

    elif STREAM_MODE == "values":
        # Full snapshot mode — after each node, yields the complete AgentState.
        # Every field is readable regardless of which node just ran.
        for state in graph.stream(initial_state, stream_mode="values"):
            # ── Access any AgentState field directly ──────────────────────────
            node_timings = state.get("node_timings", {})
            last_node    = list(node_timings)[-1] if node_timings else "start"
            selected     = state.get("selected_tool")
            retry        = state.get("retry_count", 0)
            code_ok      = state.get("code_error") is None
            eval_ok      = state.get("evaluation_passed")

            # The full AgentState is in 'state' — access anything needed, e.g.:
            #   state["selected_tool"]      → XRL tool chosen by Coordinator
            #   state["tool_args"]          → arguments for that tool
            #   state["generated_code"]     → current Coder output
            #   state["code_error"]         → last execution error traceback
            #   state["debugger_guidance"]  → Debugger's suggestion
            #   state["evaluation_passed"]  → True / False / None
            #   state["retry_count"]        → number of retries so far
            #   state["figures"]            → matplotlib figures (after XRL node)
            #   state["explanation"]        → natural language explanation (final)
            #   state["team_conversation"]  → full inter-agent log
            #   state["node_timings"]       → {node_fn_name: elapsed_seconds}
            result = state   # last snapshot is the final state

    elif STREAM_MODE == "debug":
        # Verbose metadata mode — yields one event per task start/finish/error.
        for event in graph.stream(initial_state, stream_mode="debug"):
            if event["type"] == "task":
                print(f"  [start ] {event['payload']['name']}")
            elif event["type"] == "task_result":
                node = event["payload"]["name"]
                keys = [k for k, _ in event["payload"].get("result", [])
                        if k != "node_timings"]
                print(f"  [finish] {node}  →  {keys}")
            result = event.get("payload", result)

    # ── Print final results ────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("[Result] Explanation:")
    print(result.get("explanation", "(no explanation produced)"))

    # ── Print node timing + token usage summary ───────────────────────────────
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
            t = timings.get(node, 0.0)
            u = token_usage.get(node, {"prompt": 0, "completion": 0, "total": 0})
            prompt_str = str(u["prompt"])     if u["total"] else "─"
            compl_str  = str(u["completion"]) if u["total"] else "─"
            total_str  = str(u["total"])      if u["total"] else "─"
            print(f"  {node:<30} {t:6.3f} s   {prompt_str:>8} {compl_str:>8} {total_str:>8}")
        print(f"{'─' * 80}")
        print(f"  {'TOTAL':<30} {sum(timings.values()):6.3f} s   "
              f"{total_prompt:>8} {total_completion:>8} {total_tokens:>8}")
    print(f"{'─' * 80}")
