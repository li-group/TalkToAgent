"""
LangGraph workflow assembly.

Connects nodes and edges to build the full XRL explanation pipeline.

Overall graph structure:

  START
    │
    ▼
  coordinator  ──────────────────────────────────────────────────────┐
    │                                                                │
    │  (branches on selected_tool)                                   │
    ├─ feature_importance_global ─────────────────────────┐          │
    ├─ feature_importance_local ──────────────────────────┤          │
    ├─ contrastive_action ────────────────────────────────┤          │
    ├─ contrastive_behavior ──────────────────────────────┤─→ explainer → END
    ├─ q_decompose ───────────────────────────────────────┘          │
    │                                                                │
    └─ contrastive_policy                                            │
          │                                                          │
          ▼                                                          │
        cp_init                                                      │
          │                                                          │
          ▼                                                          │
        cp_executor ──(error)──→ debugger → cp_coder_refine ──┐      │
          │                                                   │      │
          └──(success)──→ evaluator ──(failed)────────────────┘      │
                              │                                      │
                              └──(passed)──→ cp_viz ─────────────────┘
"""

import time
import functools
import contextvars

from langgraph.graph import StateGraph, END

from langgraph_agent.Lang_state import AgentState
from langgraph_agent.Lang_nodes import (
    coordinator_node,
    fi_global_node,
    fi_local_node,
    ca_node,
    cb_node,
    q_decompose_node,
    cp_init_node,
    cp_executor_node,
    debugger_node,
    cp_coder_refine_node,
    evaluator_node,
    cp_viz_node,
    explainer_node,
)


# ══════════════════════════════════════════════════════════════════════════════
# Token tracking via OpenAI class-level monkey-patch
# ══════════════════════════════════════════════════════════════════════════════

# Context variable to accumulate token counts during a single node's execution.
# Value is a mutable dict {"prompt": int, "completion": int, "total": int},
# or None when no node is currently being tracked.
_node_token_accumulator: contextvars.ContextVar[dict | None] = (
    contextvars.ContextVar("_node_token_accumulator", default=None)
)


def _patch_openai_for_token_tracking() -> None:
    """
    Monkey-patch openai.resources.chat.completions.Completions.create at the
    class level so that every client instance (including those inside sub-agents)
    automatically feeds token usage into _node_token_accumulator.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    try:
        from openai.resources.chat.completions import Completions
    except ImportError:
        return  # OpenAI package not available; skip silently

    if getattr(Completions, "_token_tracking_patched", False):
        return  # Already patched

    _original_create = Completions.create

    @functools.wraps(_original_create)
    def _patched_create(self, *args, **kwargs):
        response = _original_create(self, *args, **kwargs)
        acc = _node_token_accumulator.get()
        if acc is not None:
            usage = getattr(response, "usage", None)
            if usage is not None:
                acc["prompt"]     += getattr(usage, "prompt_tokens",     0) or 0
                acc["completion"] += getattr(usage, "completion_tokens", 0) or 0
                acc["total"]      += getattr(usage, "total_tokens",      0) or 0
        return response

    Completions.create = _patched_create
    Completions._token_tracking_patched = True


# Apply the patch immediately when this module is imported
_patch_openai_for_token_tracking()


# ══════════════════════════════════════════════════════════════════════════════
# Node timing wrapper
# ══════════════════════════════════════════════════════════════════════════════

def timed_node(fn):
    """
    Wrap a node function to measure wall-clock execution time AND LLM token
    usage (prompt / completion / total).

    Both metrics are stored in AgentState under the function's own name:
      - node_timings[fn.__name__]     → elapsed seconds (float)
      - node_token_usage[fn.__name__] → {"prompt": int, "completion": int, "total": int}

    Applied at graph.add_node() so that the original node functions stay clean.
    """
    @functools.wraps(fn)
    def wrapper(state: dict) -> dict:
        # ── Start token accumulator for this node ─────────────────────────
        token_acc = {"prompt": 0, "completion": 0, "total": 0}
        ctx_token = _node_token_accumulator.set(token_acc)

        # ── Execute the node ──────────────────────────────────────────────
        start  = time.perf_counter()
        result = fn(state) or {}
        elapsed = round(time.perf_counter() - start, 4)

        # ── Restore previous accumulator context ──────────────────────────
        _node_token_accumulator.reset(ctx_token)

        # ── Merge timing into cumulative dict ─────────────────────────────
        timings = dict(state.get("node_timings") or {})
        timings[fn.__name__] = elapsed
        result["node_timings"] = timings

        # ── Merge token usage into cumulative dict ────────────────────────
        token_usage = dict(state.get("node_token_usage") or {})
        token_usage[fn.__name__] = dict(token_acc)
        result["node_token_usage"] = token_usage

        return result
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# Conditional routing functions
# ══════════════════════════════════════════════════════════════════════════════

def route_by_tool(state: AgentState) -> str:
    """Route to the XRL node selected by the Coordinator."""
    return state["selected_tool"]


def route_after_executor(state: AgentState) -> str:
    """
    Branch after code execution:
      - error + retries remaining  →  debugger (debug then refine)
      - error + retries exhausted  →  END (failure)
      - success                    →  evaluator (trajectory validation)
    """
    if state.get("code_error"):
        if state["retry_count"] >= state["max_retries"]:
            print("[Graph] Max retries exceeded during execution. Stopping.")
            return "exceeded"
        return "error" if state.get("use_debugger", True) else "no_debugger"
    return "success"


def route_after_evaluator(state: AgentState) -> str:
    """
    Branch after trajectory evaluation:
      - passed                     →  cp_viz (visualize then explain)
      - failed + retries remaining →  cp_coder_refine (regenerate code)
      - failed + retries exhausted →  END (failure)
    """
    if state.get("evaluation_passed"):
        return "passed"
    if state["retry_count"] >= state["max_retries"]:
        print("[Graph] Max retries exceeded during evaluation. Stopping.")
        return "exceeded"
    return "failed"


# ══════════════════════════════════════════════════════════════════════════════
# Graph factory function
# ══════════════════════════════════════════════════════════════════════════════

def create_xrl_graph():
    """
    Build and compile the XRL workflow as a LangGraph StateGraph.

    Returns:
        CompiledGraph: executable workflow; call graph.invoke(state) to run.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes (all wrapped with timed_node for execution timing) ───
    graph.add_node("coordinator",               timed_node(coordinator_node))

    # XRL tool nodes
    graph.add_node("feature_importance_global", timed_node(fi_global_node))
    graph.add_node("feature_importance_local",  timed_node(fi_local_node))
    graph.add_node("contrastive_action",        timed_node(ca_node))
    graph.add_node("contrastive_behavior",      timed_node(cb_node))
    graph.add_node("q_decompose",               timed_node(q_decompose_node))

    # Contrastive Policy sub-flow nodes
    graph.add_node("cp_init",           timed_node(cp_init_node))
    graph.add_node("cp_executor",       timed_node(cp_executor_node))
    graph.add_node("debugger",          timed_node(debugger_node))
    graph.add_node("cp_coder_refine",   timed_node(cp_coder_refine_node))
    graph.add_node("evaluator",         timed_node(evaluator_node))
    graph.add_node("cp_viz",            timed_node(cp_viz_node))

    # Shared explainer node
    graph.add_node("explainer",         timed_node(explainer_node))

    # ── Entry point ──────────────────────────────────────────────────────────
    graph.set_entry_point("coordinator")

    # ── Coordinator → XRL tool (conditional branch) ──────────────────────────
    graph.add_conditional_edges(
        "coordinator",
        route_by_tool,
        {
            "feature_importance_global": "feature_importance_global",
            "feature_importance_local":  "feature_importance_local",
            "contrastive_action":        "contrastive_action",
            "contrastive_behavior":      "contrastive_behavior",
            "contrastive_policy":        "cp_init",
            "q_decompose":               "q_decompose",
        },
    )

    # ── Simple XRL tools → Explainer (direct edges) ──────────────────────────
    for tool_node in [
        "feature_importance_global",
        "feature_importance_local",
        "contrastive_action",
        "contrastive_behavior",
        "q_decompose",
    ]:
        graph.add_edge(tool_node, "explainer")

    # ── Contrastive Policy sub-flow edges ────────────────────────────────────
    # Initialization → first execution attempt
    graph.add_edge("cp_init", "cp_executor")

    # Branch after execution
    graph.add_conditional_edges(
        "cp_executor",
        route_after_executor,
        {
            "error":       "debugger",          # error → Debugger → Coder refinement
            "no_debugger": "cp_coder_refine",   # error, debugger skipped → direct refinement
            "success":     "evaluator",         # success → Evaluator validation
            "exceeded":    END,                 # retries exhausted → stop
        },
    )

    # Debugger → Coder refinement → re-execution (cycle)
    graph.add_edge("debugger", "cp_coder_refine")
    graph.add_edge("cp_coder_refine", "cp_executor")

    # Branch after evaluation
    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {
            "passed":   "cp_viz",          # passed → visualization
            "failed":   "cp_coder_refine", # failed → Coder refinement (cycle)
            "exceeded": END,               # retries exhausted → stop
        },
    )

    # Visualization → Explainer → END
    graph.add_edge("cp_viz", "explainer")
    graph.add_edge("explainer", END)

    return graph.compile()
