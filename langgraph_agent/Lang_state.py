"""
AgentState — a single state object that flows through the entire XRL workflow.

Each LangGraph node receives this TypedDict as input and returns only the
keys it modifies; LangGraph automatically merges the partial update back
into the shared state.
"""

from typing import TypedDict, Annotated, Optional, Any
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Conversation history ─────────────────────────────────────────────────
    messages: Annotated[list, add_messages]   # accumulated via add_messages reducer
    user_query: str                           # original user query string

    # ── Coordinator outputs ──────────────────────────────────────────────────
    selected_tool: Optional[str]              # name of the selected XRL tool
    tool_args: Optional[dict]                 # arguments for the selected tool

    # ── XRL results ──────────────────────────────────────────────────────────
    figures: Optional[list]                   # list of matplotlib Figure objects
    explanation: Optional[str]                # natural language explanation (Explainer output)

    # ── Code generation loop (contrastive_policy only) ───────────────────────
    generated_code: Optional[str]             # current code produced by Coder
    code_error: Optional[str]                 # error traceback from the last execution attempt
    debugger_guidance: Optional[str]          # debugging guidance provided by Debugger
    evaluation_passed: Optional[bool]         # result of Evaluator validation
    retry_count: int                          # number of retries so far
    max_retries: int                          # maximum number of allowed retries

    # ── Shared runtime resources ─────────────────────────────────────────────
    rl_agent: Any                             # trained stable-baselines3 agent
    data: dict                                # rollout data {algo: {x, u, r}}

    # ── Contrastive Policy sub-flow state ────────────────────────────────────
    begin_index: Optional[int]                # start index of the CE time window
    end_index: Optional[int]                  # end index of the CE time window
    horizon: Optional[int]                    # length of the visualization horizon
    env_ce: Any                               # noise-free CE environment instance
    evaluator_obj: Any                        # pcgym evaluator object (used for plotting)
    data_actual: Optional[dict]               # rollout data from the actual policy
    data_ce: Optional[dict]                   # rollout data from the CE policy
    coder: Any                                # Coder instance (preserves conversation history)

    # ── Logging ──────────────────────────────────────────────────────────────
    team_conversation: list                   # inter-agent conversation log

    # ── Execution timing ─────────────────────────────────────────────────────
    node_timings: dict                        # wall-clock time (s) per node, e.g. {"coordinator_node": 1.23}

    # ── Token usage ───────────────────────────────────────────────────────────
    node_token_usage: dict                    # LLM token counts per node, e.g. {"coordinator_node": {"prompt": 120, "completion": 45, "total": 165}}
