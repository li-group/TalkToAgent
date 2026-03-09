import io

import streamlit as st
from PIL import Image

from params import get_running_params, get_LLM_configs
from internal_tools import train_agent, get_rollout_data
from langgraph_agent.Lang_graph import create_xrl_graph

# ── Page config ───────────────────────────────────────────────────────────────
_, MODEL = get_LLM_configs()

st.set_page_config(page_title="XRL Explainer", layout="wide")
st.title("XRL Explainer")
st.markdown(f"**Model**: `{MODEL}`")

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("rl_agent",      None),
    ("data",          None),
    ("graph",         None),
    ("history_pairs", []),   # list of (query, explanation, figures, timings, token_usage)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar: settings and initialization ─────────────────────────────────────
st.sidebar.header("Agent Settings")

use_debugger = st.sidebar.toggle("Use Debugger on code errors", value=True)
max_retries  = st.sidebar.slider("Max retries", min_value=1, max_value=10, value=5)

if st.sidebar.button("Initialize Agent & Rollout Data"):
    with st.spinner("Training agent and collecting rollout data..."):
        params   = get_running_params()
        rl_agent = train_agent(lr=params["learning_rate"], gamma=params["gamma"])
        data     = get_rollout_data(rl_agent)
        graph    = create_xrl_graph()

        st.session_state.rl_agent = rl_agent
        st.session_state.data     = data
        st.session_state.graph    = graph

    st.sidebar.success(f"Ready — system: `{params['system']}`")

# ── Sidebar: past Q&A history ─────────────────────────────────────────────────
st.sidebar.markdown("### Past Questions")
selected_idx = None
if st.session_state.history_pairs:
    labels = [
        f"{i + 1}. {q[:60]}"
        for i, (q, *_) in enumerate(st.session_state.history_pairs)
    ]
    selected_idx = st.sidebar.selectbox(
        "Select a previous question",
        options=list(range(len(labels))),
        format_func=lambda x: labels[x],
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def render_figures(figs: list) -> None:
    for i, fig in enumerate(figs):
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        with st.expander(f"Figure {i + 1}", expanded=True):
            st.image(Image.open(buf))


def render_stats(timings: dict, token_usage: dict) -> None:
    if not timings and not token_usage:
        return
    all_nodes        = list(dict.fromkeys(list(timings) + list(token_usage)))
    total_prompt     = sum(v["prompt"]     for v in token_usage.values())
    total_completion = sum(v["completion"] for v in token_usage.values())
    total_tokens     = sum(v["total"]      for v in token_usage.values())

    rows = []
    for node in all_nodes:
        u = token_usage.get(node, {"prompt": 0, "completion": 0, "total": 0})
        rows.append({
            "Node":       node,
            "Time (s)":   f"{timings.get(node, 0.0):.3f}",
            "Prompt":     str(u["prompt"])     if u["total"] else "─",
            "Completion": str(u["completion"]) if u["total"] else "─",
            "Total":      str(u["total"])      if u["total"] else "─",
        })
    rows.append({
        "Node":       "TOTAL",
        "Time (s)":   f"{sum(timings.values()):.3f}",
        "Prompt":     str(total_prompt),
        "Completion": str(total_completion),
        "Total":      str(total_tokens),
    })

    with st.expander("Node timing & token usage", expanded=False):
        st.table(rows)


# ── Main chat ─────────────────────────────────────────────────────────────────
query = st.chat_input("Enter your question...")

if query:
    if st.session_state.rl_agent is None:
        st.warning("Please initialize the agent first (sidebar).")
    else:
        st.chat_message("user").markdown(query)

        initial_state = {
            "messages":          [],
            "user_query":        query,
            "selected_tool":     None,
            "tool_args":         None,
            "figures":           None,
            "explanation":       None,
            "generated_code":    None,
            "code_error":        None,
            "debugger_guidance": None,
            "evaluation_passed": None,
            "retry_count":       0,
            "max_retries":       max_retries,
            "use_debugger":      use_debugger,
            "rl_agent":          st.session_state.rl_agent,
            "data":              st.session_state.data,
            "begin_index":       None,
            "end_index":         None,
            "horizon":           None,
            "env_ce":            None,
            "evaluator_obj":     None,
            "data_actual":       None,
            "data_ce":           None,
            "coder":             None,
            "team_conversation": [],
            "node_timings":      {},
            "node_token_usage":  {},
        }

        final_state = {}
        with st.status("Running XRL pipeline...", expanded=True) as pipeline_status:
            for state in st.session_state.graph.stream(initial_state, stream_mode="values"):
                node_timings = state.get("node_timings") or {}
                if node_timings:
                    last_node = list(node_timings)[-1]
                    elapsed   = node_timings[last_node]
                    pipeline_status.update(label=f"[{last_node}]  {elapsed:.2f}s")
                final_state = state
            pipeline_status.update(label="Pipeline complete", state="complete")

        explanation = final_state.get("explanation") or "(no explanation produced)"
        figures     = final_state.get("figures")     or []
        timings     = final_state.get("node_timings")     or {}
        token_usage = final_state.get("node_token_usage") or {}

        with st.chat_message("assistant"):
            st.markdown(explanation)
            render_figures(figures)
            render_stats(timings, token_usage)

        st.session_state.history_pairs.append(
            (query, explanation, figures, timings, token_usage)
        )

# ── Past Q&A replay ───────────────────────────────────────────────────────────
if selected_idx is not None and not query:
    q, explanation, figures, timings, token_usage = (
        st.session_state.history_pairs[selected_idx]
    )
    st.markdown("### Selected Q&A")
    st.markdown(f"**Question:** {q}")
    st.markdown(f"**Answer:**\n\n{explanation}")
    render_figures(figures)
    render_stats(timings, token_usage)
