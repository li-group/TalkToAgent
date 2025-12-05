# streamlit_app.py

import os
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from internal_tools import train_agent, get_rollout_data, function_execute
from prompts import get_prompts, get_fn_json, get_fn_description, get_system_description, get_figure_description
from params import get_running_params, get_env_params
from utils import encode_fig
from PIL import Image
import base64
import io

# ==== OpenAI 설정 ====
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL = "gpt-4.1"

# ==== 페이지 UI ====
st.set_page_config(page_title="XRL Explainer", layout="wide")
st.title("🔍 XRL Explainer with OpenAI")
st.markdown(f"**Using model**: `{MODEL}`")

# ==== Session State 초기화 ====
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "data" not in st.session_state:
    st.session_state.data = None
if "team_conversation" not in st.session_state:
    st.session_state.team_conversation = []
if "figure_history" not in st.session_state:
    st.session_state.figure_history = []
if "history_pairs" not in st.session_state:
    st.session_state.history_pairs = []  # (question, answer) 튜플

# ==== Sidebar: Agent 초기화 ====
st.sidebar.header("⚙️ Agent Settings")
if st.sidebar.button("Initialize Agent & Rollout Data"):
    with st.spinner("Training agent and preparing rollout data..."):
        params = get_running_params()
        env, env_data = get_env_params(params.get("system"))
        agent = train_agent(lr=params["learning_rate"], gamma=params["gamma"])
        data = get_rollout_data(agent)
        st.session_state.agent = agent
        st.session_state.data = data
        st.success("Agent initialized!")
        st.session_state.running_params = params
        st.session_state.env_params = env_data

# ==== 채팅 입력 ====
query = st.chat_input("Enter your question or follow-up here...")

if query:
    if st.session_state.agent is None or st.session_state.data is None:
        st.warning("Please initialize the agent first.")
    else:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("Processing..."):
            tools = get_fn_json()
            coordinator_prompt = get_prompts("coordinator").format(
                env_params=st.session_state.env_params,
                system_description=get_system_description(st.session_state.running_params.get("system")),
            )

            messages = [{"role": "system", "content": coordinator_prompt}] + st.session_state.messages
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                functions=tools,
                function_call="auto"
            )

            functions = function_execute(st.session_state.agent, st.session_state.data, query, st.session_state.team_conversation)
            choice = response.choices[0]

            if choice.finish_reason == "function_call":
                fn_name = choice.message.function_call.name
                args = json.loads(choice.message.function_call.arguments)
                st.info(f"[Coordinator] Calling function: `{fn_name}` with args: `{args}`")
                st.session_state.team_conversation.append({
                    "agent": "coordinator",
                    "content": f"[Calling function: {fn_name} with args: {args}]"
                })
                figs = functions[fn_name](args)

                explainer_prompt = get_prompts("explainer").format(
                    fn_name=fn_name,
                    fn_description=get_fn_description(fn_name),
                    figure_description=get_figure_description(fn_name),
                    env_params=st.session_state.env_params,
                    system_description=get_system_description(st.session_state.running_params.get("system")),
                    max_tokens=400
                )

                messages.append({"role": "user", "content": explainer_prompt})

                for fig in figs:
                    st.session_state.figure_history.append(fig)
                    fig_encoded = encode_fig(fig)
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{fig_encoded}",
                                    "detail": "auto"
                                }
                            }
                        ]
                    })

                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages
                )

                answer = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.team_conversation.append({"agent": "explainer", "content": "Multi-modal explanation generated."})

                # answer = GPT 응답
                st.session_state.history_pairs.append((query, answer, figs))

                # 결과 출력
                st.chat_message("assistant").markdown(answer)
                if figs:
                    for fig in figs:
                        fig_bytes = io.BytesIO()
                        fig.savefig(fig_bytes, format='png')
                        fig_bytes.seek(0)
                        st.image(Image.open(fig_bytes))
            else:
                st.warning("No function call was triggered.")
                st.session_state.messages.append({"role": "assistant", "content": "I couldn't trigger any reasoning function."})

st.sidebar.markdown("### 📝 Past Questions")
if st.session_state.history_pairs:
    question_list = [f"{i+1}. {q[:60]}" for i, (q, _, _) in enumerate(st.session_state.history_pairs)]
    selected_idx = st.sidebar.selectbox(
        "Select a previous question",
        options=list(range(len(question_list))),
        format_func=lambda x: question_list[x]
    )
else:
    selected_idx = None

# 선택된 Q&A 표시
if selected_idx is not None:
    q, a, figs = st.session_state.history_pairs[selected_idx]
    st.markdown("### 🔁 Selected Q&A")
    st.markdown(f"**🙋 Question:**\n> {q}")
    st.markdown(f"**🤖 Answer:**\n{a}")

    if figs:
        st.markdown("**📊 Associated Figures:**")
        for i, fig in enumerate(figs):
            with st.expander(f"Figure {i+1}", expanded=False):
                fig_bytes = io.BytesIO()
                fig.savefig(fig_bytes, format='png')
                fig_bytes.seek(0)
                st.image(Image.open(fig_bytes))


# # ==== 채팅 로그 출력 ====
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#
# if st.session_state.figure_history:
#     st.subheader("All Figures So Far")
#     for fig in st.session_state.figure_history:
#         fig_bytes = io.BytesIO()
#         fig.savefig(fig_bytes, format='png')
#         fig_bytes.seek(0)
#         st.image(Image.open(fig_bytes))

# # ==== Sidebar: History 요약 ====
# st.sidebar.markdown("### 🕒 History Summary")
#
# # 최근 3개 메시지 (사용자 + assistant)
# last_messages = [msg for msg in st.session_state.messages if msg["role"] in ("user", "assistant")][-3:]
# for msg in last_messages:
#     role = "🙋 User" if msg["role"] == "user" else "🤖 Assistant"
#     st.sidebar.markdown(f"**{role}:** {msg['content'][:80]}{'...' if len(msg['content']) > 80 else ''}")
#
# # 최근 그림 한 장만 썸네일
# if st.session_state.figure_history:
#     st.sidebar.markdown("**🖼 Last Figure:**")
#     fig = st.session_state.figure_history[-1]
#     fig_bytes = io.BytesIO()
#     fig.savefig(fig_bytes, format='png')
#     fig_bytes.seek(0)
#     st.sidebar.image(Image.open(fig_bytes), use_column_width=True)