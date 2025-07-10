import os
import streamlit as st
import io
from openai import OpenAI
import json
from dotenv import load_dotenv
from PIL import Image
from internal_tools import (
    train_agent,
    get_rollout_data,
    function_execute
)
from prompts import get_prompts, get_fn_json, get_fn_description, get_system_description, get_figure_description
from params import running_params, env_params
from utils import encode_fig


# %% OpenAI and streamlit setting
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL = 'gpt-4.1'
# MODEL = 'gpt-4o'

st.set_page_config(layout='wide')

st.title("TalktoAgent: Talk to your chemical process controller")
st.markdown(f"**Using model**: `{MODEL}`")

running_params = running_params()
env, env_params = env_params(running_params.get("system"))
print(f"System: {running_params.get('system')}")

# 1. Prepare environment and agent
@st.cache_resource
def init_agent():
    agent = train_agent(lr=running_params["learning_rate"], gamma=running_params["gamma"])
    data = get_rollout_data(agent)
    return agent, data

agent, data = init_agent()

# 2. Call OpenAI API with function calling enabled
tools = get_fn_json()
coordinator_prompt = get_prompts('coordinator_prompt').format(
    env_params=env_params,
    system_description=get_system_description(running_params.get("system")),
)

team_conversation = []
messages = [{"role": "system", "content": coordinator_prompt}]

# query = "How do the process states globally influence the agent's decisions of v1?" # Global FI
# query = "Which feature makes great contribution to the agent's decisions at timestep 4000?" # Local FI
# query = "How would the action variable change if the state variables vary at timestep 4000?" #
# query = "How does action vary with the state variables change generally?" #
# query = "What is the agent trying to achieve in the long run by doing this action at timestep 4000?" # Future_intention
# query = "What would happen if I reduce the value of v1 action to 2.5 from 4000 to 4200, instead of optimal action?" # CF_action
# query = "What would happen if a more conservative control of 0.3 was taken from 4000 to 4200, instead of optimal policy?" # CF_behavior
# query = "What if we use the bang-bang controller instead of the current RL policy? What hinders the bang-bang controller from using it?" # CF_policy
# query = "Why don't we just set v1 as maximum when the h1 is below 0.2?" # CF_policy

# Step 2. User query input
default_query = "How do the process states globally influence the agent's decisions of v1?"
query = st.text_area("Enter your question to explain the RL agent's behavior:", value=default_query)

if st.button("Run"):
    with st.spinner("Thinking..."):
        tools = get_fn_json()
        coordinator_prompt = get_prompts("coordinator_prompt").format(
            env_params=env_params,
            system_description=get_system_description(running_params.get("system")),
        )

        team_conversation = []
        messages = [{"role": "system", "content": coordinator_prompt}]
        messages.append({"role": "user", "content": query})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            functions=tools,
            function_call="auto"
        )

        functions = function_execute(agent, data, team_conversation)

        choice = response.choices[0]
        if choice.finish_reason == "function_call":
            fn_name = choice.message.function_call.name
            args = json.loads(choice.message.function_call.arguments)
            st.info(f"[Coordinator] Calling function: `{fn_name}` with args: `{args}`")
            team_conversation.append({
                "agent": "coordinator",
                "content": f"[Calling function: {fn_name} with args: {args}]"
            })
            figs = functions[fn_name](args)
        else:
            st.warning("No function call was triggered.")
            figs = []
            fn_name = None

        # Step 3. Natural language explanation
        if fn_name:
            explainer_prompt = get_prompts("explainer_prompt").format(
                fn_name=fn_name,
                fn_description=get_fn_description(fn_name),
                figure_description=get_figure_description(fn_name),
                env_params=env_params,
                system_description=get_system_description(running_params.get("system")),
                max_tokens=400
            )

            messages.append({"role": "user", "content": explainer_prompt})

            for fig in figs:
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
            explanation = response.choices[0].message.content
            team_conversation.append({"agent": "explainer", "content": "Multi-modal explanations are generated."})

            st.subheader("Explanation")
            st.markdown(explanation)

            if figs:
                st.subheader("Figures")
                for fig in figs:
                    fig_bytes = io.BytesIO()
                    fig.savefig(fig_bytes, format='png')
                    fig_bytes.seek(0)
                    st.image(Image.open(fig_bytes))
