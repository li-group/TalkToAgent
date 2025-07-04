# @title Import necessary libraries
import os
import asyncio
from dotenv import load_dotenv
from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts

from internal_tools_adk import (
    train_agent,
    get_rollout_data,
    function_execute,
    cluster_states,
    feature_importance_global,
    feature_importance_local,
    partial_dependence_plot_global,
    partial_dependence_plot_local,
    trajectory_sensitivity,
    trajectory_counterfactual,
    q_decompose
)

from prompts import get_prompts, get_fn_json, get_fn_description, get_system_description, get_figure_description
from params import running_params, env_params

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

# %% OpenAI setting
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

import logging
logging.basicConfig(level=logging.ERROR)

print("Libraries imported.")

# %%
# --- Verify Keys (Optional Check) ---
print("API Keys Set:")
print(f"Google API Key set: {'Yes' if os.environ.get('GOOGLE_API_KEY') and os.environ['GOOGLE_API_KEY'] != 'YOUR_GOOGLE_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")
print(f"OpenAI API Key set: {'Yes' if os.environ.get('OPENAI_API_KEY') and os.environ['OPENAI_API_KEY'] != 'YOUR_OPENAI_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")

# %%
# --- Define Model Constants for easier use ---
MODEL = "openai/gpt-4.1" # You can also try: gpt-4.1-mini, gpt-4o etc.
AGENT_MODEL = LiteLlm(MODEL) # Starting with Gemini
print("\nEnvironment configured.")

# %%
print(f"========= XRL Explainer using {MODEL} model =========")
# query = input("Enter your query:)
# 1. Prepare environment and agent
running_params = running_params()
env, env_params = env_params(running_params.get("system"))
print(f"System: {running_params.get('system')}")

agent = train_agent(lr = running_params['learning_rate'],
                    gamma = running_params['gamma'])
data = get_rollout_data(agent)

# %% Pre-2. Prompt definition
system_description_prompt = get_prompts('system_description_prompt').format(
    env_params=env_params,
    system_description=get_system_description(running_params.get("system")),
)

reward_decomposer_prompt = get_prompts('reward_decomposer_prompt')

fn_name = 'kk'
explainer_prompt = get_prompts('explainer_prompt').format(
    fn_name = fn_name,
    fn_description = get_fn_description(fn_name),
    figure_description = get_figure_description(fn_name),
    max_tokens = 400
)

coordinator_prompt = get_prompts('coordinator_prompt')

function_execute(agent, data)


# %% 2. Agent definition
# decomposer = LlmAgent(
#     name = "Reward_decomposer",
#     model = AGENT_MODEL,
#     description = "Decomposes reward function code into multiple semantic components",
#     instruction = reward_decomposer_prompt,
#     output_key = ['file_path', 'function_name']
# )

explainer = LlmAgent(
    name = "XRL_Explainer",
    model = AGENT_MODEL,
    description = "Explains the resulting XRL figures and relate into its physical meanings",
    instruction = explainer_prompt,
    tools = [get_fn_description, get_figure_description]
)

executor = LlmAgent(
    name = "XRL_coordinator",
    model = AGENT_MODEL,
    description = "Understands user's queries and their intentions, and selects appropriate tool",
    instruction = coordinator_prompt,
    tools = [
        cluster_states(agent, data),
        feature_importance_global(agent, data),
        feature_importance_local(agent, data),
        partial_dependence_plot_global(agent, data),
        partial_dependence_plot_local(agent, data),
        trajectory_sensitivity(agent, data),
        trajectory_counterfactual(agent, data),
        # q_decompose(agent, data)
    ],
    # sub_agents = [explainer],
    output_key = 'fn_name'
)

root_agent = SequentialAgent(
    name= 'XRL_Agent',
    sub_agents = [executor, explainer],
    description = 'Executes proper Explainable RL techniques and interprets the results'
)

print(f"Agent '{root_agent.name}' created using model '{MODEL}'.")

# %%
# @title Setup Session Service and Runner
if __name__ == "__main__":

    # --- Session Management ---
    # Key Concept: SessionService stores conversation history & state.
    # InMemorySessionService is simple, non-persistent storage for this tutorial.
    session_service = InMemorySessionService()

    # Define constants for identifying the interaction context
    APP_NAME = "XRL_LLM"
    USER_ID = "HCK"
    SESSION_ID = "session_001" # Using a fixed ID for simplicity

    # Create the specific session where the conversation will happen
    import asyncio
    session = asyncio.run(session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    ))
    print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

    # --- Runner ---
    # Key Concept: Runner orchestrates the agent execution loop.
    runner = Runner(
        agent=root_agent, # The agent we want to run
        app_name=APP_NAME,   # Associates runs with our app
        session_service=session_service # Uses our session manager
    )
    print(f"Runner created for agent '{runner.agent.name}'.")

    # %%
    # @title Define Agent Interaction Function

    from google.genai import types # For creating message Content/Parts

    def logger(event, max_len = 100):
        parts = event.content.parts[0]
        if parts.function_call:
            if parts.function_call.name == 'transfer_to_agent':
                log =  f"- Transferred to Agent {parts.function_call.args['agent_name']}"
            else:
                log = f"- Function {parts.function_call.name} with args {parts.function_call.args} called"
        elif parts.function_response:
            log = f"- Function response: {parts.function_response.response}"
        elif parts.text:
            log =  f"- Text: {parts.text}"

        if len(log) > max_len:
            return log[:max_len]
        return log

    async def call_agent_async(query: str, runner, user_id, session_id):
        """Sends a query to the agent and prints the final response."""
        print(f"\n>>> User Query: {query}")

        # Prepare the user's message in ADK format
        content = types.Content(role='user', parts=[types.Part(text=query)])

        final_response_text = "Agent did not produce a final response." # Default

        events = []

        # Key Concept: run_async executes the agent logic and yields Events.
        # We iterate through events to find the final answer.
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            events.append(event.content.parts[0])
            # events.append(event.content.parts[0])
            # print(f"Content: {event.content.parts[0]}")
            print(logger(event))

            # Key Concept: is_final_response() marks the concluding message for the turn.
            if event.is_final_response():
                if event.content and event.content.parts:
                    # Assuming text response in the first part
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate: # Handle potential errors/escalations
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                    # Add more checks here if needed (e.g., specific error codes)
                    break # Stop processing events once the final response is found

        print(f"<<< Agent Response: {final_response_text}")
        return events

    # %% Run the Initial Conversation
    async def run_conversation(query):
        results = await call_agent_async(query = query,
                               runner=runner,
                               user_id=USER_ID,
                               session_id=SESSION_ID
                               )
        return results

        # Possible queries
        query = "How do the process states globally influence the agent's decisions of v1 by SHAP?"
        # query = "Which feature makes great contribution to the agent's decisions at timestep 150?"
        # query = "I want to know at which type of states have the low q values of an actor."
        # query = "What would happen if I execute 300˚C as Tc action value instead of optimal action at timestep 150?"
        # query = "What would happen if I execute 9.5 as v1 action value instead of optimal action at timestep 200?"
        # query = "What would happen if I slight vary v1 action value at timestep 200?"
        # query = "How would the action variable change if the state variables vary at timestep 200?"
        # query = "How does action vary with the state variables change generally?"
        # query = "What is the agent trying to achieve in the long run by doing this action at timestep 180?"

        try:
            results = asyncio.run(run_conversation(query))
        except Exception as e:
            print(f"An error occurred: {e}")
