import os
from openai import OpenAI
import json
from dotenv import load_dotenv
from internal_tools import (
    train_agent,
    get_rollout_data,
    function_execute
)
from prompts import get_prompts, get_fn_json, get_fn_description, get_system_description, get_figure_description
from params import running_params, env_params
from submission.EX_Queries import get_queries

os.chdir("..")

# %% Main parameters for experiments
MODEL = 'gpt-4.1'
# MODEL = 'gpt-4o'

USE_DEBUGGER = True
# USE_DEBUGGER = False

MODELS = ['gpt-4.1', 'gpt-4o']
USE_DEBUGGERS = [True, False]

NUM_EXPERIMENTS = 1

# %% OpenAI setting
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
print(f"========= XRL Explainer using {MODEL} model =========")

# Prepare environment and agent
running_params = running_params()
env, env_params = env_params(running_params.get("system"))
print(f"System: {running_params.get('system')}")

agent = train_agent(lr = running_params['learning_rate'],
                    gamma = running_params['gamma'])
data = get_rollout_data(agent)

# %% Constructing dataset
_, _, _, _, CF_P_queries = get_queries()

tools = get_fn_json()
coordinator_prompt = get_prompts('coordinator_prompt').format(
    env_params=env_params,
    system_description=get_system_description(running_params.get("system")),
)

average_error_result = {}
error_messages_result = {}


# %% Execution

for MODEL in MODELS:
    for USE_DEBUGGER in USE_DEBUGGERS:
        team_conversations = []
        error_messages = []

        for i, query in enumerate(CF_P_queries):
            query = "Use 'counterfactual policy' tool to answer the following question:" + query
            team_conversation = []
            messages = [{"role": "system", "content": coordinator_prompt}]
            messages.append({"role": "user", "content": query})

            # Coordinator agent
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                functions=tools,
                function_call="auto"
            )

            # 3. Execute returned function call (if any)
            functions = function_execute(agent, data, team_conversation)

            choice = response.choices[0]
            if choice.finish_reason == "function_call":
                fn_name = choice.message.function_call.name
                args = json.loads(choice.message.function_call.arguments)
                args['use_debugger'] = USE_DEBUGGER
                print(f"[Coordinator] Calling function: {fn_name} with args: {args}")
                team_conversation.append(
                    {"agent": "coordinator", "content": f"[Calling function: {fn_name} with args: {args}]"})
                figs = functions[fn_name](args)
            else:
                print("No function call was triggered.")

            team_conversations.append(team_conversation)
            error_messages.extend([entry['error'] for entry in team_conversation if 'error' in entry])

        # %%
        error_counts = [
            sum(1 for item in inner_list if 'error_message' in item)
            for inner_list in team_conversations
        ]

        average_errors = sum(error_counts) / len(error_counts) if error_counts else 0
        kk = 'with user debugger' if USE_DEBUGGER else ''
        print(f"Average error counts for model {MODEL} {kk}: {average_errors}")
        average_error_result[f'{MODEL} {kk}'] = average_errors
        error_messages_result[f'{MODEL} {kk}'] = error_messages
