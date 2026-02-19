import json

from internal_tools import (
    train_agent,
    get_rollout_data,
    function_execute
)
from prompts import get_prompts, get_fn_json, get_fn_description, get_system_description, get_figure_description
from params import get_running_params, get_env_params, get_LLM_configs, set_LLM_configs
from utils import encode_fig

# %% OpenAI setting
client, MODEL = get_LLM_configs()

print(f"========= XRL Explainer using {MODEL} model =========")

# %% Prepare environment and agent
running_params = get_running_params()
env, env_params = get_env_params(running_params.get("system"))
print(f"System: {running_params.get('system')}")

agent = train_agent(lr = running_params['learning_rate'],
                    gamma = running_params['gamma'])
data = get_rollout_data(agent)

# 2. Call OpenAI API with function calling enabled
tools = get_fn_json()
coordinator_prompt = get_prompts('coordinator').format(
    env_params=env_params,
    system_description=get_system_description(running_params.get("system")),
)

team_conversation = []
messages = [{"role": "system", "content": coordinator_prompt}]

queries = [
    # "Which state variable makes great contribution to the agent's decisions at t=4820?", # FI
    # "What is the agent trying to achieve in the long run at t=4800?", # EO
    # "Why don't we set the value of v1 action to 2.5 and v2 action to 7.5 from 4800 to 5000?", # CE_A
    # "Why don't we act a more conservative control from t=4800 to 5000?", # CE_B
    "What would happen if we replaced the current RL policy with an on-off controller between 4800 and 5000 seconds,"
    "such that $v_1 = 15.0$ whenever the error of $h_2 > 0.0$, and $v_1 = 5.0$ otherwise;"
    "and similarly, $v_2 = 15.0$ whenever the error of $h_1 > 0.0$, and $v_2 = 5.0$ otherwise?", # CE_P
    ]

# Obtain XRL results for all queries in 'queries'
for query in queries:
    messages.append({"role": "user", "content": query})

    # Coordinator agent
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        functions=tools,
        function_call="auto"
    )

    # Execute returned function call (if any)
    functions = function_execute(agent, data, query, team_conversation)

    choice = response.choices[0]

    # Invoke predefined XRL functions
    if choice.finish_reason == "function_call":
        fn_name = choice.message.function_call.name
        args = json.loads(choice.message.function_call.arguments)
        print(f"[Coordinator] Calling function: {fn_name} with args: {args}")
        team_conversation.append({"agent": "Coordinator", "content": f"[Calling function: {fn_name} with args: {args}]"})
        figs = functions[fn_name](args)

        # Summarize explanation results in natural language form
        explainer_prompt = get_prompts('explainer').format(
            user_query = query,
            fn_name = fn_name,
            fn_description = get_fn_description(fn_name),
            figure_description = get_figure_description(fn_name),
            env_params=env_params,
            system_description=get_system_description(running_params.get("system")),
            max_tokens = 200
        )

        messages.append(
                {"role": "user", "content": explainer_prompt}
        )

        # Encode XRL result figures
        for fig in figs:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encode_fig(fig)}",
                                "detail": "auto"
                            }
                        }
                    ]
                }
            )

        # Explainer agent
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            seed = 21,
            temperature=0,
            top_p=0
        )

        explanation = response.choices[0].message.content
        team_conversation.append({"agent": "Explainer", "content": "Multi-modal explanations are generated."})
        print(explanation)

    else:
        print("No function call was triggered.")
