import json

from src.prompts import get_prompts, get_fn_json, get_system_description
from src.params import get_running_params, get_env_params, get_LLM_configs

running_params = get_running_params()
env, env_params = get_env_params(running_params['system'])
system = running_params['system']

# %% Coordinator agent
class Coordinator:
    def __init__(self):
        self.history = []

    def select_tool(self, user_query, coordinator_prompt_override=None):
        """
        Select the appropriate XRL tool for the user query via OpenAI function-calling.
        Args:
            user_query (str): The user's XRL query
            coordinator_prompt_override (str, optional): Custom system prompt to inject
                (e.g. for RQ1 ablations) instead of the default coordinator prompt
        Returns:
            selected_tool (str): Name of the chosen XRL function
            tool_args (dict): Arguments to pass to that function
        """
        client, MODEL = get_LLM_configs()   # refresh at call time for multi-model experiments
        tools = get_fn_json()

        coordinator_prompt = coordinator_prompt_override or get_prompts("coordinator").format(
            env_params=env_params,
            system_description=get_system_description(system),
        )

        messages = [
            {"role": "system", "content": coordinator_prompt},
            {"role": "user", "content": user_query},
        ]

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            functions=tools,
            function_call="auto",
        )

        fn_call = response.choices[0].message.function_call
        selected_tool = fn_call.name
        tool_args = json.loads(fn_call.arguments)

        return selected_tool, tool_args
