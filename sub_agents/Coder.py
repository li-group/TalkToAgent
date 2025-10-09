import ast
import json

from prompts import get_system_description, get_prompts
from utils import py2str, str2py, py2func
from internal_tools import raise_error
from params import get_running_params, get_env_params, get_LLM_configs

running_params = get_running_params()
env, env_params = get_env_params(running_params['system'])
client, MODEL = get_LLM_configs()


# %% PolicyGenerator agent
class Coder:
    def __init__(self):
        self.messages = []
        self.prev_codes = []
        self.system = running_params['system']

    def generate(self, message, original_policy):
        """
        Use LLMs to generate the contrastive policy, based on coordinator's message.
        Args:
            message (str): Coordinator's instruction for generating CE policy
            original_policy (Stable-baselines3 BaseAlgorithm): Original RL controller algorithm
        Returns:
            dec_code (str): Generated code, if validation with evaluator was successful
        """
        self.task = 'generate'
        output_example = original_policy.predict(env_params['x0'])[0]

        generator_prompt = """
        You are a coding expert that generates rule-based control logic, based on user queries.
        Your job is to write a code for the following Python class structure, named 'CE_policy': 

        ========================
        import numpy as np
        np.random.seed(21)

        class CE_policy():
            def __init__(self, env, original_policy):
                self.env = env
                self.original_policy = original_policy

            def predict(self, state, deterministic=True):
                # INSERT YOUR RULE-BASED LOGIC HERE
                return action    

        ========================

        Please consider the following points when writing the 'predict' method:
        - If the instruction requires you to modify the original policy, free to use the 'self.original_policy.predict(state)' method
        - The output of the 'predict' method (i.e., the action) should be within the range \[-1,1\], as it will be used by an external function that expects scaled values.
            You can scale the actions values by using the method: 'self.env._scale_U(u)', if needed.
            You can also descale the states by using the method: 'self.env._descale_X(x)', if needed.
        - The input 'state' is also scaled. Ensure that your if-then logic works with scaled variables.
            To scale raw state values, you may use: 'self.env._scale_X(x)'.
        - The input for the 'predict' method ('state') is the same shape with the initial state 'x0'.
        - The output for the 'predict' method ('action') is the same shape with the output shape of original policy. Example output) {output_example}
        - If your code requires any additional Python modules, make sure to import them at the beginning of your code.
        - Only return the code of 'CE_policy' class, WITHOUT ANY ADDITIONAL COMMENTS.
        - if the user requested controllers other than rule-based ones (e.g.)MPC, PID), trigger the 'raise_error' tool.


        For accurate policy generation, here are some descriptions of the control system:
        {system_description}

        Also, environment parameters used in process control:
        {env_params}

        You will get a great reward if you correctly generate the contrastive policy function!
        """

        tools = [
            {
                "type": "function",
                "name": "raise_error",
                "description": "Raise an error when the request violates constraints (e.g., asks for MPC, PID).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Error message explaining why the request is not supported."
                        }
                    },
                    "required": ["message"]
                }
            }
        ]

        self.messages.append({
            "role": "system",
            "content": generator_prompt.format(
                system_description=get_system_description(self.system),
                env_params=vars(env),
                output_example=output_example
            )
        })

        self.messages.append({
            "role": "user",
            "content": f"Would you make a controller policy that satisfies the requirements below:"
                       f"{message}"
        })

        response = client.chat.completions.create(
            model=MODEL,
            messages=self.messages,
            functions=tools,
        )

        if response.choices[0].message.function_call is not None and response.choices[
            0].message.function_call.name == 'raise_error':
            error_message = json.loads(response.choices[0].message.function_call.arguments)['message']
            raise_error(error_message)
        else:
            content = response.choices[0].message.content

            dec_code = self._sanitize(content)
            self.prev_codes.append(dec_code)
            return dec_code

    def decompose(self, file_path, function_name):
        """
        Use LLMs to decompose the reward function
        Args:
            file_path (str): file path where the reward function code is located
            function_name (str): name of the reward function
        Returns:
            new_reward_f (function): Generated decomposed reward function
            component_names (list): List of reward components
        """
        self.task = 'decompose'
        code = py2str(file_path, function_name)

        decomposer_prompt = """
        Your job is to decompose reward function into multiple components.
        You will get a python code of reward function used to train the RL controller agent,
        and your job is to return its corresponding decomposed reward function.

        Here are some requirements help you decompose the reward.
            1. While the original reward function gives scalar reward, the decomposed reward should be in tuple format,
            which contains each component reward.

            2. When returning answer, please only return the following two outputs:
                1) The resulting python function code. It would be better if necessary python packages are imported.
                    Remove unnecessary strings like "'''" or "''' python".
                2) List of concise names of each control objective components.
                    The format should in python list, so that we can directly translate into list object using ast module.
                These two outputs should be separated by separating signal '\n---\n'

            3. You will be also given a brief description of the system. Please follow the description to appropriately decompose the reward.

            4. Also, the function's name should be in the form of '(original function name)_decomposed'.

        Here are the description of the current system and corresponding environment parameters that may help you decompose the reward function.
        System description:
            {system_description}

        Environment parameters:
            {env_params}

        You will get a great reward if you correctly decompose the reward!
        """

        self.messages.append({
            "role": "system",
            "content": decomposer_prompt.format(
                system_description=get_system_description('four_tank'),
                env_params=env_params
            )
        })

        self.messages.append({
            "role": "user",
            "content": f"The original reward function is given as below: \n {code}"
        })

        response = client.chat.completions.create(
            model=MODEL,
            messages=self.messages,
        )

        content = response.choices[0].message.content
        content = self._sanitize(content)

        dec_code, component_names = content.split("\n---\n")
        dec_code = self._sanitize(dec_code)
        component_names = ast.literal_eval(component_names)

        str2py(dec_code, file_path=f'./explainer/reward_fs/{function_name}_decomposed.py')
        new_reward_f = py2func(file_path=f'./explainer/reward_fs/{function_name}_decomposed.py',
                               function_name=f'{function_name}_decomposed')
        return new_reward_f, component_names

    def refine_with_error(self, error_message):
        """
        Use LLMs to refine the contrastive policy, based on the errors raised after executing the code.
        Args:
            error_message (str): Error message raised
        Returns:
            dec_code (str): Refined code (contrastive policy / decomposed reward)
        """

        if self.task == 'generate':
            refining_input = f"""
            You previously generated the following code for a contrastive policy:
    
            {self.prev_codes[-1]}
    
            However, the following error occurred during simulation:
    
            {error_message}
    
            Please revise the code to fix the error. Only return the corrected 'CE_policy' class.
            Also, you still have to follow the instructions from the initial prompt when modifying the code.
            """

        elif self.task == 'decompose':
            refining_input = f"""
            You previously generated the following code for a decomposed reward function:

            {self.prev_codes[-1]}

            However, the following error occurred during simulation:

            {error_message}

            Please revise the code to fix the error. Only return the corrected '(original function name)_decomposed' function.
            Also, you still have to follow the instructions from the initial prompt when modifying the code.
            """

        messages = self.messages + [{"role": "user", "content": refining_input}]

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )

        content = response.choices[0].message.content

        dec_code = self._sanitize(content)
        self.prev_codes.append(dec_code)

        return dec_code

    def refine_with_guidance(self, error_message, guidance):
        """
        Use LLMs to refine the contrastive policy, based on the guidance provided by Debugger agent.
        Args:
            error_message (str): Error message raised
            guidance (str): Debugging guidance provided by Debugger agent
        Returns:
            dec_code (str): Refined code (contrastive policy / decomposed reward)
        """

        if self.task == 'generate':
            refining_input = f"""
                You previously generated the following code for a contrastive policy:
                {self.prev_codes[-1]}
    
                However, the following error occurred during simulation:
                {error_message}
    
                In order to debug this error, our Debugger agent suggested the following below:
                {guidance}
    
                Please revise the code to fix the error. Only return the corrected CE_policy class.
                Also, you still have to follow the instructions from the initial prompt when modifying the code.
                """

        elif self.task == 'decompose':
            refining_input = f"""
                You previously generated the following code for a decomposed reward function:
                {self.prev_codes[-1]}

                However, the following error occurred during simulation:
                {error_message}

                In order to debug this error, our Debugger agent suggested the following below:
                {guidance}

                Please revise the code to fix the error. Only return the corrected '(original function name)_decomposed' function.
                Also, you still have to follow the instructions from the initial prompt when modifying the code.
                """

        messages = self.messages + [{"role": "user", "content": refining_input}]

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )

        content = response.choices[0].message.content

        dec_code = self._sanitize(content)
        self.prev_codes.append(dec_code)

        return dec_code

    def _sanitize(self, code):
        """
        Sanitizes the code, when there exist some Markdown-style characters
        Args:
            code (str): Generated code
        Returns:
            (str) Sanitized code
        """
        start_token = "```python"
        end_token = "```"

        start_idx = code.find(start_token)
        if start_idx == -1:
            return code

        sub_str = code[start_idx + len(start_token):]

        end_idx = sub_str.find(end_token)
        if end_idx == -1:
            return sub_str.strip()

        return sub_str[:end_idx].strip()
