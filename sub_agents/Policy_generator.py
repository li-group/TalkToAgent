import os
import ast
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from prompts import get_system_description, get_prompts
from internal_tools import raise_error

from params import running_params, env_params

running_params = running_params()
env, env_params = env_params(running_params['system'])

# LLM settings
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL = 'gpt-4.1'

class PolicyGenerator:
    def __init__(self):
        self.messages = []
        self.prev_codes = []
        self.system = running_params['system']

    def generate(self, message, original_policy):
        """
        Use LLMs to generate the counterfactual policy, based on coordinator's message.
        Returns:
            message (str): Coordinator's instruction for generating CF policy
            original_policy (Stable-baselines3 BaseAlgorithm): Original RL controller algorithm
        """
        output_example = original_policy.predict(env_params['x0'])[0]

        generator_prompt = """
        You are a coding expert that generates rule-based control logic, based on user queries.
        Your job is to write a code for the following Python class structure, named 'CF_policy': 
    
        ========================
        import numpy as np
        np.random.seed(21)
        
        class CF_policy():
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
        - The input 'state' is also scaled. Ensure that your if-then logic works with scaled variables.
            To scale raw state values, you may use: 'self.env._scale_X(x)'.
        - The input for the 'predict' method ('state') is the same shape with the initial state 'x0'.
        - The output for the 'predict' method ('action') is the same shape with the output shape of original policy. Example output) {output_example}
        - If your code requires any additional Python modules, make sure to import them at the beginning of your code.
        - Only return the code of 'CF_policy' class, WITHOUT ANY ADDITIONAL COMMENTS.
        - if the user requested controllers other than rule-based ones (e.g.)MPC, PID), trigger the 'raise_error' tool.

    
        For accurate policy generation, here are some descriptions of the control system:
        {system_description}
        
        Also, environment parameters used in process control:
        {env_params}
    
        You will get a great reward if you correctly generate the counterfactual policy function!
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
                env_params = vars(env),
                output_example = output_example
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
            functions = tools
        )

        if response.choices[0].message.function_call is not None and response.choices[0].message.function_call.name == 'raise_error':
            error_message = json.loads(response.choices[0].message.function_call.arguments)['message']
            raise_error(error_message)
        else:
            content = response.choices[0].message.content

            dec_code = self._sanitize(content)
            self.prev_codes.append(dec_code)
            return dec_code

    def refine_with_error(self, error_message):
        """
        Use LLMs to refine the counterfactual policy, based on the errors raised after executing the code.
        Returns:
            CF_policy (CF_policy object) : Refined CF policy class
        """
        refining_input = f"""
        You previously generated the following code for a counterfactual policy:

        {self.prev_codes[-1]}

        However, the following error occurred during simulation:

        {error_message}

        Please revise the code to fix the error. Only return the corrected 'CF_policy' class.
        Also, you still have to follow the instructions from the initial prompt when modifying the code.
        """

        messages = self.messages + [{"role": "user", "content": refining_input}]
        # self.messages.append({"role": "user",
        #                       "content": refining_input
        #      })

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2
        )

        content = response.choices[0].message.content

        dec_code = self._sanitize(content)
        self.prev_codes.append(dec_code)

        return dec_code


    def refine_with_guidance(self, error_message, guidance):
        """
        Use LLMs to refine the counterfactual policy, based on the guidance provided by Debugger agent.
        Args:
            error_message (str): Error message
            guidance (str): Debugging guidance provided by Debugger agent
        Returns:
            CF_policy (CF_policy object) : Refined CF policy class
        """
        refining_input = f"""
            You previously generated the following code for a counterfactual policy:
            {self.prev_codes[-1]}

            However, the following error occurred during simulation:
            {error_message}
            
            In order to debug this error, our Debugger agent suggested the following below:
            {guidance}

            Please revise the code to fix the error. Only return the corrected CF_policy class.
            Also, you still have to follow the instructions from the initial prompt when modifying the code.
            """

        messages = self.messages + [{"role": "user", "content": refining_input}]
        # self.messages.append({"role": "user",
        #                       "content": refining_input
        #                       })

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2
        )

        content = response.choices[0].message.content

        dec_code = self._sanitize(content)
        self.prev_codes.append(dec_code)

        return dec_code

    def _sanitize(self, code):
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