import os
import ast
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from prompts import get_system_description, get_prompts
from utils import py2str, str2py, py2func

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

    def generate(self, message):
        """
        Use LLMs to generate the counterfactual policy, based on coordinator's message.
        Returns:
            new_reward_f (Callable): Decomposed reward function
            component_label (list): List of component labels
        """

        generator_prompt = """
        You are a coding expert that generates rule-based control logic, based on user queries.
        Your job is to write a code for the following Python class structure, named 'CF_policy': 
    
        ========================
        class CF_policy():
            def __init__(self, env):
                self.env = env
    
            def predict(self, state, deterministic=True):
                # INSERT YOUR RULE-BASED LOGIC HERE
                return action    
    
        ========================
    
        Please consider the following points when writing the 'predict' method:
        - The output of the 'predict' method (i.e., the action) should be within the range \[-1,1\], as it will be used by an external function that expects scaled values.
            You can scale the actions values by using the method: 'self.env._scale_U(u)', if needed.
        - The input 'state' is also scaled. Ensure that your if-then logic works with scaled variables.
            To scale raw state values, you may use: 'self.env._scale_X(x)'.
        - The input for the 'predict' method, which is 'state' is the same shape with the initial state 'x0'.
        - If your code requires any additional Python modules, make sure to import them at the beginning of your code.
        - Only return the 'CF_policy' class, without "'''" or "'''python".
    
        For accurate policy generation, here are some descriptions of the control system:
        {system_description}
        
        Also, environment parameters used in process control:
        {env_params}
    
        You will get a great reward if you correctly generate the counterfactual policy function!
        """

        self.messages.append({
            "role": "system",
            "content": generator_prompt.format(
                system_description=get_system_description(running_params['system']),
                env_params = vars(env)
            )
        })

        self.messages.append({"role": "user",
                              "content": f"Would you make a controller policy that satisfies the requirements below:"
                                         f"{message}"
             })

        response = client.chat.completions.create(
            model=MODEL,
            messages=self.messages,
        )

        content = response.choices[0].message.content

        dec_code = content
        self.prev_codes.append(dec_code)

        str2py(dec_code, file_path=f'./explainer/cf_policies/CF_policy.py')
        CF_policy = py2func(f'./explainer/cf_policies/CF_policy.py', 'CF_policy')(env)
        return CF_policy

    def refine(self, error_message):
        """
        Use LLMs to refine the counterfactual policy, based on the suggestions raised by Evaluator agent.
        Returns:
            new_reward_f (Callable): Decomposed reward function
            component_label (list): List of component labels
        """
        refining_input = f"""
        You previously generated the following code for a counterfactual policy:

        {self.prev_codes[-1]}

        However, the following error occurred during simulation:

        {error_message}

        Please revise the code to fix the error. Only return the corrected function code.
        Also, you still have to follow the instructions from the initial prompt when modifying the code.
        """

        self.messages.append({"role": "user",
                              "content": refining_input
             })

        response = client.chat.completions.create(
            model=MODEL,
            messages=self.messages,
            temperature=0.2
        )

        content = response.choices[0].message.content

        dec_code = content
        self.prev_codes.append(dec_code)

        str2py(dec_code, file_path=f'./explainer/cf_policies/CF_policy.py')
        CF_policy = py2func(f'./explainer/cf_policies/CF_policy.py', 'CF_policy')(env)
        return CF_policy