import os
import ast
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from prompts import get_system_description, get_prompts
from utils import py2str, str2py, py2func
from sub_agents.BasicCoder import BasicCoder

from params import running_params, env_params

running_params = running_params()
env, env_params = env_params(running_params['system'])

# LLM settings
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
MODEL = 'gpt-4.1'


class TrajectoryGenerator(BasicCoder):
    def __init__(self):
        super(TrajectoryGenerator, self).__init__()

    def generate(self, message, original_trajectory):
        """
        Use LLMs to generate the counterfactual policy, based on coordinator's message.
        Returns:
            message (str): Coordinator's instruction for generating CF policy
            original_policy (Stable-baselines3 BaseAlgorithm): Original RL controller algorithm
        """
        generator_prompt = """
        Your job is to generate an array of counterfactual trajectory based on original action trajectory.

        Instructions for generating output:
        - The input is given as shape of (action_dim, timestep_dim).
        - If the user provides specific numeric values or targets, you must base the counterfactual trajectory directly on those numbers.
        - In most cases, the user will use qualitative terms such as "opposite", "aggressive", or "conservative". These describe changes in control behavior, not absolute action values.
        - Therefore, interpret such adjectives in terms of how the action values change over time.
        - For example, if the original trajectory shows increasing actions, and the user requests "opposite" behavior, you should generate decreasing actions over the same interval.
        
        Output format instructions:
        - The output should be a single Python list (nested if needed), representing a NumPy-compatible array.
        - If some time interval for perturbation specified, only perturb the actions within these time steps. 
            In this case, note that the indices across 'timestep_dim' are timesteps, not simulation time itself. You may have to use Delta_t to get corresponding timesteps for queried time intervals.
        - Do NOT include any explanation, markdown formatting, code block markers, or extra text.
        - Just print something like: [[0.1, -0.2], [0.3, 0.4], ...]

        Constraints:
        - Values should be between -1 and 1.
        - The shape of the output must exactly match the original trajectory: shape = {shape}
        
        For accurate policy generation, here are environment parameters used in our control problem:
        Action variables: {actions}
        Time interval between each timestep (Delta_t): {delta_t} 
        """

        self.messages.append({
            "role": "system",
            "content": generator_prompt.format(
                actions=env_params['actions'],
                delta_t=env_params['delta_t'],
                shape=original_trajectory.shape
            )
        })

        self.messages.append({
            "role": "user",
            "content": f"""Would you make a counterfactual trajectory that satisfies the requirements below:
                       {message}
                       
                       Here are the original control trajectory that you may have to refer to:
                       {original_trajectory}
                       """
        })

        response = client.chat.completions.create(
            model=MODEL,
            messages=self.messages,
        )

        content = response.choices[0].message.content.strip()

        try:
            parsed = ast.literal_eval(content)  # Safe parsing
            trajectory_array = np.array(parsed, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output into np.array: {e}\nRaw content:\n{content}")

        # Optional: validate shape match
        if trajectory_array.shape != original_trajectory.shape:
            raise ValueError(f"Generated trajectory shape mismatch: expected {original_trajectory.shape}, got {trajectory_array.shape}")

        return trajectory_array

    def refine(self, error_message):
        """
        Use LLMs to refine the counterfactual policy, based on the errors raised by Evaluator agent.
        Returns:
            CF_policy (CF_policy object) : Refined CF policy class
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
        CF_policy = py2func(f'./explainer/cf_policies/CF_policy.py', 'CF_policy')(env, self.original_policy)
        return CF_policy

    def refine_new(self, error_message, guidance):
        """
        Use LLMs to refine the counterfactual policy, based on the guidance provided by Evaluator agent.
        Args:
            error_message (str): Error message
            guidance (str): Debugging guidance provided by Evaluator agent
        Returns:
            CF_policy (CF_policy object) : Refined CF policy class
        """
        refining_input = f"""
                You previously generated the following code for a counterfactual policy:
                {self.prev_codes[-1]}

                However, the following error occurred during simulation:
                {error_message}

                In order to debug this error, our Evaluator suggested for the following below:
                {guidance}

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
        CF_policy = py2func(f'./explainer/cf_policies/CF_policy.py', 'CF_policy')(env, self.original_policy)
        return CF_policy