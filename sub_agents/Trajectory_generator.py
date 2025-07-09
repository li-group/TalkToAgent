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


class TrajectoryGenerator:
    def __init__(self):
        self.messages = []
        self.prev_codes = []

    def generate(self, message, original_trajectory):
        """
        Use LLMs to generate the counterfactual policy, based on coordinator's message.
        Returns:
            message (str): Coordinator's instruction for generating CF policy
            original_policy (Stable-baselines3 BaseAlgorithm): Original RL controller algorithm
        """
        # generator_prompt = """
        # Your job is to modify an action trajectory in control problem.
        # When the user use qualitative terms such as "opposite", "aggressive", or "conservative",
        # you may have to perturbate the trajectory to meet the control behavior.
        #
        # Here are some instructions for generating modified trajectory:
        # - The input is given as shape of (action_dim, timestep_dim).
        # - Interpret given qualitative demands in terms of the CHANGES of action values, not action values themselves.
        # - For example, if the original trajectory shows increasing actions, and the user requests "opposite" behavior, you should generate decreasing actions over the same interval.
        #
        # Output format instructions:
        # - The output should be a single Python list (nested if needed), representing a NumPy-compatible array.
        # - If some time interval for perturbation is specified, only perturb the actions within these time steps.
        #     Note that the indices of the trajectory are steps, not simulation time itself. You might want to use divide queried time intervals by {delta_t} to obtain corresponding timesteps.
        # - Only return the array of MODIFIED trajectory, WITHOUT ANY ADDITIONAL COMMENTS OR MARKDOWN FORMATTING.
        #
        # Constraints:
        # - Values should be between -1 and 1.
        # - The shape of the output must exactly match the original trajectory: shape = {shape}
        #
        # For accurate policy generation, here are environment parameters used in our control problem:
        # Action variables: {actions}
        # """

        # Use chain of prompting
        generator_prompt = """
        Your job is to modify an action trajectory in control problem.
        When the user use qualitative terms such as "opposite", "aggressive", or "conservative",
        you may have to perturbate the trajectory to meet the control behavior.
        
        Let's do this together, step by step!
        
        1. First, you will be given some time interval to be explained.
            However, since the array is indexed with time step, we might first have to translate the timestamp into step_index.
            You will do this by dividing timestamp by {delta_t}, and then rounding.
            
        2. Next, you have to track for the INCREMENTS in the action values, not the action values itself.
            For example, when you were given action array of [0.1, 0.2, 0.3, 0.3], the increments array would be [0.1, 0.1, 0.0, 0.0].
            
        3. Based on the increments array and qualitatie terms, transform the action values by adjusting the increments.
            For example, 

        Here are some instructions for generating modified trajectory:
        - The input is given as shape of (action_dim, timestep_dim).
        - Interpret given qualitative demands in terms of the CHANGES of action values, not action values themselves.
        - For example, if the original trajectory shows increasing actions, and the user requests "opposite" behavior, you should generate decreasing actions over the same interval.

        Output format instructions:
        - The output should be a single Python list (nested if needed), representing a NumPy-compatible array.
        - If some time interval for perturbation is specified, only perturb the actions within these time steps. 
            Note that the indices of the trajectory are steps, not simulation time itself. You might want to use divide queried time intervals by {delta_t} to obtain corresponding timesteps.
        - Only return the array of MODIFIED trajectory, WITHOUT ANY ADDITIONAL COMMENTS OR MARKDOWN FORMATTING.

        Constraints:
        - Values should be between -1 and 1.
        - The shape of the output must exactly match the original trajectory: shape = {shape}

        For accurate policy generation, here are environment parameters used in our control problem:
        Action variables: {actions}
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
        content = self._sanitize(content)

        try:
            parsed = ast.literal_eval(content)  # Safe parsing
            trajectory_array = np.array(parsed, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output into np.array: {e}\nRaw content:\n{content}")

        # Optional: validate shape match
        if trajectory_array.shape != original_trajectory.shape:
            raise ValueError(f"Generated trajectory shape mismatch: expected {original_trajectory.shape}, got {trajectory_array.shape}")

        self.trajectory = trajectory_array
        return trajectory_array

    # def validate(self, orig_traj, cf_traj):
    #     if np.isclose(orig_traj, cf_traj, 1e-3).all():
    #         return "The original trajectory and counterfactual trajectory are the same. Please check whether you have perturbed the action trajectory correctly."
    #     return None



    def refine(self, error_message):
        """
        Use LLMs to refine the counterfactual policy, based on the guidance provided by Evaluator agent.
        Args:
            error_message (str): Error message
            guidance (str): Debugging guidance provided by Evaluator agent
        Returns:
            CF_policy (CF_policy object) : Refined CF policy class
        """
        refining_input = f"""
                You previously generated the following array for counterfactual trajectory:
                {self.trajectory}

                However, the following error occurred during simulation:
                {error_message}

                Please revise the array to fix the error. 
                ONLY RETURN THE NUMPY ARRAY, WITHOUT ANY FURTHER EXPLANATIONS.
                Also, you still have to follow the instructions from the initial prompt when modifying the trajectory.
                """

        self.messages.append({"role": "user",
                              "content": refining_input
                              })

        response = client.chat.completions.create(
            model=MODEL,
            messages=self.messages,
            temperature=0.2
        )

        content = response.choices[0].message.content.strip()

        try:
            parsed = ast.literal_eval(content)  # Safe parsing
            trajectory_array = np.array(parsed, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output into np.array: {e}\nRaw content:\n{content}")

        # Optional: validate shape match
        if trajectory_array.shape != self.trajectory.shape:
            raise ValueError(f"Generated trajectory shape mismatch: expected {self.trajectory.shape}, got {trajectory_array.shape}")

        self.trajectory = trajectory_array
        return trajectory_array

