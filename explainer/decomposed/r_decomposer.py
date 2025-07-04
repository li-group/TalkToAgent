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


def decompose_r(file_path, function_name):
    """
    Use LLMs to decompose the reward function
    Returns:
        new_reward_f (Callable): Decomposed reward function
        component_label (list): List of component labels
    """
    code = py2str(file_path, function_name)

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    MODEL = 'gpt-4.1'

    decomposer_prompt = """
    Your job is to decompose reward function into multiple components.
    You will get a python code of reward function used to train the RL controller agent, and your job is to return its corresponding decomposed reward function.
    
    Here are some requirements help you decompose the reward.
        1. While the original reward function gives scalar reward, the decomposed reward should be in tuple format, which contains each component reward.
    
        2. When returning answer, please only return the following two outputs:
            1) The resulting python function code. It would be better if necessary python packages are imported. Remove unnecessary strings like "'''" or "''' python".
            2) List of concise names of each control objective components.
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

    messages = [
        {"role": "system", "content": decomposer_prompt.format(
            system_description = get_system_description('four_tank'),
            env_params = env_params
        )},
        {"role": "user", "content": f"The original reward function is given as below: \n {code}"}
        ]

    response = client.chat.completions.create(
        model = MODEL,
        messages = messages,
    )

    content = response.choices[0].message.content

    dec_code, component_label = content.split("\n---\n")
    import ast
    component_label = ast.literal_eval(component_label)

    str2py(dec_code, file_path =f'./explainer/decomposed/reward_fs/{function_name}_decomposed.py')
    new_reward_f = py2func(file_path =f'./explainer/decomposed/reward_fs/{function_name}_decomposed.py',
                           function_name = f'{function_name}_decomposed')
    return new_reward_f, component_label


# def check_reward_consistency(original_fn, decomposed_fn, x, u, con):
#     r_scalar = original_fn(x, u, con)
#     r_components = decomposed_fn(x, u, con)
#     return np.isclose(r_scalar, sum(r_components), atol=1e-5)

# original_fn = load_function_from_file(file_path, function_name)
# decomposed_fn = load_function_from_file(f'./reward_fs/{function_name}_decomposed.py', f'{function_name}_decomposed')
# # check_reward_consistency(original_fn, decomposed_fn)


