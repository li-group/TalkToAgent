import os
import ast
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from prompts import get_system_description, get_prompts

from params import running_params, env_params

running_params = running_params()
env, env_params = env_params(running_params['system'])

def extract_function_from_file(file_path, function_name):
    with open(file_path, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            func_code = ast.get_source_segment(source, node)
            return func_code

    raise ValueError(f"Function '{function_name}' not found in {file_path}")

def save_decomposed_reward_to_file(code_str, file_path="four_tank_reward_decomposed.py"):
    """
    Args:
        code_str: [str] — Code string
        file_path: str — 저장할 .py 경로
    """
    with open(file_path, "w") as f:
        f.write(code_str)
    print(f"Decomposed reward function saved to {file_path}")

# def check_reward_consistency(original_fn, decomposed_fn, x, u, con):
#     r_scalar = original_fn(x, u, con)
#     r_components = decomposed_fn(x, u, con)
#     return np.isclose(r_scalar, sum(r_components), atol=1e-5)


# def decompose_r(file_path, function_name):
if __name__ == "__main__":
    file_path = "../../custom_reward.py"
    function_name = "four_tank_reward"
    code = extract_function_from_file(file_path, function_name)

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    MODEL = 'gpt-4.1'

    role_message = """
    Your job is to decompose reward function into multiple components.
    You will get a python code of reward function used to train the RL controller agent, and your job is to return its corresponding decomposed reward function.
    
    Here are some requirements help you decompose the reward.
        1. While the original reward function gives scalar reward, the decomposed reward should be in tuple format, which contains each component reward.
    
        2. When returning answer, please only return the following two outputs:
            1) The resulting python function code. It would be better if necessary python packages are imported.
            2) List of concise names of each control objective components.
            These two outputs should be separated by separating signal '\n---\n'
        
        3. You will be also given a brief description of the system. Please follow the description to appropriately decompose the reward.
        
        4. Also, the function's name should be in the form of '(original function name)_decomposed'.
    
    You will get a great reward if you correctly decompose the reward!
    """

    messages = [
        {"role": "system", "content": role_message},
        {"role": "user", "content": f"The brief description of the controlled system is given below \n {get_system_description('four_tank')}"},
        {"role": "user", "content": f"The current environment parameter is given below \n {env_params}"},
        {"role": "user", "content": f"The original reward function is given as below: \n {code}"}
        ]

    response = client.chat.completions.create(
        model = MODEL,
        messages = messages,
    )

    content = response.choices[0].message.content

    dec_code, component_label = content.split("\n---\n")

    # %%
    save_decomposed_reward_to_file(dec_code, file_path = f'./reward_fs/{function_name}_decomposed.py')

    import importlib.util
    def load_function_from_file(file_path, function_name):
        """
        file_path: str — .py 파일 경로
        function_name: str — 함수 이름
        return: 함수 객체 (callable)
        """
        spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        fn = getattr(module, function_name)
        return fn

    original_fn = load_function_from_file(file_path, function_name)
    decomposed_fn = load_function_from_file(f'./reward_fs/{function_name}_decomposed.py', f'{function_name}_decomposed')
    # check_reward_consistency(original_fn, decomposed_fn)


