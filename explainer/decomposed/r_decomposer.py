import ast

def extract_function_from_file(file_path, function_name):
    with open(file_path, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            func_code = ast.get_source_segment(source, node)
            return func_code

    raise ValueError(f"Function '{function_name}' not found in {file_path}")


code = extract_function_from_file("../../custom_reward.py", "four_tank_reward")
print(code)


def decompose_r():

def save_decomposed_reward_to_file(code_str, file_path="four_tank_reward_decomposed.py"):
    """
    code_str: str — 함수 전체 코드 (decomposed reward 함수)
    file_path: str — 저장할 .py 경로
    """
    with open(file_path, "w") as f:
        f.write(code_str)
    print(f"✅ Decomposed reward function saved to {file_path}")
