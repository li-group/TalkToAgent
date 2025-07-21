import os
import ast
from importlib import util

# %% Adding directories
current_dir = os.getcwd()
figure_dir = os.path.join(current_dir, 'figures')
policies_dir = os.path.join(current_dir, 'policies')
curves_dir = os.path.join(current_dir, 'learning_curves')

os.makedirs(figure_dir, exist_ok=True)
os.makedirs(policies_dir, exist_ok=True)
os.makedirs(curves_dir, exist_ok=True)

# %% Supplementary functions
def encode_fig(fig):
    """
    Encodes a matplotlib Figure object into str code
    Args:
        fig (matplotlib Figure)
    Returns:
        img_base64 (str): Encoded figure
    """
    from io import BytesIO
    import base64
    def fig_to_bytes(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return buf
    buf = fig_to_bytes(fig)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def py2str(file_path, function_name):
    """
    Transforms a function in python file into string
    Args:
        file_path (str): File path where .py file is saved
        function_name (str): Name of the reward function to be decomposed
    Return:
        func_code (str): Function code in string format
    """
    with open(file_path, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            func_code = ast.get_source_segment(source, node)
            return func_code

    raise ValueError(f"Function '{function_name}' not found in {file_path}")

def str2py(code_str, file_path):
    """
    Transforms a code string into .py file
    Args:
        code_str (str): Code string
        file_path (str): File path where .py file is saved
    """
    with open(file_path, "w") as f:
        f.write(code_str)

def py2func(file_path, function_name):
    """
    Loads python function from file
    Args:
        file_path (str): File path where .py file is saved
        function_name (str): Name of the reward function to be decomposed
    Return:
        fn (Callable) : Python function
    """
    spec = util.spec_from_file_location("dynamic_module", file_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fn = getattr(module, function_name)
    return fn