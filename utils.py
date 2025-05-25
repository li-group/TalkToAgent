import re
import os
import io
import sys
import importlib
from contextlib import redirect_stdout
import signal
import copy
from typing import Union
from io import StringIO

def extract_component_descriptions(models_dict):
    ref_model_dict = models_dict['model_representation']["components"]
    component_descriptions = copy.deepcopy(ref_model_dict)
    return component_descriptions


def replace(src_code: str, old_code: str, new_code: str) -> str:
    """
    TAKEN FROM AUTOGEN: https://microsoft.github.io/autogen/docs/notebooks/agentchat_nestedchat_optiguide/
    Inserts new code into the source code by replacing a specified old
    code block.

    Args:
        src_code (str): The source code to modify.
        old_code (str): The code block to be replaced.
        new_code (str): The new code block to insert.

    Returns:
        str: The modified source code with the new code inserted.

    Raises:
        None

    Example:
        src_code = 'def hello_world():\n    # CODE GOES HERE'
        old_code = '# CODE GOES HERE'
        new_code = 'print("Bonjour, monde!")\nprint("Hola, mundo!")'
        modified_code = _replace(src_code, old_code, new_code)
        print(modified_code)
        # Output:
        # def hello_world():
        #     print("Bonjour, monde!")
        #     print("Hola, mundo!")
    """
    pattern = r"( *){old_code}".format(old_code=old_code)
    head_spaces = re.search(pattern, src_code, flags=re.DOTALL).group(1)
    new_code = "\n".join([head_spaces + line for line in new_code.split("\n")])
    rst = re.sub(pattern, new_code, src_code)
    return rst


def insert_code(src_code: str, new_lines: str, code_type: str) -> str:
    """
    ADAPTED FROM AUTOGEN: https://microsoft.github.io/autogen/docs/notebooks/agentchat_nestedchat_optiguide/

    insert a code patch into the source code.
    """
    # # for now, we have # OPTICHAT REVISION CODE GOES HERE and # OPTICHAT PRINT CODE GOES HERE
    # return replace(src_code, '# CODE GOES HERE', new_lines)
    if code_type == 'REVISION':
        return replace(src_code, f"# OPTICHAT {code_type} CODE GOES HERE", new_lines)
    elif code_type == 'PRINT':
        return replace(src_code, f"# OPTICHAT {code_type} CODE GOES HERE", new_lines)
    else:
        raise ValueError(f"Invalid code type: {code_type}")


def run_with_exec(src_code: str):
    locals_dict = {}
    output = io.StringIO()

    try:
        with redirect_stdout(output):
            exec(src_code, locals_dict, locals_dict)
        return output.getvalue()
    except Exception as e:
        import traceback
        return output.getvalue() + "\n" + traceback.format_exc()


def var_in_con(constraint_expr):
    vars_list = list(identify_variables(constraint_expr))
    return vars_list


def param_in_con(constraint_expr):
    params_list = list(identify_mutable_parameters(constraint_expr))
    return params_list


def get_files_generator(folder_name):
    """
    Get all the .py files in the folder
    folder_name = "video_showcase"
    py_file_names = get_files_generator(folder_name)
    a generator of ['video_showcase/pdi_inf_1.py', 'video_showcase/pdi_inf_2.py']
    """
    files_and_dirs = os.listdir(folder_name)
    for f in files_and_dirs:
        if os.path.isfile(os.path.join(folder_name, f)) and f.endswith('.py'):
            yield os.path.join(folder_name, f)


def get_files(folder_name):
    """
    Get all the .py files in the folder
    folder_name = "video_showcase"
    py_file_names = get_files_generator(folder_name)
    ['video_showcase/pdi_inf_1.py', 'video_showcase/pdi_inf_2.py']
    then split it into two lists, one is for feasible models, the other is for infeasible models
    """
    files_and_dirs = os.listdir(folder_name)
    infeasible_files = []
    feasible_files = []
    for f in files_and_dirs:
        if os.path.isfile(os.path.join(folder_name, f)) and f.endswith('.py'):
            if '_inf_' in f:
                infeasible_files.append(os.path.join(folder_name, f))
            else:
                feasible_files.append(os.path.join(folder_name, f))
    return infeasible_files, feasible_files


# def get_skipJSON_old(model_representation):
#     """
#     get the model description and description of every component
#     skipJSON is the json that help skip the process of calling interpreter (interpret, illustrate, infer)
#     """
#     skipJSON = {"model description": model_representation["model description"],
#                  "components": {"sets": {},
#                                 "parameters": {},
#                                 "variables": {},
#                                 "constraints": {},
#                                 "objective": {}}
#                  }
#     for component_type in ["sets", "parameters", "variables", "constraints", "objective"]:
#         for component_name, component_dict in model_representation["components"][component_type].items():
#             skipJSON["components"][component_type][component_name] = component_dict["description"]
#
#     return skipJSON


def get_skipJSON(model_representation):
    """
    get the model description and description of every component
    skipJSON is the json that help skip the process of calling interpreter (interpret, illustrate, infer)
    """
    COMPONENT_TYPES = ["sets", "parameters", "variables", "constraints", "objective"]
    skipJSON = {
        "model description": model_representation["model description"],
        "components": {
            component_type: {
                component_name: component_dict["description"]
                for component_name, component_dict in model_representation["components"][component_type].items()
            }
            for component_type in COMPONENT_TYPES
        }
    }
    return skipJSON


def feed_skipJSON(skipJSON, models_dict, queried_model='model_1'):
    """
    feed the skipJSON to the models_dict['queried_model']
    """
    COMPONENT_TYPES = ["sets", "parameters", "variables", "constraints", "objective"]
    model_dict = models_dict[queried_model]
    model_dict['model description'] = skipJSON['model description']
    for component_type in COMPONENT_TYPES:
        for component_name, component_dict in model_dict["components"][component_type].items():
            component_dict['description'] = skipJSON['components'][component_type][component_name]
    return models_dict