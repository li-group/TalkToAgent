"""
LangGraph node function collection.

Each function receives an AgentState dict and returns only the keys it
modifies. The existing sub_agents/, explainer/, and internal_tools.py
modules are reused without modification; only the orchestration logic
(who calls whom, and when) is managed here.
"""

import json
import traceback
import numpy as np
import pandas as pd

from params import get_running_params, get_env_params, get_LLM_configs
from prompts import (
    get_prompts,
    get_fn_json,
    get_fn_description,
    get_system_description,
    get_figure_description,
)
from utils import encode_fig, str2py, py2func
from src.pcgym import make_env
from sub_agents.Coder import Coder
from sub_agents.Debugger import Debugger
from sub_agents.Evaluator import Evaluator

# Module-level shared configuration (mirrors the pattern used in existing code)
running_params = get_running_params()
env, env_params = get_env_params(running_params["system"])
client, MODEL = get_LLM_configs()
system = running_params["system"]
algo = running_params["algo"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. COORDINATOR  —  Parse the user query and select the appropriate XRL tool
# ══════════════════════════════════════════════════════════════════════════════

def coordinator_node(state: dict) -> dict:
    """
    Select the appropriate XRL tool for the user query via OpenAI function-calling.

    Returns:
        selected_tool (str): name of the chosen XRL function
        tool_args (dict): arguments to pass to that function
    """
    user_query = state["user_query"]
    tools = get_fn_json()

    coordinator_prompt = get_prompts("coordinator").format(
        env_params=env_params,
        system_description=get_system_description(system),
    )

    messages = [
        {"role": "system", "content": coordinator_prompt},
        {"role": "user", "content": user_query},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        functions=tools,
        function_call="auto",
    )

    fn_call = response.choices[0].message.function_call
    selected_tool = fn_call.name
    tool_args = json.loads(fn_call.arguments)

    print(f"[Coordinator] Tool: {selected_tool} | Args: {tool_args}")

    team_conversation = list(state["team_conversation"])
    team_conversation.append({
        "agent": "Coordinator",
        "content": f"Selected tool: {selected_tool}",
        "args": tool_args,
    })

    return {
        "selected_tool": selected_tool,
        "tool_args": tool_args,
        "team_conversation": team_conversation,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. XRL TOOL NODES  —  One node per explanation method
# ══════════════════════════════════════════════════════════════════════════════

def fi_global_node(state: dict) -> dict:
    """Compute SHAP-based global feature importance across all timesteps."""
    from internal_tools import feature_importance_global

    figures = feature_importance_global(
        state["rl_agent"],
        state["data"],
        actions=state["tool_args"].get("actions"),
    )
    return {"figures": figures}


def fi_local_node(state: dict) -> dict:
    """Compute SHAP-based local feature importance at a specific timestep."""
    from internal_tools import feature_importance_local

    figures = feature_importance_local(
        state["rl_agent"],
        state["data"],
        t_query=state["tool_args"].get("t_query"),
        actions=state["tool_args"].get("actions"),
    )
    return {"figures": figures}


def ca_node(state: dict) -> dict:
    """Simulate a contrastive scenario using a manually specified action."""
    from internal_tools import contrastive_action

    args = state["tool_args"]
    figures = contrastive_action(
        state["rl_agent"],
        t_begin=args.get("t_begin"),
        t_end=args.get("t_end"),
        actions=args.get("actions"),
        values=args.get("values"),
    )
    return {"figures": figures}


def cb_node(state: dict) -> dict:
    """Simulate a contrastive scenario with aggressive or conservative action scaling."""
    from internal_tools import contrastive_behavior

    args = state["tool_args"]
    figures = contrastive_behavior(
        state["rl_agent"],
        t_begin=args.get("t_begin"),
        t_end=args.get("t_end"),
        actions=args.get("actions"),
        alpha=args.get("alpha", 1.0),
    )
    return {"figures": figures}


def q_decompose_node(state: dict) -> dict:
    """Decompose Q-values into individual reward components."""
    from internal_tools import q_decompose

    figures = q_decompose(
        state["data"],
        t_query=state["tool_args"].get("t_query"),
        team_conversation=state["team_conversation"],
        max_retries=state["max_retries"],
    )
    return {"figures": figures}


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONTRASTIVE POLICY SUB-FLOW NODES
#    Replaces the while-loop in CE_policy.py with a LangGraph cycle.
#
#    cp_init → cp_executor ──(error)──→ debugger → cp_coder_refine  ─┐
#                  │                                                 │
#                  └──(success)──→ evaluator ──(failed)──────────────┘
#                                      │
#                                      └──(passed)──→ cp_viz
# ══════════════════════════════════════════════════════════════════════════════

def cp_init_node(state: dict) -> dict:
    """
    Initialize the contrastive policy generation loop:
      1) Convert the queried time window to step indices
      2) Create a noise-free CE environment for reproducibility
      3) Run the actual policy rollout to obtain the baseline trajectory
      4) Generate the first candidate policy code with the Coder agent
    """
    args = state["tool_args"]
    rl_agent = state["rl_agent"]

    t_begin = args.get("t_begin")
    t_end = args.get("t_end")
    message = args.get("message")
    horizon = 20

    begin_index = int(np.round(t_begin / env_params["delta_t"]))
    end_index = int(np.round(t_end / env_params["delta_t"]))
    horizon += (end_index - begin_index + 1)

    # Disable noise for reproducibility
    env_params_ce = {**env_params, "noise": False}
    env_ce = make_env(env_params_ce)

    # Baseline (actual) policy rollout
    evaluator_obj, data_actual = env_ce.get_rollouts({"Actual": rl_agent}, reps=1)

    # Generate the first CE policy code
    coder = Coder()
    code = coder.generate(message, rl_agent)
    print("[Coder] Initial contrastive policy generated")

    team_conversation = list(state["team_conversation"])
    team_conversation.append({
        "agent": "Coder",
        "content": "Initial policy generated",
        "code_length": len(code),
    })

    return {
        "begin_index": begin_index,
        "end_index": end_index,
        "horizon": horizon,
        "env_ce": env_ce,
        "evaluator_obj": evaluator_obj,
        "data_actual": data_actual,
        "coder": coder,
        "generated_code": code,
        "code_error": None,
        "retry_count": 0,
        "team_conversation": team_conversation,
    }


def cp_executor_node(state: dict) -> dict:
    """
    Save the generated CE policy code to a file, dynamically load it, and run
    a rollout. Returns data_ce on success, or code_error (traceback) on failure.
    """
    code = state["generated_code"]
    rl_agent = state["rl_agent"]
    env_ce = state["env_ce"]
    begin_index = state["begin_index"]
    end_index = state["end_index"]
    retry_count = state["retry_count"]

    try:
        file_path = f"./policies/[{system}] ce_policy.py"
        str2py(code, file_path=file_path)
        CE_policy = py2func(file_path, "CE_policy")(env_ce, rl_agent)

        ce_settings = {
            "CE_mode": "policy",
            "begin_index": begin_index,
            "end_index": end_index,
            "CE_policy": CE_policy,
        }
        _, data_ce = env_ce.get_rollouts(
            {"New policy": rl_agent}, reps=1, ce_settings=ce_settings
        )

        print("[Executor] Policy executed successfully")
        return {"data_ce": data_ce, "code_error": None}

    except Exception as e:
        error_message = traceback.format_exc()
        retry_count += 1
        print(f"[Executor] Error (trial {retry_count}): {str(e)}")

        team_conversation = list(state["team_conversation"])
        team_conversation.append({
            "agent": "Executor",
            "content": f"[Trial {retry_count}] Execution error",
            "error_message": str(e),
            "error_type": type(e).__name__,
        })

        return {
            "code_error": error_message,
            "retry_count": retry_count,
            "team_conversation": team_conversation,
        }


def debugger_node(state: dict) -> dict:
    """Analyze the execution error and provide debugging guidance to the Coder agent."""
    debugger = Debugger()
    guidance = debugger.debug(state["generated_code"], state["code_error"])
    print("[Debugger] Guidance generated")

    team_conversation = list(state["team_conversation"])
    team_conversation.append({"agent": "Debugger", "content": guidance})

    return {"debugger_guidance": guidance, "team_conversation": team_conversation}


def cp_coder_refine_node(state: dict) -> dict:
    """
    Refine the generated code using Debugger guidance (if available) or
    the raw error message. Preserves the Coder instance's conversation history
    across multiple refinement calls.
    """
    coder = state["coder"]
    error_message = state["code_error"]
    guidance = state.get("debugger_guidance")

    if guidance:
        code = coder.refine_with_guidance(error_message, guidance)
    else:
        code = coder.refine_with_error(error_message)

    print(f"[Coder] Code refined (retry {state['retry_count']})")

    team_conversation = list(state["team_conversation"])
    team_conversation.append({
        "agent": "Coder",
        "content": f"[Trial {state['retry_count']}] Refined policy generated.",
        "code_length": len(code),
    })

    return {
        "generated_code": code,
        "debugger_guidance": None,   # clear guidance after use
        "team_conversation": team_conversation,
    }


def evaluator_node(state: dict) -> dict:
    """
    Validate via LLM whether the CE policy trajectory matches the user's intent.
    Sets evaluation_passed=True on acceptance, or False + code_error on rejection.
    """
    data_ce = state["data_ce"]
    begin_index = state["begin_index"]
    end_index = state["end_index"]
    user_query = state["user_query"]
    message = state["tool_args"].get("message", user_query)
    retry_count = state["retry_count"]

    # Slice the trajectory to the contrastive time window
    data_interval = {k: v[:, begin_index:end_index, :] for k, v in data_ce["New policy"].items()}
    x = data_interval["x"].squeeze(axis=2)   # (nx, T)
    u = data_interval["u"].squeeze(axis=2)   # (nu, T)
    trajectory = np.vstack([x, u]).T         # (T, nx+nu)

    # env.env_params is a copy made before feature_names was appended in params.py,
    # so use the module-level env_params dict which contains the complete key set.
    state_names = env_params["feature_names"]
    input_names = env.model.info()["inputs"]
    traj_df = pd.DataFrame(trajectory, columns=state_names + input_names)
    traj_as_json = traj_df.to_json(orient="records")

    try:
        ev = Evaluator()
        passed = ev.evaluate(traj_as_json, message=message)
        print(f"[Evaluator] Policy {'accepted' if passed else 'rejected'}")
        return {"evaluation_passed": passed, "code_error": None}

    except Exception as e:
        retry_count += 1
        error_message = str(e)
        print(f"[Evaluator] Policy rejected (trial {retry_count}): {error_message}")

        team_conversation = list(state["team_conversation"])
        team_conversation.append({
            "agent": "Evaluator",
            "content": f"[Trial {retry_count}] Trajectory rejected",
            "error_message": error_message,
        })

        return {
            "evaluation_passed": False,
            "code_error": error_message,
            "retry_count": retry_count,
            "team_conversation": team_conversation,
        }


def cp_viz_node(state: dict) -> dict:
    """
    Generate a comparison plot of the actual vs. CE policy trajectories
    using the pcgym evaluator object.
    """
    evaluator_obj = state["evaluator_obj"]
    data_actual = state["data_actual"]
    data_ce = state["data_ce"]
    begin_index = state["begin_index"]
    horizon = state["horizon"]
    rl_agent = state["rl_agent"]

    # Merge CE data into the evaluator object for combined plotting
    evaluator_obj.n_pi += 1
    evaluator_obj.policies["New policy"] = rl_agent
    evaluator_obj.data = data_actual | data_ce

    interval = [begin_index - 1, begin_index + horizon]
    figures = [evaluator_obj.plot_data(evaluator_obj.data, interval=interval)]

    log = "[Coder] Code successfully generated. Rollout complete."
    team_conversation = list(state["team_conversation"])
    team_conversation.append({"agent": "Coder", "content": log, "status": "Success"})
    print(log)

    return {"figures": figures, "team_conversation": team_conversation}


# ══════════════════════════════════════════════════════════════════════════════
# 4. EXPLAINER  —  Translate XRL figures into a natural language explanation
# ══════════════════════════════════════════════════════════════════════════════

def explainer_node(state: dict) -> dict:
    """
    Pass XRL analysis figures to a Vision LLM and generate a concise
    natural language explanation of the results.
    """
    figures = state.get("figures") or []
    user_query = state["user_query"]
    selected_tool = state["selected_tool"]

    explainer_prompt = get_prompts("explainer").format(
        user_query=user_query,
        fn_name=selected_tool,
        fn_description=get_fn_description(selected_tool),
        figure_description=get_figure_description(selected_tool),
        env_params=env_params,
        system_description=get_system_description(system),
        max_tokens=200,
    )

    messages = [{"role": "system", "content": explainer_prompt}]

    # Attach each figure as a base64-encoded vision input
    for fig in figures:
        encoded = encode_fig(fig)
        messages.append({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            }],
        })

    response = client.chat.completions.create(model=MODEL, messages=messages)
    explanation = response.choices[0].message.content

    print(f"\n[Explainer] {explanation}")

    team_conversation = list(state["team_conversation"])
    team_conversation.append({"agent": "Explainer", "content": explanation})

    return {"explanation": explanation, "team_conversation": team_conversation}
