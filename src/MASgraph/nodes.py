"""
LangGraph node collection — pure orchestration.

Each function receives an AgentState dict and returns only the keys it
modifies. The five framework agents (Coordinator, Coder, Debugger,
Evaluator, Explainer) live as classes in agents/; the XRL/ modules
provide the explanation methods. rl_setup.py is used only for
environment/agent bootstrap (train_agent, get_rollout_data) by the
entry-point scripts.
"""

import traceback
import numpy as np
import pandas as pd

from src.params import get_running_params, get_env_params
from src.utils import str2py, py2func
from src.pcgym import make_env
from src.agents.Coder import Coder
from src.agents.Debugger import Debugger
from src.agents.Evaluator import Evaluator
from src.agents.Coordinator import Coordinator
from src.agents.Explainer import Explainer

# Module-level shared configuration (mirrors the pattern used in existing code)
running_params = get_running_params()
env, env_params = get_env_params(running_params["system"])
system = running_params["system"]
algo = running_params["algo"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. COORDINATOR  —  Parse the user query and select the appropriate XRL tool
# ══════════════════════════════════════════════════════════════════════════════

def coordinator_node(state: dict, verbose=1) -> dict:
    """
    Select the appropriate XRL tool for the user query (Coordinator agent).

    Returns:
        selected_tool (str): name of the chosen XRL function
        tool_args (dict): arguments to pass to that function
    """
    coordinator = Coordinator()
    selected_tool, tool_args = coordinator.select_tool(
        state["user_query"],
        coordinator_prompt_override=state.get("coordinator_prompt_override"),
    )

    print(f"[Coordinator] Tool: {selected_tool} | Args: {tool_args}") if verbose==1 else None

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
    """
    Compute SHAP-based global feature importance across all timesteps.

    Use when: You want to understand which features most influence the
    agent's policy across all states.
    Example:
        1) "How do the process states globally influence the agent's decisions?"
        2) "Which feature makes great contribution to the agent's decisions generally?"
    """
    from src.XRL.FI_SHAP import SHAP

    agent = state["rl_agent"]
    data = state["data"]
    actions = state["tool_args"].get("actions")
    feature_names = env_params.get("feature_names")

    if algo == "DDPG":
        actor = agent.actor.mu
    elif algo == "SAC":
        from torch.nn import Sequential
        actor = Sequential(*agent.actor.latent_pi, agent.actor.mu)  # sequentially connect the two networks

    X = data[algo]["x"].reshape(data[algo]["x"].shape[0], -1).T

    explainer = SHAP(model=actor, bg=X, feature_names=feature_names, algo=algo, env_params=env_params)
    explainer.explain(X=X)
    figures = explainer.plot(local=False, actions=actions)
    return {"figures": figures}


def fi_local_node(state: dict) -> dict:
    """
    Compute SHAP-based local feature importance at a specific timestep.

    Use when: You want to inspect how features affected the agent's decision
    at a specific time point.
    Example:
        1) "How do the state variables influence actions at t=400?"
        2) "Which state variable influenced the agent's action most at timestep 120?"
    """
    from src.XRL.FI_SHAP import SHAP

    agent = state["rl_agent"]
    data = state["data"]
    t_query = state["tool_args"].get("t_query")
    actions = state["tool_args"].get("actions")

    step_index = round(t_query / env_params["delta_t"])
    feature_names = env_params.get("feature_names")

    if algo == "DDPG":
        actor = agent.actor.mu
    elif algo == "SAC":
        import torch.nn as nn
        actor = nn.Sequential(
            agent.actor.features_extractor,
            agent.actor.latent_pi,   # MLP
            agent.actor.mu,          # final linear layer producing mean action
            nn.Tanh(),
        )

    X = data[algo]["x"].reshape(data[algo]["x"].shape[0], -1).T

    explainer = SHAP(model=actor, bg=X, feature_names=feature_names, algo=algo, env_params=env_params)
    instance = X[step_index, :]
    explainer.explain(X=instance)
    figures = explainer.plot(local=True, actions=actions)
    return {"figures": figures}


def ca_node(state: dict) -> dict:
    """
    Simulate a contrastive scenario using a manually specified action.

    Use when: You want to simulate a contrastive scenario with a manually
    chosen action.
    Example:
        1) "Why don't we apply a different action of a=100 at t=400 instead?"
        2) "What would have happened if we had chosen action = 300 from t=200 to t=400?"
    """
    from src.XRL.CE_action import ce_by_action

    args = state["tool_args"]
    figures, data = ce_by_action(
        t_begin=args.get("t_begin"),
        t_end=args.get("t_end"),
        actions=args.get("actions"),
        values=args.get("values"),
        policy=state["rl_agent"],
        horizon=20,
    )
    return {"figures": figures, "ce_rollout_data": data}


def cb_node(state: dict) -> dict:
    """
    Simulate a contrastive scenario with aggressive or conservative action scaling.

    Use when: You want to simulate a contrastive scenario with different
    control behaviors.
    Example:
        1) "What would happen if the agent had a more aggressive behavior than our current agent?"
        2) "Why don't we just control the system in an opposite direction from t=4000 to 4200?"
    """
    from src.XRL.CE_behavior import ce_by_behavior

    args = state["tool_args"]
    figures, data = ce_by_behavior(
        t_begin=args.get("t_begin"),
        t_end=args.get("t_end"),
        actions=args.get("actions"),
        alpha=args.get("alpha", 1.0),
        policy=state["rl_agent"],
        horizon=20,
    )
    return {"figures": figures, "ce_rollout_data": data}


def q_decompose_node(state: dict) -> dict:
    """
    Decompose Q-values into individual reward components.

    Use when: You want to know the agent's intention behind a certain action,
    by decomposing Q-values into both semantic and temporal dimensions.
    Example:
        1) "What is the agent trying to achieve in the long run by doing this action at timestep 180?"
        2) "What is the agent's intention behind the action at timestep 200?"
    """
    from src.XRL.EO_Qdecompose import decompose_forward

    t_query = state["tool_args"].get("t_query")
    horizon = 10

    figures, r_trajs, component_names = decompose_forward(
        t_query=t_query,
        data=state["data"],
        env=env,
        team_conversation=state["team_conversation"],
        max_retries=state["max_retries"],
        horizon=horizon,
    )
    eo_rollout_data = {
        "r_trajs": r_trajs,
        "component_names": component_names,
        "t_query": t_query,
        "horizon": horizon,
    }
    return {"figures": figures, "eo_rollout_data": eo_rollout_data}


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

    start, end = max(0, begin_index), min(env_params["N"]+1, begin_index + horizon)
    interval = [start, end]  # Interval to watch the control results
    ce_data = evaluator_obj.data
    figures = [evaluator_obj.plot_data(ce_data, interval=interval)]

    log = "[Coder] Code successfully generated. Rollout complete."
    team_conversation = list(state["team_conversation"])
    team_conversation.append({"agent": "Coder", "content": log, "status": "Success"})
    print(log)

    return {"figures": figures, "ce_rollout_data": ce_data, "team_conversation": team_conversation}


# ══════════════════════════════════════════════════════════════════════════════
# 4. EXPLAINER  —  Translate XRL figures into a natural language explanation
# ══════════════════════════════════════════════════════════════════════════════

def explainer_node(state: dict) -> dict:
    """
    Translate the XRL analysis figures into a natural language explanation
    (Explainer agent). Uses a separate EXPLAINER_MODEL (set in params.py).
    """
    explainer = Explainer()
    explanation = explainer.explain(
        figures=state.get("figures") or [],
        user_query=state["user_query"],
        selected_tool=state["selected_tool"],
        ce_rollout_data=state.get("ce_rollout_data"),
        eo_rollout_data=state.get("eo_rollout_data"),
    )

    print(f"\n[Explainer] {explanation}")

    team_conversation = list(state["team_conversation"])
    team_conversation.append({"agent": "Explainer", "content": explanation})

    return {"explanation": explanation, "team_conversation": team_conversation}
