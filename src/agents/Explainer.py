import numpy as np
import pandas as pd

from src.prompts import (
    get_prompts,
    get_fn_description,
    get_figure_description,
    get_system_description,
)
from src.params import get_running_params, get_env_params, get_explainer_LLM_configs
from src.utils import encode_fig

running_params = get_running_params()
env, env_params = get_env_params(running_params['system'])
system = running_params['system']

# %% Explainer agent
class Explainer:
    def __init__(self):
        self.history = []

    def explain(self, figures, user_query, selected_tool,
                ce_rollout_data=None, eo_rollout_data=None):
        """
        Pass XRL analysis figures to a Vision LLM and generate a concise
        natural language explanation of the results.
        Uses a separate EXPLAINER_MODEL (set in params.py) which can be a reasoning model.
        Args:
            figures (list): List of matplotlib figures produced by the XRL tool
            user_query (str): The original user query
            selected_tool (str): Name of the XRL tool that was executed
            ce_rollout_data (dict, optional): CE rollout data for grounding (ca / cb / cp tools)
            eo_rollout_data (dict, optional): EO (Q-decomposition) rollout data for grounding
        Returns:
            explanation (str): Natural language explanation
        """
        explainer_client, explainer_model = get_explainer_LLM_configs()

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

        # Build data summary for CE tools (ca / cb / cp) to ground the explanation in numbers
        if ce_rollout_data is not None:
            messages.append({"role": "user", "content": self._summarize_ce_rollout_data(ce_rollout_data)})

        # Build data summary for EO tool (q_decompose) to ground the explanation in numbers
        if eo_rollout_data is not None:
            messages.append({"role": "user", "content": self._summarize_eo_rollout_data(eo_rollout_data)})

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

        response = explainer_client.chat.completions.create(model=explainer_model, messages=messages)
        explanation = response.choices[0].message.content

        return explanation

    def _summarize_ce_rollout_data(self, data: dict) -> str:
        """
        Build a concise text summary of CE rollout data for the Explainer LLM.
        Provides median trajectories and factual constraint violation info
        so the LLM can ground its explanation in numbers rather than visual inference.
        """
        state_names = env.model.info()["states"]
        input_names = env.model.info()["inputs"]
        time_scale = env.env_params["time_scale"]
        t = np.linspace(0, env.tsim, env.N+1)
        lines = []

        for pi_name, traj in data.items():
            lines.append(f"=== Policy: {pi_name} ===")

            N = env.env_params["N"]
            t_traj = t[:N+1]  # handle interval-sliced data

            x_med = np.median(traj["x"], axis=2)   # (Nx, N+1)
            u_med = np.median(traj["u"], axis=2)    # (Nu, N+1)

            rows = {s: x_med[i] for i, s in enumerate(state_names)}
            rows.update({a: u_med[j] for j, a in enumerate(input_names)})
            df = pd.DataFrame(rows, index=t_traj)
            df.index.name = f"time ({time_scale})"
            lines.append(df.to_string())

            # Constraint violations — factual ground truth
            if "g" in traj and env.constraint_active:
                g = traj["g"]   # (n_con, N+1, 1, reps)
                viol_per_step = np.sum(g[:, :, 0, :], axis=2)  # (n_con, N+1)
                for ci, con_name in enumerate(env.constraints):
                    viol_indices = np.where(viol_per_step[ci] > 0)[0]
                    con_val = env.constraints[con_name]
                    if len(viol_indices) > 0:
                        viol_times = np.round(t_traj[viol_indices], 4).tolist()
                        lines.append(
                            f"Constraint '{con_name}' (bound={con_val}): "
                            f"violated at {time_scale}={viol_times}"
                        )
                    else:
                        lines.append(f"Constraint '{con_name}' (bound={con_val}): No violations.")
            lines.append("")

        return "\n".join(lines)

    def _summarize_eo_rollout_data(self, eo_data: dict) -> str:
        """
        Build a concise text summary of EO (Q-decomposition) rollout data for the Explainer LLM.
        Provides per-timestep decomposed reward component tables so the LLM can ground its
        explanation in numbers rather than visual inference.
        """
        r_trajs = eo_data["r_trajs"]
        component_names = eo_data["component_names"]
        t_query = eo_data["t_query"]
        horizon = eo_data["horizon"]
        delta_t = env.env_params["delta_t"]
        time_scale = env.env_params["time_scale"]
        lines = []

        for traj_name, rewards in r_trajs.items():
            lines.append(f"=== Trajectory: {traj_name} ===")
            dec_segment = rewards[:horizon]   # (horizon, C)
            t_axis = np.round(t_query + np.arange(len(dec_segment)) * delta_t, 6)
            df = pd.DataFrame(dec_segment, index=t_axis, columns=component_names)
            df.index.name = f"time ({time_scale})"
            lines.append(df.to_string())
            lines.append("")

        return "\n".join(lines)
