# %% System descriptions
def get_system_description(system):
    cstr_description = """
    ### Description & Equations
    The continuously stirred tank reactor (CSTR) is a system which converts species A to species B via the reaction: A → B.
    The reactor's temperature is controlled by a cooling jacket. The following system of equations describes the system:
    
    dC_A/dt = (q/V)(C_{A_f} - C_A) - kC_A * exp(-E_A / (R * T))
    
    dT/dt = (q/V)(T_f - T) - (ΔH_R / (ρ * C_p)) * kC_A * exp(-E_A / (R * T)) + (UA / (ρ * C_p * V)) * (T_c - T)
    
    where:
    - C_A: concentration of species A in the reactor
    - T: temperature of the reactor
    - x = [C_A, T]^T ∈ ℝ²: state variables
    - u = T_c: cooling water temperature (action variable)
    
    ### Observation
    The observation of the CSTR environment provides information on the state variables and their associated errors between current values and setpoints at the current timestep.
    Therefore, the observation when there exists a error for C_A is [C_A, T, Errors_C_A].
    Errors are defined by (setpoint - current value).
    The observation space and initial conditions are defined as (env_params).
    
    ### Action
    The action corresponds to a jacket temperature, which is bounded by action_space in (env_params).
    
    ### Reward
    The reward is a continuous value corresponding to the square error of the state and its setpoint.
    For multiple states, these are scaled with a factor (r_scale) and summed to give a single value.
    """

    four_tank_description = """
    ### Description & Equations
    The four-tank system is a multivariable process consisting of four interconnected water tanks.
    The model describes the change in water levels in each tank based on the inflows and outflows.

    Equations:
        dh1/dt = -(a1/A1)*sqrt(2*g_a*h1) + (a3/A1)*sqrt(2*g_a*h3) + (gamma1*k1/A1)*v1
        dh2/dt = -(a2/A2)*sqrt(2*g_a*h2) + (a4/A2)*sqrt(2*g_a*h4) + (gamma2*k2/A2)*v2
        dh3/dt = -(a3/A3)*sqrt(2*g_a*h3) + ((1 - gamma2)*k2/A3)*v2
        dh4/dt = -(a4/A4)*sqrt(2*g_a*h4) + ((1 - gamma1)*k1/A4)*v1
        
    where:
    - h_i: Water level
    - A_i: Cross-section area of the tank
    - a_i: Corss-section area of the outlet hole
    (i = 1,2,3,4)

    ## Observation Space
    The observation of the 'four_tank' environment provides information on the state variables and their associated setpoints (if they exist) at the current timestep.
    The observation is an array of shape (1, 4 + N_SP) where N_SP is the number of setpoints.
    For example, the observation when there are errors for h1 and h2 is [h1, h2, h3, h4, Errors_h1, Errors_h2].
    Errors are defined by (setpoint - current value).

    ## Action Space
    The action space consists of two variables (v1 and v2), which represent the voltages to the respective pumps.

    ## Reward
    The reward is a continuous value corresponding to the square error of the state and its setpoint.
    For multiple states, these are scaled with a factor r_scale and summed to give a single value.
    The goal of this environment is to drive the x1 state to the origin.
    """

    if system == 'cstr':
        return cstr_description
    elif system == 'four_tank':
        return four_tank_description
    else:
        raise Exception("System not correctly configured!")


# %% Function descriptions
feature_importance_global_fn_description = """
Use when: You want to understand which features most influence the agent’s policy across all states.
"""

feature_importance_local_fn_description = """
Use when: You want to inspect how features affected the agent's decision at a specific point.
"""

partial_dependence_plot_global_fn_description = """
Use when: You want to examine how changing one input feature influences the agent's action.
"""

partial_dependence_plot_local_fn_description = """
Use when: You want to examine how changing one input feature AT SPECIFIC TIME POINT influences the agent's action.
"""

counterfactual_action_fn_description = """
Use when: You want to simulate a counterfactual scenario with manually chosen action.
"""

counterfactual_behavior_fn_description = """
Use when: You want to simulate a counterfactual scenario with different control behaviors
"""

counterfactual_policy_fn_description = """
Use when: You want to what would the trajectory would be if we chose alternative policy, or to compare the optimal policy with other policies.
"""

q_decompose_fn_description = """
Use when: You want to know the agent's intention behind certain action, by decomposing q values into both semantic and temporal dimension.
"""

# %% Get prompts
def get_prompts(prompt):
    system_description_prompt = """
    You are a chemical process operator and your role is to briefly explain the system that are simulated and controlled based on reinforcement learning.
    
    The brief explanation of control system is given below:
    {system_description}
    -----
    
    Furthermore, the environment parameters are given below:
    {env_params}
    -----
    
    If user requires the system to be explained first, you would:
        - Start with a brief description of the system, including what the observation and action variables are.
        - Explain how the action variable can affect the observation variables
        - Clarify what the controller is trying to achieve.
        - Explain what constraints are imposed on the system, if available.
    Otherwise, keep these descriptions in memory and infer them when explaining XRL results.
    """

    explainer_prompt = """
    You're an expert in both explainable reinforcement learning (XRL).
    Your role is to explain the XRL results and figures triggered by XRL functions in natural language form.
    
    - Below are the name of the XRL function triggered and it's description:
        Function name:
            {fn_name}
        
        Function description:
            {fn_description}
        
    - Also, for more clear explanation, the description of the system and its environment parameters are given as below:
        System description:
            {system_description}
        
        Environment parameters:
            {env_params}
            
    - If XRL visualization are inputted, briefly explain how to interpret the all given visualization results.
        Figure description:
            {figure_description}
        
    - If there are multiple agent actions to be explained, you will get sets of the plots. Make sure to interpret them individually.
    - Make sure to emphasize how the XRL results relates to the task of chemical process control, based on the given system description.
    - The explanation output must be concise and short enough (below {max_tokens} tokens), because users may be distracted by too much information.
    - Try to concentrate on providing only the explanation results, not on additional importance of the explanation.
        
    Make sure the explanation must be coherent and easy to understand for the users who are experts in chemical process,
    but not quite informed at explainable artificial intelligence tools and their interpretations.  
    """

    coordinator_prompt = """
    Your task is to choose the next function to work on the problem based on the given function tools and user queries.
    
    The brief explanation of control system is given below:
    {system_description}
    -----
    
    Furthermore, the environment parameters are given below:
    {env_params}
    -----
    
    Here are a few points that you have to consider while calling a function:
    - When calling a function with 'action' argument, make sure the action is within env_params["actions"].
      Otherwise raise an error.
    - When queried for a certain time interval, make sure to use the queried time itself when calling the function, without dividing by 'delta_t' parameter.
    - Also, don't scale neither state or action value, since it will be scaled at the subsequent functions.  
    """

    if prompt == 'coordinator_prompt':
        return coordinator_prompt
    elif prompt == 'explainer_prompt':
        return explainer_prompt
    elif prompt == 'system_description_prompt':
        return system_description_prompt

def get_fn_description(fn_name):
    if fn_name == "feature_importance_global":
        return feature_importance_global_fn_description
    elif fn_name == "feature_importance_local":
        return feature_importance_local_fn_description
    elif fn_name == "partial_dependence_plot_global":
        return partial_dependence_plot_global_fn_description
    elif fn_name == "partial_dependence_plot_local":
        return partial_dependence_plot_local_fn_description
    elif fn_name == "counterfactual_action":
        return counterfactual_action_fn_description
    elif fn_name == "counterfactual_behavior":
        return counterfactual_behavior_fn_description
    elif fn_name == "counterfactual_policy":
        return counterfactual_policy_fn_description
    elif fn_name == "q_decompose":
        return q_decompose_fn_description


def get_fn_json():
    fn_json = [
        {
            "type": "function",
            "name": "feature_importance_global",
            "description": feature_importance_global_fn_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Name of the agent action to be explained"
                    },
                },
                "required": ["agent", "data"]
            }
        },
        {
            "type": "function",
            "name": "feature_importance_local",
            "description": feature_importance_local_fn_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Name of the agent action to be explained"
                    },
                    "t_query": {
                        "type": "number",
                        "description": "Time points to query for feature importance"
                    },
                },
                "required": ["agent", "data", "t_query"]
            }
        },
        {
            "type": "function",
                "name": "partial_dependence_plot_global",
                "description": partial_dependence_plot_global_fn_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Name of the agent action to be explained"
                        },
                        "features": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of names of the state variable whose impact are to be explained"
                        },
                    },
                    "required": ["agent", "data"]
            }
        },
        {
            "type": "function",
            "name": "partial_dependence_plot_local",
            "description": partial_dependence_plot_local_fn_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Name of the agent action to be explained"
                    },
                    "features": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of names of the state variable whose impact are to be explained"
                    },
                    "t_query": {
                        "type": "number",
                        "description": "Time points to query for sensitivity analysis"
                    },
                },
                "required": ["agent", "data", "t_query"]
            }
        },
        {
            "type": "function",
            "name": "counterfactual_action",
            "description": counterfactual_action_fn_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "t_begin": {
                        "type": "number",
                        "description": "Start timestep of the counterfactual intervention."
                    },
                    "t_end": {
                        "type": "number",
                        "description": "End timestep of the counterfactual intervention."
                    },
                    "actions": {
                        "type": "array",
                        "description": "List of action names (variables) to which counterfactual values should be applied.",
                        "items": {
                            "type": "string"
                        },
                        "example": ["v1", "v2"]
                    },
                    "values": {
                        "type": "array",
                        "description": "List of counterfactual values corresponding to each action in 'actions'. Must be the same length.",
                        "items": {
                            "type": "number"
                        },
                        "example": [0.5, -0.2]
                    }
                },
                "required": ["t_begin", "t_end", "actions", "values"]
            }
        },
        {
            "type": "function",
            "name": "counterfactual_behavior",
            "description": counterfactual_behavior_fn_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "t_begin": {
                        "type": "number",
                        "description": "Start timestep of the counterfactual intervention."
                    },
                    "t_end": {
                        "type": "number",
                        "description": "End timestep of the counterfactual intervention."
                    },
                    "actions": {
                        "type": "array",
                        "description": "List of action names (variables) to which counterfactual behavior should be applied.",
                        "items": {
                            "type": "string"
                        },
                        "example": ["v1"]
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Magnitude and direction of behavioral change in control actions. Default value of 1.0 means original control behavior."
                                       "Higher value implies aggressive controller, while lower values means conservative one. Negative value means opposite behavior"
                                       "It would be better to set alpha below 2.0, since too much alpha will cause instability of the controller.",
                        "example": 1.8
                    }
                },
                "required": ["t_begin", "t_end", "actions", "alpha"]
            }
        },
        {
            "type": "function",
            "name": "counterfactual_policy",
            "description": counterfactual_policy_fn_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "t_begin": {
                        "type": "number",
                        "description": "Start timestep of the counterfactual intervention."
                    },
                    "t_end": {
                        "type": "number",
                        "description": "End timestep of the counterfactual intervention."
                    },
                    "message": {
                        "type": "string",
                        "description": "Brief instruction for constructing the counterfactual policy. It is used as prompts for the Coder agent."
                                       "Currently, only rule-based control are used for the alternative policy."
                    },
                },
                "required": ["t_begin", "t_end", "message"]
            }
        },
        {
            "type": "function",
            "name": "q_decompose",
            "description": q_decompose_fn_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "t_query": {
                        "type": "number",
                        "description": "Time points to query for counterfactual analysis"
                    },
                },
                "required": ["agent", "data", "t_query"]
            }
        },
        {
            "type": "function",
            "name": "raise_error",
            "description": "Raises error based on given message",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Error message"
                    },
                },
                "required": ["message"]
            }
        }
    ]
    return fn_json

def get_figure_description(fn_name):
    control_term_description = """
    Here are some control-related terms that you can use to determine and describe the counterfactual behavior:
        - Overshoot: When the system output temporarily exceeds the desired target before settling.
        - Undershoot: When the system output temporarily drops below the target value before converging.
        - Settling time: The time required for the system response to remain within a small error band (e.g., ±2%) around the target value.
        - Opposite behavior: The control action moves in the reverse direction compared to the expected response (e.g., increasing instead of decreasing).
        - Critically damped response: The fastest åresponse without oscillation, reaching the target in minimum time without overshoot.
        - Over-damped response: A slow, smooth response with no oscillation but taking longer to reach the target.
        - Under-damped response: A response with oscillations around the target before eventually settling.
        - Steady-state error: The difference between the system’s final output and the desired target value after all transient effects have decayed."""


    feature_importance_global_figure_description = """ fn_name is feature_importance_global.
    You will get sets of three plots as results:
        First plot is the bar plot, which compares feature importance values of features.
        Second plot is the beeswarm plot, which shows both the magnitude and direction of feature attribution to action variables.
        Third plot is the decision plot, which shows how each data deviates from the reference value, affected by each feature variable."""

    feature_importance_local_figure_description = """ fn_name is feature_importance_local.
    You will get one plot as results:
        The waterfall plot compares how each state variable affect the action values.
        Make sure to discuss both the magnitude and direction of contributions of each feature.
        Also, relate the values of the state variables against the observation space defined in 'env_params' to determine their relative magnitudes.
        Then, relate these magnitudes to the SHAP values to deduce how high or low state variables influence the agent's actions."""

    partial_dependence_plot_global_figure_description = """ fn_name is partial_dependence_plot_global.
    You will get one plot as results:
        The PDP plot displays how the action value will change as each state variables differ.
        In global PDP plot, results of all the states are displayed and mean tendency is visualized by separate color."""

    partial_dependence_plot_local_figure_description = """ fn_name is partial_dependence_plot_local.
    You will get one plot as results:
        The ICE plot displays how the action value will change as each state variables differ.
        In local ICE plot, results of a singel state at queried timestep are displayed."""

    counterfactual_figure_description = f"""
    You will get a plot of future trajectory of actual and counterfactual scenarios.
    Your job is to explain how the environment(e.g.) states, rewards) would change, in terms of both short and long perspective.
    Explain in those three perspectives, then make a summary of whether the counterfactual scenario exceled at controlling the system and why.
        1. States
        2. Actions
        3. Rewards
        
    It would be better if you can explain why the action yielded by the actor was the best, instead of other actions.
    Interpret the graph of region after 't_begin' only, not before 't_begin'.
    Focus on comparing the actual trajectory with counterfactual trajectory.

    Relate the behavior of both actual and counterfactual behavior with control-related contexts, based on the defininitions below:
        - Overshoot: When the system output temporarily exceeds the desired target before settling.
        - Undershoot: When the system output temporarily drops below the target value before converging.
        - Settling time: The time required for the system response to remain within a small error band (e.g., ±2%) around the target value.
        - Opposite behavior: The control action moves in the reverse direction compared to the expected response (e.g., increasing instead of decreasing).
        - Critically damped response: The fastest åresponse without oscillation, reaching the target in minimum time without overshoot.
        - Over-damped response: A slow, smooth response with no oscillation but taking longer to reach the target.
        - Under-damped response: A response with oscillations around the target before eventually settling.
        - Steady-state error: The difference between the system’s final output and the desired target value after all transient effects have decayed.
    """

    counterfactual_action_figure_description = f"""fn_name is counterfactual_action.
    You will get one plot as results:
        The plot shows future trajectory when executed an action with various counterfactual action values.
        You will have to explain how the environment would change, in terms of both short and long perspective.
        It would be better if you can explain why the action yielded by the actor was the best, instead of other actions.
        Interpret the graph of region after 't_begin' only, not before 't_begin'.
        Focus on comparing the actual trajectory with counterfactual trajectory.
        
    {control_term_description}
    """

    counterfactual_behavior_figure_description = f"""fn_name is counterfactual_action.
    You will get one plot as results:
        The plot compares future trajectory from original controller and with one from the counterfactual control behavior.
        You will have to explain how the environment would change, in terms of both short and long perspective.
        It would be better if you can compare the two trajectories in terms of settling time or overshooting behavior, and concluding the overall performance of two control trajectories.
        
    {control_term_description}
    """

    counterfactual_policy_figure_description = f"""fn_name is counterfactual_policy.
    You will get one plot as results:
        The plot compares potential rollout between our RL policy and the counterfactual policy made by coder agent.
        You will have to explain how does the two policies differ in acting and which one is better in controlling the system.
        If CF policy failed to control the system, it would be better to analyze the potential cause of the failure, based on the CF policy itself and system descriptions.
    
    {control_term_description}
    """

    q_decompose_figure_description = """fn_name is q_decompose.
    You will get one plot as results:
        The plot shows Q-values decomposed in both temporal and semantic dimension.
        You will have to explain what the agent has achieved by executing the action at the queried time step.
        If possible, it is better to describe which reward the agent prioritized to obtain among various reward components, in both short and long time scale. 
        Make sure that the rewards are being visualized in negative fashion, so bigger portion of bar means more negative reward.
    """

    if fn_name == "feature_importance_global":
        return feature_importance_global_figure_description
    elif fn_name == "feature_importance_local":
        return feature_importance_local_figure_description
    elif fn_name == "partial_dependence_plot_global":
        return partial_dependence_plot_global_figure_description
    elif fn_name == "partial_dependence_plot_local":
        return partial_dependence_plot_local_figure_description
    elif fn_name == "counterfactual_action":
        return counterfactual_action_figure_description
    elif fn_name == "counterfactual_behavior":
        return counterfactual_behavior_figure_description
    elif fn_name == "counterfactual_policy":
        return counterfactual_policy_figure_description
    elif fn_name == "q_decompose":
        return q_decompose_figure_description
