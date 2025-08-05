# %% System descriptions
def get_system_description(system):
    """
    Returns the system description
    Args:
        system (str): Name of the system (environment)
    Returns:
        description (str): Description of the system
    """
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
    
    It consists of two upper tanks (Tank 3 and Tank 4) and two lower tanks (Tank 1 and Tank 2), where the lower tanks' water levels (h1, h2) are the controlled variables.
    
    Two pumps supply water with voltages (v1, v2), and the flow from each pump is split by valves with ratios (\gamma_1, \gamma_2):valve 1 directs the flow to both Tank 1(h1) and Tank 3(h3), while valve 2 directs the flow to both Tank 2(h2) and Tank 4(h4).
    
    Due to hydraulic coupling, a change in one pump affects multiple tank levels simultaneously.
    
    The water in Tank 3(h3) flows into Tank 1(h1), while the water in Tank 4(h4) flows into Tank 2(h2) under the influence of gravity.

    ## Observation Space
    The observation is an array of shape (1, 4 + N_error) where N_error is the number of error terms.
    For example, the observation when there are errors for h1 and h2 is [h1, h2, h3, h4, Errors_h1, Errors_h2].
    Errors are defined by (setpoint - current value).

    ## Action Space
    The action space consists of two variables (v1 and v2), which represent the voltages to the respective pumps.

    ## Reward
    The reward is a continuous value corresponding to the square error of the state and its setpoint.
    For multiple states, these are scaled with a factor r_scale and summed to give a single value.
    The goal of this environment is to drive the x1 state to the origin.
    """

    multistage_extraction_description = """
    ### Description & Equations
    The multistage extraction column is a key unit operation in chemical engineering that enables mass transfer between
    liquid and gas phases across multiple theoretical stages, described by coupled differential equations representing
    the dynamic concentration changes in each phase:
    
    dXi/dt = LV/L * (Xi-1 - Xi) - KLa * (Xi - Yim)
    dYi/dt = GV/G * (Yi+1 - Yi) + KLa * (Xi - Yim)
    
    Where:
    - Xi and Yi are the solute concentrations in the liquid and gas at each stage.
    - State vector: x = [X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5] ∈ R^10
    - Action variables: u = [L, G] ∈ R^2 (liquid and gas flow rates)
    
    ### Observation
    The observation provides the current state variables and error values of target states.
    - Observation shape: (1, 10 + N_SP)
    - Example observation when targets are X5 and Y1:
      [X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, error_X5, error_Y1]
    
    Observation space bounds:
    [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[-1,1],[-1,1]]
    
    Example initial conditions:
    [0.55, 0.3, 0.45, 0.25, 0.4, 0.20, 0.35, 0.15, 0.25, 0.1, 0.0, 0.0]
    
    ### Action
    ContinuousBox([[5,10],[500,1000]])
    - Liquid phase flowrate: 5–500 m³/hr
    - Gas phase flowrate: 10–1000 m³/hr
    
    ### Reward
    The reward is the negative squared error between the current state and its setpoint.
    For multiple states, these are scaled by r_scale and summed.
    """

    if system == 'cstr':
        return cstr_description
    elif system == 'four_tank':
        return four_tank_description
    elif system == 'multistage_extraction':
        return multistage_extraction_description
    else:
        raise Exception("System not correctly configured!")


# %% Function descriptions
feature_importance_global_fn_description = """
Use when: You want to understand which features most influence the agent’s policy across all states.
Example:
    1) "How do the process states globally influence the agent's decisions?"
    2) "Which feature makes great contribution to the agent's decisions generally?"
"""

feature_importance_local_fn_description = """
Use when: You want to inspect how features affected the agent's decision at a specific point.
Example:
    1) "How do the state variables influence actions at t=400?"
    2) "Which state variable influenced the agent's action most at timestep 120?"
"""

counterfactual_action_fn_description = """
Use when: You want to simulate a counterfactual scenario with manually chosen action.
Example:
    1) "Why don't we apply a different action of a=100 at t=400 instead?"
    2) "What would have happened if we had chosen action = 300 from t=200 to t=400?"
"""

counterfactual_behavior_fn_description = """
Use when: You want to simulate a counterfactual scenario with different control behaviors
Example:
    1) "What would happen if the agent had a more aggressive behavior than our current agent?"
    2) "Why don't we just control the system in an opposite direction from t=4000 to 4200?"
"""

counterfactual_policy_fn_description = """
Use when: You want to what would the trajectory would be if we chose alternative policy,
        or to compare the optimal policy with other policies.
Example:
    1) "What would the trajectory change if I use the on-off controller instead of the current RL policy?"
    2) "What if a simple threshold rule was applied between timestep 4000 and 4400, setting v1 = 0.1 whenever h3 > 0.9 and v1 = 3.0 whenever h3 < 0.4, instead of using the RL policy?"
"""

q_decompose_fn_description = """
Use when: You want to know the agent's intention behind certain action, by decomposing q values into both semantic and temporal dimension.
Example:
    1) "What is the agent trying to achieve in the long run by doing this action at timestep 180?"
    2) "Why is the agent's intention behind the action at timestep 200?"
"""

# %% Get prompts
def get_prompts(agent_name):
    """
    Get prompts for coordinator and explainer agents. Prompts for (Coder, Evaluator and Debugger) are embedded in their own agent .py file.
    Args:
        agent_name (str): Name of the agent
    Returns:
        prompt (str): Corresponding prompt of the agent
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

    explainer_prompt = """
    You're an expert in both explainable reinforcement learning (XRL) and process control.
    Your role is to explain the user queries based on XRL results and related figures triggered by XRL functions.
    
    User query: {user_query}
    
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
            
    - If XRL visualization are available, briefly explain how to interpret the all given visualization results.
        Figure description:
            {figure_description}
        
    - If there are multiple agent actions to be explained, you will get sets of the plots. Make sure to interpret them individually.
    - IMPORTANT! Make sure to relate the XRL results to input-output relationship within the system, based on the given system description.
    - The explanation output must be concise and short enough (below {max_tokens} tokens), because users may be distracted by too much information.
    - Try to concentrate on providing only the explanation results, not on additional importance of the explanation.
    """

    if agent_name == 'coordinator_prompt':
        return coordinator_prompt
    elif agent_name == 'explainer_prompt':
        return explainer_prompt

def get_fn_description(fn_name):
    """
    Returns the function description
    Args:
        fn_name (str): Name of the predefined XRL function
    Returns:
        fn_description (str): Corresponding description of the XRL function
    """
    if fn_name == "feature_importance_global":
        return feature_importance_global_fn_description
    elif fn_name == "feature_importance_local":
        return feature_importance_local_fn_description
    elif fn_name == "counterfactual_action":
        return counterfactual_action_fn_description
    elif fn_name == "counterfactual_behavior":
        return counterfactual_behavior_fn_description
    elif fn_name == "counterfactual_policy":
        return counterfactual_policy_fn_description
    elif fn_name == "q_decompose":
        return q_decompose_fn_description


def get_fn_json():
    """
    Get json for XRL functions, which are used in the coordinator agent to select appropriate tools
    Returns:
        fn_json (dict): JSON representation of the XRL functions
    """
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
    """
    Return description about the expected figures visualized by the XRL tools.
    This helps Explainer agent to read the figure and relate it into domain context.
    Args:
        fn_name (str): Name of the predefined XRL tool
    Returns:
        figure_description (str): Corresponding description of the visualization results triggered by XRL tool
    """
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

    # # Use this figure description when the expected decomposed rewards are also compared between actual and counterfactual policies
    # counterfactual_figure_description = f"""
    # You will get two plots as results, and your job is to explain why a certain action trajectory is better in control than the other:
    #     - The first plot compares future trajectory from original controller and with one from the counterfactual control behavior.
    #         - From this plot, you will have to explain how the environment(e.g.) states, rewards) would change, in terms of both instant and long-term perspective.
    #
    #     - The second plot compares the future decomposed reward of executing actual and counterfactual action trajectory for short-time period.
    #         - From this plot, you should focus on explaining which reward components indicate that one control behavior outperforms the others.
    #
    #     Here are some points that you might have to consider when generating explanations
    #     - It would be really great if you select a specific time interval that was critical for deciding the control aptitude of two trajectories.
    #     - Also, you might compare the two trajectories in terms of settling time or overshooting behavior, and concluding the overall performance of two control trajectories.
    #     - If Counterfactual trajectory failed to control the system, it would be better to analyze the potential cause of the failure.
    #     - Lastly, make a summary of whether the counterfactual scenario exceled at controlling the system and why.
    #
    # Interpret the graph of region after 't_begin' only, not before 't_begin'.
    # Focus on comparing the actual trajectory with counterfactual trajectory.
    # """

    counterfactual_figure_description = f"""
    You will get one plot as results, and your job is to explain why a certain action trajectory is better in control than the other:
        - The first plot compares future trajectory from original controller and with one from the counterfactual control behavior.
            - From this plot, you will have to explain how the environment(e.g.) states, rewards) would change, in terms of both instant and long-term perspective.

        Here are some points that you might have to consider when generating explanations
        - It would be really great if you select a specific time interval that was critical for deciding the control aptitude of two trajectories.
        - Also, you might compare the two trajectories in terms of settling time or overshooting behavior, and concluding the overall performance of two control trajectories.
        - If Counterfactual trajectory failed to control the system, it would be better to analyze the potential cause of the failure.
        - Lastly, make a summary of whether the counterfactual scenario exceled at controlling the system and why.

    Interpret the graph of region after 't_begin' only, not before 't_begin'.
    Focus on comparing the actual trajectory with counterfactual trajectory.
    """

    q_decompose_figure_description = """fn_name is q_decompose.
    You will get one plot as results:
        - The plot shows reward values decomposed in both temporal and semantic dimension.
        - You will have to explain what the agent has achieved by executing the action at the queried time step.
        - If possible, it is better to describe which reward the agent prioritized to obtain among various reward components, in both short and long time scale.
        - If possible, make sure that explain how does each of the reward component change, and whenever there exist a sacrifice of one reward component for another, please mention it.
        - Make sure that the rewards are being visualized in negative fashion, so bigger portion of bar means more negative reward.
    """

    if fn_name == "feature_importance_global":
        return feature_importance_global_figure_description
    elif fn_name == "feature_importance_local":
        return feature_importance_local_figure_description
    elif fn_name == "counterfactual_action":
        return counterfactual_figure_description
    elif fn_name == "counterfactual_behavior":
        return counterfactual_figure_description
    elif fn_name == "counterfactual_policy":
        return counterfactual_figure_description
    elif fn_name == "q_decompose":
        return q_decompose_figure_description
