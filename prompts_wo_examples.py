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

    photo_production_description = """
    ### Description & Equations
    A model describing the photo production of phycocyanin from Cyanobacteria Arthrospira platensis.
    The system is described by three state variables representing biomass concentration (c_x), nitrate concentration (c_N), and phycocyanin concentration (c_q).
    
    The system dynamics are governed by the following equations:
    
        dc_x/dt = (μ_m I / (I + k_s + I^2 / k_i)) · (c_x c_N / (c_N + k_N)) − μ_d c_x  
        dc_N/dt = −Y_NX · (μ_m I / (I + k_s + I^2 / k_i)) · (c_x c_N / (c_N + k_N)) + F_N  
        dc_q/dt = (k_m I / (I + k_{sq} + I^2 / k_{iq})) · c_x − (k_d c_q / (c_N + K_{Nq}))
    
    where x = [c_x, c_N, c_q]^T ∈ ℝ³ represents the state vector and u = [I, F_N]^T represents the input vector consisting of light intensity (I) and nitrate feed rate (F_N).
    
    ### Initial Conditions
    The initial conditions for the state variables are defined as follows:  
    x₀ = [0.1, 20.0, 0.01]: Initial state vector representing the initial concentrations of biomass, nitrate, and phycocyanin, respectively.
    
    ### Model Parameters
    The model includes the following parameters:  
    - μ_m = 0.0572: Maximum specific growth rate  
    - μ_d = 0.0: Death rate  
    - Y_NX = 504.5: Yield coefficient  
    - k_m = 0.00016: Product formation rate  
    - k_d = 0.281: Product degradation rate  
    - k_{sq} = 23.51: Light saturation constant for product formation  
    - K_{Nq} = 16.89: Nitrate saturation constant for product degradation  
    - k_{iq} = 800.0: Light inhibition constant for product formation
    
    ### States
    The model has three states:  
    - c_x: Biomass concentration  
    - c_N: Nitrate concentration  
    - c_q: Phycocyanin concentration
    - qx_ratio: Ratio of c_q over c_x
    
    ### Inputs
    The system has two control inputs:  
    - I: Light intensity  
    - F_N: Nitrate feed rate
    
    ### Reward
    The reward function combines (1 − tanh(c_q)) element with quadratic soft penalties for violating constraints for c_N and qx_ratio respectively, and a quadratic penalty on changes in the control inputs.
    For multiple constraints, these are summed to give a single value.
    This encourages the system to maximize the amount of Phycocyanin (c_q) while constraining state variables and penalizing large control input variations.
    """

    if system == 'cstr':
        return cstr_description
    elif system == 'four_tank':
        return four_tank_description
    elif system == 'photo_production':
        return photo_production_description
    else:
        raise Exception("System not correctly configured!")


# %% Function descriptions
feature_importance_global_fn_description = """
Use when: You want to understand which features most influence the agent’s policy across all states.
"""

feature_importance_local_fn_description = """
Use when: You want to inspect how features affected the agent's decision at a specific point.
"""

contrastive_action_fn_description = """
Use when: You want to simulate a contrastive scenario with manually chosen action.
"""

contrastive_behavior_fn_description = """
Use when: You want to simulate a contrastive scenario with different control behaviors
"""

contrastive_policy_fn_description = """
Use when: You want to what would the trajectory would be if we chose alternative policy,
        or to compare the optimal policy with other policies.
"""

q_decompose_fn_description = """
Use when: You want to know the agent's intention behind certain action, by decomposing q values into both semantic and temporal dimension.
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
    
    Explain the results within a single paragraph.
    """

    if agent_name == 'coordinator':
        return coordinator_prompt
    elif agent_name == 'explainer':
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
    elif fn_name == "contrastive_action":
        return contrastive_action_fn_description
    elif fn_name == "contrastive_behavior":
        return contrastive_behavior_fn_description
    elif fn_name == "contrastive_policy":
        return contrastive_policy_fn_description
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
                    "actions": {
                        "type": "array",
                        "description": "List of action names (variables) to be explained. List all actions if not specified.",
                        "items": {
                            "type": "string"
                        },
                        "example": ["v1", "v2"]
                    },
                },
                "required": ["agent", "data", "actions"]
            }
        },
        {
            "type": "function",
            "name": "feature_importance_local",
            "description": feature_importance_local_fn_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "description": "List of action names (variables) to be explained. List all actions if not specified.",
                        "items": {
                            "type": "string"
                        },
                        "example": ["v1", "v2"]
                    },
                    "t_query": {
                        "type": "number",
                        "description": "Time points to query for feature importance."
                    },
                },
                "required": ["agent", "data", "actions", "t_query"]
            }
        },
        {
            "type": "function",
            "name": "contrastive_action",
            "description": contrastive_action_fn_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "t_begin": {
                        "type": "number",
                        "description": "Start timestep of the contrastive intervention."
                    },
                    "t_end": {
                        "type": "number",
                        "description": "End timestep of the contrastive intervention."
                    },
                    "actions": {
                        "type": "array",
                        "description": "List of action names (variables) to which contrastive values should be applied.",
                        "items": {
                            "type": "string"
                        },
                        "example": ["v1"]
                    },
                    "values": {
                        "type": "array",
                        "description": "List of contrastive values corresponding to each action in 'actions'."
                                       "Must return an array of numbers, not the number itself (e.g., [3.2], NOT 3.2).",
                        "items": {
                            "type": "number"
                        },
                        "example": [0.5],
                    }
                },
                "required": ["t_begin", "t_end", "actions", "values"]
            }
        },
        {
            "type": "function",
            "name": "contrastive_behavior",
            "description": contrastive_behavior_fn_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "t_begin": {
                        "type": "number",
                        "description": "Start timestep of the contrastive intervention."
                    },
                    "t_end": {
                        "type": "number",
                        "description": "End timestep of the contrastive intervention."
                    },
                    "actions": {
                        "type": "array",
                        "description": "List of action names (variables) to which contrastive behavior should be applied.",
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
            "name": "contrastive_policy",
            "description": contrastive_policy_fn_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "t_begin": {
                        "type": "number",
                        "description": "Start timestep of the contrastive intervention."
                    },
                    "t_end": {
                        "type": "number",
                        "description": "End timestep of the contrastive intervention."
                    },
                    "message": {
                        "type": "string",
                        "description": "Brief instruction for constructing the contrastive policy. It is used as prompts for the Coder agent."
                                       "Currently, only rule-based control are used for the alternative policy."
                                       "Include the policy rules only, without any timestep-related information."
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
                        "description": "Time points to query for contrastive analysis"
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

    # Use this figure description when the expected decomposed rewards are also compared between actual and contrastive policies
    # contrastive_figure_description = f"""
    # You will get two plots as results, and your job is to explain why a certain action trajectory is better in control than the other:
    #     - The first plot compares future trajectory from original controller and with one from the contrastive control behavior.
    #         - From this plot, you will have to explain how the environment(e.g.) states, rewards) would change, in terms of both instant and long-term perspective.
    #
    #     - The second plot compares the future decomposed reward of executing actual and contrastive action trajectory for short-time period.
    #         - From this plot, you should focus on explaining which reward components indicate that one control behavior outperforms the others.
    #
    #     Here are some points that you might have to consider when generating explanations
    #     - It would be really great if you select a specific time interval that was critical for deciding the control aptitude of two trajectories.
    #     - Also, you might compare the two trajectories in terms of settling time or overshooting behavior, and concluding the overall performance of two control trajectories.
    #     - If contrastive trajectory failed to control the system, it would be better to analyze the potential cause of the failure.
    #     - Lastly, make a summary of whether the contrastive scenario exceled at controlling the system and why.
    #
    # Interpret the graph of region after 't_begin' only, not before 't_begin'.
    # Focus on comparing the actual trajectory with contrastive trajectory.
    # """

    contrastive_figure_description = f"""
    You will get one plot as results, and your job is to explain why a certain action trajectory is better in control than the other:
        - The first plot compares future trajectory from original controller and with one from the contrastive control behavior.
            - From this plot, you will have to explain how the environment(e.g.) states, rewards) would change, in terms of both instant and long-term perspective.

        Here are some points that you might have to consider when generating explanations
        - It would be really great if you select a specific time interval that was critical for deciding the control aptitude of two trajectories.
        - Also, you might compare the two trajectories in terms of settling time or overshooting behavior, and concluding the overall performance of two control trajectories.
        - If contrastive trajectory failed to control the system, it would be better to analyze the potential cause of the failure.
        - Lastly, make a summary of whether the contrastive scenario excelled at controlling the system and why.

    Interpret the graph of region after 't_begin' only, not before 't_begin'.
    Focus on comparing the actual trajectory with contrastive trajectory.
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
    elif fn_name == "contrastive_action":
        return contrastive_figure_description
    elif fn_name == "contrastive_behavior":
        return contrastive_figure_description
    elif fn_name == "contrastive_policy":
        return contrastive_figure_description
    elif fn_name == "q_decompose":
        return q_decompose_figure_description
