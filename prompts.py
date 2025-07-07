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
train_agent_fn_description = """
Use when: You want to train or load a reinforcement learning agent on the specified environment.
Example:
    1) "Train a DDPG agent for the CSTR environment."
    2) "Load a pretrained PPO model and skip training."
"""

get_rollout_data_fn_description = """
Use when: You want to simulate and extract state-action-reward data after training.
Example:
    1) "Evaluate the agent's policy through rollouts."
    2) "Get the Q-values and state trajectories from the rollout."
"""

cluster_states_fn_description = """
Use when: You want to perform unsupervised clustering of states and classify each cluster's characteristic.
Example:
    1) "Cluster the agent's behavior using HDBSCAN on the state-action space."
    2) "Visualize states using t-SNE and group into behavioral clusters."
"""

feature_importance_global_fn_description = """
Use when: You want to understand which features most influence the agent’s policy across all states.
Example:
    1) "How do the process states globally influence the agent's decisions?"
    2) "Which feature makes great contribution to the agent's decisions generally?"
"""

feature_importance_local_fn_description = """
Use when: You want to inspect how features affected the agent's decision at a specific point.
Example:
    1) "Provide local SHAP values for a single instance."
    2) "What influenced the agent most at timestep 120?"
"""

partial_dependence_plot_global_fn_description = """
Use when: You want to examine how changing one input feature influences the agent's action.
Example:
    1) "Plot ICE and PDP curves to understand sensitivity to temperature."
    2) "How does action vary with concentration change generally?"
    3) "How would the action variables change if the state variables vary?"
"""

partial_dependence_plot_local_fn_description = """
Use when: You want to examine how changing one input feature AT SPECIFIC TIME POINT influences the agent's action.
Example:
    1) "Plot ICE curves to understand sensitivity to temperature at timestep 180."
    2) "How does action can vary with concentration change now?"
"""

trajectory_sensitivity_fn_description = """
Use when: You want to simulate how small action perturbations influence future trajectory.
Example:
    1) "Evaluate sensitivity of state trajectory to action perturbations at t=180."
    2) "How robust is the policy to action noise?"
"""

trajectory_counterfactual_fn_description = """
Use when: You want to simulate a counterfactual scenario with manually chosen action.
Example:
    1) "What would have happened if we had chosen action = 300 from t=200 to t=400?"
    2) "Show the trajectory if a different control input is applied."
"""

q_decompose_fn_description = """
Use when: You want to know the agent's intention behind certain action, by decomposing q values into both semantic and temporal dimension.
Example:
    1) "What is the agent trying to achieve in the long run by doing this action at timestep 180?"
    2) "Why is the agent's intention behind the action at timestep 200?"
"""

policy_counterfactual_fn_description = """
    Use when: You want to what would the trajectory would be if we chose alternative policy,
            or to compare the optimal policy with other policies.
    Example:
        1) "What would the trajectory change if I use the bang-bang controller instead of the current RL policy?"
        2) "Why don't we just use the PID controller instead of the RL policy?"
        3) "Would you compare the predicted trajectory between our RL policy and bang-bang controller after t-300?"
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
        
    - Also, for more clear explanation, the description of the system and its enviroinment parameters are given as below:
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
    """

    reward_decomposer_prompt = """
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

    if prompt == 'coordinator_prompt':
        return coordinator_prompt
    elif prompt == 'explainer_prompt':
        return explainer_prompt
    elif prompt == 'system_description_prompt':
        return system_description_prompt
    elif prompt == 'reward_decomposer_prompt':
        return reward_decomposer_prompt

def get_fn_description(fn_name):
    if fn_name == "train_agent":
        return train_agent_fn_description
    elif fn_name == "get_rollout_data":
        return get_rollout_data_fn_description
    elif fn_name == "cluster_states":
        return cluster_states_fn_description
    elif fn_name == "feature_importance_global":
        return feature_importance_global_fn_description
    elif fn_name == "feature_importance_local":
        return feature_importance_local_fn_description
    elif fn_name == "partial_dependence_plot_global":
        return partial_dependence_plot_global_fn_description
    elif fn_name == "partial_dependence_plot_local":
        return partial_dependence_plot_local_fn_description
    elif fn_name == "trajectory_sensitivity":
        return trajectory_sensitivity_fn_description
    elif fn_name == "trajectory_counterfactual":
        return trajectory_counterfactual_fn_description
    elif fn_name == "q_decompose":
        return q_decompose_fn_description
    elif fn_name == "policy_counterfactual":
        return policy_counterfactual_fn_description


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
                        "lime": {
                            "type": "boolean",
                            "description": "Whether to include LIME explanation"
                        },
                        "shap": {
                            "type": "boolean",
                            "description": "Whether to include SHAP explanation"
                        }
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
                "name": "cluster_states",
                "description": cluster_states_fn_description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": ["agent", "data"]
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
            "name": "trajectory_sensitivity",
            "description": trajectory_sensitivity_fn_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "t_query": {
                        "type": "number",
                        "description": "Time points to query for sensitivity analysis"
                    },
                    "action": {
                        "type": "string",
                        "description": "Name of the agent action to be explained"
                    },
                },
                "required": ["agent", "data", "t_query"]
            }
        },
        {
            "type": "function",
            "name": "trajectory_counterfactual",
            "description": trajectory_counterfactual_fn_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "t_begin": {
                        "type": "number",
                        "description": "First time step within the simulation interval to be interpreted"
                    },
                    "t_end": {
                        "type": "number",
                        "description": "Last time step within the simulation interval to be interpreted"
                    },
                    "cf_actions": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                        "description": "List of queried actions to be executed for comparison"
                    },
                    "action": {
                        "type": "string",
                        "description": "Name of the agent action to be explained"
                    },
                },
                "required": ["agent", "data", "t_begin", "t_end", "cf_actions"]
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
            "name": "policy_counterfactual",
            "description": policy_counterfactual_fn_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Brief instruction for constructing the counterfactual policy. It is used as prompts for the Coder agent."
                    },
                    "t_query": {
                        "type": "number",
                        "description": "Time points to query for counterfactual analysis"
                    },
                },
                "required": ["agent", "data", "team_conversation", "message"]
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
    feature_importance_global_figure_description = """ fn_name is feature_importance_global.
    If SHAP are being used, you will get sets of three plots as results:
        First plot is the bar plot, which compares feature importance values of features.
        Second plot is the beeswarm plot, which shows both the magnitude and direction of feature attribution to action variables.
        Third plot is the decision plot, which shows how each data deviates from the reference value, affected by each feature variable.

    If LIME are being used instead, you will get one plot as results:
        The bar plot compares feature importance values of features."""

    feature_importance_local_figure_description = """ fn_name is feature_importance_local.
    You will get one plot as results:
        The waterfall plot compares how each state variable affect the action values.
        Make sure to discuss both the magnitude and direction of contributions of each feature"""

    partial_dependence_plot_global_figure_description = """ fn_name is partial_dependence_plot_global.
    You will get one plot as results:
        The PDP plot displays how the action value will change as each state variables differ.
        In global PDP plot, results of all the states are displayed and mean tendency is visualized by separate color."""

    partial_dependence_plot_local_figure_description = """ fn_name is partial_dependence_plot_local.
    You will get one plot as results:
        The ICE plot displays how the action value will change as each state variables differ.
        In local ICE plot, results of a singel state at queried timestep are displayed."""

    cluster_states_figure_description = """ fn_name is cluster_states.
    You will get three plots as results:
        First plot is the scatter plot, each colored by various hue variables.
        From this figure, your job is to extract the correlation among scatter plots. (For example, data points with large error values tend to have relatively deficient q values...)
        Second plot is the scatter plot, where data points are colored by different clusters.
        Third plot is the violin plot, where each distribution of state variable in various clusters is visualized in violin plot.
    """

    trajectory_sensitivity_figure_description = """fn_name is trajectory_sensitivity.
    You will get one plot as results:
        The plot shows future trajectory when executed an action with various sensitivity values.
        You will have to explain how the environment would change as a result of certain actions.
        It would be better if you can explain why the action yielded by the actor was the best, instead of other actions.
    """

    trajectory_counterfactual_figure_description = """fn_name is trajectory_counterfactual.
    You will get one plot as results:
        The plot shows future trajectory when executed an action with various counterfactual action values.
        You will have to explain how the environment would change, in terms of both short and long perspective.
        It would be better if you can explain why the action yielded by the actor was the best, instead of other actions.
    """

    q_decompose_figure_description = """fn_name is q_decompose.
    You will get one plot as results:
        The plot shows Q-values decomposed in both temporal and semantic dimension.
        You will have to explain what the agent has achieved by executing the action at the queried time step.
        Make sure that the rewards are being visualized in negative fashion, so bigger portion of bar means more negative reward.
    """

    policy_counterfactual_figure_description = """fn_name is policy_counterfactual.
    You will get one plot as results:
        The plot compares potential rollout between our RL policy and the counterfactual policy made by coder agent.
        YOu will have to explain how does the two policies differ in acting and which one is better in controlling the system.
        If CF policy failed to control the system, it would be better to analyze the potential cause of the failure, based on the CF policy itself and system descriptions.
    """

    if fn_name == "cluster_states":
        return cluster_states_figure_description
    elif fn_name == "feature_importance_global":
        return feature_importance_global_figure_description
    elif fn_name == "feature_importance_local":
        return feature_importance_local_figure_description
    elif fn_name == "partial_dependence_plot_global":
        return partial_dependence_plot_global_figure_description
    elif fn_name == "partial_dependence_plot_local":
        return partial_dependence_plot_local_figure_description
    elif fn_name == "trajectory_sensitivity":
        return trajectory_sensitivity_figure_description
    elif fn_name == "trajectory_counterfactual":
        return trajectory_counterfactual_figure_description
    elif fn_name == "q_decompose":
        return q_decompose_figure_description
    elif fn_name == "policy_counterfactual":
        return policy_counterfactual_figure_description
