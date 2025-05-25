# %% System descriptions
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
Therefore, the observation when there exists a setpoint for both states is [C_A, T, Errors_C_A].
The observation space and initial conditions are defined as (env_params).

### Action
The action corresponds to a jacket temperature, which is bounded by action_space in (env_params).

### Reward
The reward is a continuous value corresponding to the square error of the state and its setpoint.
For multiple states, these are scaled with a factor (r_scale) and summed to give a single value.
"""


# %% Function descriptions
train_agent_fn_description = """
Use when: You want to train or load a reinforcement learning agent on the specified environment.
Example: "Train a DDPG agent for the CSTR environment."
Example: "Load a pretrained PPO model and skip training."
"""

get_rollout_data_fn_description = """
Use when: You want to simulate and extract state-action-reward data after training.
Example: "Evaluate the agent's policy through rollouts."
Example: "Get the Q-values and state trajectories from the rollout."
"""

cluster_states_fn_description = """
Use when: You want to reduce the dimensionality of state space and perform unsupervised clustering.
Example: "Cluster the agent's behavior using HDBSCAN on the state-action space."
Example: "Visualize states using t-SNE and group into behavioral clusters."
"""

feature_importance_global_fn_description = """
Use when: You want to understand which features most influence the agent’s policy across all states.
Example: "Explain the global feature importance using SHAP."
Example: "Visualize LIME-based feature importance for the trained agent."
"""

feature_importance_local_fn_description = """
Use when: You want to inspect how features affected the agent's decision at a specific point.
Example: "Provide local SHAP values for a single instance."
Example: "What influenced the agent most at timestep 0?"
"""

partial_dependence_plot_fn_description = """
Use when: You want to examine how changing one input feature influences the agent's action.
Example: "Plot ICE and PDP curves to understand sensitivity to temperature."
Example: "How does action vary with concentration change?"
"""

trajectory_sensitivity_fn_description = """
Use when: You want to simulate how small action perturbations influence future trajectory.
Example: "Evaluate sensitivity of state trajectory to action perturbations at t=180."
Example: "How robust is the policy to action noise?"
"""

trajectory_counterfactual_fn_description = """
Use when: You want to simulate a counterfactual scenario with manually chosen action.
Example: "What would have happened if we had chosen action = 300 at t=180?"
Example: "Show the trajectory if a different control input is applied."
"""



def get_prompts(prompt):
    need2describe_prompt = """
Here are the name of {component_type} that need to be described
-----
{component_names}
-----


"""

    model_interpretation_json = {
        "components": {
            "sets": [
                {
                    "name": "The name of the component in sets",
                    "description": "The description of the component",
                }
            ],
            "parameters": [
                {
                    "name": "The name of the component in parameters",
                    "description": "The description of the component",
                }
            ],
            "variables": [
                {
                    "name": "The name of the component in variables",
                    "description": "The description of the component",
                }
            ],
            "constraints": [
                {
                    "name": "The name of the component in constraints",
                    "description": "The description of the component",
                }
            ],
            "objective": [
                {
                    "name": "The name of the component in objective",
                    "description": "The description of the component",
                }
            ]
        }
    }

    model_interpretation_prompt = """
You are an operations research expert and your role is to use PLAIN ENGLISH to interpret an optimization model written in Pyomo. 
The Pyomo code is given below:

-----
{code}
-----


{cat_need2describe_prompt}
Your task is carefully inspect the code and write a description for each of the components. 

Then, generate a json file accordingly with the following format (STICK TO THIS FORMAT!)

{model_interpretation_json}

- description should be either physical meanings, intended use, or any other relevant information about the component.
- Note that I'm going to use python json.loads() function to parse the json file, so please make sure the format is correct (don't add ',' before enclosing '}}' or ']' characters.
- Generate the complete json file and don't omit anything.
- Use 'name' and 'description' as the keys, and provide the name and description of the component as the values.
- Use '```json' and '```' to enclose the json file.

Take a deep breath and solve the problem step by step.
"""

    system_description_prompt = """
You are a chemical process operator and your role is to briefly explain the system that are simulated and controlled based on reinforcement learning.
The environment parameters are given below:
introduce an optimization model to non-experts, based on an abstract representation of the model in json format.
The json representation is given below:

-----
{env_params}
{json_representation}
-----

- Start with a brief introduction of the model, what the problem is about, who is using the model, and what the model is trying to achieve.
- Explain what decisions (variables) are to be made
- Explain what data or information (parameters) is already known
- Explain what constraints are imposed on the decisions
- Explain what the objective is, what is being optimized

The explanation must be coherent and easy to understand for the users who are experts in the filed for which this model is built but not in optimization.
"""

    model_inference_prompt = """
You are an operations research expert and your role is to infer why an optimization model is infeasible, based on an abstract representation of the infeasible model in json format.
Particularly, your team has identified the Irreducible Infeasible Subset (IIS) of the model, which is given below:

-----
{iis_info}
-----


To understand what the parameters and the constraints mean, the json representation is given below for your reference:

-----
{json_representation}
-----


- Introduce to the user what constraints are potentially causing the infeasibility, and what parameters are involved in these constraints.
- Explain the relationship between the constraints and the parameters, and infer why the constraints are conflicting with each other.
- Provide inference by analyzing their physical meanings, and AVOID using jargon and symbols as much as possible but the explanation style must be formal. 
- Recommend some parameters that you believe can be adjusted to make the model feasible.
- Parameters recommended for adjustment MUST be changeable physically in practice. For example, molecular weight of a molecule is not changeable in practice.
- Assess the practical implications of the recommendations. For example, increasing the number of workers implies hiring more workers, which incurs additional costs.
"""

    syntax_reminder_prompt = """
You're an operator working on a pyomo model.
Your task is to identify the following arguments: 
- the component names that the user is interested in,
- the most appropriate function that can answer the user's query, 
- the model that the user is querying.
then call the predefined syntax_guidance function to generate syntax guidance.

----- Instruction to select the most appropriate function -----
you MUST select a function from ```{function_names}```, DO NOT make up your own function.
1. feasibility_restoration:
Use when: The model is infeasible and you need to find out the minimal change to specific [component name] for restoring feasibility.
Example: “How much should we adjust the [component name] to make the model feasible”
Example: "I believe changing [component name] is practical, by how much do I need to change in order to make the model feasible"
[component name] category: parameters. If only constraint name is provided in the query, you need to infer the parameters involved in the constraint.

2. components_retrival:
Use when: You need to know the current values or expressions of [component name] within the model.
Example: “What are the values of the [component name]”
Example: "How many [component name] are currently available"
[component name] category: sets, parameters, variables, constraints, or objectives.

3. sensitivity_analysis:
Use when: The model is feasible and you want to understand the impact of changing [component name] on the optimal objective value, **without specifying the extent of changes**.
Example: “How will the optimal profit change with the change in the [component name]”
Example: "How stable is the objective value in response to variations in the [component name]"
Example: "Will the optimal value be greatly affected if we have more [component name]"
[component name] category: parameters.

4. evaluate_modification:
Use when: The model is feasible and you want to understand the impact of changing [component name] on the optimal objective value, **by specifying the extent of changes**.
Example: “How will the optimal profit change with **a 10% increase** in the [component name]”
Example: "How stable is the objective value in response to the modification that [component name] is **decreased by 20 units**"
Example: "Will the optimal value be greatly affected if we have *two* more [component name]"
[component name] category: parameters or variables.

5. external_tools:
Use when: User doubts the model's optimal solution and provides a counterexample, and you want to add new constraints to implement the counterexample.
Example: “Why is it not recommended to have [component name] lower than 400 in the optimal solution”
Example: "Why isn’t [component name] and [component name] both used in the optimal scenario"
[component name] category: parameters or variables.

----- Instruction to determine the correct component name -----
The [component name] MUST be in a symbolic form, instead of its description.
Use the following dictionary to find the correct [component name] based on its description:
{component_name_meaning_pairs}

----- Instruction to find the queried model -----
In the form of 'model_integer', e.g. 'model_1'
"""

    operator_prompt = """
You're an optimization expert who helps your team to access and interact with optimization models by internal tools.

Your task is to invoke the most appropriate tool correctly based on the user's query and system reminders.
"""


    programmer_prompt = """
You're an optimization expert who helps your team to write pyomo code to answer users questions.
(1) write code snippet to revise the model, only when the user doubts the model's optimal solution and provides a counterexample
(2) write code snippet to print out the information useful for answering the user's question

Output Format:
==========
```python
CODE SNIPPET FOR REVISING THE MODEL
```

```python
CODE SNIPPET FOR PRINTING OUT USEFUL INFORMATION
```
==========

Here are some example questions and their answer codes:
----- EXAMPLE 1 -----
Question: Why is it not recommended to use just one supplier for roastery 2?

Answer Code:
```python
# user is actually interested in the case that only one supplier can supply roastery 2 and does not believe the optimal solution.
model.force_one_supplier = ConstraintList()
model.force_one_supplier.add(sum(model.z[s,'roastery2'] for s in model.suppliers) <= 1)
for s in model.suppliers:
    model.force_one_supplier.add(model.x[s,'roastery2'] <= model.capacity_in_supplier[s] * model.z[s, 'roastery2'])
```

```python
# I print out the new optimal objective value so that you can tell the user how the objective value changes if only one supplier supplies roastery 2.
print('If forcing only one supplier to supply roastery 2, the optimal objective value will become: ', model.obj())
```

----- EXAMPLE 2 -----
Question: Why is it not recommended to have production cost larger than transportation cost in the optimal setting?

Answer Code:
```python
# user does not believe the optimal solution obtained when production cost smaller than transportation cost.
# so we force production cost to be less than transportation cost to see what will happen.
model.counter_example = ConstraintList()
model.counter_example.add(model.production <= model.transportation)
```

```python
# I print out the new optimal objective value so that you can tell the user how the objective value changes.
print('If forcing production cost be smaller than transportation cost, the optimal objective value will become: ', model.obj())
```

- Code reminder has provided you with the source code of the pyomo model
- Your written code will be added to the lines with substring: "# OPTICHAT *** CODE GOES HERE"
So, you don't need to repeat the source code that has already been provided by Code reminder.
- The code for re-solving the model has already been given, 
So you don't need to add it. Solving the model repeatedly can lead to errors.
- Your written code should be accompanied by comments to explain the purpose of the code.
- Evaluator will execute the new code for you and read the execution result.
So, you MUST print out the model information that you believe is necessary for the user's question.
"""

    evaluator_prompt = """
You're an optimization expert who helps your team to review pyomo code,
based on the execution result of the code provided by the programmer.

Is the code bug-free and valid to answer the user's query?
Generate the following json file if you accept the code, and provide your own comment.
{{ "decision": "accept", "comment": "your own comment" }}
Generate the following json file if you reject the code, and provide your own comment.
{{ "decision": "reject", "comment": "your own comment" }}

- Only generate the json file, and don't generate any other text.
- Use 'decision' and 'comment' as the keys, 
- choose 'accept' or 'reject' for the decision, and provide your own comment. 
- Note that infeasibility caused by the new constraints may be acceptable. 
This is because programmers are trying to create a counterfactual example that the user is interested in, and this counterfactual example may be infeasible in nature.
"""

    test_prompt = """
You are a judge who determines if the LLM’s answer passes the test.
Criteria: The data in the execution result should be consistent with the human expert's answer.
Details: LLM may omit some data that human experts collected from other sources, but if it covers the correct objective value correctly, it should pass.

Human Expert Answer: 
{human_expert_answer}

- Return either "Pass" or "Fail."
- No additional comments or explanations.
"""

    # %% TODO: My prompts (initial)
    system_description_prompt = """
    You are a chemical process operator and your role is to briefly explain the system that are simulated and controlled based on reinforcement learning.
    The environment parameters are given below:

    -----
    {env_params}
    -----
    
    Furthermore, the brief explanation of control system is given below:
    
    -----
    {system_description}
    -----

    - Start with a brief description of the system, including what the observation and action variables are.
    - Explain how the action variable can affect the observation variables
    - Clarify what the controller is trying to achieve.
    - Explain what constraints are imposed on the system, if available.
    """

    # TODO: Explainer prompt. 각 function call에 대해서 expected outputs 기반으로 하여 어떻게 explain할건지?
    explainer_prompt = """
    You're an expert in both explainable reinforcement learning (XRL).
    Your role is to explain the XRL results triggered by XRL functions in natural language form.
    
    Below is the XRL function triggered, and it's description.
    -----
    {fn_name}
    -----
    -----
    {fn_description}
    -----
    
    Below is the XRL explanation results.
    -----
    {explanation}
    -----
    
    - Depending on the function triggered, the visualization plots can be raised. If visualization results, exist, make sure to briefly explain how to interpret the visualization results.
    - The users are not experts in XRL, but they are experts in the filed for which this model is built.
    - Provide a detailed explanation only when you believe the users need more context about optimization to understand your explanation.
    - Otherwise, the explanation must be succinct and concise, because users may be distracted by too much information.
    
    Make sure the explanation must be coherent and easy to understand for the users who are experts in chemical process,
    but not quite informed at explainable artificial intelligence tools and their interpretations.  
    """

    coordinator_prompt = """
    You're a coordinator in a team of optimization experts. The goal of the team is to help non-experts analyze an 
    optimization problem. Your task is to choose the next expert to work on the problem based on the current situation. 

    Here's the list of agents in your team:
    -----
    {agents}
    -----

    Considering the conversation, generate a json file with the following format: 
    {{ "agent_name": "Name of the agent you want to call next", "task": "The task you want the agent to carry out" }} 

    to identify the next agent to work on the problem, and also the task it has to carry out. 
    - Only generate the json file, and don't generate any other text.
    - DO NOT change the keys of the json file, only change the values. Keys are "agent_name" and "task".
    - if you think the problem is solved, generate the json file below:
    {{ "agent_name": "Explainer", "task": "DONE" }} 
    """

    if prompt == 'model_interpretation_prompt':
        return model_interpretation_prompt
    elif prompt == 'need2describe_prompt':
        return need2describe_prompt
    elif prompt == 'model_interpretation_json':
        return model_interpretation_json
    elif prompt == 'model_inference_prompt':
        return model_inference_prompt
    elif prompt == 'syntax_reminder_prompt':
        return syntax_reminder_prompt
    elif prompt == 'operator_prompt':
        return operator_prompt
    elif prompt == 'programmer_prompt':
        return programmer_prompt
    elif prompt == 'evaluator_prompt':
        return evaluator_prompt
    elif prompt == 'test_prompt':
        return test_prompt

    elif prompt == 'coordinator_prompt':
        return coordinator_prompt
    elif prompt == 'explainer_prompt':
        return explainer_prompt
    elif prompt == 'system_description_prompt':
        return system_description_prompt

def get_fn_json():
    fn_json = [
        {
            "type": "function",
                "name": "feature_importance_global",
                "description": feature_importance_global_fn_description,
                "parameters": {
                    "type": "object",
                    "properties": {
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
                "name": "partial_dependence_plot",
                "description": partial_dependence_plot_fn_description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": ["agent", "data"]
            }
        },
        {
            "type": "function",
                "name": "trajectory_sensitivity",
                "description": trajectory_sensitivity_fn_description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": ["agent", "data"]
            }
        },
        {
            "type": "function",
                "name": "trajectory_counterfactual",
                "description": trajectory_counterfactual_fn_description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": ["agent", "data"]
            }
        }
    ]

    # fn_json = [
    #     {
    #         "type": "function",
    #         "function":{
    #             "name": "feature_importance_global",
    #             "description": feature_importance_global_fn_description,
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "lime": {
    #                         "type": "boolean",
    #                         "description": "Whether to include LIME explanation"
    #                     },
    #                     "shap": {
    #                         "type": "boolean",
    #                         "description": "Whether to include SHAP explanation"
    #                     }
    #                 },
    #                 "required": ["agent", "data"]
    #             }
    #         }
    #     },
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "cluster_states",
    #             "description": cluster_states_fn_description,
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {},
    #                 "required": ["agent", "data"]
    #             }
    #         }
    #     },
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "partial_dependence_plot",
    #             "description": partial_dependence_plot_fn_description,
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {},
    #                 "required": ["agent", "data"]
    #             }
    #         }
    #     },
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "trajectory_sensitivity",
    #             "description": trajectory_sensitivity_fn_description,
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {},
    #                 "required": ["agent", "data"]
    #             }
    #         }
    #     },
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "trajectory_counterfactual",
    #             "description": trajectory_counterfactual_fn_description,
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {},
    #                 "required": ["agent", "data"]
    #             }
    #         }
    #     }
    # ]
    return fn_json



def get_tools(fn_names):
    multiple_tools = []
    single_tools = []
    none_tools = []
    all_tools = []
    for fn_name in fn_names:
        if fn_name != 'external_tools':
            multiple_tools.append(get_fn_json(fn_name, 'multiple'))
            single_tools.append(get_fn_json(fn_name, 'single'))
            none_tools.append(get_fn_json(fn_name, 'none'))
            all_tools.append(get_fn_json(fn_name, 'all'))
    return multiple_tools, single_tools, none_tools, all_tools, 'auto'


def get_syntax_guidance_tool():
    syntax_guidance_fn_json = get_syntax_guidance_fn_json()
    syntax_guidance_tool = [syntax_guidance_fn_json]
    return syntax_guidance_tool

