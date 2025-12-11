def get_queries(system):
    if system == 'cstr':
        FI_queries = {
            "Which state variables are most critical in the agent's decision of Tc at timestep 12?":
                {'t_query': 12, 'actions': ['Tc']},
            "What aspects of the state influence the control input (Tc) the most at t = 35?":
                {'t_query': 35, 'actions': ['Tc']},
            "At timestep 67, which part of the state drives the jacket temperature control decision?":
                {'t_query': 67, 'actions': ['Tc']},
            "Which components of the CSTR state guide the policy's behavior at time 83?":
                {'t_query': 83, 'actions': ['Tc']},
            "What is the most decisive state feature at timestep 50 in determining the Tc setting?":
                {'t_query': 50, 'actions': ['Tc']},
            "Can you tell which state variables the agent relies on at timestep 9 to choose the control action?":
                {'t_query': 9, 'actions': ['Tc']},
            "At time 27, which state values are most responsible for the agent’s behavior in controlling Tc?":
                {'t_query': 27, 'actions': ['Tc']},
            "Which parts of the CSTR state representation are key to the agent's decision at timestep 99?":
                {'t_query': 99, 'actions': ['Tc']},
            "At timestep 42, what elements in the state space affect the jacket temperature decision the most?":
                {'t_query': 42, 'actions': ['Tc']},
            "At time 70, which state inputs does the policy consider most important when setting Tc?":
                {'t_query': 70, 'actions': ['Tc']},
            "At timestep 18, which variables have the largest impact on the decision of Tc?":
                {'t_query': 18, 'actions': ['Tc']},
            "How much does each CSTR state variable contribute to the action (Tc) at t = 61?":
                {'t_query': 61, 'actions': ['Tc']},
            "Which observations matter the most to the agent at timestep 24 when choosing Tc?":
                {'t_query': 24, 'actions': ['Tc']},
            "Which parts of the state representation at time 89 drive the jacket temperature control logic?":
                {'t_query': 89, 'actions': ['Tc']},
            "What does the agent focus on most from the state when deciding an action at timestep 5?":
                {'t_query': 5, 'actions': ['Tc']},
            "At timestep 77, what state variables are the main drivers of the agent's Tc decision?":
                {'t_query': 77, 'actions': ['Tc']},
            "What role does each CSTR state variable play at t = 33 in determining the controller output?":
                {'t_query': 33, 'actions': ['Tc']},
            "At time 93, which dimensions of the state most influence the control policy's jacket temperature setting?":
                {'t_query': 93, 'actions': ['Tc']},
            "Which state features are prioritized at timestep 48 during control decision-making?":
                {'t_query': 48, 'actions': ['Tc']},
            "At timestep 100, what are the dominant state inputs affecting the Tc control action?":
                {'t_query': 100, 'actions': ['Tc']}
        }

        EO_queries = {
            "At timestep 15, what long-term reward is the agent aiming for by selecting this action?":
                {'t_query': 15},
            "At time 27, what future outcomes does the agent hope to achieve with its current decision?":
                {'t_query': 27},
            "How does the chosen action at timestep 43 align with the agent's long-term goals?":
                {'t_query': 43},
            "In the long run, what benefits is the agent expecting from its move at time 56?":
                {'t_query': 56},
            "Which future rewards is the agent targeting by acting at timestep 81?":
                {'t_query': 81},
            "At timestep 22, what is the ultimate objective the agent is trying to realize with this decision?":
                {'t_query': 22},
            "How does the action taken at time 64 contribute to the agent’s expected future performance?":
                {'t_query': 64},
            "At time 90, what potential future gains is the agent seeking with its action?":
                {'t_query': 90},
            "What long-term outcomes drive the agent’s decision at timestep 36?":
                {'t_query': 36},
            "At t = 12, what does the agent hope to eventually accomplish through its action?":
                {'t_query': 12},
            "How does the agent’s current action at time 77 relate to its future rewards?":
                {'t_query': 77},
            "At timestep 40, what are the expected long-term benefits of this decision?":
                {'t_query': 40},
            "What eventual achievements is the agent striving for by choosing its action at t = 8?":
                {'t_query': 8},
            "At timestep 98, what future rewards is the agent counting on?":
                {'t_query': 98},
            "What ultimate objectives does the agent have in mind with its action at time 61?":
                {'t_query': 61},
            "At timestep 24, which long-range rewards influence the agent’s decision?":
                {'t_query': 24},
            "What long-term benefits is the agent aiming to secure by acting at time 33?":
                {'t_query': 33},
            "At t = 48, how does the action contribute to the agent’s future goals?":
                {'t_query': 48},
            "At timestep 84, what eventual outcomes does the agent plan to achieve with its move?":
                {'t_query': 84},
            "At time 100, what future rewards are driving the agent’s current decision?":
                {'t_query': 100}
        }

        CE_A_queries = {
            "What would be the outcome if the agent had set Tc to 298.0 from timestep 0 to 12 instead of the chosen values?":
                {'t_begin': 0, 't_end': 12, 'actions': ['Tc'], 'values': [298.0]},
            "How would the result differ if Tc had been fixed at 302.0 between timestep 12 and 24?":
                {'t_begin': 12, 't_end': 24, 'actions': ['Tc'], 'values': [302.0]},
            "Suppose the agent had increased Tc to 300.0 from timestep 24 to 36—what impact would that have had?":
                {'t_begin': 24, 't_end': 36, 'actions': ['Tc'], 'values': [300.0]},
            "What if we raised Tc to 296.5 between t = 36 and 48 instead of the RL-predicted action?":
                {'t_begin': 36, 't_end': 48, 'actions': ['Tc'], 'values': [296.5]},
            "How would the reward trajectory change if we manually set Tc = 297.0 from timestep 48 to 60?":
                {'t_begin': 48, 't_end': 60, 'actions': ['Tc'], 'values': [297.0]},
            "Could outcomes be improved if Tc was overridden to 299.0 from t = 60 to 72?":
                {'t_begin': 60, 't_end': 72, 'actions': ['Tc'], 'values': [299.0]},
            "What if Tc had been held constant at 295.0 instead of following the policy between 72 and 84?":
                {'t_begin': 72, 't_end': 84, 'actions': ['Tc'], 'values': [295.0]},
            "Would the return change if we replaced the action with Tc = 301.0 during timestep 84 to 96?":
                {'t_begin': 84, 't_end': 96, 'actions': ['Tc'], 'values': [301.0]},
            "How would the future states evolve if Tc was clamped to 300.5 from timestep 6 to 18?":
                {'t_begin': 6, 't_end': 18, 'actions': ['Tc'], 'values': [300.5]},
            "What difference would it make if we set Tc to 297.2 between t = 18 and 30?":
                {'t_begin': 18, 't_end': 30, 'actions': ['Tc'], 'values': [297.2]},
            "Suppose we fixed Tc to 299.9 during the early stage from timestep 30 to 42—would dynamics change?":
                {'t_begin': 30, 't_end': 42, 'actions': ['Tc'], 'values': [299.9]},
            "Could stability improve if Tc was set to 296.0 from t = 42 to 54 instead of varying policy output?":
                {'t_begin': 42, 't_end': 54, 'actions': ['Tc'], 'values': [296.0]},
            "If we manually applied Tc = 295.7 between timestep 54 and 66, would the system evolve more stably?":
                {'t_begin': 54, 't_end': 66, 'actions': ['Tc'], 'values': [295.7]},
            "Would increasing Tc to 302.0 from t = 66 to 78 help mitigate concentration oscillations?":
                {'t_begin': 66, 't_end': 78, 'actions': ['Tc'], 'values': [302.0]},
            "How might the concentration change if Tc was fixed at 298.5 between timestep 78 and 90?":
                {'t_begin': 78, 't_end': 90, 'actions': ['Tc'], 'values': [298.5]},
            "What would happen if Tc = 296.8 had been used from timestep 90 to 100 instead of dynamic control?":
                {'t_begin': 90, 't_end': 100, 'actions': ['Tc'], 'values': [296.8]},
            "Assuming Tc stayed at 297.3 from t = 10 to 22, how would the temperature profile be affected?":
                {'t_begin': 10, 't_end': 22, 'actions': ['Tc'], 'values': [297.3]},
            "What difference in rewards would result from using Tc = 295.9 during timestep 22 to 34?":
                {'t_begin': 22, 't_end': 34, 'actions': ['Tc'], 'values': [295.9]},
            "If Tc was locked at 300.0 between timestep 34 and 46, would the Ca levels remain within target?":
                {'t_begin': 34, 't_end': 46, 'actions': ['Tc'], 'values': [300.0]},
            "What would be the outcome of overriding Tc to 295.2 from timestep 46 to 58?":
                {'t_begin': 46, 't_end': 58, 'actions': ['Tc'], 'values': [295.2]}
        }

        CE_B_queries = {
            "What would happen if a more conservative control strategy was applied from timestep 0 to 10 instead of following the policy?":
                {'t_begin': 0, 't_end': 10, 'actions': ['Tc'], 'alpha': 0.5},
            "How would the outcome change if the agent took a more aggressive control action between 10 and 20?":
                {'t_begin': 10, 't_end': 20, 'actions': ['Tc'], 'alpha': 2.0},
            "From t = 20 to 30, what effect would an opposite control response have had on the system?":
                {'t_begin': 20, 't_end': 30, 'actions': ['Tc'], 'alpha': -1.0},
            "What if the agent had reacted more cautiously from 30 to 40 — would the rewards improve or degrade?":
                {'t_begin': 30, 't_end': 40, 'actions': ['Tc'], 'alpha': 0.5},
            "Could the reactor have been stabilized by applying a smoother control behavior between timestep 40 and 50?":
                {'t_begin': 40, 't_end': 50, 'actions': ['Tc'], 'alpha': 0.5},
            "What would the impact be if we used a manual override with minimal adjustments from t = 50 to 60?":
                {'t_begin': 50, 't_end': 60, 'actions': ['Tc'], 'alpha': 0.5},
            "If the control had been more reactive instead of steady from timestep 60 to 70, how would outcomes change?":
                {'t_begin': 60, 't_end': 70, 'actions': ['Tc'], 'alpha': 2.0},
            "What if the agent followed a less responsive (more stable) policy between 70 and 80?":
                {'t_begin': 70, 't_end': 80, 'actions': ['Tc'], 'alpha': 0.5},
            "How would long-term performance be affected if we imposed a more conservative control from timestep 80 to 90?":
                {'t_begin': 80, 't_end': 90, 'actions': ['Tc'], 'alpha': 0.5},
            "What would happen if the control direction was reversed entirely between 90 and 100?":
                {'t_begin': 90, 't_end': 100, 'actions': ['Tc'], 'alpha': -1.0},
            "Suppose the policy had been more reactive between t = 0 and 20—would it reduce instability?":
                {'t_begin': 0, 't_end': 20, 'actions': ['Tc'], 'alpha': 2.0},
            "Could a more aggressive adjustment during t = 20–40 have helped the agent recover faster?":
                {'t_begin': 20, 't_end': 40, 'actions': ['Tc'], 'alpha': 2.0},
            "What if we keep the same actions from timestep 40 to 60?":
                {'t_begin': 40, 't_end': 60, 'actions': ['Tc'], 'alpha': 0.0},
            "Would a smoother ramp-up in control from t = 60 to 80 lead to better quality control outcomes?":
                {'t_begin': 60, 't_end': 80, 'actions': ['Tc'], 'alpha': 0.5},
            "At interval 80–100, what would be the result of replacing policy control with passive actions?":
                {'t_begin': 80, 't_end': 100, 'actions': ['Tc'], 'alpha': 0.0},
            "From timestep 10 to 30, how would a smoother control have differed in impact?":
                {'t_begin': 10, 't_end': 30, 'actions': ['Tc'], 'alpha': 0.5},
            "What if we maintained a fixed control level (no change) during the disturbance event from t = 30 to 50?":
                {'t_begin': 30, 't_end': 50, 'actions': ['Tc'], 'alpha': 0.0},
            "If the agent had been more hesitant (slower to respond) during t = 50–70, would performance improve?":
                {'t_begin': 50, 't_end': 70, 'actions': ['Tc'], 'alpha': 0.5},
            "How might the system evolve if more aggressive recovery behavior was used right after a fault between 70–90?":
                {'t_begin': 70, 't_end': 90, 'actions': ['Tc'], 'alpha': 2.0},
            "Would long-term efficiency be better if conservative control was applied only during the peak load interval t = 20–40?":
                {'t_begin': 20, 't_end': 40, 'actions': ['Tc'], 'alpha': 0.5}
        }

        CE_P_queries = [
            "What would happen if we applied a rule-based policy from timestep 20 to 40 that sets Tc to its maximum whenever Ca < 0.75, and otherwise follows the RL policy?",
            "How would the system behave if, during 20–40, Tc was clamped to 295 whenever T drops below 310, while still using the RL policy in other cases?",
            "Could we improve convergence if Tc was fixed to 300 whenever error_Ca > 0.05 from timestep 20 to 40, and RL policy used otherwise?",
            "What if, between timestep 20 and 40, we followed a hybrid policy that overrides Tc to 302 when Ca < 0.8, while deferring to RL policy otherwise?",
            "Would the agent stabilize faster if we enforced a bang-bang controller for timestep 20–40 where Tc = 302 when error_Ca < -0.08 and Tc = 295 when error_Ca > 0.08?",
            "Suppose we replaced the RL policy from 20 to 40 with a rule that sets Tc = 301 if T > 340 and Tc = 296 if T < 310 — how would the system respond?",
            "If we used a hybrid policy from timestep 20 to 40 that keeps Tc at 298 whenever Ca is within [0.85, 0.9], would the trajectory improve?",
            "How would the output trajectory change if Tc was forced to 295 when Ca < 0.76 during 20–40, while using the RL policy otherwise?",
            "Can performance be improved by applying a threshold rule between 20 and 40 that sets Tc = 300 whenever T > 335, and follows the RL policy otherwise?",
            "What would be the result of a rule-based fallback between timestep 20 and 40 that activates Tc = 302 when error_Ca < -0.05 or T > 345?",
        ]

    elif system == 'four_tank':
        FI_queries = {
            "Which state variables play a key role in the agent’s decision of v1 at timestep 2375?":
                {'t_query': 2375, 'actions': ['v1']},
            "What aspects of the state are most influential on the agent's action of pump voltage 2 at t = 6890?":
                {'t_query': 6890, 'actions': ['v2']},
            "At timestep 1500, what part of the state causes the agent to choose its current action?":
                {'t_query': 1500, 'actions': ['v1', 'v2']},
            "Which components of the environment state drive the agent’s behavior at time 7222?":
                {'t_query': 7222, 'actions': ['v1', 'v2']},
            "What is the most impactful state feature at timestep 3100 for determining the pump voltage 1 control?":
                {'t_query': 3100, 'actions': ['v1']},
            "Can you tell which state inputs the agent focuses on at timestep 5983 for v2?":
                {'t_query': 5983, 'actions': ['v2']},
            "At time 4700, what features of the state space most contribute to the agent’s decision of pump voltage 1 and pump voltage 2?":
                {'t_query': 4700, 'actions': ['v1', 'v2']},
            "Which parts of the state most significantly affect the policy output at timestep 804?":
                {'t_query': 804, 'actions': ['v1', 'v2']},
            "At timestep 3901, what state variables are the main influencers of the agent’s action of pump voltage 2?":
                {'t_query': 3901, 'actions': ['v2']},
            "What elements in the state does the agent attend to most when acting on pump voltage 1 at t = 7005?":
                {'t_query': 7005, 'actions': ['v1']},
            "At time 6240, which dimensions of the state guide the agent’s choice of action for v1 and v2?":
                {'t_query': 6240, 'actions': ['v1', 'v2']},
            "What state features are prioritized by the agent at timestep 1122?":
                {'t_query': 1122, 'actions': ['v1', 'v2']},
            "At t = 5050, which state components have the highest impact on the agent’s behavior regarding pump voltage 1?":
                {'t_query': 5050, 'actions': ['v1']},
            "Which inputs to the agent are most decisive at timestep 3499 for pump voltage 2?":
                {'t_query': 3499, 'actions': ['v2']},
            "What does the agent rely on most from the state when acting at timestep 2999?":
                {'t_query': 2999, 'actions': ['v1', 'v2']},
            "Which observations carry the most weight in the agent’s decision-making at time 6752 for pump voltage 1?":
                {'t_query': 6752, 'actions': ['v1']},
            "At timestep 4400, what are the dominant state variables behind the agent’s action of v2?":
                {'t_query': 4400, 'actions': ['v2']},
            "Which parts of the agent’s input state matter most at t = 175 when selecting v1 and v2?":
                {'t_query': 175, 'actions': ['v1', 'v2']},
            "How much does each state variable contribute to the action selected at timestep 7777?":
                {'t_query': 7777, 'actions': ['v1', 'v2']},
            "At time 808, which variables in the state representation does the agent respond to most for v1?":
                {'t_query': 808, 'actions': ['v1']}
        }

        # 2) Expected outcome queries
        EO_queries = {
            "At timestep 2375, what long-term reward is the agent aiming for by selecting this action?":
                {'t_query': 2375},
            "At time 6890, what future outcomes does the agent hope to achieve with its current decision?":
                {'t_query': 6890},
            "How does the chosen action at timestep 1500 align with the agent's long-term goals?":
                {'t_query': 1500},
            "In the long run, what benefits is the agent expecting from its move at time 7222?":
                {'t_query': 7222},
            "Which future rewards is the agent targeting by acting at timestep 3100?":
                {'t_query': 3100},
            "At timestep 5983, what is the ultimate objective the agent is trying to realize with this decision?":
                {'t_query': 5983},
            "How does the action taken at time 4700 contribute to the agent’s expected future performance?":
                {'t_query': 4700},
            "At time 804, what potential future gains is the agent seeking with its action?":
                {'t_query': 804},
            "What long-term outcomes drive the agent’s decision at timestep 3901?":
                {'t_query': 3901},
            "At t = 7005, what does the agent hope to eventually accomplish through its action?":
                {'t_query': 7005},
            "How does the agent’s current action at time 6240 relate to its future rewards?":
                {'t_query': 6240},
            "At timestep 1122, what are the expected long-term benefits of this decision?":
                {'t_query': 1122},
            "What eventual achievements is the agent striving for by choosing its action at t = 5050?":
                {'t_query': 5050},
            "At timestep 3499, what future rewards is the agent counting on?":
                {'t_query': 3499},
            "What ultimate objectives does the agent have in mind with its action at time 2999?":
                {'t_query': 2999},
            "At timestep 6752, which long-range rewards influence the agent’s decision?,":
                {'t_query': 6752},
            "What long-term benefits is the agent aiming to secure by acting at time 4400?":
                {'t_query': 4400},
            "At t = 175, how does the action contribute to the agent’s future goals?":
                {'t_query': 175},
            "At timestep 7777, what eventual outcomes does the agent plan to achieve with its move?":
                {'t_query': 7777},
            "At time 808, what future rewards are driving the agent’s current decision?":
                {'t_query': 808}
        }

        # 3-1) Contrastive explanations - action queries
        CE_A_queries = {
            "What would be the outcome if the agent had set v1 to 3.0 from timestep 4000 to 4200 instead of the chosen values?":
                {'t_begin': 4000, 't_end': 4200, 'actions': ['v1'], 'values': [3.0]},
            "How would the result differ if v2 had been fixed at 2.5 between timestep 1000 and 1200?":
                {'t_begin': 1000, 't_end': 1200, 'actions': ['v2'], 'values': [2.5]},
            "Suppose the agent had reduced v1 to 1.8 from timestep 2500 to 2700—what impact would that have had?":
                {'t_begin': 2500, 't_end': 2700, 'actions': ['v1'], 'values': [1.8]},
            "What if we increased v2 to 3.2 between t = 3000 and 3200 instead of the RL-predicted action?":
                {'t_begin': 3000, 't_end': 3200, 'actions': ['v2'], 'values': [3.2]},
            "How would the reward trajectory change if we manually set v1 = 2.0 from timestep 5000 to 5100?":
                {'t_begin': 5000, 't_end': 5100, 'actions': ['v1'], 'values': [2.0]},
            "Could outcomes be improved if v2 was overridden to 1.5 from t = 2000 to 2150?":
                {'t_begin': 2000, 't_end': 2150, 'actions': ['v2'], 'values': [1.5]},
            "What if v1 had been held constant at 2.7 instead of following the policy between 6000 and 6100?":
                {'t_begin': 6000, 't_end': 6100, 'actions': ['v1'], 'values': [2.7]},
            "Would the return change if we replaced the action with v2 = 2.0 during timestep 4300 to 4500?":
                {'t_begin': 4300, 't_end': 4500, 'actions': ['v2'], 'values': [2.0]},
            "How would the future states evolve if v1 was clamped to 3.3 from timestep 100 to 300?":
                {'t_begin': 100, 't_end': 300, 'actions': ['v1'], 'values': [3.3]},
            "What difference would it make if we set v2 to 1.2 between t = 3500 and 3700?":
                {'t_begin': 3500, 't_end': 3700, 'actions': ['v2'], 'values': [1.2]},
            "What would happen if v1 was set to 2.5 and v2 to 3.0 from timestep 7000 to 7100?":
                {'t_begin': 7000, 't_end': 7100, 'actions': ['v1', 'v2'], 'values': [2.5, 3.0]},
            "Could better control be achieved by setting v1 = 1.5 and v2 = 2.2 from t = 1200 to 1400 instead of policy actions?":
                {'t_begin': 1200, 't_end': 1400, 'actions': ['v1', 'v2'], 'values': [1.5, 2.2]},
            "What if v1 = 2.0 and v2 = 1.0 were both fixed throughout timestep 5100 to 5300?":
                {'t_begin': 5100, 't_end': 5300, 'actions': ['v1', 'v2'], 'values': [2.0, 1.0]},
            "Would process stability improve if v1 was 3.0 and v2 was 1.5 from timestep 3100 to 3250?":
                {'t_begin': 3100, 't_end': 3250, 'actions': ['v1', 'v2'], 'values': [3.0, 1.5]},
            "How might quality change if we used v1 = 2.2 and v2 = 2.7 during t = 3800 to 3900?":
                {'t_begin': 3800, 't_end': 3900, 'actions': ['v1', 'v2'], 'values': [2.2, 2.7]},
            "Suppose we forced v1 = 1.8 and v2 = 2.0 from timestep 2700 to 2800—what downstream effects would that have?":
                {'t_begin': 2700, 't_end': 2800, 'actions': ['v1', 'v2'], 'values': [1.8, 2.0]},
            "If v1 = 3.4 and v2 = 1.1 were used during the recovery interval 6100 to 6200, would the system recover faster?":
                {'t_begin': 6100, 't_end': 6200, 'actions': ['v1', 'v2'], 'values': [3.4, 1.1]},
            "Could we have prevented a spike by enforcing v1 = 2.6 and v2 = 2.4 from t = 1800 to 2000?":
                {'t_begin': 1800, 't_end': 2000, 'actions': ['v1', 'v2'], 'values': [2.6, 2.4]},
            "What if v1 was reduced to 1.6 and v2 increased to 3.3 from timestep 4200 to 4400?":
                {'t_begin': 4200, 't_end': 4400, 'actions': ['v1', 'v2'], 'values': [1.6, 3.3]},
            "How would production quality evolve if v1 = 2.9 and v2 = 2.0 during the critical load period 5500 to 5700?":
                {'t_begin': 5500, 't_end': 5700, 'actions': ['v1', 'v2'], 'values': [2.9, 2.0]}
        }

        # 3-2) Contrastive explanations - behavior queries
        CE_B_queries = {
            "What would happen if a more conservative control strategy was applied from timestep 4000 to 4200 instead of following the policy?":
                {'t_begin': 4000, 't_end': 4200, 'actions': ['v1', 'v2'], 'alpha': 0.5},
            "How would the outcome change if the agent took a more aggressive control action between 1000 and 1200?":
                {'t_begin': 1000, 't_end': 1200, 'actions': ['v1', 'v2'], 'alpha': 2.0},
            "From t = 2500 to 2700, what effect would an opposite control response have had on the system?":
                {'t_begin': 2500, 't_end': 2700, 'actions': ['v1', 'v2'], 'alpha': -1.0},
            "What if the agent had reacted more slowly and cautiously from 3000 to 3200 — would the rewards improve or degrade?":
                {'t_begin': 3000, 't_end': 3200, 'actions': ['v1', 'v2'], 'alpha': 0.5},
            "Could the process have been stabilized by applying a smoother control behavior between timestep 5000 and 5100?":
                {'t_begin': 5000, 't_end': 5100, 'actions': ['v1', 'v2'], 'alpha': 0.5},
            "What would the impact be if we used a manual override with minimal adjustments from t = 2000 to 2150?":
                {'t_begin': 2000, 't_end': 2150, 'actions': ['v1', 'v2'], 'alpha': 0.5},
            "If the control had been more reactive instead of steady from timestep 6000 to 6100, how would outcomes change?":
                {'t_begin': 6000, 't_end': 6100, 'actions': ['v1', 'v2'], 'alpha': 2.0},
            "What if the agent followed a less responsive (more stable) policy between 4300 and 4500?":
                {'t_begin': 4300, 't_end': 4500, 'actions': ['v1', 'v2'], 'alpha': 0.5},
            "How would long-term performance be affected if we imposed a more conservative control from timestep 100 to 300?":
                {'t_begin': 100, 't_end': 300, 'actions': ['v1', 'v2'], 'alpha': 0.5},
            "What would happen if the control direction was reversed entirely between 3500 and 3700?":
                {'t_begin': 3500, 't_end': 3700, 'actions': ['v1', 'v2'], 'alpha': -1.0},
            "Suppose the policy had been more reactive between t = 7000 and 7100—would it reduce instability?":
                {'t_begin': 7000, 't_end': 7100, 'actions': ['v1', 'v2'], 'alpha': 2.0},
            "Could a more aggressive adjustment during t = 1200–1400 have helped the agent recover faster?":
                {'t_begin': 1200, 't_end': 1400, 'actions': ['v1', 'v2'], 'alpha': 2.0},
            "What if we keep the same actions from timestep 2700 to 2800?":
                {'t_begin': 2700, 't_end': 2800, 'actions': ['v1', 'v2'], 'alpha': 0.0},  # passive → zero effort
            "Would a smoother ramp-up in control from t = 5100 to 5300 lead to better quality control outcomes?":
                {'t_begin': 5100, 't_end': 5300, 'actions': ['v1', 'v2'], 'alpha': 0.5},
            "At interval 800–900, what would be the result of replacing policy control with passive actions?":
                {'t_begin': 800, 't_end': 900, 'actions': ['v1', 'v2'], 'alpha': 0.0},
            "From timestep 3100 to 3250, how would a more smoother control have differed in impact?":
                {'t_begin': 3100, 't_end': 3250, 'actions': ['v1', 'v2'], 'alpha': 0.5},
            "What if we maintained a fixed control level (no change) during the disturbance event from t = 4200 to 4400?":
                {'t_begin': 4200, 't_end': 4400, 'actions': ['v1', 'v2'], 'alpha': 0.0},
            "If the agent had been more hesitant (slower to respond) during t = 1500–1600, would performance improve?":
                {'t_begin': 1500, 't_end': 1600, 'actions': ['v1', 'v2'], 'alpha': 0.5},
            "How might the system evolve if more aggressive recovery behavior was used right after a fault between 6100–6200?":
                {'t_begin': 6100, 't_end': 6200, 'actions': ['v1', 'v2'], 'alpha': 2.0},
            "Would long-term efficiency be better if conservative control was applied only during the peak load interval t = 3800–3900?":
                {'t_begin': 3800, 't_end': 3900, 'actions': ['v1', 'v2'], 'alpha': 0.5}
        }

        # 3-3) Contrastive explanations - policy queries
        CE_P_queries = [
            "What would happen if we applied a rule-based policy from timestep 4000 to 4400 that sets v1 to its maximum whenever h1 < 0.2, and otherwise follows the RL policy?",
            "How would the system behave if, during 4000–4400, v2 was forced to 0 whenever h3 exceeds 0.8, while still using the RL policy in all other conditions?",
            "Could stability improve if we used a bang-bang controller for 4000–4400 that sets v1 = 3.0 when the error of h1 > 0.1, and v1 = 0.0 when the error of h1 < -0.1?",
            "What would be the outcome if, between timestep 4000 and 4400, v1 was clamped to 2.5 whenever h2 < 0.3, but otherwise the RL policy was allowed to control v1 and v2?",
            "How would future rewards change if we enforced a hybrid policy from 4000–4400 that switches to rule-based control when h4 > 0.7 (forcing v2 = 1.5), and uses the RL policy otherwise?",
            "What if a simple threshold rule was applied between timestep 4000 and 4400, setting v1 = 0.1 whenever h3 > 0.9 and v1 = 3.0 whenever h3 < 0.4, instead of using the RL policy?",
            "Would process variance decrease if, during 4000–4400, v2 was set to 2.0 whenever |error of h2| > 0.05, but otherwise kept under the RL policy?",
            "How might the trajectory differ if we used a bang-bang rule from timestep 4000 to 4400 that forces v1 = 3.2 whenever h1 < 0.25 and forces v1 = 0 otherwise, overriding the RL policy?",
            "What would happen if a hybrid fallback rule was applied between 4000–4400 that uses RL normally, but forces v2 = 1.0 whenever h4 rises above 0.8 or error of h1 exceeds 0.15?",
            "Could we improve robustness by replacing the RL policy with a rule-based policy from timestep 4000 to 4400 that sets v1 = 2.8 whenever h2 < 0.4 and simultaneously sets v2 = 1.8 when h3 > 0.6?",
        ]

    elif system == 'photo_production':
        FI_queries = {
            "Which state variables play a key role in the agent’s decision at timestep 23?":
                {'t_query': 23, 'actions': ['I', 'F_N']},
            "What aspects of the state [c_x, c_N, c_q] are most influential on the agent's action at t = 89?":
                {'t_query': 89, 'actions': ['I', 'F_N']},
            "At timestep 15, which component among c_x, c_N, and c_q leads the agent to choose its current action?":
                {'t_query': 15, 'actions': ['I', 'F_N']},
            "Which features of the photo-production state influence the agent’s behavior most at time 222?":
                {'t_query': 222, 'actions': ['I', 'F_N']},
            "What is the most impactful state input at timestep 130 for determining the agent’s control over light intensity I and nutrient flow rate F_N?":
                {'t_query': 130, 'actions': ['I', 'F_N']},
            "At timestep 198, which of the state variables does the agent appear to rely on most when choosing I?":
                {'t_query': 198, 'actions': ['I']},
            "What elements of the state vector (c_x, c_N, c_q) most influence the decision made at time 47?":
                {'t_query': 47, 'actions': ['I', 'F_N']},
            "Which of the current state values most strongly affects the policy output at timestep 8 for the control of F_N?":
                {'t_query': 8, 'actions': ['F_N']},
            "At timestep 139, which states does the agent consider most important when choosing light intensity I?":
                {'t_query': 139, 'actions': ['I']},
            "Which variables in the photo-production process are most attended to by the agent at t = 205 for regulating F_N?":
                {'t_query': 205, 'actions': ['F_N']},
            "At time 124, which state dimensions are driving the agent’s decision?":
                {'t_query': 124, 'actions': ['I', 'F_N']},
            "What state features are prioritized by the policy at timestep 32?":
                {'t_query': 32, 'actions': ['I', 'F_N']},
            "At t = 150, which of c_x, c_N, or c_q has the highest influence on the agent’s behavior for nutrient flow control (F_N)?":
                {'t_query': 150, 'actions': ['F_N']},
            "Which state inputs dominate the decision-making process at timestep 99?":
                {'t_query': 99, 'actions': ['I', 'F_N']},
            "What does the agent rely on most from [c_x, c_N, c_q] when acting at timestep 79?":
                {'t_query': 79, 'actions': ['I', 'F_N']},
            "Which photo-production state variables carry the most weight in the agent’s choice at time 192 for adjusting light intensity (I)?":
                {'t_query': 192, 'actions': ['I']},
            "At timestep 140, which components of the state are most influential in determining the action?":
                {'t_query': 140, 'actions': ['I', 'F_N']},
            "Which of the state variables affect the agent's behavior most at t = 17?":
                {'t_query': 17, 'actions': ['I', 'F_N']},
            "How much does each state variable (c_x, c_N, c_q) contribute to the selected action at timestep 177?":
                {'t_query': 177, 'actions': ['I', 'F_N']},
            "At time 20, which observed state features is the agent responding to most strongly when adjusting F_N?":
                {'t_query': 20, 'actions': ['F_N']}
        }

        EO_queries = {
            "At timestep 23, what long-term reward is the agent aiming for by selecting this action?":
                {'t_query': 23},
            "At time 89, what future outcomes does the agent hope to achieve with its current decision?":
                {'t_query': 89},
            "How does the chosen action at timestep 15 align with the agent's long-term goals in photo production?":
                {'t_query': 15},
            "In the long run, what benefits is the agent expecting from its move at time 222?":
                {'t_query': 222},
            "Which future rewards is the agent targeting by acting at timestep 130 in the photo-production process?":
                {'t_query': 130},
            "At timestep 198, what is the ultimate objective the agent is trying to realize with its control strategy?":
                {'t_query': 198},
            "How does the action taken at time 47 contribute to the agent’s expected long-term performance in quality (c_q)?":
                {'t_query': 47},
            "At time 8, what potential future gains is the agent seeking with its action?":
                {'t_query': 8},
            "What long-term outcomes drive the agent’s decision at timestep 139 in the batch run?":
                {'t_query': 139},
            "At t = 205, what does the agent hope to eventually accomplish through its action on I and F_N?":
                {'t_query': 205},
            "How does the agent’s current control at time 124 relate to expected future improvements in c_q?":
                {'t_query': 124},
            "At timestep 32, what are the expected long-term benefits of this decision regarding light and nitrogen feed?":
                {'t_query': 32},
            "What eventual achievements is the agent striving for by choosing its action at t = 150?":
                {'t_query': 150},
            "At timestep 99, what future rewards is the agent counting on with its current strategy?":
                {'t_query': 99},
            "What ultimate objectives does the agent have in mind with its decision at time 79 in the photo process?":
                {'t_query': 79},
            "At timestep 192, which long-range rewards influence the agent’s choice?":
                {'t_query': 192},
            "What long-term benefits is the agent aiming to secure by acting at time 140?":
                {'t_query': 140},
            "At t = 17, how does the action contribute to the agent’s future goals for optimizing c_q?":
                {'t_query': 17},
            "At timestep 177, what eventual outcomes does the agent plan to achieve with its move?":
                {'t_query': 177},
            "At time 20, what future rewards are driving the agent’s current decision?":
                {'t_query': 20}
        }

        CE_A_queries = {
            "What would be the outcome if the agent had set I to 300 from timestep 0 to 24 instead of the chosen values?":
                {'t_begin': 0, 't_end': 24, 'actions': ['I'], 'values': [300]},
            "How would the result differ if F_N had been fixed at 25 between timestep 36 and 60?":
                {'t_begin': 36, 't_end': 60, 'actions': ['F_N'], 'values': [25]},
            "Suppose the agent had reduced I to 150 from timestep 72 to 96—what impact would that have had?":
                {'t_begin': 72, 't_end': 96, 'actions': ['I'], 'values': [150]},
            "What if we increased F_N to 35 between t = 108 and 132 instead of the RL-predicted action?":
                {'t_begin': 108, 't_end': 132, 'actions': ['F_N'], 'values': [35]},
            "How would the reward trajectory change if we manually set I = 200 from timestep 144 to 168?":
                {'t_begin': 144, 't_end': 168, 'actions': ['I'], 'values': [200]},
            "Could outcomes be improved if F_N was overridden to 10 from t = 180 to 204?":
                {'t_begin': 180, 't_end': 204, 'actions': ['F_N'], 'values': [10]},
            "What if I had been held constant at 350 instead of following the policy between 36 and 60?":
                {'t_begin': 36, 't_end': 60, 'actions': ['I'], 'values': [350]},
            "Would the return change if we replaced the action with F_N = 20 during timestep 72 to 96?":
                {'t_begin': 72, 't_end': 96, 'actions': ['F_N'], 'values': [20]},
            "How would the future states evolve if I was clamped to 400 from timestep 96 to 120?":
                {'t_begin': 96, 't_end': 120, 'actions': ['I'], 'values': [400]},
            "What difference would it make if we set F_N to 5 between t = 120 and 144?":
                {'t_begin': 120, 't_end': 144, 'actions': ['F_N'], 'values': [5]},
            "What would happen if I was set to 250 and F_N to 30 from timestep 0 to 24?":
                {'t_begin': 0, 't_end': 24, 'actions': ['I', 'F_N'], 'values': [250, 30]},
            "Could better control be achieved by setting I = 180 and F_N = 12 from t = 24 to 48 instead of policy actions?":
                {'t_begin': 24, 't_end': 48, 'actions': ['I', 'F_N'], 'values': [180, 12]},
            "What if I = 160 and F_N = 8 were both fixed throughout timestep 48 to 72?":
                {'t_begin': 48, 't_end': 72, 'actions': ['I', 'F_N'], 'values': [160, 8]},
            "Would process stability improve if I was 320 and F_N was 15 from timestep 60 to 84?":
                {'t_begin': 60, 't_end': 84, 'actions': ['I', 'F_N'], 'values': [320, 15]},
            "How might quality change if we used I = 240 and F_N = 28 during t = 84 to 108?":
                {'t_begin': 84, 't_end': 108, 'actions': ['I', 'F_N'], 'values': [240, 28]},
            "Suppose we forced I = 140 and F_N = 20 from timestep 108 to 132—what downstream effects would that have?":
                {'t_begin': 108, 't_end': 132, 'actions': ['I', 'F_N'], 'values': [140, 20]},
            "If I = 380 and F_N = 5 were used during the recovery interval 132 to 156, would the system recover faster?":
                {'t_begin': 132, 't_end': 156, 'actions': ['I', 'F_N'], 'values': [380, 5]},
            "Could we have prevented a spike by enforcing I = 300 and F_N = 25 from t = 156 to 180?":
                {'t_begin': 156, 't_end': 180, 'actions': ['I', 'F_N'], 'values': [300, 25]},
            "What if I was reduced to 120 and F_N increased to 35 from timestep 192 to 216?":
                {'t_begin': 192, 't_end': 216, 'actions': ['I', 'F_N'], 'values': [120, 35]},
            "How would production quality evolve if I = 275 and F_N = 18 during the load period 216 to 240?":
                {'t_begin': 216, 't_end': 240, 'actions': ['I', 'F_N'], 'values': [275, 18]}
        }

        CE_B_queries = {
            "What would happen if a more conservative control strategy was applied from timestep 0 to 24 instead of following the policy?":
                {'t_begin': 0, 't_end': 24, 'actions': ['I', 'F_N'], 'alpha': 0.5},
            "How would the outcome change if the agent took a more aggressive control action between 24 and 48?":
                {'t_begin': 24, 't_end': 48, 'actions': ['I', 'F_N'], 'alpha': 2.0},
            "From t = 48 to 72, what effect would an opposite control response have had on the system?":
                {'t_begin': 48, 't_end': 72, 'actions': ['I', 'F_N'], 'alpha': -1.0},
            "What if the agent had reacted more slowly and cautiously from 72 to 96 — would the rewards improve or degrade?":
                {'t_begin': 72, 't_end': 96, 'actions': ['I', 'F_N'], 'alpha': 0.5},
            "Could the process have been stabilized by applying a smoother control behavior between timestep 96 and 120?":
                {'t_begin': 96, 't_end': 120, 'actions': ['I', 'F_N'], 'alpha': 0.5},
            "What would the impact be if we used a manual override with less adjustments from t = 120 to 144?":
                {'t_begin': 120, 't_end': 144, 'actions': ['I', 'F_N'], 'alpha': 0.5},
            "If the control had been more reactive instead of steady from timestep 144 to 168, how would outcomes change?":
                {'t_begin': 144, 't_end': 168, 'actions': ['I', 'F_N'], 'alpha': 2.0},
            "What if the agent followed a less responsive (more stable) policy between 168 and 192?":
                {'t_begin': 168, 't_end': 192, 'actions': ['I', 'F_N'], 'alpha': 0.5},
            "How would long-term performance be affected if we imposed a more conservative control from timestep 192 to 216?":
                {'t_begin': 192, 't_end': 216, 'actions': ['I', 'F_N'], 'alpha': 0.5},
            "What would happen if the control direction was reversed entirely between 216 and 240?":
                {'t_begin': 216, 't_end': 240, 'actions': ['I', 'F_N'], 'alpha': -1.0},
            "Suppose the policy had been more reactive between t = 0 and 24—would it reduce instability?":
                {'t_begin': 0, 't_end': 24, 'actions': ['I', 'F_N'], 'alpha': 2.0},
            "Could a more aggressive adjustment during t = 24–48 have helped the agent recover faster?":
                {'t_begin': 24, 't_end': 48, 'actions': ['I', 'F_N'], 'alpha': 2.0},
            "What if we keep the same actions from timestep 48 to 72?":
                {'t_begin': 48, 't_end': 72, 'actions': ['I', 'F_N'], 'alpha': 0.0},
            "Would a smoother ramp-up in control from t = 72 to 96 lead to better quality control outcomes?":
                {'t_begin': 72, 't_end': 96, 'actions': ['I', 'F_N'], 'alpha': 0.5},
            "At interval 96–120, what would be the result of replacing policy control with passive actions?":
                {'t_begin': 96, 't_end': 120, 'actions': ['I', 'F_N'], 'alpha': 0.0},
            "From timestep 120 to 144, how would a more smoother control have differed in impact?":
                {'t_begin': 120, 't_end': 144, 'actions': ['I', 'F_N'], 'alpha': 0.5},
            "What if we maintained a fixed control level (no change) during the disturbance event from t = 144 to 168?":
                {'t_begin': 144, 't_end': 168, 'actions': ['I', 'F_N'], 'alpha': 0.0},
            "If the agent had been more hesitant (slower to respond) during t = 168–192, would performance improve?":
                {'t_begin': 168, 't_end': 192, 'actions': ['I', 'F_N'], 'alpha': 0.5},
            "How might the system evolve if more aggressive recovery behavior was used right after a fault between 192–216?":
                {'t_begin': 192, 't_end': 216, 'actions': ['I', 'F_N'], 'alpha': 2.0},
            "Would long-term efficiency be better if conservative control was applied only during the peak load interval t = 216–240?":
                {'t_begin': 216, 't_end': 240, 'actions': ['I', 'F_N'], 'alpha': 0.5}
        }

        CE_P_queries = [
            "What would happen if, between timestep 60 and 340, we used a purely rule-based policy that sets I to its maximum whenever c_n < 200 and sets I to minimum otherwise, fully replacing the RL policy?",
            "How would the system behave if, during timesteps 60–340, F_N were forced to 0 whenever qx_ratio exceeds 0.006 and held at 35 otherwise, using only rule-based control instead of the RL agent?",
            "Could performance improve if we replaced the RL policy entirely with a bang-bang controller from timestep 60 to 340 that sets I = 400 when c_n < 300 and I = 120 when c_n > 300?",
            "What would be the outcome if, during timesteps 60–340, F_N were clamped to 25 whenever qx_ratio < 0.007 and to 0 otherwise, using a fully rule-based control scheme?",
            "How would the reward trajectory change under a rule-only policy between timestep 60 and 340 that forces I = 160 whenever c_n > 600 and sets I = 300 otherwise?",
            "What if a simple threshold rule was applied between timestep 60 and 340, setting F_N = 35 whenever qx_ratio < 0.004 and F_N = 0 whenever qx_ratio > 0.013, instead of using the RL policy?",
            "Would biomass stability improve if, during 60–340, light intensity I was set to 350 whenever |c_n - 300| > 200, but otherwise kept under the RL policy?",
            "How might the growth curve differ if we used a bang-bang rule from timestep 60 to 340 that forces F_N = 40 whenever qx_ratio < 0.0015 and F_N = 0 otherwise, overriding the RL policy?",
            "What would happen if a hybrid fallback rule was applied between 60–340 that uses RL normally, but forces I = 50 whenever qx_ratio rises above 0.014 or c_n drops below 80?",
            "Could we improve robustness by replacing the RL policy with a rule-based policy from timestep 54 to 180 that sets I = 300 whenever c_n < 100 and simultaneously sets F_N = 30 when qx_ratio > 0.01?"
        ]

    return FI_queries, EO_queries, CE_A_queries, CE_B_queries, CE_P_queries
