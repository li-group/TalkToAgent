def get_queries(system):
    if system == 'cstr':
        FI_queries = {
            "Which state variables are most critical in the agent's decision at timestep 12?":
                {'t_query': 12},
            "What aspects of the state influence the agent's action the most at t = 35?":
                {'t_query': 35},
            "At timestep 67, which part of the state drives the action selection?":
                {'t_query': 67},
            "Which components of the CSTR state guide the policy's behavior at time 83?":
                {'t_query': 83},
            "What is the most decisive state feature at timestep 50 in determining the action?":
                {'t_query': 50},
            "Can you tell which state variables the agent relies on at timestep 9?":
                {'t_query': 9},
            "At time 27, which state values are most responsible for the agent’s behavior?":
                {'t_query': 27},
            "Which parts of the CSTR state representation are key to the agent's decision at timestep 99?":
                {'t_query': 99},
            "At timestep 42, what elements in the state space affect the action the most?":
                {'t_query': 42},
            "At time 70, which state inputs does the policy consider most important?":
                {'t_query': 70},
            "At timestep 18, which variables have the largest impact on action choice?":
                {'t_query': 18},
            "How much does each CSTR state variable contribute to the agent’s move at t = 61?":
                {'t_query': 61},
            "Which observations matter the most to the agent at timestep 24?":
                {'t_query': 24},
            "Which parts of the state representation at time 89 drive the control logic?":
                {'t_query': 89},
            "What does the agent focus on most from the state when deciding an action at timestep 5?":
                {'t_query': 5},
            "At timestep 77, what state variables are the main drivers of the action decision?":
                {'t_query': 77},
            "What role does each CSTR state variable play at t = 33 in determining the behavior?":
                {'t_query': 33},
            "At time 93, which dimensions of the state most influence the control policy?":
                {'t_query': 93},
            "Which state features are prioritized at timestep 48 during decision-making?":
                {'t_query': 48},
            "At timestep 100, what are the dominant state inputs affecting the agent’s action?":
                {'t_query': 100}
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
            "What would happen if we applied a rule-based policy from timestep 20 to 60 that sets Tc to its maximum whenever Ca < 0.75, and otherwise follows the RL policy?",
            "How would the system behave if, during 30–70, Tc was clamped to 295 whenever T drops below 310, while still using the RL policy in other cases?",
            "Could we improve convergence if Tc was fixed to 300 whenever error_Ca > 0.05 from timestep 10 to 50, and RL policy used otherwise?",
            "What if, between timestep 40 and 80, we followed a hybrid policy that overrides Tc to 302 when Ca < 0.8, while deferring to RL policy otherwise?",
            "Would the agent stabilize faster if we enforced a bang-bang controller for timestep 0–40 where Tc = 302 when error_Ca < -0.08 and Tc = 295 when error_Ca > 0.08?",
            "Suppose we replaced the RL policy from 60 to 100 with a rule that sets Tc = 301 if T > 340 and Tc = 296 if T < 310 — how would the system respond?",
            "If we used a hybrid policy from timestep 10 to 50 that keeps Tc at 298 whenever Ca is within [0.85, 0.9], would the trajectory improve?",
            "How would the output trajectory change if Tc was forced to 295 when Ca < 0.76 during 30–70, while using the RL policy otherwise?",
            "Can performance be improved by applying a threshold rule between 20 and 60 that sets Tc = 300 whenever T > 335, and follows the RL policy otherwise?",
            "What would be the result of a rule-based fallback between timestep 0 and 40 that activates Tc = 302 when error_Ca < -0.05 or T > 345?",
        ]

    elif system == 'four_tank':
        FI_queries = {
            "Which state variables play a key role in the agent’s decision of v1 at timestep 2375?":
                {'t_query': 2375, 'actions': ['v1']},
            "What aspects of the state are most influential on the agent's action of v2 at t = 6890?":
                {'t_query': 6890, 'actions': ['v2']},
            "At timestep 1500, what part of the state causes the agent to choose its current action?":
                {'t_query': 1500, 'actions': ['v1', 'v2']},
            "Which components of the environment state drive the agent’s behavior at time 7222?":
                {'t_query': 7222, 'actions': ['v1', 'v2']},
            "What is the most impactful state feature at timestep 3100 for determining the agent’s move on v1?":
                {'t_query': 3100, 'actions': ['v1']},
            "Can you tell which state inputs the agent focuses on at timestep 5983 for v2?":
                {'t_query': 5983, 'actions': ['v2']},
            "At time 4700, what features of the state space most contribute to the agent’s decision of v1 and v2?":
                {'t_query': 4700, 'actions': ['v1', 'v2']},
            "Which parts of the state most significantly affect the policy output at timestep 804?":
                {'t_query': 804, 'actions': ['v1', 'v2']},
            "At timestep 3901, what state variables are the main influencers of the agent’s action of v2?":
                {'t_query': 3901, 'actions': ['v2']},
            "What elements in the state does the agent attend to most when acting on v1 at t = 7005?":
                {'t_query': 7005, 'actions': ['v1']},
            "At time 6240, which dimensions of the state guide the agent’s choice of action for v1 and v2?":
                {'t_query': 6240, 'actions': ['v1', 'v2']},
            "What state features are prioritized by the agent at timestep 1122?":
                {'t_query': 1122, 'actions': ['v1', 'v2']},
            "At t = 5050, which state components have the highest impact on the agent’s behavior regarding v1?":
                {'t_query': 5050, 'actions': ['v1']},
            "Which inputs to the agent are most decisive at timestep 3499 for v2?":
                {'t_query': 3499, 'actions': ['v2']},
            "What does the agent rely on most from the state when acting at timestep 2999?":
                {'t_query': 2999, 'actions': ['v1', 'v2']},
            "Which observations carry the most weight in the agent’s decision-making at time 6752 for v1?":
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

    return FI_queries, EO_queries, CE_A_queries, CE_B_queries, CE_P_queries
