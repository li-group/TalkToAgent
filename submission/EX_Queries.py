def get_queries():
    FI_queries = {
        "Which state variables play a key role in the agent’s decision at timestep 2375?":
            {'t_query': 2375},
        "What aspects of the state are most influential on the agent's action at t = 6890?":
            {'t_query': 6890},
        "At timestep 1500, what part of the state causes the agent to choose its current action?":
            {'t_query': 1500},
        "Which components of the environment state drive the agent’s behavior at time 7222?":
            {'t_query': 7222},
        "What is the most impactful state feature at timestep 3100 for determining the agent’s move?":
            {'t_query': 3100},
        "Can you tell which state inputs the agent focuses on at timestep 5983?":
            {'t_query': 5983},
        "At time 4700, what features of the state space most contribute to the agent’s decision?":
            {'t_query': 4700},
        "Which parts of the state most significantly affect the policy output at timestep 804?":
            {'t_query': 804},
        "At timestep 3901, what state variables are the main influencers of the agent’s action?":
            {'t_query': 3901},
        "What elements in the state does the agent attend to most when acting at t = 7005?":
            {'t_query': 7005},
        "At time 6240, which dimensions of the state guide the agent’s choice of action?":
            {'t_query': 6240},
        "What state features are prioritized by the agent at timestep 1122?":
            {'t_query': 1122},
        "At t = 5050, which state components have the highest impact on the agent’s behavior?":
            {'t_query': 5050},
        "Which inputs to the agent are most decisive at timestep 3499?":
            {'t_query': 3499},
        "What does the agent rely on most from the state when acting at timestep 2999?":
            {'t_query': 2999},
        "Which observations carry the most weight in the agent’s decision-making at time 6752?":
            {'t_query': 6752},
        "At timestep 4400, what are the dominant state variables behind the agent’s action?":
            {'t_query': 4400},
        "Which parts of the agent’s input state matter most at t = 175?":
            {'t_query': 175},
        "How much does each state variable contribute to the action selected at timestep 7777?":
            {'t_query': 7777},
        "At time 808, which variables in the state representation does the agent respond to most?":
            {'t_query': 808}
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

    # 3-1) Counterfactual - action queries
    CF_A_queries = {
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

    # 3-2) Counterfactual - behavior queries
    CF_B_queries = {
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
        "Suppose the policy had been delayed by 50 timesteps between t = 7000 and 7100—would it reduce instability?":
            {'t_begin': 7000, 't_end': 7100, 'actions': ['v1', 'v2'], 'alpha': 0.5},
        "Could a more aggressive adjustment during t = 1200–1400 have helped the agent recover faster?":
            {'t_begin': 1200, 't_end': 1400, 'actions': ['v1', 'v2'], 'alpha': 2.0},
        "What if we keep the same actions from timestep 2700 to 2800?":
            {'t_begin': 2700, 't_end': 2800, 'actions': ['v1', 'v2'], 'alpha': 0.0},  # passive → zero effort
        "Would a smoother ramp-up in control from t = 5100 to 5300 lead to better quality control outcomes?":
            {'t_begin': 5100, 't_end': 5300, 'actions': ['v1', 'v2'], 'alpha': 0.5},
        "At interval 800–900, what would be the result of replacing policy control with passive actions?":
            {'t_begin': 800, 't_end': 900, 'actions': ['v1', 'v2'], 'alpha': 0.0},
        "From timestep 3100 to 3250, how would a human-like manual strategy have differed in impact?":
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

    # 3-3) Counterfactual - policy queries
    CF_P_queries = [
        "What if we use the bang-bang controller instead of the current RL policy from timestep 4000 to 4200?",
        "How would performance change if we switched to a simple rule-based policy from timestep 1000 to 1300?",
        "What if we used a fallback manual override policy whenever h1 drops below 0.3 between 2000 and 2200?",
        "How would outcomes differ if we applied a max-output policy during high-load conditions from 3500 to 3700?",
        "What happens if we implement bang-bang control for v1 between t = 3000 and 3300 instead of relying on the RL policy?",
        "Could a hybrid policy — rule-based below h2 = 0.25, RL otherwise — improve stability from 4500 to 4700?",
        "What if we replaced the learned policy with a deterministic control policy from t = 1200 to 1500?",
        "How would the return change if we enforced a constant-action policy between timestep 2700 and 2900?",
        "What if we switch to a conditional policy that sets v2 to minimum when h3 exceeds 0.8 from 6000 to 6100?",
        "Can we achieve better performance using a threshold-triggered rule instead of the RL policy from 5200 to 5400?",
        "What would happen if we applied a safety-first policy (conservative bias) from timestep 7000 to 7200?",
        "What if we define a rule: 'Set v1 to max if h1 < 0.2' and apply it from 1800 to 2000?",
        "Could we prevent the spike if we followed a bang-bang controller instead of RL between 2400 and 2600?",
        "What if the policy was overridden manually during anomalies detected from timestep 3100 to 3250?",
        "How would a hybrid policy that defaults to RL but falls back to rule-based control during faults perform from 100 to 300?",
        "What would happen if we applied a simple rule-based policy that sets v1 to its maximum whenever h1 < 0.2, from timestep 5000 to 5150?",
        "What if we imposed a conservative rule-based controller only during unstable intervals, like 3800–4000?",
        "Can we mitigate risk by switching to a deterministic safety policy from t = 6400 to 6600?",
        "What would the difference be if we used a human-engineered control policy instead of the RL agent during 5500–5700?",
        "How does the process behave if we enforce an expert-designed rule-based policy during critical load windows from 4300 to 4450?"
    ]
    return FI_queries, EO_queries, CF_A_queries, CF_B_queries, CF_P_queries