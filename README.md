# TalkToAgent

Welcome to the TalkToAgent page.
**TalkToAgent** is a human-centric explainer that connects natural language queries to a range of explainable reinforcement learning (XRL) techniques,
enabling domain experts to better understand complex RL agent behavior.

---

## 1. Motivation

While Explainable Reinforcement Learning (XRL) has improved the transparency of RL agents, its usability remains limited, especially for non-experts.
Existing tools often assume users understand which explanation technique to use and how to interpret its results.
TalkToAgent bridges this gap by interpreting user queries in natural language and returning task-appropriate XRL explanations in both textual and visual forms.

---

## 2. Methodology

<a name="chat-example"></a>
<p align="center">
<img src="images/[TTA] Fig 1. Overall Framework.png" alt="drawing" width="500"/>
</p>

_TalkToAgent_ is a multi-agent Large Language Models (LLM) framework that delivers interactive, natural language explanations for RL policies.
With five specialized LLM agents, it can generate multimodal explanations to various types user queries related to RL systems.

- **Coordinator Agent**: Maps user queries into appropriate predefined XRL functions.
- **Coder Agent**: Generates python codes of executable policies or modified rewards.
- **Evaluator Agent**: Validates whether the execution aligns with user intent.
- **Debugger Agent**: Diagnoses error messages and creates guidance to correct them.
- **Explainer Agent**: Explains the XRL visualization results in natural language form, resulting in multimodal explanations.

TalkToAgent integrates the following types of XRL types and maps them to relevant predefined XRL functions.

<a name="chat-example"></a>
<p align="center">
<img src="images/[TTA] Fig 2. XRL queries.png" alt="drawing" width="500"/>
</p>

1. **Feature Importance (FI)**
  FI explanations aim to identify which aspects of the current state most influence the agent’s specific action.  
  _Example:_ “Which state variable most affects the current action?”

2. **Expected Outcome (EO)**  
  EO explanations aim to explain an agent's behavior by analyzing anticipated future trajectories or rewards as a result of executing a particular action.  
  _Example:_ “What is the agent trying to achieve by doing this action?”

3. **Counterfactual Explanations (CF)**  
  CF approaches aim to answer contrastive questions such as "What if?" or "Why not?",
  highlighting why the agent selected the current action over plausible alternatives.  
  In TalkToAgent, three novel types of counterfactual explanations are introduced to enhance the flexibility of counterfactual reasoning in RL practices.  
   1) **Action-based Counterfactual Explanations (CF-A)**    
     CF-A approach poses contrastive actions for a certain timestep.  
     _Example:_ “Why don't we take action b, instead of action a at time t?”

   2) **Behavior-based Counterfactual Explanations (CF-B)**  
      CF-B approach constructs contrastive scenarios from qualitative descriptions about agent behavior.
      Terms like _aggressive_ or _opposite_ are translated to counterfactual trajectories by using the idea of Polyak averaging.  
      _Example:_ “Why don't we take a more conservative control from t=4000 to 4200?”
   
   3) **Policy-based Counterfactual Explanations (CF-P)**  
      CF-P approach addresses a broader question of how a fundamentally different control strategy would affect future trajectories, rather than just a localized action deviation.  
      _Example:_"What would happen if we replaced the current RL policy with an on-off controller from t=4000 to 4200?"

---

## 3. Illustration

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### Code structure


### Example results

<a name="chat-example"></a>
<p align="center">
<img src="images/[TTA] Fig 6. Explanations2.png" alt="drawing" width="800"/>
</p>
 
## 4. Citation

If you find this work useful in your research, please cite us:


