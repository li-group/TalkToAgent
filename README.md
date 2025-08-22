# TalkToAgent

**TalkToAgent** is a human-centric explainer that connects natural language queries to a range of explainable reinforcement learning (XRL) techniques, enabling domain experts to better understand complex RL agent behavior.

---

## 1. Motivation

While Explainable Reinforcement Learning (XRL) has made strides in improving the transparency of RL agents, its usability remains limited—especially for non-experts. Existing tools often assume users understand which explanation technique to use and how to interpret its results. TalkToAgent bridges this gap by interpreting user queries in natural language and returning task-appropriate XRL explanations in both textual and visual forms.

---

## 2. Methodology

TalkToAgent integrates multiple types of XRL methods and maps them to user queries based on intent:

- **Feature Importance (FI)**  
  _Sample query:_ “Which state variable most affects v1?”  
  → Identifies which inputs influence specific actions most strongly.

- **Expected Outcome (EO)**  
  _Sample query:_ “Which rewards are prioritized at the beginning?”  
  → Reveals how the agent's priorities shift over time based on reward decomposition.

- **Counterfactual Explanations (CF)**  
  _CF-A: Simple action changes_  
  _CF-B: Opposite behavioral policies_  
  _CF-P: Alternative rule-based or qualitative policies_  
  _Sample query:_ “What if the control was more conservative?”  
  → Illustrates how different control strategies impact the agent’s behavior and environment.

Each query triggers relevant code generation, execution, and validation through a pipeline of agents:
- **Coordinator**: Classifies the query type.
- **Coder Agent**: Generates executable policies or reward modifications.
- **Evaluator Agent**: Validates whether the execution aligns with user intent.
- **Debugger Agent**: Diagnoses and corrects errors during execution.

---

## 3. Illustration

### Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Code structure


---

## 4. Citation

If you find this work useful in your research, please cite us:


