import os
from internal_tools import train_agent, get_rollout_data
from explainer.CF_action import cf_by_action
from explainer.CF_behavior import cf_by_behavior
from explainer.CF_policy import cf_by_policy
from params import get_running_params, get_env_params
import matplotlib.pyplot as plt
import numpy as np

font_size = 20
plt.rcParams['axes.titlesize'] = font_size
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size-2
plt.rcParams['ytick.labelsize'] = font_size-2
plt.rcParams['legend.fontsize'] = font_size-4
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Times New Roman'

figure_dir = os.getcwd() + '/figures'
os.makedirs(figure_dir, exist_ok=True)
savedir = figure_dir + '/[RQ2-1] Counterfactual examples'
os.makedirs(savedir, exist_ok=True)

running_params = get_running_params()
env, env_params = get_env_params(running_params.get("system"))

# %%
# 1. Prepare environment and agent
print(f"System: {running_params.get('system')}")

agent = train_agent(lr = running_params['learning_rate'],
                    gamma = running_params['gamma'])
data = get_rollout_data(agent)

# 2. Draw counterfactual data
t_begin = 4000
t_end = 4200
horizon = 20

begin_index = int(np.round(t_begin / env_params['delta_t']))
end_index = int(np.round(t_end / env_params['delta_t']))
len_indices = end_index - begin_index + 1
horizon += len_indices  # Re-adjusting horizon
interval = [begin_index-1, begin_index + horizon]

_, data_cf_a = cf_by_action(t_begin = t_begin, t_end = t_end, actions = ['v1', 'v2'], values = [2.5, 7.5], policy = agent, horizon=horizon)
_, data_cf_b = cf_by_behavior(t_begin = t_begin, t_end = t_end, alpha = 0.3, actions = ['v1', 'v2'], policy = agent, horizon=horizon)
_, data_cf_p = cf_by_policy(t_begin = t_begin, t_end = t_end, policy = agent, team_conversation = [], max_retries = 10, horizon=horizon,
                            message = 'Apply a bang-bang controller in place of the RL policy for all control actions.')

trajs = {
    'CF(A) [v1=2.5, v2=7.5]': data_cf_a["CF: ['v1=2.5', 'v2=7.5']"]['u'][0,:].squeeze()[interval[0]:interval[1]],
    'CF(B) [Conservative (alpha=0.3)]': data_cf_b['Conservative, alpha = 0.3']['u'][0,:].squeeze()[interval[0]:interval[1]],
    'CF(P) [Bang-bang controller]': data_cf_p['New policy']['u'][0,:].squeeze()[interval[0]:interval[1]]
}

n_display = len(trajs)

# 3. Draw figures
t = np.linspace(0, env.tsim, env.N)
t = t[interval[0]:interval[1]]

col = ["tab:red", "tab:purple", "tab:olive", "tab:cyan"]

fig = plt.figure(figsize=(10, 2 * n_display))
for ind, (pi_name, pi_i) in enumerate(trajs.items()):
    plt.subplot(n_display, 1, ind + 1)
    plt.step(t,
        data_cf_a['Actual']['u'][0,:].squeeze()[interval[0]:interval[1]],
        color="tab:gray",
        lw=3,
        label='Actual',
    )
    plt.step(t,
        pi_i,
        color=col[ind],
        lw=3,
        label=pi_name,
    )
    plt.ylabel('v1')
    plt.legend(loc="upper right")
    plt.grid()
    plt.xlim(min(t), max(t))
plt.xlabel("Time (sec)")
plt.tight_layout()
plt.savefig(savedir + '/CF_examples.png')
plt.show()
