import os
from internal_tools import train_agent, get_rollout_data
from explainer.CE_action import ce_by_action
from explainer.CE_behavior import ce_by_behavior
from explainer.CE_policy import ce_by_policy
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
savedir = figure_dir + '/[RQ2-1] Contrastive examples'
os.makedirs(savedir, exist_ok=True)

running_params = get_running_params()
env, env_params = get_env_params(running_params.get("system"))

# %%
# 1. Prepare environment and agent
agent = train_agent(lr = running_params['learning_rate'],
                    gamma = running_params['gamma'])
data = get_rollout_data(agent)

# 2. Draw contrastive data
t_begin = 4000
t_end = 4500
horizon = 0

begin_index = int(np.round(t_begin / env_params['delta_t']))
end_index = int(np.round(t_end / env_params['delta_t']))
len_indices = end_index - begin_index + 1
horizon += len_indices  # Re-adjusting horizon
interval = [begin_index-1, begin_index + horizon]

_, data_ce_a = ce_by_action(t_begin = t_begin, t_end = t_end, actions = ['v1', 'v2'], values = [10, 10], policy = agent, horizon=horizon)
_, data_ce_b = ce_by_behavior(t_begin = t_begin, t_end = t_end, alpha = 0.3, actions = ['v1', 'v2'], policy = agent, horizon=horizon)
_, data_ce_p = ce_by_policy(t_begin = t_begin, t_end = t_end, policy = agent, team_conversation = [], max_retries = 10, horizon=horizon,
                            message = 'Use an on-off controller: set v1 = 15.0 whenever Error_h2 > 0.0 (setpoint for h2 above current h2),'
                                      'and v1 = 5.0 otherwise; similarly, set v2 = 15.0 whenever Error_h1 > 0.0, and v2 = 5.0 otherwise.',
                            query = "What would happen if we replaced the current RL policy with an on-off controller,"
                                    "such that $v_1 = 15.0$ whenever the error of $h_2 > 0.0$, and $v_1 = 5.0$ otherwise;"
                                    "and similarly, $v_2 = 15.0$ whenever the error of $h_1 > 0.0$, and $v_2 = 5.0$ otherwise?")

trajs = {
    'CE(A) [v1=10, v2=10]': data_ce_a["CE: ['v1=10', 'v2=10']"]['u'][0,:].squeeze()[interval[0]:interval[1]],
    'CE(B) [Conservative (alpha=0.3)]': data_ce_b['Conservative, alpha = 0.3']['u'][0,:].squeeze()[interval[0]:interval[1]],
    'CE(P) [Bang-bang controller]': data_ce_p['New policy']['u'][0,:].squeeze()[interval[0]:interval[1]]
}

n_display = len(trajs)

# %% 3. Draw figures
t = np.linspace(0, env.tsim, env.N)
t = t[interval[0]:interval[1]]

col = ["tab:red", "tab:purple", "tab:olive", "tab:cyan"]

fig = plt.figure(figsize=(10, 2 * n_display))
for ind, (pi_name, pi_i) in enumerate(trajs.items()):
    plt.subplot(n_display, 1, ind + 1)
    plt.step(t,
        data_ce_a['Actual']['u'][0,:].squeeze()[interval[0]:interval[1]],
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
plt.savefig(savedir + '/CE_examples.png')
plt.show()
