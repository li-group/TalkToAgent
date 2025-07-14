# Policy Evaluation Class for pc-gym
import numpy as np
import matplotlib.pyplot as plt
from .oracle import oracle


class policy_eval:
    """
    Policy Evaluation Class

    Inputs: Environment, policy and number of policy repitions

    Outputs: Plots of states/control/constraints/setpoints (complete),
             return distribution (incomplete), expected return (incomplete),
             oracle trajectories (incomplete) and lower confidence bounds (incomplete)
    """

    def __init__(
        self,
        make_env,
        policies,
        reps,
        env_params,
        oracle=False,
        MPC_params=False,
        cons_viol=False,
    ):
        self.make_env = make_env
        self.env_params = env_params
        self.env = make_env(env_params)
        self.policies = policies
        self.n_pi = len(policies)
        self.reps = reps
        self.oracle = oracle
        self.cons_viol = cons_viol

        self.MPC_params = MPC_params

    def rollout(self, policy_i, cf_settings = None):
        """
        Rollout the policy for N steps and return the total reward, states and actions

        Input:
            policy - policy to be rolled out

        Outputs:
            total_reward - total reward obtained
            states - states obtained from rollout
            actions - actions obtained from rollout

        """

        total_reward = 0
        rewards = np.zeros((1, self.env.N))
        s_rollout = np.zeros((self.env.Nx, self.env.N))
        actions = np.zeros((self.env.env_params["a_space"]["low"].shape[0], self.env.N))

        o, r = self.env.reset()
        # total_reward = r["r_init"]
        s_rollout[:, 0] = (o + 1) * (
            self.env.observation_space_base.high - self.env.observation_space_base.low
        ) / 2 + self.env.observation_space_base.low # Descaling process
        rewards[:, 0] = r["r_init"]

        for i in range(self.env.N - 1):
            a, _s = policy_i.predict(
                o, deterministic=True
            )  # Rollout with a deterministic policy

            if cf_settings is not None:
                if cf_settings['CF_mode'] == 'action':
                    begin_index = cf_settings["begin_index"]
                    end_index = cf_settings["end_index"]
                    # Replace optimal action with counterfactual action, only at queried step.
                    if begin_index <= i <= end_index:
                        a = self.env._scale_U(cf_settings["cf_traj"][:,i].squeeze())

                elif cf_settings['CF_mode'] == 'policy':
                    begin_index = cf_settings["begin_index"]
                    end_index = cf_settings["end_index"]
                    # Replace optimal action with action derived by counterfactual policy, after queried step.
                    if begin_index <= i <= end_index:
                        cf_policy = cf_settings['CF_policy']
                        a, _s = cf_policy.predict(o, deterministic=True)

            o, r, term, trunc, info = self.env.step(a)

            actions[:, i] = (a + 1) * (
                    self.env.env_params["a_space"]["high"]
                    - self.env.env_params["a_space"]["low"]
            ) / 2 + self.env.env_params["a_space"]["low"]
            s_rollout[:, i + 1] = (o + 1) * (
                    self.env.observation_space_base.high - self.env.observation_space_base.low
            ) / 2 + self.env.observation_space_base.low
            rewards[:, i] = r

        if self.env.constraint_active:
            cons_info = info["cons_info"]
        else:
            cons_info = np.zeros((1, self.env.N, 1))
        a, _s = policy_i.predict(o, deterministic=True)
        actions[:, self.env.N - 1] = (a + 1) * (
            self.env.env_params["a_space"]["high"]
            - self.env.env_params["a_space"]["low"]
        ) / 2 + self.env.env_params["a_space"]["low"]

        return rewards, s_rollout, actions, cons_info

    def get_rollouts(self, get_Q = False, cf_settings = None):
        """
        Function to plot the rollout of the policy

        Inputs:
            policy - policy to be rolled out
            reps - number of rollouts to be performed

        Outputs:
            Plot of states and actions with setpoints and constraints if they exist]

        """
        data = {}
        action_space_shape = self.env.env_params["a_space"]["low"].shape[0]
        num_states = self.env.Nx

        # Collect Oracle data
        if self.oracle:
            r_opt = np.zeros((1, self.reps))
            x_opt = np.zeros((self.env.Nx_oracle, self.env.N, self.reps))
            u_opt = np.zeros((self.env.Nu, self.env.N, self.reps))
            oracle_instance = oracle(self.make_env, self.env_params, self.MPC_params)
            for i in range(self.reps):
                x_opt[:, :, i], u_opt[:, :, i] = oracle_instance.mpc()
                for k in self.env.SP:
                    state_i = self.env.model.info()["states"].index(k)
                    r_scale = self.env_params.get("r_scale", {})
                    r_opt[:, i] += (
                        np.sum((x_opt[state_i, :, i] - self.env.SP[k]) ** 2)
                        * -1
                        * r_scale.get(k, 1)
                    )
            data.update({"oracle": {"r": r_opt, "x": x_opt, "u": u_opt}})

        # Collect RL rollouts for all policies
        for pi_name, pi_i in self.policies.items():
            states = np.zeros((num_states, self.env.N, self.reps))
            actions = np.zeros((action_space_shape, self.env.N, self.reps))
            rew = np.zeros((1, self.env.N, self.reps))
            try:
                cons_info = np.zeros((self.env.n_con, self.env.N, 1, self.reps))
            except Exception:
                cons_info = np.zeros((1, self.env.N, 1, self.reps))
            for r_i in range(self.reps):
                (
                    rew[:, :, r_i],
                    states[:, :, r_i],
                    actions[:, :, r_i],
                    cons_info[:, :, :, r_i],
                ) = self.rollout(pi_i, cf_settings)
            data.update({pi_name: {"r": rew, "x": states, "u": actions}})
            if self.env.constraint_active:
                data[pi_name].update({"g": cons_info})
            if get_Q:
              critic = pi_i.critic.qf0
              X = data[pi_name]['x']
              U = data[pi_name]['u']
              X = X.reshape(X.shape[0], -1, order='F').T
              U = U.reshape(U.shape[0], -1, order='F').T
              X, U = self.env._scale_X(X), self.env._scale_U(U)
              XU = np.hstack([X, U])
              import torch
              Q = critic(torch.tensor(np.array(XU), dtype=torch.float32)).detach().numpy()
              Q = Q.T
              Q = Q.reshape(Q.shape[0], -1, self.reps, order='F')
              data[pi_name]['q'] = Q
        self.data = data
        return data

    def plot_data(self, data, reward_dist=False, savedir = '', interval=None):
        t = np.linspace(0, self.env.tsim, self.env.N)
        if interval:
            t = t[interval[0]:interval[1]]

        len_d = 0
        # has_Q = 'q' in set().union(*(d.keys() for d in data.values()))
        n_display = self.env.Nx_oracle + self.env.Nu - self.env.Nd + 1
        # n_display = n_display+1 if has_Q else n_display

        if self.env.disturbance_active:
            len_d = len(self.env.model.info()["disturbances"])

        col = ["tab:red", "tab:purple", "tab:olive", "tab:gray", "tab:cyan"]
        if self.n_pi > len(col):
            raise ValueError(
                f"Number of policies ({self.n_pi}) is greater than the number of available colors ({len(col)})"
            )

        fig = plt.figure(figsize=(10, 2 * (n_display)))
        for i in range(self.env.Nx_oracle):
            plt.subplot(n_display, 1, i + 1)
            for ind, (pi_name, pi_i) in enumerate(self.policies.items()):
                plt.plot(
                    t,
                    np.median(data[pi_name]["x"][i, :, :], axis=1),
                    color=col[ind],
                    lw=3,
                    label=self.env.model.info()["states"][i] + " (" + pi_name + ")",
                )
                plt.gca().fill_between(
                    t,
                    np.min(data[pi_name]["x"][i, :, :], axis=1),
                    np.max(data[pi_name]["x"][i, :, :], axis=1),
                    color=col[ind],
                    alpha=0.2,
                    edgecolor="none",
                )
            if self.oracle:
                plt.plot(
                    t,
                    np.median(data["oracle"]["x"][i, :, :], axis=1),
                    color="tab:blue",
                    lw=3,
                    label="Oracle " + self.env.model.info()["states"][i],
                )
                plt.gca().fill_between(
                    t,
                    np.min(data["oracle"]["x"][i, :, :], axis=1),
                    np.max(data["oracle"]["x"][i, :, :], axis=1),
                    color="tab:blue",
                    alpha=0.2,
                    edgecolor="none",
                )
            if self.env.model.info()["states"][i] in self.env.SP:
                SP = self.env.SP[self.env.model.info()["states"][i]]
                if interval:
                    SP = SP[interval[0]:interval[1]]
                plt.step(
                    t,
                    SP,
                    where="post",
                    color="black",
                    linestyle="--",
                    label="Set Point",
                )
            if self.env.constraint_active:
                if self.env.model.info()["states"][i] in self.env.constraints:
                    plt.hlines(
                        self.env.constraints[self.env.model.info()["states"][i]],
                        0,
                        self.env.tsim,
                        color="black",
                        label="Constraint",
                    )
            plt.ylabel(self.env.model.info()["states"][i])
            plt.xlabel("Time (sec)")
            plt.legend(loc="best")
            plt.grid("True")
            plt.xlim(min(t), max(t))

        for j in range(self.env.Nu - len_d):
            plt.subplot(
                n_display,
                1,
                j + self.env.Nx_oracle + 1,
            )
            for ind, (pi_name, pi_i) in enumerate(self.policies.items()):
                plt.step(
                    t,
                    np.median(data[pi_name]["u"][j, :, :], axis=1),
                    color=col[ind],
                    lw=3,
                    label=self.env.model.info()["inputs"][j] + " (" + pi_name + ")",
                )
            if self.oracle:
                plt.step(
                    t,
                    np.median(data["oracle"]["u"][j, :, :], axis=1),
                    color="tab:blue",
                    lw=3,
                    label="Oracle " + str(self.env.model.info()["inputs"][j]),
                )
            if self.env.constraint_active:
                for con_i in self.env.constraints:
                    if self.env.model.info()["inputs"][j] == con_i:
                        plt.hlines(
                            self.env.constraints[self.env.model.info()["inputs"][j]],
                            0,
                            self.env.tsim,
                            "black",
                            label="Constraint",
                        )
            plt.ylabel(self.env.model.info()["inputs"][j])
            plt.xlabel("Time (sec)")
            plt.legend(loc="best")
            plt.grid("True")
            plt.xlim(min(t), max(t))

        if self.env.disturbance_active:
            for k in self.env.disturbances.keys():
                i = 1
                if self.env.disturbances[k].any() is not None:
                    plt.subplot(
                        n_display,
                        1,
                        i + j + self.env.Nx_oracle + 1,
                    )
                    plt.step(t, self.env.disturbances[k], color="tab:orange", label=k)
                    plt.xlabel("Time (sec)")
                    plt.ylabel(k)
                    plt.xlim(min(t), max(t))
                    i += 1

        # if has_Q:
        plt.subplot(n_display, 1, n_display)
        for ind, (pi_name, pi_i) in enumerate(self.policies.items()):
            if 'r' in data[pi_name].keys():
                plt.plot(
                  t,
                  np.median(data[pi_name]["r"][0, :, :], axis=1),
                  color=col[ind],
                  lw=3,
                  label="Reward (" + pi_name + ")",
                )
                plt.gca().fill_between(
                  t,
                  np.min(data[pi_name]["r"][0, :, :], axis=1),
                  np.max(data[pi_name]["r"][0, :, :], axis=1),
                  color=col[ind],
                  alpha=0.2,
                  edgecolor="none",
                )
            else:
                pass
        plt.ylabel("Reward")
        plt.xlabel("Time (sec)")
        plt.legend(loc="best")
        plt.grid("True")
        plt.xlim(min(t), max(t))

        plt.tight_layout()
        plt.savefig(savedir)
        plt.show()

        # Visualizing plots for constraint violation
        if self.cons_viol:
            plt.figure(figsize=(12, 3 * self.env.n_con))
            con_i = 0
            for i, con in enumerate(self.env.constraints):
                for j in range(len(self.env.constraints[str(con)])):
                    plt.subplot(self.env.n_con, 1, con_i + 1)
                    plt.title(f"{con} Constraint")
                    for ind, (pi_name, pi_i) in enumerate(self.policies.items()):
                        plt.step(
                            t,
                            np.sum(data[pi_name]["g"][con_i, :, :, :], axis=2),
                            color=col[ind],
                            label=f"{con} ({pi_name}) Violation (Sum over Repetitions)",
                        )
                    plt.grid("True")
                    plt.xlabel("Time (sec)")
                    plt.ylabel(con)
                    plt.xlim(min(t), max(t))
                    plt.legend(loc="best")
                    con_i += 1
            plt.tight_layout()
            plt.show()

        # Visualizing plots for reward distribution
        if reward_dist:
            plt.figure(figsize=(12, 8))
            plt.grid(True, linestyle="--", alpha=0.6)
            all_data = np.concatenate([data[key]["r"].flatten() for key in data.keys()])

            min_value = np.min(all_data)
            max_value = np.max(all_data)

            bins = np.linspace(min_value, max_value, self.reps)
            if self.oracle:
                plt.hist(
                    data["oracle"]["r"].flatten(),
                    bins=bins,
                    color="tab:blue",
                    alpha=0.5,
                    label="Oracle",
                    edgecolor="black",
                )
            for ind, (pi_name, pi_i) in enumerate(self.policies.items()):
                plt.hist(
                    data[pi_name]["r"].flatten(),
                    bins=bins,
                    color=col[ind],
                    alpha=0.5,
                    label=pi_name,
                    edgecolor="black",
                )

            plt.xlabel("Return", fontsize=14)
            plt.ylabel("Frequency", fontsize=14)
            plt.title("Distribution of Expected Return", fontsize=16)
            plt.legend(fontsize=12)

            plt.show()

        return fig
