import numpy as np

def cstr_reward(self,x,u,con):
    Sp_i = 0
    cost = 0 
    R = 0.1
    if not hasattr(self, 'u_prev'):
        self.u_prev = u

    for k in self.env_params["SP"]:
        i = self.model.info()["states"].index(k)
        SP = self.SP[k]
        
        o_space_low = self.env_params["o_space"]["low"][i] 
        o_space_high = self.env_params["o_space"]["high"][i] 

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        r_scale = self.env_params.get("r_scale", {})

        cost += (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2) * r_scale.get(k, 1)

        Sp_i += 1
    u_normalized = (u - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )
    u_prev_norm =  (self.u_prev - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )
    self.u_prev = u

    # Add the control cost
    cost += np.sum(R * (u_normalized-u_prev_norm)**2)
    r = -cost
    try:
        return r[0]
    except Exception:
        return r

def four_tank_reward(self, x, u, con):
    Sp_i = 0
    cost = 0
    R = 0.1
    if not hasattr(self, 'u_prev'):
        self.u_prev = u

    for k in self.env_params["SP"]:
        i = self.model.info()["states"].index(k)
        SP = self.SP[k]

        o_space_low = self.env_params["o_space"]["low"][i]
        o_space_high = self.env_params["o_space"]["high"][i]

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        r_scale = self.env_params.get("r_scale", {})

        cost += (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2) * r_scale.get(k, 1)

        Sp_i += 1
    u_normalized = (u - self.env_params["a_space"]["low"]) / (
            self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )
    u_prev_norm = (self.u_prev - self.env_params["a_space"]["low"]) / (
            self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )
    self.u_prev = u

    # Add the control cost
    cost += np.sum(R * (u_normalized - u_prev_norm) ** 2)
    r = -cost
    try:
        return r[0]
    except Exception:
        return r

def four_tank_reward_decomposed(env, x, u, con):
    Sp_i = 0
    R = 0.1
    costs = []
    if not hasattr(env, 'u_prev'):
        env.u_prev = u

    for k in env.env_params["SP"]:
        i = env.model.info()["states"].index(k)
        SP = env.SP[k]

        o_space_low = env.env_params["o_space"]["low"][i]
        o_space_high = env.env_params["o_space"]["high"][i]

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        r_scale = env.env_params.get("r_scale", {})

        cost_k = (np.sum(x_normalized - setpoint_normalized[env.t]) ** 2) * r_scale.get(k, 1)
        costs.append(cost_k)

        Sp_i += 1

    u_normalized = (u - env.env_params["a_space"]["low"]) / (
            env.env_params["a_space"]["high"] - env.env_params["a_space"]["low"]
    )
    u_prev_norm = (env.u_prev - env.env_params["a_space"]["low"]) / (
            env.env_params["a_space"]["high"] - env.env_params["a_space"]["low"]
    )
    env.u_prev = u

    # Add the control cost
    cost_u = np.sum(R * (u_normalized - u_prev_norm) ** 2)
    costs.append(cost_u)

    rs = [-cost for cost in costs]
    return rs
