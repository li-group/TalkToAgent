import numpy as np

def cstr_reward(self, x, u, con):
    Sp_i = 0
    cost = 0
    R = 0.1
    if not hasattr(self, 'u_prev'):
        self.u_prev = u

    for k in self.SP:
        i = self.model.info()["states"].index(k)
        SP = self.SP[k]

        o_space_low = self.env_params["o_space"]["low"][i]
        o_space_high = self.env_params["o_space"]["high"][i]

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        r_scale = self.env_params.get("r_scale", {})

        # cost += (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2) * r_scale.get(k, 1)
        cost += np.tanh(100 * (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2)) * r_scale.get(k, 1)

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

def multistage_extraction_reward(self, x, u, con):
    Sp_i = 0
    cost = 0
    R = 0.1
    if not hasattr(self, 'u_prev'):
        self.u_prev = u

    for k in self.SP:
        i = self.model.info()["states"].index(k)
        SP = self.SP[k]

        o_space_low = self.env_params["o_space"]["low"][i]
        o_space_high = self.env_params["o_space"]["high"][i]

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        r_scale = self.env_params.get("r_scale", {})

        cost += (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2) * r_scale.get(k, 1)
        # cost += np.tanh(100 * (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2)) * r_scale.get(k, 1)

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

def crystallization_reward(self, x, u, con):
    Sp_i = 0
    cost = 0
    R = 0.1
    if not hasattr(self, 'u_prev'):
        self.u_prev = u

    for k in self.SP:
        i = self.model.info()["states"].index(k)
        SP = self.SP[k]

        o_space_low = self.env_params["o_space"]["low"][i]
        o_space_high = self.env_params["o_space"]["high"][i]

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        r_scale = self.env_params.get("r_scale", {})

        cost += (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2) * r_scale.get(k, 1)
        # cost += np.tanh(100 * (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2)) * r_scale.get(k, 1)

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

def four_tank_reward(self, x, u, con):
    Sp_i = 0
    cost = 0
    R = 10
    if not hasattr(self, 'u_prev'):
        self.u_prev = u

    for k in self.SP:
        i = self.model.info()["states"].index(k)
        SP = self.SP[k]

        o_space_low = self.env_params["o_space"]["low"][i]
        o_space_high = self.env_params["o_space"]["high"][i]

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        r_scale = self.env_params.get("r_scale", {})

        # cost += (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2) * r_scale.get(k, 1)
        cost += np.tanh(40 * (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2)) * r_scale.get(k, 1)

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

def photo_production_reward(self, x, u, con):
    cost = 0
    R = np.diag([3.125 * 1e-8, 3.125 * 1e-6])
    if not hasattr(self, 'u_prev'):
        self.u_prev = u

    for k in self.env_params["targets"]:
        i = self.model.info()["states"].index(k)

        o_space_low = self.env_params["o_space"]["low"][i]
        o_space_high = self.env_params["o_space"]["high"][i]

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)

        r_scale = self.env_params.get("r_scale", {})

        cost += 1 - np.tanh(6 * x_normalized)

    # Soft constraint setting by adding penalty term for c_N
    i = self.model.info()["states"].index('c_N')
    cost += max(0, (x[i] - 800)) ** 2 * 1e-6
    # cost += (x[i] - 800) ** 2 * 1e-4

    # Add the control cost
    delta = u - self.u_prev
    cost += delta.T @ R @ delta
    r = -cost
    try:
        return r[0]
    except Exception:
        return r
