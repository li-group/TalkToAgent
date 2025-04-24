import os
import numpy as np

current_dir = os.getcwd()
figure_dir = os.path.join(current_dir, 'figures')
os.makedirs(figure_dir, exist_ok=True)

# %% SHAP module
class Base_explainer:
    def __init__(self, model, target, algo, system):
        """
        :argument
            model: [pd.DataFrame] Data to be interpreted
            target: [str] Target variable string
        """
        # TODO: Actor, critic distinguish 반영해야. verbose도 추가하면 좋을 듯
        self.model = model
        self.target = target
        self.algo = algo
        self.system = system
        self.savedir = os.path.join(figure_dir, f'[{algo}][{system}]')
        os.makedirs(self.savedir, exist_ok=True)

        # if self.model == 'DDPG':
        #     print(f"Interpreting actor network, which outputs deterministic output from observations")
        # elif self.model == 'SAC':
        #     print(f"Interpreting actor network, which outputs deterministic output from observations")
        # else:
        #     print(f"Interpreting value network, which outputs each Q value from observations")

    def explain(self, X, feature_names):
        pass

    def plot(self, values, max_display=10):
        pass

    def scale(self, X, U, observation_space, action_space):
        self.observation_space = observation_space
        low = self.observation_space['low'][np.newaxis, :]
        high = self.observation_space['high'][np.newaxis, :]
        X_scaled = 2 * (X - low) / (high - low) - 1

        self.action_space = action_space
        low = self.action_space['low'][np.newaxis, :]
        high = self.action_space['high'][np.newaxis, :]
        U_scaled = 2 * (U - low) / (high - low) - 1
        return X_scaled, U_scaled

    def descale_X(self, X_scaled):
        low = self.observation_space['low'][np.newaxis, :]
        high = self.observation_space['high'][np.newaxis, :]
        return (high - low) * (X_scaled + 1) / 2 + low

    def descale_U(self, U_scaled):
        low = self.action_space['low'][np.newaxis, :]
        high = self.action_space['high'][np.newaxis, :]
        return (high - low) * (U_scaled + 1) / 2 + low

    def descale_Uattr(self, U_scaled):
        low = self.action_space['low'][np.newaxis, :]
        high = self.action_space['high'][np.newaxis, :]
        return (high - low) * (U_scaled) / 2
