import os
import numpy as np

current_dir = os.getcwd()
figure_dir = os.path.join(current_dir, 'figures')
os.makedirs(figure_dir, exist_ok=True)

# %% SHAP module
class Base_explainer:
    def __init__(self, model, bg, feature_names, target, algo, env_params):
        """
        :argument
            model: [pd.DataFrame] Data to be interpreted
            bg: Background data
            target: [str] Target variable string
        """
        # TODO: Actor, critic distinguish 반영해야. verbose도 추가하면 좋을 듯
        self.model = model
        self.feature_names = feature_names
        self.target = target
        self.algo = algo
        self.env_params = env_params
        self.o_space = env_params['o_space']
        self.a_space = env_params['a_space']
        system = env_params['model']
        self.savedir = os.path.join(figure_dir, f'[{algo}][{system}]')
        os.makedirs(self.savedir, exist_ok=True)

        self.bg = bg
        self.bg = self._scale_X(bg)

        # if self.model == 'DDPG':
        #     print(f"Interpreting actor network, which outputs deterministic output from observations")
        # elif self.model == 'SAC':
        #     print(f"Interpreting actor network, which outputs deterministic output from observations")
        # else:
        #     print(f"Interpreting value network, which outputs each Q value from observations")

    def explain(self, X):
        pass

    def plot(self, values, max_display=10):
        pass

    def _scale_X(self, X):
        low = self.o_space['low'][np.newaxis, :]
        high = self.o_space['high'][np.newaxis, :]
        X_scaled = 2 * (X - low) / (high - low) - 1
        return X_scaled

    def _scale_U(self, U):
        low = self.a_space['low'][np.newaxis, :]
        high = self.a_space['high'][np.newaxis, :]
        U_scaled = 2 * (U - low) / (high - low) - 1
        return U_scaled

    def _descale_X(self, X_scaled):
        low = self.o_space['low'][np.newaxis, :]
        high = self.o_space['high'][np.newaxis, :]
        return (high - low) * (X_scaled + 1) / 2 + low

    def _descale_U(self, U_scaled):
        low = self.a_space['low'][np.newaxis, :]
        high = self.a_space['high'][np.newaxis, :]
        return (high - low) * (U_scaled + 1) / 2 + low

    def _descale_Uattr(self, U_scaled):
        low = self.a_space['low'][np.newaxis, :]
        high = self.a_space['high'][np.newaxis, :]
        return (high - low) * (U_scaled) / 2
