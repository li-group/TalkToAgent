from explainer.base_explainer import Base_explainer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import torch

class Interestingness():
    def __init__(self, model, data, env):

        pass

    def explaine(self):

    def get_Q(self, data):
        critic = DDPG_CSTR.critic.qf0
        X = data['DDPG']['x']
        X = X.reshape(X.shape[0], -1).T
        U = data['DDPG']['u']
        U = U.reshape(U.shape[0], -1).T
        X, U = env._scale_X(X), env._scale_U(U)
        XU = np.hstack([X, U])
        critic(torch.tensor(np.array(XU), dtype=torch.float32))
        pass

    def plot(self, values, max_display = 10):
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
