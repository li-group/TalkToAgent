import os
import shap
import torch
import pickle
import numpy as np

# %% SHAP module
current_dir = os.getcwd()
figure_dir = os.path.join(current_dir, 'figures')


class SHAP:
    def __init__(self, model, bg, feature_names, algo, env_params):
        """
        Args:
            model (nn.Sequential): DNN model to be interpreted
            bg (np.ndarray): Background data to be compared with explicands
            feature_names (list): List of feature (state) names
            algo (str): Name of the RL algorithm being used
            env_params (dict): Environment parameters
        """
        self.model = model
        self.feature_names = feature_names
        self.algo = algo
        self.env_params = env_params
        self.o_space = env_params['o_space']
        self.a_space = env_params['a_space']
        system = env_params['model']
        self.savedir = os.path.join(figure_dir, f'[{algo}][{system}]')
        os.makedirs(self.savedir, exist_ok=True)
        self.bg = self._scale_X(bg)

        self.device = next(model.parameters()).device

        if isinstance(self.bg, np.ndarray):
            self.bg = torch.tensor(self.bg, dtype = torch.float32).to(self.device)

        self.explainer = shap.DeepExplainer(model=self.model, data=self.bg)
        self.explainer.feature_names = feature_names
        self.explainer.masker = None

        self.savedir = os.path.join(self.savedir, 'SHAP')
        os.makedirs(self.savedir, exist_ok=True)

    def explain(self, X):
        """
        Computes SHAP values for all features
        Args:
            X (np.ndarray): Single instance (local) or multiple instances (global) to be explained
        Return:
            shap_values: [np.ndarray] Matrix containing SHAP values of all features and instances
        """
        print("Waiting for SHAP analysis...", end='')

        self.X = self._scale_X(X)
        if isinstance(self.X, np.ndarray):
            self.X = torch.tensor(self.X, dtype = torch.float32)

        mean_prediction = np.array(
            self.model(torch.tensor(self.bg, dtype=torch.float32)).detach().cpu().numpy().mean(axis=0)) # If keepdims=True -> (1,1)

        # Obtain, then Descale SHAP values
        self.result = self.explainer(self.X)
        self.result.data = self._descale_X(self.X.numpy()) # shape: (instance, states)
        self.result.values = self._descale_Uattr(self.result.values) # shape: (instance, states, actions)
        self.result.feature_names = self.feature_names
        self.result.base_values = self._descale_U(mean_prediction).reshape(-1)

        # Saves SHAP values into pickle format
        with open(self.savedir + '/SHAP_values.pickle', 'wb') as handle:
            pickle.dump(self.result, handle, protocol=pickle.HIGHEST_PROTOCOL)

        shap_values = self.result.values
        return shap_values

    def plot(self, local, actions = None, max_display = 10):
        """
        Provides visual aids for the explanation.
        Args:
            local: [bool] Whether to visualize local explanations
            actions: [list] List of agent actions to be explained
            max_display: [int] Maximum number of features to display
        Additional Info (Types of visualizations):
            Waterfall: Absolute values and directions of attributions (local)
            Bar: Mean absolute values of attributions for every feature (global)
            Beeswarm: Absolute values and directions of attributions (global)
            Decision: Directions of attributions (global)
        """
        def _plot_result(result, figures, action):
            if local:
                print("Plots for local explanations: Waterfall plots")
                fig = self._plot_waterfall(result, action=action)
                figures.append(fig)

            else:
                print("Plots for global explanations: Bar, Beeswarm and Decision plots")
                fig_bar = self._plot_bar(result, action=action)
                fig_bee = self._plot_beeswarm(result, action=action)
                fig_dec = self._plot_decision(result, action=action)
                figures.extend([fig_bar, fig_bee, fig_dec])

        figures = []
        if not actions:
            # If actions not specified by LLM, we extract figures for all agent actions.
            for a in self.env_params['actions']:
                result = self.result[:, :, self.env_params['actions'].index(a)]
                result.base_values = result.base_values[self.env_params['actions'].index(a)]
                _plot_result(result, figures, a)
        else:
            for a in actions:
                result = self.result[:, :, self.env_params['actions'].index(a)]
                result.base_values = result.base_values[self.env_params['actions'].index(a)]
                _plot_result(result, figures, a)
        print("Done!")
        return figures

    def _plot_waterfall(self, result, action=''):
        for i in range(len(result)):
            savename = self.savedir + f'/[{action}] Waterfall.png'
            fig = shap.plots.waterfall(result[i],
                                       show=True,
                                       savedir=savename,
                                       # title=f'Agent action: {action}'
                                       )
            return fig

    def _plot_bar(self, result, max_display = 10, action=''):
        savename = self.savedir + f'/[{action}] Bar.png'
        fig = shap.plots.bar(result,
                             savedir=savename,
                             max_display=max_display,
                             # title= f'Agent action: {action}'
                             )
        return fig

    def _plot_beeswarm(self, result, max_display = 10, action=''):
        savename = self.savedir + f'/[{action}] Beeswarm.png'
        fig = shap.plots.beeswarm(result,
                                  show=True,
                                  max_display=max_display,
                                  savedir=savename,
                                  # title= f'Agent action: {action}'
                                  )
        return fig

    def _plot_decision(self, result, action=''):
        savename = self.savedir + f'/[{action}] Decision.png'
        fig = shap.plots.decision(result.base_values,
                                  result.values,
                                  feature_display_range=range(20, -1, -1),
                                  feature_names=self.explainer.feature_names,
                                  # title=f'Agent action: {action}',
                                  savedir=savename,
                                  ignore_warnings=True,
                                  return_objects=True)
        return fig

    # Scaling and descaling functions for explanations
    def _scale_X(self, X):
        low = self.o_space['low'][np.newaxis, :]
        high = self.o_space['high'][np.newaxis, :]
        X_scaled = 2 * (X - low) / (high - low) - 1
        return X_scaled

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
