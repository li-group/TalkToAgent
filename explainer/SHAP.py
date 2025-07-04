from explainer.base_explainer import Base_explainer
import os
import shap
import torch
import pickle
import numpy as np
from copy import deepcopy

# %% SHAP module
class SHAP(Base_explainer):
    def __init__(self, model, bg, feature_names, algo, env_params):
        """
        Args:
            model: [pd.DataFrame] Data to be interpreted
        """
        super(SHAP, self).__init__(model, bg, feature_names, algo, env_params)
        self.clustered_shap = False

        if isinstance(self.bg, np.ndarray):
            self.bg = torch.tensor(self.bg, dtype = torch.float32)

        self.explainer = shap.DeepExplainer(model=self.model,
                                            data=self.bg)

        self.explainer.feature_names = feature_names
        self.explainer.masker = None

        self.savedir = os.path.join(self.savedir, 'SHAP')
        os.makedirs(self.savedir, exist_ok=True)


    def explain(self, X):
        """
        Computes SHAP values for all features
        Args:
            X: Single instance (local) or multiple instances (global)
        Return:
            shap_values: [np.ndarray] Matrix containing SHAP values of all features and instances
        """
        print("Waiting for SHAP analysis...", end='')

        self.X = self._scale_X(X)
        if isinstance(self.X, np.ndarray):
            self.X = torch.tensor(self.X, dtype = torch.float32)

        # Descaling SHAP values
        self.result = self.explainer(self.X)

        self.result.data = self._descale_X(self.X.numpy()) # shape: (instance, states)
        self.result.values = self._descale_Uattr(self.result.values) # shape: (instance, states, actions)
        self.result.feature_names = self.feature_names
        mean_prediction = np.array(self.model(torch.tensor(X, dtype=torch.float32)).detach().numpy().mean(axis=0)) # If keepdims=True -> (1,1)
        self.result.base_values = self._descale_U(mean_prediction).reshape(-1)

        # Saves SHAP values into pickle format
        with open(self.savedir + '/SHAP_values.pickle', 'wb') as handle:
            pickle.dump(self.result, handle, protocol=pickle.HIGHEST_PROTOCOL)

        shap_values = self.result.values
        return shap_values

    def plot(self, local, action = None, max_display = 10, cluster_labels = None):
        """
        Provides visual aids for the explanation.
        Args:
            local: [bool] Whether to visualize local explanations
            action: [str] Agent action to be explained
            max_display: [int] Maximum number of features to display
        Additional Info (Types of visualizations):
            Bar: Mean absolute values of attributions for every feature (global)
            Beeswarm: Absolute values and directions of attributions (global)
            Waterfall: Absolute values and directions of attributions (local)
            Force: Absolute values and directions of attributions (local)
            Decision: Directions of attributions (global)
            Scatter: Attributions against feature values (global)
            Dependence: Attributions against feature values, colored by other feature values(global)
        """
        def _plot_result(result, figures):
            if local:
                print("Plots for local explanations: Waterfall and Force plots")
                fig = self._plot_waterfall(result, action=action)
                self._plot_force(result, action=action)
                figures.append(fig)

            else:
                print("Plots for global explanations: Bar, Beeswarm and Decision plots")
                if not cluster_labels:
                    self.label = ''
                    fig_bar = self._plot_bar(result, action=action)
                    fig_bee = self._plot_beeswarm(result, action=action)
                    fig_dec = self._plot_decision(result, action=action)
                    figures.extend([fig_bar, fig_bee, fig_dec])
                else:
                    result = deepcopy(self.result)
                    label_sets = set(cluster_labels)
                    label_sets.remove(-1) # Remove unclustered data
                    for label in label_sets:
                        self.label = label
                        group_index = (cluster_labels == label)
                        result.data = self.result.data[group_index]
                        result.values = self.result.values[group_index]
                        fig_bar = self._plot_bar(result, action=action)
                        fig_bee = self._plot_beeswarm(result, action=action)
                        fig_dec = self._plot_decision(result, action=action)
                # return fig_bar, fig_bee, fig_dec

        figures = []
        if not action:
            # If action not specified by LLM, we extract figures for all agent actions.
            for action in self.env_params['actions']:
                result = self.result[:, :, self.env_params['actions'].index(action)]
                result.base_values = result.base_values[self.env_params['actions'].index(action)]
                _plot_result(result, figures)
        else:
            result = self.result[:, :, self.env_params['actions'].index(action)]
            result.base_values = result.base_values[self.env_params['actions'].index(action)]
            _plot_result(result, figures)
        print("Done!")
        return figures

    def _plot_waterfall(self, result, action=''):
        for i in range(len(result)):
            savename = self.savedir + f'/[{action}] Waterfall.png'
            fig = shap.plots.waterfall(result[i],
                                       show=True,
                                       savedir=savename,
                                       title=f'Agent action: {action}'
                                       )
            return fig

    def _plot_force(self, result, action=''):
        for i in range(len(result)):
            shap.plots.force(result[i],
                             matplotlib=True,
                             show=True
                             )

    def _plot_bar(self, result, max_display = 10, action=''):
        savename = self.savedir + f'/[{action}]{self.label} Bar.png'
        fig = shap.plots.bar(result,
                             # order=feature_order,
                             savedir=savename,
                             max_display=max_display,
                             title= f'Agent action: {action}'
                             )
        return fig

    def _plot_beeswarm(self, result, max_display = 10, action=''):
        savename = self.savedir + f'/[{action}]{self.label} Beeswarm.png'
        fig = shap.plots.beeswarm(result,
                                  show=True,
                                  # order=feature_order,
                                  max_display=max_display,
                                  savedir=savename,
                                  title= f'Agent action: {action}'
                                  )
        return fig

    def _plot_decision(self, result, max_display = 10, action=''):
        savename = self.savedir + f'/[{action}]{self.label} Decision.png'
        fig = shap.plots.decision(result.base_values,
                                  result.values,
                                  # feature_order=feature_order,
                                  feature_display_range=range(20, -1, -1),
                                  feature_names=self.explainer.feature_names,
                                  title=f'Agent action: {action}',
                                  savedir=savename,
                                  ignore_warnings=True,
                                  return_objects=True)
        return fig

    def _plot_scatter(self, result, action=''):
        for i, feature in enumerate(self.feature_names):
            savename = self.savedir + f'/[{action}]{self.label} Scatter_{feature}.png'
            shap.plots.scatter(result[:, i],
                               savedir=savename,
                               show=True)

    def _plot_dependence(self, result, action=''):
        for i, feature in enumerate(self.feature_names):
            savename = self.savedir + f'/[{action}]{self.label} Dependence_{feature}.png'
            shap.dependence_plot(feature,
                                 result.values,
                                 self.X,
                                 savedir = savename)
