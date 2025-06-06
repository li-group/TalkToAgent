from explainer.base_explainer import Base_explainer
import os
import copy
import shap
import torch
import pickle
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# %% SHAP module
class SHAP(Base_explainer):
    def __init__(self, model, bg, feature_names, algo, env_params):
        """
        :argument
            model: [pd.DataFrame] Data to be interpreted
        """
        super(SHAP, self).__init__(model, bg, feature_names, algo, env_params)
        self.clustered_shap = False

        if isinstance(self.bg, np.ndarray):
            self.bg = torch.tensor(self.bg, dtype = torch.float32)

        self.explainer = shap.DeepExplainer(model=self.model,
                                            data=self.bg)
        # self.explainer = shap.KernelExplainer(model=self.model,
        #                                       data=self.bg)

        self.explainer.feature_names = feature_names
        self.explainer.masker = None

        self.savedir = os.path.join(self.savedir, 'SHAP')
        os.makedirs(self.savedir, exist_ok=True)


    def explain(self, X):
        """
        Computes SHAP values for all features
        :argument
            X: Single instance (local) or multiple instances (global)
        :returns
            shap_values: [np.ndarray] Matrix containing SHAP values of all features and instances
        """
        print("Waiting for SHAP analysis...", end='')

        self.X = self._scale_X(X)
        if isinstance(self.X, np.ndarray):
            self.X = torch.tensor(self.X, dtype = torch.float32)

        # Descaling SHAP values
        self.result = self.explainer(self.X)

        self.result.data = self._descale_X(self.X.numpy())
        self.result.values = self._descale_Uattr(self.result.values.squeeze())
        self.result.feature_names = self.feature_names
        mean_prediction = np.array(self.model(torch.tensor(X, dtype=torch.float32)).detach().numpy().mean())
        self.result.base_values = np.float32(self._descale_U(mean_prediction).squeeze())

        # Saves SHAP values into pickle format
        with open(self.savedir + '/SHAP_values.pickle', 'wb') as handle:
            pickle.dump(self.result, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Reshaping single output SHAP values into multi output form
        if len(self.env_params['targets']) == 1:
            self.result.values = self.result.values[:,:,np.newaxis]
            self.result.data = self.result.data[:, :, np.newaxis]
            self.result.base_values = self.result.base_values[np.newaxis]

        shap_values = self.result.values
        return shap_values

    def plot(self, local, target = None, max_display = 10, cluster_labels = None):
        """
        Provides visual aids for the explanation.
        :argument
            local: [bool] Whether to visualize local explanations
            target: [str] target action to be explained
            values: [np.ndarray] Shap values
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
                fig = self._plot_waterfall(result)
                self._plot_force(result)
                figures.append(fig)

            else:
                print("Plots for global explanations: Bar, Beeswarm and Decision plots")
                if cluster_labels is None:
                    self.label = ''
                    fig_bar = self._plot_bar(result)
                    fig_bee = self._plot_beeswarm(result)
                    fig_dec = self._plot_decision(result)
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
                        fig_bar = self._plot_bar(result)
                        fig_bee = self._plot_beeswarm(result)
                        fig_dec = self._plot_decision(result)
                # return fig_bar, fig_bee, fig_dec

        figures = []
        if target is None:
            # If target not specified by LLM, we extract figures for all target actions.
            for target in self.env_params['targets']:
                result = self.result[:, :, self.env_params['targets'].index(target)]
                _plot_result(result, figures)
        else:
            result = self.result[:, :, self.env_params['targets'].index(target)]
            _plot_result(result, figures)
        print("Done!")
        return figures

    def _plot_waterfall(self, result, target=''):
        for i in range(len(result)):
            savename = self.savedir + f'/[{target}] Waterfall.png'
            fig = shap.plots.waterfall(result[i],
                                       show=True,
                                       savedir=savename
                                       )
            return fig

    def _plot_force(self, result, target=''):
        for i in range(len(result)):
            shap.plots.force(result[i],
                             matplotlib=True,
                             show=True
                             )

    def _plot_bar(self, result, max_display = 10, target=''):
        savename = self.savedir + f'/[{target}]{self.label} Bar.png'
        fig = shap.plots.bar(result,
                             # order=feature_order,
                             savedir=savename,
                             max_display=max_display
                             )
        return fig

    def _plot_beeswarm(self, result, max_display = 10, target=''):
        savename = self.savedir + f'/[{target}]{self.label} Beeswarm.png'
        fig = shap.plots.beeswarm(result,
                                  show=True,
                                  # order=feature_order,
                                  max_display=max_display,
                                  savedir=savename
                                  )
        return fig

    def _plot_decision(self, result, max_display = 10, target=''):
        savename = self.savedir + f'/[{target}]{self.label} Decision.png'
        fig = shap.plots.decision(result.base_values,
                                  result.values,
                                  # feature_order=feature_order,
                                  feature_display_range=range(20, -1, -1),
                                  feature_names=self.explainer.feature_names,
                                  # title='Groups',
                                  savedir=savename,
                                  ignore_warnings=True,
                                  return_objects=True)
        return fig

    def _plot_scatter(self, result, target=''):
        for i, feature in enumerate(self.feature_names):
            savename = self.savedir + f'/[{target}]{self.label} Scatter_{feature}.png'
            shap.plots.scatter(result[:, i],
                               savedir=savename,
                               show=True)

    def _plot_dependence(self, result, target=''):
        for i, feature in enumerate(self.feature_names):
            savename = self.savedir + f'/[{target}]{self.label} Dependence_{feature}.png'
            shap.dependence_plot(feature,
                                 result.values,
                                 self.X,
                                 savedir = savename)
