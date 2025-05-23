from explainer.base_explainer import Base_explainer
import os
import copy
import shap
import torch
import pickle
import numpy as np
from copy import deepcopy

# %% SHAP module
class SHAP(Base_explainer):
    def __init__(self, model, bg, feature_names, target, algo, env_params):
        """
        :argument
            model: [pd.DataFrame] Data to be interpreted
            target: [str] Target variable string
        """
        super(SHAP, self).__init__(model, bg, feature_names, target, algo, env_params)
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
            feature_names: [list] List of feature names (state variables)
        :returns
            self.result.values: [np.ndarray] Matrix containing SHAP values of all features and instances
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
        with open(self.savedir + '/SHAP_{}.pickle'.format(self.target), 'wb') as handle:
            pickle.dump(self.result, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.result.values

    def plot(self, values, max_display = 10, cluster_labels = None):
        """
        Provides visual aids for the explanation.
        :argument
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
        if self.result.shape[0] == 1 or self.result.data.ndim == 1:
            print("Plots for local explanations: Waterfall and Force plots")
            self._plot_waterfall()
            self._plot_force()

        else:
            print("Plots for global explanations: Bar, Beeswarm and Decision plots")
            if cluster_labels is None:
                self.label = ''
                self._plot_bar()
                self._plot_beeswarm()
                self._plot_decision()
            else:
                global_result = deepcopy(self.result)
                label_sets = set(cluster_labels)
                label_sets.remove(-1) # Remove unclustered data
                for label in label_sets:
                    self.label = label
                    group_index = (cluster_labels == label)
                    self.result.data = global_result.data[group_index]
                    self.result.values = global_result.values[group_index]
                    self._plot_bar()
                    self._plot_beeswarm()
                    self._plot_decision()
        print("Done!")

    def _plot_waterfall(self):
        for i in range(len(self.result)):
            savename = self.savedir + f'/[{self.target}] Waterfall.png'
            shap.plots.waterfall(self.result[i],
                                 show=True,
                                 # title=f'Sample_{i}',
                                 savedir=savename
                                 )

    def _plot_force(self):
        for i in range(len(self.result)):
            shap.plots.force(self.result[i],
                             matplotlib=True,
                             show=True
                             )

    def _plot_bar(self, max_display = 10):
        savename = self.savedir + f'/[{self.target}]{self.label} Bar.png'
        shap.plots.bar(self.result,
                       # order=feature_order,
                       savedir=savename,
                       max_display=max_display
                       )

    def _plot_beeswarm(self, max_display = 10):
        savename = self.savedir + f'/[{self.target}]{self.label} Beeswarm.png'
        shap.plots.beeswarm(self.result,
                            show=True,
                            # order=feature_order,
                            max_display=max_display,
                            savedir=savename
                            )

    def _plot_decision(self, max_display = 10):
        savename = self.savedir + f'/[{self.target}]{self.label} Decision.png'
        shap.plots.decision(self.result.base_values,
                            self.result.values,
                            # feature_order=feature_order,
                            feature_display_range=range(20, -1, -1),
                            feature_names=self.explainer.feature_names,
                            title='Groups',
                            savedir=savename,
                            ignore_warnings=True)

    def _plot_scatter(self):
        for i, feature in enumerate(self.feature_names):
            savename = self.savedir + f'/[{self.target}]{self.label} Scatter_{feature}.png'
            shap.plots.scatter(self.result[:, i],
                               savedir=savename,
                               show=True)

    def _plot_dependence(self):
        for i, feature in enumerate(self.feature_names):
            savename = self.savedir + f'/[{self.target}]{self.label} Dependence_{feature}.png'
            shap.dependence_plot(feature,
                                 self.result.values,
                                 self.X,
                                 savedir = savename)
