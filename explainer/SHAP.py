from explainer.base_explainer import Base_explainer
import os
import copy
import shap
import torch
import pickle
import numpy as np

# %% SHAP module
class SHAP(Base_explainer):
    def __init__(self, model, target, algo, system):
        """
        :argument
            model: [pd.DataFrame] Data to be interpreted
            target: [str] Target variable string
        """
        # TODO: Actor, critic distinguish 반영해야. verbose도 추가하면 좋을 듯
        super(SHAP, self).__init__(model, target, algo, system)
        self.clustered_shap = False


    def explain(self, X, feature_names):
        """
        Computes SHAP values for all features
        :argument
            load_data: [bool] Whether to load the saved SHAP data. If false, compute the SHAP values
        :returns
            self.result.values: [np.ndarray] Matrix containing SHAP values of all features and instances
        """
        print("Waiting for SHAP analysis...", end='')

        self.X = X
        self.feature_names = feature_names

        if isinstance(self.X, np.ndarray):
            self.X = torch.tensor(self.X, dtype = torch.float32)

        self.explainer = shap.DeepExplainer(model=self.model,
                                            data=self.X)

        # self.explainer = shap.KernelExplainer(model=self.model,
        #                                       data=self.X)

        self.explainer.feature_names = feature_names
        self.explainer.masker = None

        # Descaling SHAP values
        self.result = self.explainer(self.X)

        self.result.data = self.descale_X(self.X.numpy())
        self.result.values = self.descale_Uattr(self.result.values.squeeze())
        self.result.feature_names = feature_names
        # self.result.base_value = self.result.base_values[0]
        # self.result.base_values = self.result.base_value * np.ones(shape=(self.result.shape[0],))

        # Saves SHAP values into pickle format
        with open(self.savedir + '/SHAP_{}.pickle'.format(self.target), 'wb') as handle:
            pickle.dump(self.result, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.result.values

    def plot(self, values, visuals, max_display = 10):
        """
        Provides visual aids for the explanation.
        :argument
            visuals: [List] List of visual aids preferred to be drawn.
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
        if 'Bar' in visuals:
            print("Visualizing Bar plots...", end='')
            savename = self.savedir + f'/[{self.target}] Bar.png'
            shap.plots.bar(self.result,
                           # order=feature_order,
                           savename=savename,
                           max_display=max_display
                           )

            if self.clustered_shap:
                for label in self.label_set:
                    savename = self.savedir + f'/[{self.target}] Feature importance_Group_{label}.png'
                    result = self.result.copy()
                    result.values = self.values[label]
                    result.base_values = self.base_values[label]
                    shap.plots.bar(result,
                                   # order=feature_order,
                                   savename=savename,
                                   max_display=max_display
                                   )

        if 'Beeswarm' in visuals:
            print("Visualizing Beeswarm plots...", end='')
            savename = self.savedir + f'/[{self.target}] Beeswarm.png'
            shap.plots.beeswarm(self.result,
                                show=True,
                                # order=feature_order,
                                max_display = max_display,
                                savedir=savename
                                )
            if self.clustered_shap:
                for label in self.label_set:
                    result = self.result.copy()
                    result.data = self.Xs[label]
                    result.values = self.values[label]
                    result.base_values = self.base_values[label]
                    savename = self.savedir + f'/[{self.target}] Group {label} Beeswarm.png'
                    shap.plots.beeswarm(result,
                                        show=True,
                                        # order=feature_order,
                                        xlim=(-0.2, 0.2),
                                        savedir=savename)

        if 'Waterfall' in visuals:
            print("Visualizing Waterfall plots...", end='')
            try:
                for label in self.label_set:
                    for i in range(self.Xs[label].shape[0]):
                        result = self.result.copy()
                        result.data = self.Xs[label][i]
                        result.values = self.values[label][i]
                        group_dir = os.path.join(self.savedir, f'Group_{label}')
                        os.makedirs(group_dir, exist_ok=True)
                        savename = group_dir + f'/Sample_{i}.png'
                        shap.plots.waterfall(result,
                                             show=True,
                                             title=f'Sample_{i}',
                                             savedir=savename)
            except:
                for i in range(100):
                    result = copy.deepcopy(self.result)
                    result.data = self.result.data[i,:]
                    result.values = self.result.values[i]
                    result.base_value = self.result.base_values
                    savename = self.savedir + f'/Sample_{i}.png'
                    shap.plots.waterfall(result,
                                         show=True,
                                         # title=f'Sample_{i}',
                                         # savedir=savename
                                         )

        if 'Force' in visuals:
            print("Visualizing Force plots...", end='')
            for label in self.label_set:
                for i in range(self.Xs[label].shape[0]):
                    result = self.result.copy()
                    result.data = self.Xs[label][i]
                    result.values = self.values[label][i]
                    group_dir = os.path.join(self.savedir, f'Group_{label}')
                    os.makedirs(group_dir, exist_ok=True)
                    savename = group_dir + f'/Sample_{i}.png'
                    shap.plots.force(result,
                                     matplotlib=True,
                                     show=True)

        if 'Decision' in visuals:
            print("Visualizing Decision plots...", end='')
            savename = self.savedir + f'/[{self.target}] Decision.png'
            shap.plots.decision(self.result.base_values,
                                self.result.values,
                                # feature_order=feature_order,
                                feature_display_range=range(20, -1, -1),
                                feature_names=self.explainer.feature_names,
                                title='Groups',
                                # savedir=savename,
                                ignore_warnings=True)
            if self.clustered_shap:
                for label in self.label_set:
                    result = self.result.copy()
                    result.values = self.values[label]
                    result.base_values = self.base_values[label]
                    savename = self.savedir + f'/[{self.target}] Group {label} Decision.png'
                    shap.plots.decision(result.base_value,
                                        result.values,
                                        # feature_order=feature_order,
                                        feature_display_range=range(20, -1, -1),
                                        feature_names=self.explainer.feature_names,
                                        title='Group {}'.format(label),
                                        # savedir=savename,
                                        ignore_warnings=True)

        if 'Scatter' in visuals:
            print("Extracting scatter plot...", end='')
            for i, feature in enumerate(self.feature_names):
                savename = self.savedir + f'/[{self.target}] Scatter_{feature}.png'
                shap.plots.scatter(self.result[:, i],
                                   # savedir=savename,
                                   show=True)

            if self.clustered_shap:
                for label in self.label_set:
                    result = self.result.copy()
                    result.data = self.Xs[label]
                    result.values = self.values[label]
                    result.base_values = self.base_values[label]
                    group_dir = os.path.join(self.savedir, 'Group {}'.format(label))
                    os.makedirs(group_dir, exist_ok=True)
                    for i, feature in enumerate(self.feature_names):
                        savename = group_dir + f'/[{self.target}] Scatter_{feature}.png'
                        shap.plots.scatter(result[:, feature],
                                           # savedir=savename,
                                           show=False)

        if 'Dependence' in visuals:
            print("Extracting Dependence plot...", end='')
            for i, feature in enumerate(self.feature_names):
                savename = self.savedir + f'/[{self.target}] Dependence_{feature}.png'
                shap.dependence_plot(feature,
                                     self.result.values,
                                     self.X,
                                     savedir = savename)

        print("Done!")
