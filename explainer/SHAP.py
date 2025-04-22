import os
import copy
import shap
import pickle
import numpy as np

# %% SHAP module
class SHAP:
    def __init__(self, model, target, config):
        """
        :argument
            model: [pd.DataFrame] Data to be interpreted
            target: [str] Target variable string
        """
        # TODO: Actor, critic distinguish 반영해야. verbose도 추가하면 좋을 듯
        self.model = model
        self.target = target
        self.config = config
        self.clustered_shap = False

        save_path = config['save_path']
        self.shap_path = os.path.join(save_path, 'shap')
        os.makedirs(self.shap_path, exist_ok=True)


    def __call__(self, X, load_data = False):
        """
        Computes SHAP values for all features
        :argument
            load_data: [bool] Whether to load the saved SHAP data. If false, compute the SHAP values
        :returns
            self.result.values: [np.ndarray] Matrix containing SHAP values of all features and instances
        """
        print("Waiting for SHAP analysis...", end='')

        self.X = X

        if load_data:
            with open(self.shap_path + '/SHAP_{}.pickle'.format(self.target), 'rb') as handle:
                self.result =pickle.load(handle)

        else:
            # self.explainer = shap.DeepExplainer(model=self.model,
            #                                     data=self.X)

            self.explainer = shap.KernelExplainer(model=self.model,
                                                  data=self.X)
            self.explainer.feature_names = [f'Feature {i}' for i in range(self.X.shape[1])]
            self.explainer.masker = None

            # Descaling SHAP values
            self.result = self.explainer(self.X)
            self.result.data = self.X
            self.result.values = self.result.values
            self.result.base_value = self.result.base_values[0]
            self.result.base_values = self.result.base_value * np.ones(shape=(self.result.shape[0],))

            # Saves SHAP values into pickle format
            with open(self.shap_path + '/SHAP_{}.pickle'.format(self.target), 'wb') as handle:
                pickle.dump(self.result, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.result.values

    def SHAP_plot(self, visuals, add_info = '', max_display = 10):
        """
        Provides visual aids for the explanation.
        :argument
            visuals: [List] List of visual aids preferred to be drawn.
            add_info: [str] Additional information for saving plots
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
        savename = self.shap_path + f'/[{self.target}] Bar{add_info}.png'
        feature_order = shap.plots.bar(self.result,
                                       savename=savename,
                                       max_display = max_display,
                                       )

        if 'Bar' in visuals:
            print("Visualizing Bar plots...", end='')
            savename = self.shap_path + f'/[{self.target}] Bar{add_info}.png'
            shap.plots.bar(self.result,
                           order=feature_order,
                           savename=savename,
                           max_display=max_display
                           )

            if self.clustered_shap:
                for label in self.label_set:
                    savename = self.shap_path + f'/[{self.target}] Feature importance_Group_{label}{add_info}.png'
                    result = self.result.copy()
                    result.values = self.values[label]
                    result.base_values = self.base_values[label]
                    shap.plots.bar(result,
                                   order=feature_order,
                                   savename=savename,
                                   max_display=max_display
                                   )

        if 'Beeswarm' in visuals:
            print("Visualizing Beeswarm plots...", end='')
            savename = self.shap_path + f'/[{self.target}] Beeswarm{add_info}.png'
            shap.plots.beeswarm(self.result,
                                show=True,
                                order=feature_order,
                                max_display = max_display,
                                savedir=savename
                                )
            if self.clustered_shap:
                for label in self.label_set:
                    result = self.result.copy()
                    result.data = self.Xs[label]
                    result.values = self.values[label]
                    result.base_values = self.base_values[label]
                    savename = self.shap_path + f'/[{self.target}] Group {label} Beeswarm{add_info}.png'
                    shap.plots.beeswarm(result,
                                        show=True,
                                        order=feature_order,
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
                        group_dir = os.path.join(self.shap_path, f'Group_{label}')
                        os.makedirs(group_dir, exist_ok=True)
                        savename = group_dir + f'/Sample_{i}{add_info}.png'
                        shap.plots.waterfall(result,
                                             show=True,
                                             title=f'Sample_{i}',
                                             savedir=savename)
            except:
                for i in range(100):
                    result = copy.deepcopy(self.result)
                    result.data = self.X.iloc[i,:]
                    result.values = self.result.values[i]
                    result.base_value = self.result.base_value[i]
                    result.base_values = self.result.base_values[i]
                    savename = self.shap_path + f'/Sample_{i}{add_info}.png'
                    shap.plots.waterfall(result,
                                         show=True,
                                         title=f'Sample_{i}',
                                         savedir=savename)

        if 'Force' in visuals:
            print("Visualizing Force plots...", end='')
            for label in self.label_set:
                for i in range(self.Xs[label].shape[0]):
                    result = self.result.copy()
                    result.data = self.Xs[label][i]
                    result.values = self.values[label][i]
                    group_dir = os.path.join(self.shap_path, f'Group_{label}')
                    os.makedirs(group_dir, exist_ok=True)
                    savename = group_dir + f'/Sample_{i}{add_info}.png'
                    shap.plots.force(result,
                                     matplotlib=True,
                                     show=True)

        if 'Decision' in visuals:
            print("Visualizing Decision plots...", end='')
            savename = self.shap_path + f'/[{self.target}] Decision{add_info}.png'
            shap.plots.decision(self.result.base_value[0],
                                self.result.values,
                                feature_order=feature_order,
                                feature_display_range=range(20, -1, -1),
                                feature_names=self.X.columns.tolist(),
                                title='Groups',
                                savedir=savename,
                                ignore_warnings=True)
            if self.clustered_shap:
                for label in self.label_set:
                    result = self.result.copy()
                    result.values = self.values[label]
                    result.base_values = self.base_values[label]
                    savename = self.shap_path + f'/[{self.target}] Group {label} Decision{add_info}.png'
                    shap.plots.decision(result.base_value,
                                        result.values,
                                        feature_order=feature_order,
                                        feature_display_range=range(20, -1, -1),
                                        feature_names=self.X.columns.tolist(),
                                        title='Group {}'.format(label),
                                        savedir=savename,
                                        ignore_warnings=True)

        if 'Scatter' in visuals:
            print("Extracting scatter plot...", end='')
            for i, feature in enumerate(self.X.columns):
                savename = self.shap_path + f'/[{self.target}] Scatter_{feature}{add_info}.png'
                shap.plots.scatter(self.result[:, i],
                                   savedir=savename,
                                   show=True)

            if self.clustered_shap:
                for label in self.label_set:
                    result = self.result.copy()
                    result.data = self.Xs[label]
                    result.values = self.values[label]
                    result.base_values = self.base_values[label]
                    group_dir = os.path.join(self.shap_path, 'Group {}'.format(label))
                    os.makedirs(group_dir, exist_ok=True)
                    for i, feature in enumerate(self.X.columns):
                        savename = group_dir + f'/[{self.target}] Scatter_{feature}{add_info}.png'
                        shap.plots.scatter(result[:, feature],
                                           savedir=savename,
                                           show=False)

        if 'Dependence' in visuals:
            print("Extracting Dependence plot...", end='')
            for i, feature in enumerate(self.X.columns):
                savename = self.shap_path + f'/[{self.target}] Dependence_{feature}{add_info}.png'
                shap.dependence_plot(feature,
                                     self.result.values,
                                     self.X,
                                     savedir = savename)

        print("Done!")
