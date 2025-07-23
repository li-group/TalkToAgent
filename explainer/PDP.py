import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from explainer.base_explainer import Base_explainer

# %%
class PDP(Base_explainer):
    def __init__(self, model, bg, feature_names, algo, env_params, grid_points):
        super(PDP, self).__init__(model, bg, feature_names, algo, env_params)
        self.grid_points = grid_points

        # self.predictor = lambda x: self.model.predict(x).reshape(-1)
        self.predictor = lambda x: self.model(torch.tensor(x, dtype=torch.float32)).detach().numpy()

        self.savedir = os.path.join(self.savedir, 'PDP')
        os.makedirs(self.savedir, exist_ok=True)

    def explain(self, X, action = None, features = None, max_samples = 60, device = 'cpu'):
        print("Extracting partial dependence data...")
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.X = self._scale_X(X)
        self.e_features = features if features else self.feature_names
        self.device = device

        ice_curves_all = {}
        if not action:
            for action in self.env_params['actions']:
                ice_curves_all[action] = self._draw_curve(X, action)
        else:
            ice_curves_all[action] = self._draw_curve(X, action)

        return ice_curves_all

    def plot(self, ice_curves_all, cluster_labels = None):
        """
        Plots ICE curves for each feature using data from `explain()`.

        Parameters:
            ice_curves_all (dict): output from self.explain(), mapping feature name → list of ICE curves
        """
        figures = []
        for action in ice_curves_all.keys():
            if cluster_labels is None:
                self.label = ''
                figures.append(self._plot_PDP(ice_curves_all[action], action))
            else:
                label_sets = set(cluster_labels)
                label_sets.remove(-1)  # Remove unclustered data
                for label in label_sets:
                    self.label = label
                    group_index = (cluster_labels == label)
                    ice_curves_group = ice_curves_all[action].copy()
                    for state in ice_curves_all[action].keys():
                        ice_curves_group[state] = np.array(ice_curves_all[action][state])[group_index]
                    figures.append(self._plot_PDP(ice_curves_group, action))
        return figures

    def _plot_PDP(self, ice_curves_all, action = None):
        n_features = len(ice_curves_all)
        fig, axes = plt.subplots(n_features, 1, figsize=(6, 3 * n_features), sharex=False)

        if n_features == 1:
            axes = [axes]  # ensure iterable

        for ax, (feature_name, curves) in zip(axes, ice_curves_all.items()):
            curves = np.array(curves)  # shape: (num_samples, grid_points)

            if curves.shape[0] == 1:
                # Plot single ICE
                ax.plot(self.x_vals[feature_name], curves[0], color='blue', linewidth=2, label='ICE')

            else:
                # Plot ICE curves
                for i, curve in enumerate(curves):
                    label = f'ICEs - Group {self.label}' if i == 0 else None
                    ax.plot(self.x_vals[feature_name], curve, color='gray', alpha=0.4, label=label)

                # Plot PDP (mean)
                mean_curve = np.mean(curves, axis=0)
                ax.plot(self.x_vals[feature_name], mean_curve, color='red', linewidth=2, label='PDP (mean)')

            ax.set_title(f'ICE Curves of Feature {feature_name} to Action {action}')
            ax.set_xlabel(f'{feature_name}')
            ax.set_ylabel(f'{action} Output')
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        savedir = self.savedir + f'/[{action}]{self.label} PDP.png'
        plt.savefig(savedir)
        plt.show()
        return fig

    def _draw_curve(self, X, action):
        _x_vals = {}
        ice_curves_dict = {}
        action_index = self.env_params['actions'].index(action)

        for i, feature in enumerate(self.e_features):
            x_vals = np.linspace(self.env_params['o_space']['low'], self.env_params['o_space']['high'], self.grid_points)
            ice_curves = []

            for sample_idx in range(X.shape[0]):
                x_sample = X[sample_idx].copy()
                preds = []
                for val in x_vals.T[i]:
                    x_sample[i] = val
                    x = self._scale_X(x_sample)
                    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        y = self._descale_U(self.model(x_tensor).detach().numpy()).squeeze()

                    if len(self.env_params['actions']) == 1:
                        preds.append(y)
                    else:
                        preds.append(y[action_index]) # Only append relevant actions
                ice_curves.append(preds)
            ice_curves_dict[feature] = ice_curves
            _x_vals[feature] = x_vals[:,i]
        self.x_vals = _x_vals

        return ice_curves_dict