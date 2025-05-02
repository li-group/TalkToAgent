from explainer.base_explainer import Base_explainer
import pandas as pd
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
import numpy as np

import torch

# %%

class PDP(Base_explainer):
    def __init__(self, model, bg, feature_names, target, algo, env_params, grid_points):
        super(PDP, self).__init__(model, bg, feature_names, target, algo, env_params)
        self.grid_points = grid_points

        # self.predictor = lambda x: self.model.predict(x).reshape(-1)
        self.predictor = lambda x: self.model(torch.tensor(x, dtype=torch.float32)).detach().numpy()

    def explain(self, X, feature = None, max_samples = 60, device = 'cpu'):
        print("Extracting partial dependence data...")
        self.X = self._scale_X(X)
        self.e_features = feature if feature is not None else self.feature_names
        self.x_vals = {}
        ice_curves_all = {}

        for i, feature in enumerate(self.e_features):
            x_vals = np.linspace(self.env_params['o_space']['low'], self.env_params['o_space']['high'], self.grid_points)
            x_vals = self._scale_X(x_vals)
            ice_curves = []

            for sample_idx in range(X.shape[0]):
                x_sample = X[sample_idx].copy()
                preds = []
                for val in x_vals.T[i]:
                    x_sample[i] = val
                    x_tensor = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        y = self._descale_U(self.model(x_tensor).detach().numpy())
                    preds.append(y.item())
                ice_curves.append(preds)
            ice_curves_all[feature] = ice_curves
            # self.x_vals[feature] = self._descale_X(x_vals)
            self.x_vals[feature] = x_vals

        return ice_curves_all


    def plot(self, ice_curves_all):
        """
        Plots ICE curves for each feature using data from `explain()`.

        Parameters:
            ice_curves_all (dict): output from self.explain(), mapping feature name → list of ICE curves
        """
        n_features = len(ice_curves_all)
        fig, axes = plt.subplots(n_features, 1, figsize=(8, 4 * n_features), sharex=False)

        if n_features == 1:
            axes = [axes]  # ensure iterable

        for ax, (feature_name, curves) in zip(axes, ice_curves_all.items()):
            curves = np.array(curves)  # shape: (num_samples, grid_points)

            # Plot ICE curves
            for curve in curves:
                ax.plot(self.x_vals[feature_name], curve, color='gray', alpha=0.4)

            # Plot PDP (mean)
            mean_curve = np.mean(curves, axis=0)
            ax.plot(self.x_vals[feature_name], mean_curve, color='red', linewidth=2, label='PDP (mean)')

            ax.set_title(f'ICE Curves for Feature: {feature_name}')
            ax.set_xlabel(f'{feature_name}')
            ax.set_ylabel('Model Output')
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()



def plot_pdp_ice(model, X, feature_index, grid_points=20, ice_samples=10, device='cpu'):
    """
    Plots PDP and ICE for a given torch model and one input feature.

    Parameters:
        model (torch.nn.Module): Trained PyTorch model
        X (np.ndarray or torch.Tensor): Input data of shape (N, D)
        feature_index (int): Which feature to analyze
        grid_points (int): Number of points to evaluate the feature at
        ice_samples (int): Number of individual ICE curves to plot
        device (str): 'cpu' or 'cuda'
    """
    model.eval()
    model.to(device)

    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    N, D = X.shape

    # Prepare evaluation grid for the selected feature
    x_vals = np.linspace(X[:, feature_index].min(), X[:, feature_index].max(), grid_points)

    # Pick ICE samples from data
    ice_indices = np.random.choice(N, size=ice_samples, replace=False)
    ice_curves = []

    for idx in ice_indices:
        x_sample = X[idx].copy()
        preds = []
        for val in x_vals:
            x_sample[feature_index] = val
            x_tensor = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                y = model(x_tensor)
            preds.append(y.item())
        ice_curves.append(preds)

    # PDP is just the average across many samples
    pdp_preds = []
    for val in x_vals:
        X_modified = X.copy()
        X_modified[:, feature_index] = val
        x_tensor = torch.tensor(X_modified, dtype=torch.float32).to(device)
        with torch.no_grad():
            y = model(x_tensor).cpu().numpy()
        pdp_preds.append(np.mean(y))

    # Plotting
    plt.figure(figsize=(8, 6))

    # ICE
    for curve in ice_curves:
        plt.plot(x_vals, curve, color='gray', alpha=0.4)

    # PDP
    plt.plot(x_vals, pdp_preds, color='red', linewidth=2, label='PDP')

    plt.title(f'PDP and ICE for feature {feature_index}')
    plt.xlabel(f'Feature {feature_index}')
    plt.ylabel('Model output')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()