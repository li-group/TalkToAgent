import os
import hdbscan
import umap.umap_ as umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

current_dir = os.getcwd()
figure_dir = os.path.join(current_dir, 'figures')
os.makedirs(figure_dir, exist_ok=True)

# %% Params for DR, and clustering
def cluster_params(num_data):
    """
    Adjusts hyperparameters for UMAP and HDBSCAN, based on the size of data
    Args:
        num_data: [int] Size of data
    Returns:
        params: [dict] Hyperparameter dictionary
    """
    return {
        # Hyperparameters for UMAP
        'n_neighbors': int(num_data / 200),
        'min_dist': 0.01,

        # Hyperparameters for TSNE
        'perplexity': 80,
        'learning_rate': 200,
        'n_iter': 1000,

        # Hyperparameters for HDBSCAN
        'min_samples': int(num_data / 85),
        'min_cluster_size': int(num_data / 100),
        }

# %% Functions for dimensionality reduction
class Reducer():
    def __init__(self, params):
        self.params = params
        return
    def reduce(self, X, algo = False):
        if algo == False:
            X_reduced = X
        elif algo == 'PCA':
            X_reduced = self._PCA(X, n_components=2)
        elif algo == 'UMAP':
            X_reduced = self._UMAP(X, n_components=2)
        elif algo == 'TSNE':
            X_reduced = self._TSNE(X, n_components=2)
        else:
            raise Exception(f'{algo} not configured')
        return X_reduced

    def _PCA(self, X, n_components):
        pc_list = []
        for components in range(n_components):
            pc_list.append('PC' + str(components + 1))

        pca = PCA(n_components=n_components)
        mapper = pca.fit(X)
        X_PCA = mapper.transform(X)

        return X_PCA


    def _UMAP(self, X, n_components):
        print("Waiting for UMAP projection...", end='')
        n_neighbors, min_dist = self.params['n_neighbors'], self.params['min_dist']
        print(n_neighbors)

        mapper = umap.UMAP(n_neighbors=n_neighbors,
                           min_dist=min_dist,
                           n_components=n_components,
                           metric='euclidean',
                           random_state=42).fit(X)

        X_umap = mapper.transform(X)
        print("Done!")
        return X_umap

    def _TSNE(self, X, n_components):
        from sklearn.manifold import TSNE
        print("Waiting for t-SNE projection...", end='')
        print(self.params.get('perplexity'))

        tsne = TSNE(n_components=n_components,
                    perplexity=self.params.get('perplexity'),
                    learning_rate=self.params.get('learning_rate'),
                    n_iter=self.params.get('n_iter'),
                    random_state=42)

        X_tsne = tsne.fit_transform(X)
        print("Done!")
        return X_tsne

    def plot_scatter(self, X_reduced, hue=None):
        plt.figure(figsize=(8, 8))
        if hue is None:
            sns.scatterplot(x=X_reduced[:, 0],
                            y=X_reduced[:, 1],
                            color = 'black',
                            alpha=0.8)
        else:
            sns.scatterplot(x=X_reduced[:, 0],
                            y=X_reduced[:, 1],
                            hue=hue,
                            palette="viridis",
                            alpha=0.8)
        plt.legend()
        plt.xlabel('Dim1')
        plt.ylabel('Dim2')
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_scatter_grid(self, X_reduced, hue=None, hue_names=None):
        """
        Plot scatter plots with multiple hues as subplots.

        Parameters:
            X_reduced : np.ndarray of shape (N, 2)
            hue       : np.ndarray or pd.DataFrame of shape (N, K)
            hue_names : list of strings, length K
        """
        if hue is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.scatter(X_reduced[:, 0], X_reduced[:, 1], color='black', alpha=0.8)
            ax.set_xlabel('Dim1')
            ax.set_ylabel('Dim2')
            ax.grid()
            plt.tight_layout()
            plt.show()
            return

        # Ensure hue is DataFrame for easy handling
        if isinstance(hue, np.ndarray):
            hue = pd.DataFrame(hue, columns=hue_names)
        elif isinstance(hue, pd.DataFrame):
            if hue_names is None:
                hue_names = hue.columns.tolist()

        num_hues = hue.shape[1]
        n_cols = 2
        n_rows = int(np.ceil(num_hues / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

        for i, name in enumerate(hue_names):
            row, col = divmod(i, n_cols)
            ax = axes[row][col]
            sns.scatterplot(
                x=X_reduced[:, 0],
                y=X_reduced[:, 1],
                hue=hue[name],
                palette='viridis',
                alpha=0.8,
                ax=ax,
                legend=False
            )
            ax.set_title(f'Hue: {name}')
            ax.set_xlabel('Dim1')
            ax.set_ylabel('Dim2')
            ax.grid(True)

        # Remove empty subplots if hue count < 4
        for j in range(num_hues, n_rows * n_cols):
            row, col = divmod(j, n_cols)
            fig.delaxes(axes[row][col])

        plt.tight_layout()
        plt.show()


class Cluster:
    def __init__(self, params, feature_names):
        self.params = params
        self.feature_names = feature_names
        return

    def cluster(self, X, y = None, algo = False):
        # from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler()
        # X = scaler.fit_transform(X)

        if algo == 'HDBSCAN':
            cluster_labels = self._HDBSCAN(X, y)
        elif algo == 'OPTICS':
            cluster_labels = self._OPTICS(X, y)
        else:
            raise Exception(f'{algo} not configured')
        return cluster_labels

    def _HDBSCAN(self, X, y=None):
        if y is not None:
            y = y * 10
            X = np.hstack([X, y[:,np.newaxis]])
        print("Waiting for HDBSCAN clustering...", end='')
        min_samples, min_cluster_size = self.params['min_samples'], self.params['min_cluster_size']
        print(min_samples)
        hdbscan_labels = hdbscan.HDBSCAN(min_samples=min_samples,
                                         min_cluster_size=min_cluster_size).fit_predict(X)
        self.clustered = (hdbscan_labels >= 0)
        print("Done!")
        return hdbscan_labels

    def _OPTICS(self, X, y=None):
        from sklearn.cluster import OPTICS
        if y is not None:
            y = y * 10  # bias 강조
            X = np.hstack([X, y[:, np.newaxis]])

        print("Waiting for OPTICS clustering...", end='')
        # min_samples = self.params['min_samples']
        min_samples = 120
        print(min_samples)

        optics = OPTICS(min_samples=min_samples, metric='euclidean')
        optics_labels = optics.fit_predict(X)

        self.clustered = (optics_labels >= 0)
        print("Done!")
        return optics_labels

    def plot_scatter(self, X, cluster_labels):
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=X[~self.clustered, 0],
                        y=X[~self.clustered, 1],
                        color=(0.5, 0.5, 0.5), s=6, alpha=0.5)
        sns.scatterplot(x=X[self.clustered, 0],
                        y=X[self.clustered, 1],
                        hue=cluster_labels[self.clustered],
                        palette='tab20')
        # plt.legend()
        plt.xlabel('Dim1')
        plt.ylabel('Dim2')
        plt.grid()
        plt.tight_layout()
        plt.show()

    import matplotlib.pyplot as plt
    import numpy as np

    def plot_scatter_with_arrows(self, X, cluster_labels):
        """
        Plots a 2D scatter plot with arrows showing transitions over time.

        Parameters:
            X: np.ndarray of shape (N, 2) – reduced 2D coordinates
            t: optional np.ndarray of shape (N,) – time index to sort by
        """
        # Scatter points
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=X[~self.clustered, 0],
                        y=X[~self.clustered, 1],
                        color=(0.5, 0.5, 0.5), s=6, alpha=0.5)
        sns.scatterplot(x=X[self.clustered, 0],
                        y=X[self.clustered, 1],
                        hue=cluster_labels[self.clustered],
                        palette='tab20')

        # Draw arrows between consecutive points
        dx = X[1:, 0] - X[:-1, 0]
        dy = X[1:, 1] - X[:-1, 1]
        plt.quiver(
            X[:-1, 0], X[:-1, 1],  # arrow start positions
            dx, dy,  # arrow direction
            angles='xy',
            scale_units='xy',
            scale=1,
            color='red',
            width=0.003,
            headwidth=3,
            headlength=4,
            alpha=0.8
        )

        plt.xlabel('Dim1')
        plt.ylabel('Dim2')
        plt.grid()
        plt.tight_layout()
        plt.show()

    def groupby(self, X, cluster_labels):
        X = pd.DataFrame(X, columns = self.feature_names)
        X['label'] = cluster_labels
        X_mean = X.groupby(by=['label']).mean()
        X_std = X.groupby(by=['label']).std()
        result = {'mean': X_mean, 'std': X_std}
        return result

    def plot_violin(self, X, cluster_labels):
        """
        Plots a violin plots of feature between groups. Recommended for two groups
        Args:
            data_labeled: [pd.DataFrame] Data to be plotted, with target label included
            target: [str] Target
        """
        X = pd.DataFrame(X, columns=self.feature_names)
        X['label'] = cluster_labels
        for feature in X.columns[:-1]:
            plt.figure(figsize=(8, 6))
            sns.violinplot(data=X, x='label', y=feature)
            plt.xlabel('Clustered group')
            plt.ylabel('Feature distribution')
            plt.title("Clustered results {}".format(feature))
            plt.grid()
            plt.tight_layout()
            plt.show()
