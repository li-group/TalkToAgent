from explainer.base_explainer import Base_explainer
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import torch

class LIME(Base_explainer):
    def __init__(self, model, bg, feature_names, target, algo, env_params):
        super(LIME, self).__init__(model, bg, feature_names, target, algo, env_params)

        self.explainer = LimeTabularExplainer(
            self.bg,
            feature_names=feature_names,
            mode="regression"
        )
        # self.predictor = lambda x: self.model.predict(x).reshape(-1)
        self.predictor = lambda x: self.model(torch.tensor(x, dtype=torch.float32)).detach().numpy()

    def explain(self, X):
        print("Explaining key variables via LIME...")
        self.X = self._scale(X)

        explanations = []
        for i in range(self.X.shape[0]):
            explanation = self.explainer.explain_instance(
                self.X[i],
                self.predictor,
                num_features=len(self.feature_names)
            )
            explanations.append(explanation)

        feature_importance = {feature: [] for feature in self.feature_names}

        # Collect feature attributions from every explanation
        for explanation in explanations:
            local_importance = explanation.local_exp[0]
            for feature_idx, weight in local_importance:
                feature = self.feature_names[feature_idx]
                feature_importance[feature].append(abs(weight))

        lime_values = pd.DataFrame(feature_importance)
        return lime_values

    def plot(self, values, max_display = 10):
        mean_importance = values.abs().mean().sort_values(ascending=True)

        plt.figure(figsize=(10, 6))
        mean_importance.plot(kind='barh', color = 'yellowgreen')
        plt.title('LIME Feature Importance (Mean Absolute Value)')
        plt.xlabel('Mean |Importance|')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(self.savedir + '/LIME importance.png')
        plt.show()
