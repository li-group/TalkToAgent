import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import torch

class LIME:
    def __init__(self, model = None):
        self.model = model

    def explain(self, X, feature_names):
        print("Explaining key variables via LIME...")
        self.X = X

        explainer = LimeTabularExplainer(
            self.X,
            feature_names=feature_names,
            mode="regression"
        )
        # predict_fn = lambda x: self.model.predict(x).reshape(-1)
        predict_fn = lambda x: self.model(torch.tensor(x, dtype=torch.float32)).detach().numpy()

        explanations = []
        for i in range(self.X.shape[0]):
            explanation = explainer.explain_instance(
                self.X[i],
                predict_fn,
                num_features=len(feature_names)
            )
            explanations.append(explanation)

        feature_importance = {feature: [] for feature in feature_names}

        # Collect feature attributions from every explanation
        for explanation in explanations:
            local_importance = explanation.local_exp[0]
            for feature_idx, weight in local_importance:
                feature = feature_names[feature_idx]
                feature_importance[feature].append(abs(weight))

        lime_values = pd.DataFrame(feature_importance)
        return lime_values

    def plot(self, lime_values):
        # 평균 절대값 계산
        mean_importance = lime_values.abs().mean().sort_values(ascending=True)

        # 시각화
        plt.figure(figsize=(10, 6))
        mean_importance.plot(kind='barh')
        plt.title('LIME Feature Importance (Mean Absolute Value)')
        plt.xlabel('Mean |Importance|')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()

