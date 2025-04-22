import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
from tqdm import tqdm

# Define a simple Neural Network for regression
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Single output for regression

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Integrated Gradients-based Feature Explainer
class IntegratedGradientsExplainer:
    def __init__(self, X, y, n_bg, model=None):
        self.X = X
        self.y = y
        self.X_torch = torch.tensor(X.values, dtype=torch.float32)
        self.y_torch = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)  # Ensure proper shape
        self.n_bg = n_bg
        self.bg = X.sample(n_bg, random_state = 21)

        if model is None:
            print("Training new PyTorch model...")
            self.model = self.train_model()
            print("Training complete!")
        else:
            self.model = model

        self.model.eval()  # Set model to evaluation mode

        # Initialize Integrated Gradients
        self.ig = IntegratedGradients(self.model)

    def train_model(self, epochs=100, lr=0.01):
        """ Train a simple neural network on the given data """
        input_dim = self.X_torch.shape[1]
        model = SimpleNN(input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(self.X_torch)
            loss = criterion(outputs, self.y_torch)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        return model

    def explain(self):
        print("Explaining key variable through Integrated gradients...")
        feature_importance = {feature: [] for feature in self.X.columns}

        # Calculate Integrated Gradients for every instance and every background sample
        for i in tqdm(range(len(self.bg))):
            baseline = torch.tensor(self.bg[i:i+1].values)
            baseline = baseline.to(torch.float32)

            for j in range(self.X.values.shape[0]):
                input_tensor = self.X_torch[j].unsqueeze(0)
                input_tensor = input_tensor.to(torch.float32)

                # Compute Integrated Gradients
                attributions = self.ig.attribute(input_tensor, baseline, target=None).detach().numpy()
                abs_attributions = np.abs(attributions).flatten()

                # Store feature importance
                for k, feature in enumerate(self.X.columns):
                    feature_importance[feature].append(abs_attributions[k])

        for k, feature in enumerate(self.X.columns):
            feature_importance[feature] = np.array(feature_importance[feature])

        # Convert to DataFrame
        ig_values = pd.DataFrame(feature_importance)

        # Take absolute and mean over all background samples
        ig_values_mean = np.abs(ig_values.values.reshape(len(self.bg), self.X.values.shape[0], len(self.X.columns))).mean(axis = 0)
        return ig_values_mean
